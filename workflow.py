"""Run pretraining, linear evaluation or open-set learning from a JSON config."""
import argparse
import importlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from run_experiment import validate_split_roots


class ArrayView(Dataset):
    """Expose aligned NPY arrays without loading the complete dataset into RAM."""

    def __init__(self, values, labels, indices=None):
        self.values = values
        self.labels = labels
        self.indices = np.arange(len(values)) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        return (torch.from_numpy(np.array(self.values[row], dtype=np.float32, copy=True)),
                torch.tensor(int(self.labels[row]), dtype=torch.long))


def absolute_path(root, value):
    path = Path(value).expanduser()
    return Path(os.path.abspath(str(path if path.is_absolute() else root / path)))


def load_pair(specification, root, classes=None, allow_unknown=False):
    raw = np.load(absolute_path(root, specification["raw"]), mmap_mode="r", allow_pickle=False)
    sequence = np.load(absolute_path(root, specification["sequence"]), mmap_mode="r", allow_pickle=False)
    if raw.ndim != 3 or raw.shape[1] != 2:
        raise ValueError("Raw IQ must have shape [N, 2, L].")
    if sequence.ndim != 3 or sequence.shape[1] != 10:
        raise ValueError("Sequence data must have shape [N, 10, L].")
    if raw.shape[0] != sequence.shape[0] or raw.shape[-1] != sequence.shape[-1] or not len(raw):
        raise ValueError("Raw and sequence arrays must have matching, nonzero sample counts and lengths.")
    if classes is None:
        # Pretraining does not require class annotations.
        labels = np.zeros(len(raw), dtype=np.int64)
    else:
        labels = np.load(absolute_path(root, specification["labels"]), allow_pickle=False)
        if labels.ndim == 2:
            labels = labels.argmax(axis=1)
        if labels.ndim != 1 or len(labels) != len(raw):
            raise ValueError("Labels must have one entry per sample.")
        if not np.issubdtype(labels.dtype, np.integer) and not np.all(labels == np.floor(labels)):
            raise ValueError("Class indices must be integers.")
        labels = labels.astype(np.int64)
        maximum = classes if allow_unknown else classes - 1
        if labels.min() < 0 or labels.max() > maximum:
            raise ValueError("Labels lie outside the configured known/unknown class range.")
    return raw, sequence, labels


def pair_loaders(pair, batch_size, workers=0, indices=None):
    raw, sequence, labels = pair
    options = dict(batch_size=batch_size, shuffle=False, num_workers=workers)
    return (DataLoader(ArrayView(raw, labels, indices), **options),
            DataLoader(ArrayView(sequence, labels, indices), **options))


def initialize_model(module, device):
    """Initialize the same branches used by the preserved experiment scripts."""
    module.device = device
    module.model_z = module.Encoder1().to(device)
    module.model_LSTM = module.EncoderD().to(device)
    return module.pre_model(module.model_z, module.model_LSTM, module.model_r,
                            module.model_s, module.model_s1).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate inputs without training or writing outputs.")
    args = parser.parse_args()
    configuration = args.config.expanduser().absolute()
    cfg = json.loads(configuration.read_text(encoding="utf-8"))
    task = cfg["task"]
    if task not in ("pretrain", "linear", "open"):
        raise ValueError("task must be pretrain, linear or open.")
    root = absolute_path(configuration.parent, cfg.get("data_root", "."))
    output = absolute_path(root, cfg["output_dir"])
    batch = int(cfg.get("batch_size", 64))
    epochs = int(cfg.get("epochs", 100))
    classes = int(cfg.get("num_known_classes", 11))
    if min(batch, epochs, classes) < 1:
        raise ValueError("batch_size, epochs and num_known_classes must be positive.")
    if task == "open" and "num_known_classes" not in cfg:
        raise ValueError("The open task requires an explicit num_known_classes.")
    train_pair = load_pair(cfg["train"], root, classes=None if task == "pretrain" else classes)
    validation_pair = None
    if task != "pretrain":
        validation_pair = load_pair(cfg["validation"], root, classes=classes, allow_unknown=task == "open")
        for key in ("raw", "sequence", "labels"):
            if absolute_path(root, cfg["train"][key]) == absolute_path(root, cfg["validation"][key]):
                raise ValueError("Training and validation must use separate input files.")
        pretrained = absolute_path(root, cfg["pretrained_checkpoint"])
        if not pretrained.is_file():
            raise FileNotFoundError(pretrained)
    if task == "open":
        validation_root = absolute_path(root, cfg["validation_snr_root"])
        test_root = absolute_path(root, cfg["test_snr_root"])
        validate_split_roots(validation_root, test_root)
        snrs = cfg.get("snrs", [-4, 0, 4])
        if not snrs or any(not isinstance(snr, int) for snr in snrs):
            raise ValueError("snrs must be a nonempty list of integers.")
        for folder, split in ((validation_root, "val"), (test_root, "test")):
            for snr in snrs:
                names = [f"raw_{split}_{snr}dB_data.npy", f"sequence_{split}_{snr}dB_data.npy", f"{split}_{snr}dB_label.npy"]
                for name in names:
                    if not (folder / f"snr_{snr}dB" / name).is_file():
                        raise FileNotFoundError(folder / f"snr_{snr}dB" / name)
        os.environ["RADIO_SSL_SNRS"] = ",".join(map(str, snrs))
    report = {"task": task, "training_samples": len(train_pair[0]), "output_dir": str(output)}
    if args.check:
        print(json.dumps(report, indent=2))
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output_dir to avoid overwriting a previous run.")
    device = torch.device(cfg.get("device", "cuda:0"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Choose device='cpu' for a small execution check.")
    seed = int(cfg.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(cfg.get("cpu_threads", 1)))
    os.environ["RADIO_SSL_NUM_KNOWN_CLASSES"] = str(classes)
    module_name = "open" if task == "open" or cfg.get("module") == "open" else "1"
    module = importlib.import_module(module_name)
    model = initialize_model(module, device)
    output.mkdir(parents=True)
    (output / "configuration.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    workers = int(cfg.get("num_workers", 0))
    if task == "pretrain":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 0.01)),
                                      weight_decay=float(cfg.get("weight_decay", 0.001)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best, history = float("inf"), []
        for epoch in range(1, epochs + 1):
            indices = np.random.RandomState(seed + epoch).permutation(len(train_pair[0]))
            raw, sequence = pair_loaders(train_pair, batch, workers, indices)
            losses = module.model_pretrain(model, optimizer, scheduler, sequence, raw, device)
            values = [float(loss) for loss in losses]
            if not all(np.isfinite(values)):
                raise RuntimeError("Pretraining produced a non-finite loss.")
            history.append({"epoch": epoch, "loss": values[0], "sequence_loss": values[1], "raw_loss": values[2]})
            if values[0] < best:
                best = values[0]
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "configuration": cfg}, output / "best_pretrained.pt")
        report.update(history=history, checkpoint="best_pretrained.pt")
    elif task == "linear":
        checkpoint = torch.load(pretrained, map_location=device)
        ft, classifier, optimizer, scheduler, *_unused = module.build_model(0.001, device, checkpoint)
        best, history = -1.0, []
        for epoch in range(1, epochs + 1):
            indices = np.random.RandomState(seed + epoch).permutation(len(train_pair[0]))
            raw, sequence = pair_loaders(train_pair, batch, workers, indices)
            loss, accuracy, *_other = module.model_finetune_fast(ft, raw, sequence, device, scheduler, classifier, optimizer)
            raw_val, sequence_val = pair_loaders(validation_pair, batch, workers)
            metrics = module.model_validate(ft, sequence_val, raw_val, device, classifier=classifier)
            history.append({"epoch": epoch, "training_loss": float(loss), "training_accuracy": float(accuracy),
                            "validation_accuracy": float(metrics[0])})
            if metrics[0] > best:
                best = metrics[0]
                torch.save({"encoder_state": ft.state_dict(), "classifier_state": classifier.state_dict(),
                            "epoch": epoch, "configuration": cfg}, output / "best_linear.pt")
        report.update(history=history, checkpoint="best_linear.pt")
    else:
        train_raw, train_sequence = pair_loaders(train_pair, batch, workers)
        val_raw, val_sequence = pair_loaders(validation_pair, batch, workers)
        by_snr = module.build_loaders_by_snr(str(validation_root), split="val", snr_list=snrs, batch_size=batch)
        result = module.Trainer1(train_raw, train_sequence, val_raw, val_sequence, by_snr, device,
                                 str(output), pretrained_path=str(pretrained), test_dataset_root=str(test_root),
                                 epochs=epochs, batch_size=batch)
        report.update(overall={key: float(value) for key, value in result["overall"].items()},
                      checkpoint=str(Path(result["checkpoint_path"]).relative_to(output)),
                      spreadsheet=str(Path(result["excel_path"]).relative_to(output)))
    report.update(elapsed_seconds=round(time.perf_counter() - started, 3), seed=seed,
                  device=str(device), python_version=os.sys.version, torch_version=torch.__version__)
    (output / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
