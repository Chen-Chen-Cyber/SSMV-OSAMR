"""Preflight or run the preserved author entry points with explicit data roots.

The closed entry point preserves the supplied plotting/evaluation workflow;
it is not advertised as an end-to-end pretraining command.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent


def resolve_from(root, value):
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def validate_split_roots(validation_root, test_root):
    """Reject identical or nested validation/test directories."""
    validation = Path(validation_root).resolve()
    test = Path(test_root).resolve()
    if validation == test or validation in test.parents or test in validation.parents:
        raise ValueError("Validation and test roots must be distinct, non-nested directories.")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", choices=("closed", "open"), required=True)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="Root containing the author's prepared data and checkpoints.")
    parser.add_argument("--rml2016a", default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--validation-root", default="open_set_data/validation/by_snr")
    parser.add_argument("--test-root", default="open_set_data/test/by_snr")
    parser.add_argument("--num-known-classes", type=int,
                        help="Explicit known-class count for the open workflow classifier.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of open-set training epochs; use 1 for an execution check.")
    parser.add_argument("--check", action="store_true", help="List missing inputs without loading arrays or training.")
    args = parser.parse_args()
    root = args.work_dir.expanduser().absolute().resolve()
    raw_dataset = resolve_from(root, args.rml2016a)
    val_root = resolve_from(root, args.validation_root)
    test_root = resolve_from(root, args.test_root)
    if args.workflow == "open":
        validate_split_roots(val_root, test_root)
    env = {
        "RADIO_SSL_WORKDIR": str(root),
        "RADIO_SSL_RML2016A": str(raw_dataset),
        "RADIO_SSL_OPEN_VAL_ROOT": str(val_root),
        "RADIO_SSL_OPEN_TEST_ROOT": str(test_root),
    }
    os.environ.update(env)
    if args.epochs < 1:
        raise ValueError("The epoch count must be positive.")
    os.environ["RADIO_SSL_EPOCHS"] = str(args.epochs)
    if args.num_known_classes is not None:
        if args.num_known_classes < 1:
            raise ValueError("The known-class count must be positive.")
        os.environ["RADIO_SSL_NUM_KNOWN_CLASSES"] = str(args.num_known_classes)
    os.environ.setdefault("MPLBACKEND", "Agg")
    script = HERE / ("1.py" if args.workflow == "closed" else "open.py")
    tree = ast.parse(script.read_text(encoding="utf-8"))
    required = {raw_dataset}


    main_block = [n for n in tree.body if isinstance(n, ast.If) and
                  isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                  and n.test.left.id == "__name__"][-1]
    for node in ast.walk(main_block):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.endswith((".npy", ".pt", ".pth")):
                required.add(resolve_from(root, node.value))
    if args.workflow == "open":
        required.update((val_root, test_root))
    missing = sorted(str(path) for path in required if not path.exists())
    report = {"workflow": args.workflow, "work_dir": str(root),
              "required_paths": sorted(map(str, required)), "missing_paths": missing,
              "note": "File existence is not evidence of sample independence or full reproducibility."}
    print(json.dumps(report, indent=2))
    if missing:
        return 2
    if args.check:
        return 0
    if args.workflow == "open" and args.num_known_classes is None:
        raise ValueError("Specify --num-known-classes from the manuscript configuration; the source default is 33.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The preserved experiment entry points require CUDA. Use demo.py for a CPU smoke test.")
    # Archive inputs/versions with each run, without placing them in the source tree.
    output = root / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(report, source_sha256=file_hash(script), python=sys.version,
                    torch=torch.__version__, cuda=torch.version.cuda,
                    device=torch.cuda.get_device_name(0), environment_paths=env,
                    num_known_classes=args.num_known_classes, epochs=args.epochs)
    (output / (args.workflow + "_run_metadata.json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.chdir(root)
    sys.path.insert(0, str(HERE))
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
