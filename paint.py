from pathlib import Path
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix


RML2016_MODS = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]

RML2016_10B_MODS = [
    "8PSK",
    "AM-DSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]

OPEN_CLASS_NAMES = ['BPSK', 'PAM4', 'AM-SSB', 'CPFSK', 'GFSK','unknown']


def _as_class_indices(labels):
    if labels.ndim > 1:
        labels = labels.argmax(dim=1)
    return labels.long()


def _extract_logits(classifier_output):
    if torch.is_tensor(classifier_output):
        return classifier_output
    if isinstance(classifier_output, (tuple, list)):
        return classifier_output[0]
    if isinstance(classifier_output, dict):
        for key in ("logits", "output", "pred"):
            if key in classifier_output:
                return classifier_output[key]
    raise TypeError(
        "Unsupported classifier output. Expected a tensor, tuple/list, "
        "or a dict containing 'logits', 'output', or 'pred'."
    )


def _write_matrix_csv(path, matrix, class_names, value_format):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["True / Predicted", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *[value_format(value) for value in row]])


def plot_confusion_matrix_18db_loader(
    checkpoint_path,
    device,
    feature_dl,
    feature_dl_mask,
    build_model_func,
    pre_ck=None,
    save_path="confusion_matrix_18db.png",
    use_lstm=False,
    class_names=None,
    normalize="true",
    annotate=True,
    annotation_min=0,
    verify_paired_labels=True,
    flatten_before_classifier=False,
    show=True,
    title=None,
):
    """
    Evaluate a paired-input classifier and draw a publication-ready matrix.

    Parameters
    ----------
    normalize : {"true", "pred", "all", None}
        "true" is recommended: each row sums to 100%, so every cell is the
        percentage of a true class assigned to a predicted class.
    annotation_min : float
        Normalized cells below this percentage are left blank. Ignored when
        normalize is None, where zero-count cells are left blank.
    flatten_before_classifier : bool
        Keep False when the trained classifier expects sequence/channel
        dimensions and performs operations such as permute internally.
        Set True only for a classifier trained on flattened feature vectors.
    """
    if class_names is None:
        class_names = OPEN_CLASS_NAMES
    class_names = list(class_names)

    valid_normalizations = {"true", "pred", "all", None}
    if normalize not in valid_normalizations:
        raise ValueError(f"normalize must be one of {valid_normalizations}.")

    if hasattr(feature_dl, "__len__") and hasattr(feature_dl_mask, "__len__"):
        if len(feature_dl) != len(feature_dl_mask):
            raise ValueError(
                "feature_dl and feature_dl_mask contain different numbers "
                "of batches; zip() would silently truncate the evaluation."
            )

    (
        ft_model,
        classifier,
        cls_opt,
        cls_sched,
        ft_opt,
        ft_sched,
        LSTM,
        LSTM_optimizer,
        LSTM_scheduler,
    ) = build_model_func(5e-4, device, chkpoint=pre_ck)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ft_model.load_state_dict(checkpoint["encoder_state"])
    classifier.load_state_dict(checkpoint["classifier_state"])

    ft_model.eval()
    classifier.eval()
    if use_lstm and LSTM is not None:
        LSTM.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_index, (
            (data, labels),
            (data_mask, labels_mask),
        ) in enumerate(zip(feature_dl, feature_dl_mask)):
            data = data.float().to(device)
            data_mask = data_mask.float().to(device)
            labels = _as_class_indices(labels.to(device))
            labels_mask = _as_class_indices(labels_mask.to(device))

            if verify_paired_labels and not torch.equal(labels, labels_mask):
                raise ValueError(
                    f"Label mismatch in paired batch {batch_index}. Ensure "
                    "both DataLoaders use the same sample order and shuffle=False."
                )

            features = ft_model(data_mask, data)
            if use_lstm and LSTM is not None:
                features = LSTM(features)
                if isinstance(features, (tuple, list)):
                    features = features[0]

            if flatten_before_classifier and features.ndim > 2:
                features = features.reshape(features.size(0), -1)

            try:
                logits = _extract_logits(classifier(features))
            except RuntimeError as error:
                raise RuntimeError(
                    "Classifier inference failed with input shape "
                    f"{tuple(features.shape)}. The classifier must receive the "
                    "same feature shape used during training. Keep "
                    "flatten_before_classifier=False for sequence-based "
                    "classifiers, or set it to True only if training flattened "
                    "the encoder output."
                ) from error
            predictions = logits.argmax(dim=1)

            y_true.append(labels.cpu().numpy())
            y_pred.append(predictions.cpu().numpy())

    if not y_true:
        raise ValueError("No samples were produced by the DataLoaders.")

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    label_ids = np.arange(len(class_names))
    unexpected = np.setdiff1d(
        np.union1d(np.unique(y_true), np.unique(y_pred)),
        label_ids,
    )
    if len(unexpected):
        raise ValueError(
            "Found labels outside the class_names range: "
            f"{unexpected.tolist()}."
        )

    counts = confusion_matrix(y_true, y_pred, labels=label_ids)
    overall_accuracy = accuracy_score(y_true, y_pred)

    if normalize == "true":
        denominator = counts.sum(axis=1, keepdims=True)
        colorbar_label = "Samples per true class (%)"
    elif normalize == "pred":
        denominator = counts.sum(axis=0, keepdims=True)
        colorbar_label = "Samples per predicted class (%)"
    elif normalize == "all":
        denominator = np.array([[counts.sum()]])
        colorbar_label = "All samples (%)"
    else:
        denominator = None
        colorbar_label = "Number of samples"

    if denominator is None:
        display_matrix = counts.astype(float)
        vmax = max(1.0, float(display_matrix.max()))
    else:
        display_matrix = np.divide(
            counts,
            denominator,
            out=np.zeros_like(counts, dtype=float),
            where=denominator != 0,
        ) * 100.0
        vmax = 100.0

    # Restrained, perceptually ordered teal palette suitable for print.
    nature_teal = LinearSegmentedColormap.from_list(
        "nature_teal",
        ["#F7F8F8", "#DDE9E8", "#A9CFCC", "#66AAA4", "#277C76", "#0B4744"],
        N=256,
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    # 88 mm wide: Nature Communications single-column figure.
    fig, ax = plt.subplots(
        figsize=(88 / 25.4, 105 / 25.4),
        constrained_layout=True,
    )
    image = ax.imshow(
        display_matrix,
        cmap=nature_teal,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xlabel("Predicted label", fontsize=7)
    ax.set_ylabel("True label", fontsize=7)
    ax.set_xticks(label_ids)
    ax.set_yticks(label_ids)
    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=5.5,
    )
    ax.set_yticklabels(class_names, fontsize=5.5)
    ax.tick_params(axis="both", which="major", length=2.5, width=0.7, pad=1.5)

    # Fine white cell boundaries retain structure without a heavy grid.
    ax.set_xticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    if title:
        ax.set_title(title, fontsize=7, fontweight="bold", pad=4)

    if annotate:
        text_threshold = 0.52 * vmax
        for row in range(display_matrix.shape[0]):
            for column in range(display_matrix.shape[1]):
                value = display_matrix[row, column]
                if normalize is None:
                    if counts[row, column] == 0:
                        continue
                    text = f"{counts[row, column]:d}"
                else:
                    if value < annotation_min:
                        continue
                    text = f"{value:.1f}"

                ax.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value >= text_threshold else "#202020",
                )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    colorbar.set_label(colorbar_label, fontsize=6)
    colorbar.ax.tick_params(labelsize=5.5, length=2, width=0.6)
    colorbar.outline.set_linewidth(0.6)

    base_path = Path(save_path)
    if base_path.suffix:
        base_path = base_path.with_suffix("")
    base_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(base_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")

    _write_matrix_csv(
        base_path.with_name(base_path.name + "_counts.csv"),
        counts,
        class_names,
        lambda value: f"{int(value)}",
    )
    _write_matrix_csv(
        base_path.with_name(base_path.name + "_normalized.csv"),
        display_matrix,
        class_names,
        lambda value: f"{float(value):.4f}",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Samples: {len(y_true)}")
    print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Confusion matrix saved to: {base_path}.png/.pdf/.svg")

    return {
        "counts": counts,
        "matrix": display_matrix,
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": overall_accuracy,
        "class_names": class_names,
    }
