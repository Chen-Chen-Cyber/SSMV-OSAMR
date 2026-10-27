from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import gaussian_kde
from tqdm import tqdm


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _class_indices(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim > 1:
        labels = labels.argmax(dim=1)
    return labels.reshape(-1).long()


@torch.no_grad()
def collect_classifier_input_features(
    checkpoint_path,
    device,
    feature_dl,
    feature_dl_mask,
    build_model_func,
    pre_ck,
    unknown_label=None,
    use_lstm=False,
    feature_mode="fused",
):
    """Collect fused or branch-specific normalized 128-D features."""
    valid_modes = {"fused", "augmentation_branch", "cross_domain_branch"}
    if feature_mode not in valid_modes:
        raise ValueError(f"feature_mode must be one of {sorted(valid_modes)}")
    if use_lstm and feature_mode != "fused":
        raise ValueError("use_lstm is supported only when feature_mode='fused'.")
    (
        ft_model,
        classifier,
        _,
        _,
        _,
        _,
        lstm,
        _,
        _,
    ) = build_model_func(5e-4, device, chkpoint=pre_ck)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "encoder_state" in checkpoint:
            ft_model.load_state_dict(checkpoint["encoder_state"])
            if "classifier_state" in checkpoint:
                classifier.load_state_dict(checkpoint["classifier_state"])
        elif "model_state_dict" not in checkpoint:
            raise KeyError(
                "Expected 'encoder_state' or 'model_state_dict' in checkpoint."
            )
        # A pretraining checkpoint has already been loaded through pre_ck by
        # build_model_func, so no second state-dict load is needed here.

    ft_model.eval()
    classifier.eval()
    if use_lstm and lstm is not None:
        lstm.eval()

    if len(feature_dl) != len(feature_dl_mask):
        raise ValueError(
            "feature_dl and feature_dl_mask have different numbers of batches."
        )

    captured = []

    def capture_linear_input(module, inputs):
        # Capture the feature after LayerNorm and immediately before Linear.
        captured.append(inputs[0].detach().cpu())

    if not hasattr(classifier, "classifier"):
        raise AttributeError(
            "classifier must expose the final block as classifier.classifier."
        )

    classifier_layers = list(classifier.classifier.children())
    if not classifier_layers:
        raise ValueError("classifier.classifier contains no layers.")

    hook = None
    if feature_mode == "fused":
        hook = classifier_layers[-1].register_forward_pre_hook(
            capture_linear_input
        )
    labels_all = []

    try:
        iterator = zip(feature_dl, feature_dl_mask)
        for (raw, labels), (sequence, sequence_labels) in tqdm(
            iterator,
            total=len(feature_dl),
            desc="Collect classifier features",
        ):
            labels = _class_indices(labels)
            sequence_labels = _class_indices(sequence_labels)

            if not torch.equal(labels.cpu(), sequence_labels.cpu()):
                raise ValueError(
                    "The raw and sequence dataloaders are not sample-aligned."
                )

            raw = raw.float().to(device, non_blocking=True)
            sequence = sequence.float().to(device, non_blocking=True)

            # FinetuneEncoder.forward(x_inter, x_mask) is called in 1.py as
            # ft_model(sequence_data, raw_data).
            if feature_mode == "augmentation_branch":
                # This is the first branch used by FinetuneEncoder.forward.
                embeddings = ft_model.encoder(sequence)
            elif feature_mode == "cross_domain_branch":
                # This is the second branch used by FinetuneEncoder.forward.
                embeddings = ft_model.encoder_masked(raw)
            else:
                embeddings = ft_model(sequence, raw)

            if use_lstm and lstm is not None:
                embeddings = lstm(embeddings)
                if isinstance(embeddings, (tuple, list)):
                    embeddings = embeddings[0]

            if feature_mode == "fused":
                # Running the classifier triggers the hook immediately before
                # its LayerNorm/Linear block and captures the decision feature.
                classifier(embeddings)
            else:
                # Match target_classifier through every operation before the
                # final Linear layer: pooling, L2 normalization, LayerNorm.
                if embeddings.shape[1] != 128:
                    embeddings = embeddings.permute(0, 2, 1)
                branch_features = classifier.pool(embeddings).squeeze(-1)
                branch_features = F.normalize(branch_features, dim=1)
                for layer in classifier_layers[:-1]:
                    branch_features = layer(branch_features)
                captured.append(branch_features.detach().cpu())
            labels_all.append(labels.cpu())
    finally:
        if hook is not None:
            hook.remove()

    if not captured:
        raise ValueError("No features were collected from the dataloaders.")

    features = torch.cat(captured, dim=0).numpy().astype(np.float32)
    labels = torch.cat(labels_all, dim=0).numpy().astype(np.int64)

    if unknown_label is not None:
        known = labels != int(unknown_label)
        features = features[known]
        labels = labels[known]

    if len(np.unique(labels)) < 2:
        raise ValueError("At least two known classes are required.")

    return features, labels


def compute_similarity_distributions(
    features,
    labels,
    pairs_per_class=5000,
    seed=2026,
):
    """Return class-balanced intra-class and inter-class cosine similarities."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)

    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("Expected features=(N, D) and labels=(N,).")

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-12)

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    indices = {label: np.flatnonzero(labels == label) for label in classes}
    classes = np.asarray([label for label in classes if len(indices[label]) >= 2])

    if len(classes) < 2:
        raise ValueError("At least two classes with two samples each are required.")

    intra_parts = []
    for label in classes:
        class_idx = indices[label]
        first = rng.choice(class_idx, pairs_per_class, replace=True)
        second = rng.choice(class_idx, pairs_per_class, replace=True)
        duplicate = first == second
        while duplicate.any():
            second[duplicate] = rng.choice(class_idx, duplicate.sum(), replace=True)
            duplicate = first == second
        intra_parts.append(np.einsum("ij,ij->i", features[first], features[second]))

    intra = np.concatenate(intra_parts).astype(np.float32)
    inter_parts = []

    # Sample the same number of inter-class pairs for every class.
    for label in classes:
        other_classes = classes[classes != label]
        other_labels = rng.choice(other_classes, pairs_per_class, replace=True)
        first = rng.choice(indices[label], pairs_per_class, replace=True)
        second = np.fromiter(
            (rng.choice(indices[other]) for other in other_labels),
            dtype=np.int64,
            count=pairs_per_class,
        )
        inter_parts.append(np.einsum("ij,ij->i", features[first], features[second]))

    inter = np.concatenate(inter_parts).astype(np.float32)
    return intra, inter


def _density(values, grid, bw_method=None):
    values = np.asarray(values, dtype=np.float64)
    if np.std(values) < 1e-8:
        values = values + np.linspace(-1e-6, 1e-6, len(values))
    return gaussian_kde(values, bw_method=bw_method)(grid)


def plot_similarity_density(
    intra_similarity,
    inter_similarity,
    save_path,
    show=False,
    y_max=7.0,
    bw_method=None,
):
    """Plot and export a publication-style similarity density panel."""
    intra_similarity = np.asarray(intra_similarity)
    inter_similarity = np.asarray(inter_similarity)
    grid = np.linspace(-1.0, 1.0, 600)

    intra_density = _density(intra_similarity, grid, bw_method=bw_method)
    inter_density = _density(inter_similarity, grid, bw_method=bw_method)
    overlap = float(np.trapz(np.minimum(intra_density, inter_density), grid))

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    intra_color = "#2A788E"
    inter_color = "#D97721"

    ax.fill_between(
        grid,
        intra_density,
        color=intra_color,
        alpha=0.28,
        linewidth=0,
    )
    ax.plot(grid, intra_density, color=intra_color, lw=1.6, label="Intra-class")
    ax.fill_between(
        grid,
        inter_density,
        color=inter_color,
        alpha=0.25,
        linewidth=0,
    )
    ax.plot(grid, inter_density, color=inter_color, lw=1.6, label="Inter-class")

    ax.axvline(
        intra_similarity.mean(),
        color=intra_color,
        lw=0.9,
        ls="--",
    )
    ax.axvline(
        inter_similarity.mean(),
        color=inter_color,
        lw=0.9,
        ls="--",
    )

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Probability density")
    ax.legend(loc="upper left")
    ax.text(
        0.62,
        0.95,
        f"Overlap = {overlap:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )
    ax.tick_params(direction="out", width=0.8, length=3)
    fig.tight_layout(pad=0.5)

    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return overlap


def plot_branch_similarity_comparison(
    checkpoint_path,
    device,
    feature_dl,
    feature_dl_mask,
    build_model_func,
    pre_ck,
    save_path="./figures/branch_similarity_comparison.png",
    unknown_label=None,
    pairs_per_class=5000,
    seed=2026,
    y_max=7.0,
    bw_method=None,
    show=False,
):
    """Create a three-panel comparison with shared axes and one legend."""
    specifications = [
        ("augmentation_branch", "Augmentation-domain branch"),
        ("cross_domain_branch", "Cross-domain branch"),
        ("fused", "Fused representation"),
    ]
    results = {}

    for mode, title in specifications:
        features, labels = collect_classifier_input_features(
            checkpoint_path=checkpoint_path,
            device=device,
            feature_dl=feature_dl,
            feature_dl_mask=feature_dl_mask,
            build_model_func=build_model_func,
            pre_ck=pre_ck,
            unknown_label=unknown_label,
            use_lstm=False,
            feature_mode=mode,
        )
        intra, inter = compute_similarity_distributions(
            features=features,
            labels=labels,
            pairs_per_class=pairs_per_class,
            seed=seed,
        )
        results[mode] = {
            "title": title,
            "features": features,
            "labels": labels,
            "intra_similarity": intra,
            "inter_similarity": inter,
        }

    grid = np.linspace(-1.0, 1.0, 600)
    intra_color = "#2A788E"
    inter_color = "#D97721"
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.35),
        sharex=True,
        sharey=True,
    )

    for panel_index, (ax, (mode, title)) in enumerate(
        zip(axes, specifications)
    ):
        intra = results[mode]["intra_similarity"]
        inter = results[mode]["inter_similarity"]
        intra_density = _density(intra, grid, bw_method=bw_method)
        inter_density = _density(inter, grid, bw_method=bw_method)
        overlap = float(
            np.trapz(np.minimum(intra_density, inter_density), grid)
        )
        results[mode]["overlap"] = overlap

        ax.fill_between(
            grid,
            intra_density,
            color=intra_color,
            alpha=0.28,
            linewidth=0,
        )
        ax.plot(
            grid,
            intra_density,
            color=intra_color,
            lw=1.5,
            ls="-",
            label="Intra-class pairs",
        )
        ax.fill_between(
            grid,
            inter_density,
            color=inter_color,
            alpha=0.23,
            linewidth=0,
        )
        ax.plot(
            grid,
            inter_density,
            color=inter_color,
            lw=1.5,
            ls="--",
            label="Inter-class pairs",
        )
        ax.axvline(intra.mean(), color=intra_color, lw=0.8, ls=":")
        ax.axvline(inter.mean(), color=inter_color, lw=0.8, ls=":")

        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Cosine similarity")
        ax.tick_params(direction="out", width=0.8, length=3)
        ax.text(
            0.53,
            0.92,
            f"OVL = {overlap:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
        )
        ax.text(
            -0.16,
            1.03,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )

    axes[0].set_ylabel("Probability density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=False,
        handlelength=2.8,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.19, top=0.78, wspace=0.12)

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")

    source_data = {}
    for mode, _ in specifications:
        source_data[f"{mode}_labels"] = results[mode]["labels"]
        source_data[f"{mode}_intra_similarity"] = results[mode][
            "intra_similarity"
        ]
        source_data[f"{mode}_inter_similarity"] = results[mode][
            "inter_similarity"
        ]
    np.savez_compressed(
        stem.with_name(stem.name + "_source_data").with_suffix(".npz"),
        **source_data,
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    print("Branch similarity comparison")
    for mode, title in specifications:
        print(f"{title}: OVL={results[mode]['overlap']:.4f}")
    print(f"Figure saved to: {stem.with_suffix('.png')}")
    return results


def plot_feature_similarity_from_checkpoint(
    checkpoint_path,
    device,
    feature_dl,
    feature_dl_mask,
    build_model_func,
    pre_ck,
    save_path="./figures/feature_similarity_density.png",
    unknown_label=None,
    pairs_per_class=5000,
    seed=2026,
    use_lstm=False,
    feature_mode="fused",
    show=False,
):
    """Load a trained checkpoint, extract features, and create the figure."""
    features, labels = collect_classifier_input_features(
        checkpoint_path=checkpoint_path,
        device=device,
        feature_dl=feature_dl,
        feature_dl_mask=feature_dl_mask,
        build_model_func=build_model_func,
        pre_ck=pre_ck,
        unknown_label=unknown_label,
        use_lstm=use_lstm,
        feature_mode=feature_mode,
    )

    intra, inter = compute_similarity_distributions(
        features=features,
        labels=labels,
        pairs_per_class=pairs_per_class,
        seed=seed,
    )

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    np.savez_compressed(
        stem.with_name(stem.name + "_source_data").with_suffix(".npz"),
        features=features,
        labels=labels,
        intra_similarity=intra,
        inter_similarity=inter,
    )

    overlap = plot_similarity_density(
        intra_similarity=intra,
        inter_similarity=inter,
        save_path=save_path,
        show=show,
    )

    summary_path = stem.with_name(stem.name + "_summary").with_suffix(".csv")
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["samples", len(labels)])
        writer.writerow(["classes", len(np.unique(labels))])
        writer.writerow(["intra_mean", float(intra.mean())])
        writer.writerow(["inter_mean", float(inter.mean())])
        writer.writerow(["density_overlap", overlap])

    print(f"Features: {features.shape}; classes: {len(np.unique(labels))}")
    print(f"Intra-class mean similarity: {intra.mean():.4f}")
    print(f"Inter-class mean similarity: {inter.mean():.4f}")
    print(f"Density overlap: {overlap:.4f}")
    print(f"Figure saved to: {stem.with_suffix('.png')}")

    return {
        "features": features,
        "labels": labels,
        "intra_similarity": intra,
        "inter_similarity": inter,
        "overlap": overlap,
    }
