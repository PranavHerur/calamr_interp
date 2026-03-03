#!/usr/bin/env python3
"""PCA comparison: prove the GNN learns discriminative representations.

Extracts raw input embeddings and GNN final-layer embeddings, computes
separability metrics for both, and produces a 4-panel comparison figure.

Usage:
    uv run python scripts/run_pca_comparison.py \\
        --checkpoint results/hybridgcn/best_model.pt \\
        --model-name HybridGCN

    uv run python scripts/run_pca_comparison.py \\
        --checkpoint results/hybridgcn/best_model.pt \\
        --model-name HybridGCN \\
        --output-dir results/my_experiment/ \\
        --n-pcs 15
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from calamr_interp.utils.data_loading import load_dataset
from calamr_interp.utils.model_loading import load_model_checkpoint
from calamr_interp.phase5_embeddings import LayerEmbeddingExtractor
from calamr_interp.pca_comparison import (
    RawEmbeddingExtractor,
    SeparabilityMetrics,
    PCAComparisonVisualizer,
    SupplementaryVisualizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pca_comparison")


def make_serializable(obj):
    """Recursively convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    elif hasattr(obj, "item"):
        return obj.item()
    else:
        return str(obj)


def find_last_conv_layer(layer_embeddings: dict) -> str:
    """Find the last (deepest) convolutional layer by name.

    Heuristic: pick the layer with the highest numeric suffix,
    or simply the last key in insertion order.
    """
    names = list(layer_embeddings.keys())
    if not names:
        raise ValueError("No layer embeddings extracted — check model hooks")

    # Try to find layers with numeric indices (e.g., gcn_layers.2)
    best_name = names[-1]
    best_idx = -1
    for name in names:
        parts = name.replace(".", "_").split("_")
        for part in reversed(parts):
            if part.isdigit():
                idx = int(part)
                if idx > best_idx:
                    best_idx = idx
                    best_name = name
                break
    return best_name


def main():
    parser = argparse.ArgumentParser(
        description="PCA comparison: raw input vs GNN embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing .pt graph files (default: medhallu/v8/labeled)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to best_model.pt checkpoint",
    )
    parser.add_argument(
        "--model-name", type=str, required=True,
        help="Model architecture name (e.g. HybridGCN, GraphTransformer)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/<model_name>/)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cpu or cuda (default: auto-detect)",
    )
    parser.add_argument(
        "--n-pcs", type=int, default=10,
        help="Number of principal components for FDA bar chart (default: 10)",
    )
    parser.add_argument(
        "--projection", type=str, default="umap", choices=["umap", "tsne"],
        help="2D projection method for scatter panels: umap or tsne (default: umap)",
    )
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results") / args.model_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PCA Comparison: Raw Input vs GNN Embeddings")
    logger.info("=" * 60)
    logger.info("Data dir:    %s", args.data_dir or "(default)")
    logger.info("Checkpoint:  %s", args.checkpoint)
    logger.info("Model:       %s", args.model_name)
    logger.info("Output:      %s", output_dir)
    logger.info("Device:      %s", device)
    logger.info("N PCs:       %d", args.n_pcs)

    # ---- Load data ----
    logger.info("Loading dataset...")
    dataset = load_dataset(args.data_dir)
    labels = np.array([d.y.item() for d in dataset])
    logger.info("Dataset: %d samples (%d truth, %d hallu)",
                len(dataset), (labels == 0).sum(), (labels == 1).sum())

    # ---- Extract raw input embeddings ----
    logger.info("Extracting raw input embeddings (mean-pooled 771-dim)...")
    t0 = time.time()
    raw_extractor = RawEmbeddingExtractor()
    raw_embeddings = raw_extractor.extract(dataset)
    logger.info("Raw embeddings: shape %s (%.1fs)", raw_embeddings.shape, time.time() - t0)

    # ---- Extract GNN embeddings ----
    logger.info("Loading model and extracting GNN embeddings...")
    t0 = time.time()
    model = load_model_checkpoint(args.checkpoint, args.model_name, device=device)
    logger.info("Model loaded: %s", type(model).__name__)

    extractor = LayerEmbeddingExtractor(model, device=device)
    extractor.register_hooks()
    layer_embeddings = extractor.extract_graph_embeddings(dataset)
    extractor.clear_hooks()

    layer_shapes = {k: list(v.shape) for k, v in layer_embeddings.items()}
    logger.info("Layer shapes: %s", layer_shapes)

    # Pick the last conv layer as "GNN Layer 2"
    last_layer = find_last_conv_layer(layer_embeddings)
    gnn_embeddings = layer_embeddings[last_layer]
    logger.info("Using layer '%s' as final GNN layer: shape %s (%.1fs)",
                last_layer, gnn_embeddings.shape, time.time() - t0)

    # ---- Compute metrics ----
    logger.info("Computing separability metrics...")
    t0 = time.time()

    raw_name = "Raw Input"
    gnn_name = f"GNN {last_layer}"

    raw_metrics = SeparabilityMetrics.compute_all(raw_embeddings, labels, args.n_pcs)
    gnn_metrics = SeparabilityMetrics.compute_all(gnn_embeddings, labels, args.n_pcs)
    logger.info("Metrics computed in %.1fs", time.time() - t0)

    # Log key comparisons
    logger.info("")
    logger.info("%-25s %-15s %-15s", "Metric", raw_name, gnn_name)
    logger.info("-" * 55)
    for key, label in [
        ("silhouette", "Silhouette"),
        ("linear_probe_f1_mean", "Linear Probe F1"),
        ("mahalanobis", "Mahalanobis Dist."),
        ("lda_cohens_d", "LDA Cohen's d"),
        ("fda_sum_top10", "Sum FDA (top 10)"),
    ]:
        logger.info("%-25s %-15.3f %-15.3f", label, raw_metrics[key], gnn_metrics[key])

    # ---- Generate figure ----
    logger.info("Generating 4-panel figure...")
    representations = {
        raw_name: raw_embeddings,
        gnn_name: gnn_embeddings,
    }

    visualizer = PCAComparisonVisualizer()
    suffix = f"_{args.projection}" if args.projection != "umap" else ""
    fig_path = str(fig_dir / f"fig_pca_comparison{suffix}.pdf")
    fig = visualizer.plot(
        representations, labels, save_path=fig_path,
        n_pcs=args.n_pcs, projection=args.projection,
    )
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    logger.info("Figure saved to %s (and .png)", fig_path)

    # ---- Supplementary figures ----
    logger.info("")
    logger.info("Generating supplementary figures...")

    # Build pair indices (consecutive hallu/truth)
    pair_indices = []
    for i in range(0, len(labels) - 1, 2):
        if labels[i] == 1 and labels[i + 1] == 0:
            pair_indices.append((i, i + 1))
    logger.info("Found %d truth/hallu pairs", len(pair_indices))

    # Get model predictions for confidence & misclassification figures
    logger.info("Running model inference for prediction probabilities...")
    pred_probs = np.zeros(len(dataset))
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataset):
            data_dev = data.clone().to(device)
            if not hasattr(data_dev, "batch") or data_dev.batch is None:
                data_dev.batch = torch.zeros(
                    data_dev.x.shape[0], dtype=torch.long, device=device
                )
            logit = model(data_dev)
            pred_probs[i] = torch.sigmoid(logit).cpu().item()
    preds = (pred_probs >= 0.5).astype(int)
    acc = (preds == labels).mean()
    logger.info("Model accuracy: %.1f%% (%d/%d correct)",
                acc * 100, (preds == labels).sum(), len(labels))

    sv = SupplementaryVisualizer()
    proj = args.projection

    # Fig S1: Pair displacement histogram
    logger.info("  [S1] Pair displacement histogram...")
    sv.plot_pair_displacement(
        raw_embeddings, gnn_embeddings, labels, pair_indices,
        save_path=str(fig_dir / f"fig_s1_pair_displacement{suffix}.pdf"),
    )
    _plt.close("all")

    # Fig S2: Layer-by-layer filmstrip
    logger.info("  [S2] Layer-by-layer filmstrip...")
    sv.plot_layer_filmstrip(
        layer_embeddings, labels, projection=proj, seed=42,
        save_path=str(fig_dir / f"fig_s2_layer_filmstrip{suffix}.pdf"),
    )
    _plt.close("all")

    # Fig S3: Decision boundary
    logger.info("  [S3] Decision boundary overlay...")
    sv.plot_decision_boundary(
        raw_embeddings, gnn_embeddings, labels,
        raw_name=raw_name, gnn_name=gnn_name,
        projection=proj, seed=42,
        save_path=str(fig_dir / f"fig_s3_decision_boundary{suffix}.pdf"),
    )
    _plt.close("all")

    # Fig S4: Confidence vs displacement
    logger.info("  [S4] Confidence vs displacement...")
    sv.plot_confidence_vs_displacement(
        gnn_embeddings, labels, pred_probs, pair_indices,
        save_path=str(fig_dir / f"fig_s4_confidence_displacement{suffix}.pdf"),
    )
    _plt.close("all")

    # Fig S5: Misclassification anatomy
    logger.info("  [S5] Misclassification anatomy...")
    sv.plot_misclassification_anatomy(
        raw_embeddings, gnn_embeddings, labels, pred_probs,
        raw_name=raw_name, gnn_name=gnn_name,
        projection=proj, seed=42,
        save_path=str(fig_dir / f"fig_s5_misclassification{suffix}.pdf"),
    )
    _plt.close("all")

    logger.info("All supplementary figures saved to %s", fig_dir)

    # ---- Save metrics JSON ----
    # Strip non-serializable fields (pca_coords, lda_projection)
    def clean_metrics(m: dict) -> dict:
        return {k: v for k, v in m.items() if k not in ("pca_coords", "lda_projection")}

    results = {
        "raw_input": clean_metrics(raw_metrics),
        "gnn_layer": clean_metrics(gnn_metrics),
        "gnn_layer_name": last_layer,
        "n_samples": len(dataset),
        "n_pcs": args.n_pcs,
        "raw_dim": int(raw_embeddings.shape[1]),
        "gnn_dim": int(gnn_embeddings.shape[1]),
        "all_layer_shapes": layer_shapes,
    }

    metrics_path = output_dir / "pca_comparison_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PCA COMPARISON COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
