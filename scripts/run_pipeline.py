#!/usr/bin/env python3
"""Unified interpretability pipeline for GNN hallucination detectors.

Runs selected analysis phases on a given dataset and model checkpoint,
saving JSON results to the output directory. Designed for headless
execution on Linux servers.

Usage:
    # Run all phases on the default dataset
    python scripts/run_pipeline.py \\
        --checkpoint results/hybridgcn/best_model.pt \\
        --model-name HybridGCN

    # Run specific phases on a custom dataset
    python scripts/run_pipeline.py \\
        --data-dir /path/to/dataset/ \\
        --checkpoint /path/to/best_model.pt \\
        --model-name HybridGCN \\
        --phases 1 2 5 7 \\
        --output-dir results/my_dataset/

    # Phase 3 (ablation) requires retraining — use run_phase3.py instead.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Headless matplotlib before any imports that touch it
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from calamr_interp.utils.data_loading import load_dataset, split_dataset
from calamr_interp.utils.model_loading import load_model_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


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


def save_results(results: dict, path: Path, phase_name: str):
    """Save phase results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    logger.info("Saved %s results to %s", phase_name, path)


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


def run_phase1(dataset, output_dir):
    """Phase 1: Statistical baselines & feature distributions."""
    from calamr_interp.phase1_baselines import (
        GraphFeatureExtractor, StatisticalBaseline, distribution_analysis,
    )

    extractor = GraphFeatureExtractor()
    features_df, labels = extractor.extract_batch(dataset)

    # Baselines
    baseline = StatisticalBaseline()
    cv_results = baseline.evaluate(features_df.values, labels)

    # Distribution analysis
    dist_df = distribution_analysis(features_df, labels)

    # Feature importance
    importance_df = baseline.feature_importance(
        features_df.values, labels, features_df.columns.tolist()
    )

    results = {
        "n_samples": len(dataset),
        "n_features": len(features_df.columns),
        "feature_names": features_df.columns.tolist(),
        "baselines": cv_results,
        "distribution_analysis": dist_df.to_dict("records"),
        "feature_importance": importance_df.to_dict("records"),
    }
    save_results(results, output_dir / "phase1_baselines.json", "Phase 1")
    return results


def run_phase2(dataset, output_dir):
    """Phase 2: Structural analysis."""
    from calamr_interp.phase2_structural import StructuralAnalyzer

    analyzer = StructuralAnalyzer()
    features_df, labels = analyzer.extract_batch(dataset, verbose=True)

    # Basic statistics by class
    truth_mask = labels == 0
    hallu_mask = labels == 1
    summary = {}
    for col in features_df.columns:
        summary[col] = {
            "truth_mean": float(features_df.loc[truth_mask, col].mean()),
            "truth_std": float(features_df.loc[truth_mask, col].std()),
            "hallu_mean": float(features_df.loc[hallu_mask, col].mean()),
            "hallu_std": float(features_df.loc[hallu_mask, col].std()),
        }

    results = {
        "n_samples": len(dataset),
        "n_features": len(features_df.columns),
        "feature_names": features_df.columns.tolist(),
        "feature_summary": summary,
    }
    save_results(results, output_dir / "phase2_structural.json", "Phase 2")
    return results


def run_phase4(model, dataset, device, output_dir):
    """Phase 4: Node & edge attribution."""
    from calamr_interp.phase4_attribution import (
        GradientSaliency, IntegratedGradientsAttribution,
        EdgeTypeImportanceAggregator,
    )

    # Gradient saliency
    gs = GradientSaliency(model, device=device)
    gs_df = EdgeTypeImportanceAggregator.batch_aggregate(
        dataset, gs.attribute, importance_key="edge_saliency"
    )

    gs_node_cats = []
    for data in dataset:
        attr = gs.attribute(data)
        cats = EdgeTypeImportanceAggregator.node_feature_saliency_by_category(
            attr["node_saliency"]
        )
        cats["label"] = data.y.item()
        gs_node_cats.append(cats)

    gs_node_df = {
        cat: {
            "mean": float(np.mean([r[cat] for r in gs_node_cats])),
            "std": float(np.std([r[cat] for r in gs_node_cats])),
        }
        for cat in ["node_type", "component_type", "sbert", "metadata"]
    }

    # Integrated gradients
    ig = IntegratedGradientsAttribution(model, n_steps=30, device=device)
    ig_df = EdgeTypeImportanceAggregator.batch_aggregate(
        dataset, ig.attribute, importance_key="edge_ig"
    )

    ig_node_cats = []
    for data in dataset:
        attr = ig.attribute(data)
        cats = EdgeTypeImportanceAggregator.node_feature_saliency_by_category(
            attr["node_ig"]
        )
        cats["label"] = data.y.item()
        ig_node_cats.append(cats)

    ig_node_df = {
        cat: {
            "mean": float(np.mean([r[cat] for r in ig_node_cats])),
            "std": float(np.std([r[cat] for r in ig_node_cats])),
        }
        for cat in ["node_type", "component_type", "sbert", "metadata"]
    }

    # Edge importance aggregation
    gs_edge_agg = {}
    ig_edge_agg = {}
    if len(gs_df) > 0:
        for col in ["internal/role", "alignment"]:
            if col in gs_df.columns:
                gs_edge_agg[col] = {
                    "mean": float(gs_df[col].mean()),
                    "std": float(gs_df[col].std()),
                }
    if len(ig_df) > 0:
        for col in ["internal/role", "alignment"]:
            if col in ig_df.columns:
                ig_edge_agg[col] = {
                    "mean": float(ig_df[col].mean()),
                    "std": float(ig_df[col].std()),
                }

    results = {
        "gradient_saliency": {
            "node_feature_categories": gs_node_df,
            "edge_importance_by_type": gs_edge_agg,
        },
        "integrated_gradients": {
            "node_feature_categories": ig_node_df,
            "edge_importance_by_type": ig_edge_agg,
        },
        "n_samples": len(dataset),
    }
    save_results(results, output_dir / "phase4_attribution.json", "Phase 4")
    return results


def run_phase5(model, dataset, device, output_dir):
    """Phase 5: Embedding space analysis."""
    from calamr_interp.phase5_embeddings import (
        LayerEmbeddingExtractor, ProbingClassifier,
        CKAAnalysis, cosine_similarity_analysis,
    )

    labels = np.array([d.y.item() for d in dataset])

    extractor = LayerEmbeddingExtractor(model, device=device)
    extractor.register_hooks()
    layer_embeddings = extractor.extract_graph_embeddings(dataset)
    extractor.clear_hooks()

    # Layer shapes
    layer_shapes = {k: list(v.shape) for k, v in layer_embeddings.items()}

    # Linear probing
    prober = ProbingClassifier(seed=42)
    probe_df = prober.probe(layer_embeddings, labels)

    # Cosine similarity
    cosine_results = {}
    for lname, emb in layer_embeddings.items():
        cosine_results[lname] = cosine_similarity_analysis(emb, labels)

    # CKA
    cka_matrix, cka_names = CKAAnalysis.compute_cka_matrix(layer_embeddings)

    results = {
        "layer_shapes": layer_shapes,
        "probing": probe_df.to_dict("records"),
        "cosine_similarity": cosine_results,
        "cka_matrix": cka_matrix.tolist(),
        "cka_layer_names": cka_names,
    }
    save_results(results, output_dir / "phase5_embeddings.json", "Phase 5")
    return results


def run_phase6(model, dataset, device, output_dir):
    """Phase 6: Case study selection & summary statistics."""
    from calamr_interp.phase6_case_studies import CaseStudySelector

    selector = CaseStudySelector(model, device=device)
    cases = selector.select(dataset, n_per_category=3)

    results = {}
    for cat, case_list in cases.items():
        results[cat] = [
            {
                "index": c["index"],
                "pred_prob": c["pred_prob"],
                "label": c["label"],
                "n_nodes": c["data"].x.shape[0],
                "n_edges": c["data"].edge_index.shape[1],
            }
            for c in case_list
        ]

    save_results(results, output_dir / "phase6_case_studies.json", "Phase 6")
    return results


def run_phase7(model, dataset, train_data, test_data, device, num_layers, output_dir):
    """Phase 7: Advanced interpretability (neuron concepts, explainers, counterfactuals)."""
    from calamr_interp.phase7_advanced import run_all

    results = run_all(
        model=model,
        dataset=dataset,
        train_dataset=train_data[:50],
        test_dataset=test_data,
        device=device,
        num_layers=num_layers,
    )
    save_results(results, output_dir / "phase7_advanced.json", "Phase 7")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

AVAILABLE_PHASES = [1, 2, 4, 5, 6, 7]


def main():
    parser = argparse.ArgumentParser(
        description="Run GNN interpretability pipeline on a dataset.",
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
        "--phases", nargs="+", type=int, default=AVAILABLE_PHASES,
        help=f"Phases to run (default: {AVAILABLE_PHASES}). Phase 3 requires run_phase3.py.",
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
        "--num-layers", type=int, default=3,
        help="Number of GNN message-passing layers (for Phase 7 GraphMask)",
    )
    parser.add_argument(
        "--split-seed", type=int, default=42,
        help="Random seed for train/val/test split",
    )
    args = parser.parse_args()

    # Validate phases
    for p in args.phases:
        if p not in AVAILABLE_PHASES:
            parser.error(f"Phase {p} not available. Choose from {AVAILABLE_PHASES}. Phase 3 uses run_phase3.py.")

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

    logger.info("=" * 60)
    logger.info("GNN Interpretability Pipeline")
    logger.info("=" * 60)
    logger.info("Data dir:    %s", args.data_dir or "(default)")
    logger.info("Checkpoint:  %s", args.checkpoint)
    logger.info("Model:       %s", args.model_name)
    logger.info("Phases:      %s", args.phases)
    logger.info("Output:      %s", output_dir)
    logger.info("Device:      %s", device)

    # Load data
    logger.info("Loading dataset...")
    dataset = load_dataset(args.data_dir)
    train_sub, val_sub, test_sub = split_dataset(dataset, seed=args.split_seed)
    train_data = [train_sub[i] for i in range(len(train_sub))]
    test_data = [test_sub[i] for i in range(len(test_sub))]
    logger.info("Dataset: %d total, %d train, %d test", len(dataset), len(train_data), len(test_data))

    # Load model
    logger.info("Loading model checkpoint...")
    model = load_model_checkpoint(args.checkpoint, args.model_name, device=device)
    logger.info("Model loaded: %s", type(model).__name__)

    # Run phases
    t_total = time.time()
    completed = []

    for phase in sorted(args.phases):
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE %d", phase)
        logger.info("=" * 60)
        t0 = time.time()

        try:
            if phase == 1:
                run_phase1(dataset, output_dir)
            elif phase == 2:
                run_phase2(dataset, output_dir)
            elif phase == 4:
                run_phase4(model, test_data, device, output_dir)
            elif phase == 5:
                run_phase5(model, dataset, device, output_dir)
            elif phase == 6:
                run_phase6(model, test_data, device, output_dir)
            elif phase == 7:
                run_phase7(model, dataset, train_data, test_data, device, args.num_layers, output_dir)

            elapsed = time.time() - t0
            logger.info("Phase %d completed in %.1fs", phase, elapsed)
            completed.append(phase)

        except Exception as e:
            logger.error("Phase %d FAILED: %s", phase, e, exc_info=True)

    # Summary
    elapsed_total = time.time() - t_total
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Completed phases: %s", completed)
    logger.info("Total time: %.1fs", elapsed_total)
    logger.info("Results in: %s", output_dir)

    # Save run metadata
    run_meta = {
        "data_dir": str(args.data_dir or "default"),
        "checkpoint": str(args.checkpoint),
        "model_name": args.model_name,
        "phases_requested": args.phases,
        "phases_completed": completed,
        "n_samples": len(dataset),
        "device": str(device),
        "split_seed": args.split_seed,
        "total_time_s": round(elapsed_total, 1),
    }
    save_results(run_meta, output_dir / "run_metadata.json", "metadata")


if __name__ == "__main__":
    main()
