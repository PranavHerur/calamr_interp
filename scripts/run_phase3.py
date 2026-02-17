#!/usr/bin/env python3
"""CLI runner for Phase 3: Ablation Studies.

Usage:
    python scripts/run_phase3.py --epochs 10 --seeds 42 123 456
    python scripts/run_phase3.py --ablations A1_no_sbert A3_no_alignment_edges --epochs 50
    python scripts/run_phase3.py --models GraphTransformer EdgeAwareGAT GPS --epochs 50
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import numpy as np

from calamr_interp.utils.data_loading import load_and_split
from calamr_interp.utils.model_loading import create_model
from calamr_interp.phase3_ablations import AblationRunner, ABLATION_TYPES


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["GraphTransformer", "EdgeAwareGAT", "GPS"],
        help="Model names to evaluate",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=None,
        help="Ablation types (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per run")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456, 789, 1024],
        help="Random seeds",
    )
    parser.add_argument("--output-dir", type=str, default="results/ablations", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Data split seed")

    args = parser.parse_args()

    print("Loading dataset...")
    train_data, val_data, test_data = load_and_split(args.data_dir, seed=args.seed)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Build model constructors
    model_constructors = {}
    for name in args.models:
        model_constructors[name] = lambda n=name: create_model(n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    runner = AblationRunner(
        models=model_constructors,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        device=device,
        output_dir=args.output_dir,
    )

    print(f"\nRunning ablations: {args.ablations or 'all'}")
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")

    results_df = runner.run(
        ablation_types=args.ablations,
        seeds=args.seeds,
        epochs=args.epochs,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)

    analysis = AblationRunner.analyze_results(results_df)

    print("\n--- F1 Scores (mean +/- std) ---")
    summary = analysis["summary"]
    for _, row in summary.iterrows():
        print(f"  {row['model']:20s} | {row['ablation']:25s} | F1={row['f1_mean']:.3f} +/- {row['f1_std']:.3f}")

    print("\n--- Delta F1 from Full Model ---")
    delta = analysis["delta_f1"]
    for _, row in delta[delta["ablation"] != "full"].iterrows():
        sign = "+" if row["delta_f1"] > 0 else ""
        print(f"  {row['model']:20s} | {row['ablation']:25s} | {sign}{row['delta_f1']:.3f}")

    if not analysis["significance"].empty:
        print("\n--- Statistical Significance (Bonferroni corrected) ---")
        for _, row in analysis["significance"].iterrows():
            sig = "*" if row.get("significant", False) else ""
            print(f"  {row['model']:20s} | {row['ablation']:25s} | p={row.get('p_corrected', row['p_value']):.4f} {sig}")

    print(f"\nResults saved to {args.output_dir}/ablation_results.csv")


if __name__ == "__main__":
    main()
