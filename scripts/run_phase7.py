"""Run Phase 7 advanced interpretability analyses."""

import json
import logging
import sys
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from calamr_interp.utils.model_loading import load_model_checkpoint
from calamr_interp.utils.data_loading import load_dataset, split_dataset
from calamr_interp.phase7_advanced import run_all


def make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    elif hasattr(obj, "item"):
        return obj.item()
    else:
        return str(obj)


def main():
    device = torch.device("cpu")
    model = load_model_checkpoint(
        "results/hybridgcn/best_model.pt", "HybridGCN", device=device
    )

    dataset = load_dataset()
    train_sub, val_sub, test_sub = split_dataset(dataset)

    train_data = [train_sub[i] for i in range(len(train_sub))]
    test_data = [test_sub[i] for i in range(len(test_sub))]

    logger.info(
        "Dataset: %d total, %d train, %d test",
        len(dataset), len(train_data), len(test_data),
    )

    results = run_all(
        model=model,
        dataset=dataset,
        train_dataset=train_data[:50],
        test_dataset=test_data,
        device=device,
        num_layers=3,
    )

    results_ser = make_serializable(results)
    with open("results/phase7_advanced.json", "w") as f:
        json.dump(results_ser, f, indent=2)
    logger.info("Results saved to results/phase7_advanced.json")

    # Print summary
    print("\n=== PHASE 7 SUMMARY ===")
    for key, val in results_ser.items():
        if isinstance(val, dict):
            print(f"\n{key}:")
            for k2, v2 in val.items():
                if isinstance(v2, (str, int, float, bool)):
                    print(f"  {k2}: {v2}")
                elif isinstance(v2, list) and len(v2) <= 5:
                    print(f"  {k2}: {v2}")
                elif isinstance(v2, list):
                    print(f"  {k2}: [{len(v2)} items]")
                else:
                    print(f"  {k2}: ...")
        elif isinstance(val, list):
            print(f"\n{key}: [{len(val)} items]")
            for item in val[:3]:
                print(f"  {item}")


if __name__ == "__main__":
    main()
