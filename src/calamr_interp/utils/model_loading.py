"""Model loading utilities for trained GNN checkpoints."""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json

import torch
import torch.nn as nn
from torch_geometric.data import Data

# Ensure calamr_pyg is importable from sibling project
_CALAMR_PYG_SRC = Path(__file__).resolve().parents[4] / "calamr_pyg" / "src"
if str(_CALAMR_PYG_SRC) not in sys.path and _CALAMR_PYG_SRC.exists():
    sys.path.insert(0, str(_CALAMR_PYG_SRC))


# Model registry: name -> (module_path, class_name, default_kwargs)
MODEL_REGISTRY = {
    "GraphTransformer": {
        "class": "graph_transformer.GraphTransformer",
        "defaults": {
            "input_dim": 771,
            "hidden_dim": 256,
            "num_layers": 4,
            "edge_dim": 4,
            "num_heads": 8,
            "dropout": 0.2,
            "pooling": "both",
        },
    },
    "EdgeAwareGAT": {
        "class": "edge_aware_gat.EdgeAwareGAT",
        "defaults": {
            "input_dim": 771,
            "hidden_dim": 256,
            "num_layers": 3,
            "edge_dim": 4,
            "num_heads": 4,
            "dropout": 0.2,
            "pooling": "attention",
        },
    },
    "GPS": {
        "class": "gps.GPS",
        "defaults": {
            "input_dim": 771,
            "hidden_dim": 256,
            "num_layers": 4,
            "edge_dim": 4,
            "num_heads": 4,
            "dropout": 0.2,
            "pooling": "mean",
        },
    },
    "GATv2": {
        "class": "gatv2.GATv2",
        "defaults": {
            "input_dim": 771,
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.2,
            "pooling": "mean",
        },
    },
    "HybridGCN": {
        "class": "hybrid_gcn.HybridGCN",
        "defaults": {
            "input_dim": 771,
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.2,
            "pooling": "attention",
        },
    },
    "GIN": {
        "class": "gin.GIN",
        "defaults": {
            "input_dim": 771,
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.2,
            "pooling": "add",
        },
    },
}

CALAMR_PYG_ROOT = Path(__file__).resolve().parents[4] / "calamr_pyg"


class ExplainerModelWrapper(nn.Module):
    """Wraps models that take Data objects for PyG Explainer compatibility.

    PyG's Explainer framework calls model(x, edge_index, **kwargs), but our
    GNN models expect a single Data object. This wrapper bridges that gap.

    Args:
        model: Trained GNN model whose forward() takes a Data object.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch=None, edge_attr=None, **kwargs):
        data = Data(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        return self.model(data)


def _import_model_class(class_path: str):
    """Dynamically import a model class from calamr_pyg.models."""
    import importlib

    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(f"calamr_pyg.models.{module_name}")
    return getattr(module, class_name)


def create_model(model_name: str, **override_kwargs) -> nn.Module:
    """Create a model instance with default or overridden kwargs.

    Args:
        model_name: Name from MODEL_REGISTRY.
        **override_kwargs: Override default model parameters.

    Returns:
        Instantiated model.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    info = MODEL_REGISTRY[model_name]
    kwargs = {**info["defaults"], **override_kwargs}
    cls = _import_model_class(info["class"])
    return cls(**kwargs)


def load_model_checkpoint(
    checkpoint_path: str,
    model_name: str,
    device: Optional[torch.device] = None,
    **override_kwargs,
) -> nn.Module:
    """Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to best_model.pt state dict.
        model_name: Model architecture name.
        device: Device to load onto.
        **override_kwargs: Override model constructor params.

    Returns:
        Model with loaded weights, in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(model_name, **override_kwargs)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def find_checkpoints(results_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Find all trained model checkpoints in a results directory.

    Args:
        results_dir: Root results directory. Defaults to calamr_pyg/results/experiments/.

    Returns:
        Dict mapping model_name -> {"checkpoint_path": ..., "config": ..., "metrics": ...}
    """
    if results_dir is None:
        results_dir = CALAMR_PYG_ROOT / "results" / "experiments"
    else:
        results_dir = Path(results_dir)

    checkpoints = {}
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        for model_dir in sorted(exp_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            # Look for best_model.pt in subdirectories
            for pt_file in model_dir.rglob("best_model.pt"):
                # Try to find corresponding results.json
                results_json = pt_file.parent / "results.json"
                config = {}
                metrics = {}
                if results_json.exists():
                    with open(results_json) as f:
                        data = json.load(f)
                    config = data.get("training_config", {})
                    metrics = data.get("results", {})
                    name = data.get("model", model_dir.name)
                else:
                    name = model_dir.name

                checkpoints[name] = {
                    "checkpoint_path": str(pt_file),
                    "config": config,
                    "metrics": metrics,
                    "experiment_dir": str(exp_dir),
                }

    return checkpoints
