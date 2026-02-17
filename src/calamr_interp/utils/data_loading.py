"""Data loading utilities wrapping calamr_pyg dataset functions."""

from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Subset
from torch_geometric.data import Data

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[4] / "calamr" / "calamr-master" / "medhallu" / "v8" / "labeled"


def load_dataset(path: Optional[str] = None) -> List[Data]:
    """Load all .pt graph files from a directory.

    Args:
        path: Directory containing .pt files. Defaults to the standard medhallu labeled dir.

    Returns:
        List of PyG Data objects.
    """
    path = Path(path) if path else DEFAULT_DATA_DIR
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")

    data_files = sorted(path.glob("*.pt"))
    if not data_files:
        raise ValueError(f"No .pt files found in {path}")

    all_data = []
    for f in data_files:
        d = torch.load(f, weights_only=False)
        if isinstance(d, list):
            all_data.extend(d)
        elif isinstance(d, Data):
            all_data.append(d)
        else:
            raise ValueError(f"Unexpected data type in {f}: {type(d)}")

    return all_data


def split_dataset(
    dataset: List[Data],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Split dataset into train/val/test using the same split as calamr_pyg.

    Args:
        dataset: List of PyG Data objects.
        ratios: (train, val, test) ratios.
        seed: Random seed for reproducibility.

    Returns:
        (train_subset, val_subset, test_subset)
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, list(ratios), generator=gen)


def load_and_split(
    path: Optional[str] = None,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Load dataset and split into train/val/test.

    Args:
        path: Directory containing .pt files.
        ratios: (train, val, test) ratios.
        seed: Random seed.

    Returns:
        (train_subset, val_subset, test_subset)
    """
    dataset = load_dataset(path)
    return split_dataset(dataset, ratios, seed)
