"""Shared plotting helpers with consistent style and color schemes."""

from typing import Optional, List, Dict, Any

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

# Consistent color scheme
COLORS = {
    "hallu": "#e74c3c",       # Red for hallucination
    "truth": "#2ecc71",       # Green for truth
    "primary": "#3498db",     # Blue
    "secondary": "#9b59b6",   # Purple
    "accent": "#f39c12",      # Orange
    "neutral": "#95a5a6",     # Gray
}

EDGE_TYPE_COLORS = {
    "role": "#3498db",        # Blue
    "internal": "#95a5a6",    # Gray
    "alignment": "#e74c3c",   # Red
}

LABEL_NAMES = {0: "Truth", 1: "Hallucination"}
LABEL_COLORS = {0: COLORS["truth"], 1: COLORS["hallu"]}


def setup_style():
    """Set up consistent matplotlib/seaborn style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "font.family": "sans-serif",
    })


def label_palette() -> Dict[int, str]:
    """Return seaborn-compatible palette for binary labels."""
    return LABEL_COLORS


def violin_comparison(
    data_dict: Dict[str, Dict[int, np.ndarray]],
    title: str = "",
    ncols: int = 3,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create violin plots comparing feature distributions by label.

    Args:
        data_dict: {feature_name: {0: array_truth, 1: array_hallu}}
        title: Super title.
        ncols: Columns per row.
        figsize: Figure size.
        save_path: If set, save figure.

    Returns:
        matplotlib Figure.
    """
    import pandas as pd

    features = list(data_dict.keys())
    nrows = (len(features) + ncols - 1) // ncols
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx, feat_name in enumerate(features):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        records = []
        for label, values in data_dict[feat_name].items():
            for v in values:
                records.append({"value": v, "label": LABEL_NAMES[label]})
        df = pd.DataFrame(records)

        sns.violinplot(
            data=df, x="label", y="value", ax=ax,
            palette={"Truth": COLORS["truth"], "Hallucination": COLORS["hallu"]},
            inner="box", cut=0,
        )
        ax.set_title(feat_name)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide empty axes
    for idx in range(len(features), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def grouped_bar_chart(
    data: Dict[str, Dict[str, float]],
    title: str = "",
    ylabel: str = "F1 Score",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create grouped bar chart (e.g., model x ablation).

    Args:
        data: {group_name: {bar_name: value}}
        title: Plot title.
        ylabel: Y-axis label.
        figsize: Figure size.
        save_path: If set, save figure.

    Returns:
        matplotlib Figure.
    """
    import pandas as pd

    df = pd.DataFrame(data).T
    fig, ax = plt.subplots(figsize=figsize)
    df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str = "",
    cmap: str = "RdBu_r",
    center: Optional[float] = 0,
    fmt: str = ".2f",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create annotated heatmap.

    Args:
        matrix: 2D array of values.
        row_labels: Row tick labels.
        col_labels: Column tick labels.
        title: Plot title.
        cmap: Colormap.
        center: Center value for diverging colormap.
        fmt: Annotation format.
        figsize: Figure size.
        save_path: If set, save figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix, annot=True, fmt=fmt, cmap=cmap, center=center,
        xticklabels=col_labels, yticklabels=row_labels, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def roc_curves(
    curves: Dict[str, Dict[str, np.ndarray]],
    title: str = "ROC Curves",
    figsize: tuple = (8, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot multiple ROC curves.

    Args:
        curves: {name: {"fpr": array, "tpr": array, "auc": float}}
        title: Plot title.
        figsize: Figure size.
        save_path: If set, save figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, data in curves.items():
        ax.plot(data["fpr"], data["tpr"], label=f'{name} (AUC={data["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig
