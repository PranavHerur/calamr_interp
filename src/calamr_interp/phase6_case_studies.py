"""Phase 6: Case Studies & Qualitative Analysis.

Selects exemplar cases, creates attribution-overlaid visualizations,
and maps predictions back to original text for interpretability.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class CaseStudySelector:
    """Select exemplar cases by prediction confidence.

    Selects 2 True Positives, 2 True Negatives, 2 False Positives, 2 False Negatives
    ordered by confidence (most confident first).

    Args:
        model: Trained GNN model.
        device: Torch device.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def select(
        self,
        dataset: List[Data],
        n_per_category: int = 2,
        threshold: float = 0.5,
    ) -> Dict[str, List[Dict]]:
        """Select exemplar cases from a dataset.

        Args:
            dataset: List of PyG Data objects.
            n_per_category: Cases per category (TP, TN, FP, FN).
            threshold: Classification threshold.

        Returns:
            {category: [{data, pred_prob, label, index}, ...]}
        """
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for data in dataset:
                data_d = data.clone().to(self.device)
                if not hasattr(data_d, "batch") or data_d.batch is None:
                    data_d.batch = torch.zeros(data_d.x.shape[0], dtype=torch.long, device=self.device)
                out = self.model(data_d)
                prob = torch.sigmoid(out).item()
                all_probs.append(prob)
                all_labels.append(data.y.item())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs >= threshold).astype(int)

        categories = {
            "TP": (preds == 1) & (all_labels == 1),  # Predicted hallu, is hallu
            "TN": (preds == 0) & (all_labels == 0),  # Predicted truth, is truth
            "FP": (preds == 1) & (all_labels == 0),  # Predicted hallu, is truth
            "FN": (preds == 0) & (all_labels == 1),  # Predicted truth, is hallu
        }

        result = {}
        for cat_name, mask in categories.items():
            indices = np.where(mask)[0]
            if len(indices) == 0:
                result[cat_name] = []
                continue

            # Sort by confidence (distance from threshold)
            confidences = np.abs(all_probs[indices] - threshold)
            sorted_idx = indices[np.argsort(-confidences)]

            selected = []
            for idx in sorted_idx[:n_per_category]:
                selected.append({
                    "data": dataset[idx],
                    "pred_prob": float(all_probs[idx]),
                    "label": int(all_labels[idx]),
                    "index": int(idx),
                })
            result[cat_name] = selected

        return result


class AlignmentGraphVisualizer:
    """Visualize bipartite alignment graphs with attribution overlays.

    Source nodes on the left, summary nodes on the right,
    with edge width proportional to flow and color indicating importance.
    """

    def __init__(self):
        from calamr_interp.utils.visualization import COLORS
        self.colors = COLORS

    def plot_bipartite(
        self,
        data: Data,
        edge_importance: Optional[torch.Tensor] = None,
        node_importance: Optional[torch.Tensor] = None,
        title: str = "",
        figsize: tuple = (14, 10),
        max_edges: int = 50,
    ) -> plt.Figure:
        """Plot bipartite alignment graph.

        Args:
            data: PyG Data object.
            edge_importance: Per-edge importance (for color).
            node_importance: Per-node importance (for size).
            title: Plot title.
            figsize: Figure size.
            max_edges: Max alignment edges to show (top by importance/flow).

        Returns:
            matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        comp_labels = data.component_labels.numpy()
        source_nodes = np.where(comp_labels == 0)[0]
        summary_nodes = np.where(comp_labels == 1)[0]

        # Layout: source on left, summary on right
        pos = {}
        for i, node in enumerate(source_nodes):
            pos[node] = (0, -i * 1.0)
        for i, node in enumerate(summary_nodes):
            pos[node] = (3, -i * 1.0)

        # Center vertically
        if len(source_nodes) > 0 and len(summary_nodes) > 0:
            source_center = -len(source_nodes) / 2
            summary_center = -len(summary_nodes) / 2
            for node in source_nodes:
                pos[node] = (pos[node][0], pos[node][1] - source_center)
            for node in summary_nodes:
                pos[node] = (pos[node][0], pos[node][1] - summary_center)

        # Draw nodes
        node_sizes = np.ones(data.x.shape[0]) * 50
        if node_importance is not None:
            imp = node_importance.numpy() if isinstance(node_importance, torch.Tensor) else node_importance
            if imp.ndim > 1:
                imp = imp.sum(axis=-1)
            # Normalize to [30, 200]
            imp_min, imp_max = imp.min(), imp.max()
            if imp_max > imp_min:
                node_sizes = 30 + 170 * (imp - imp_min) / (imp_max - imp_min)

        for node in source_nodes:
            ax.scatter(*pos[node], s=node_sizes[node], c=self.colors["primary"],
                       zorder=5, edgecolors="black", linewidth=0.5)
        for node in summary_nodes:
            ax.scatter(*pos[node], s=node_sizes[node], c=self.colors["accent"],
                       zorder=5, edgecolors="black", linewidth=0.5)

        # Draw alignment edges
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        is_alignment = edge_attr[:, 3] == 1.0
        align_mask = is_alignment.numpy()
        align_indices = np.where(align_mask)[0]

        # Sort by importance or flow, take top max_edges
        if edge_importance is not None:
            imp = edge_importance.numpy() if isinstance(edge_importance, torch.Tensor) else edge_importance
            if imp.ndim > 1:
                imp = imp.sum(axis=-1)
            sort_idx = np.argsort(-imp[align_indices])
        else:
            flows = edge_attr[align_mask, 2].numpy()
            sort_idx = np.argsort(-flows)

        top_edges = align_indices[sort_idx[:max_edges]]

        # Normalize edge importance for coloring
        if edge_importance is not None:
            all_imp = imp[top_edges]
        else:
            all_imp = edge_attr[top_edges, 2].numpy()

        imp_min, imp_max = all_imp.min(), all_imp.max()
        if imp_max > imp_min:
            norm_imp = (all_imp - imp_min) / (imp_max - imp_min)
        else:
            norm_imp = np.ones_like(all_imp) * 0.5

        cmap = plt.cm.Reds

        for i, edge_idx in enumerate(top_edges):
            src = int(edge_index[0, edge_idx].item())
            tgt = int(edge_index[1, edge_idx].item())
            if src in pos and tgt in pos:
                flow = float(edge_attr[edge_idx, 2].item())
                width = max(0.5, min(3.0, flow / 5.0))
                color = cmap(norm_imp[i])
                ax.plot(
                    [pos[src][0], pos[tgt][0]],
                    [pos[src][1], pos[tgt][1]],
                    color=color, linewidth=width, alpha=0.6, zorder=1,
                )

        # Legend
        source_patch = mpatches.Patch(color=self.colors["primary"], label=f"Source ({len(source_nodes)} nodes)")
        summary_patch = mpatches.Patch(color=self.colors["accent"], label=f"Summary ({len(summary_nodes)} nodes)")
        ax.legend(handles=[source_patch, summary_patch], loc="upper right")

        ax.set_title(title)
        ax.set_xlim(-0.5, 3.5)
        ax.axis("off")
        fig.tight_layout()

        return fig


class TextMapper:
    """Map node indices back to AMR concept descriptions and text spans.

    Requires the CALAMR DataFrame (from export_alignment_graph_with_df).
    """

    @staticmethod
    def get_node_descriptions(
        data: Data, df: pd.DataFrame
    ) -> Dict[int, str]:
        """Map node indices to descriptive text.

        Args:
            data: PyG Data object.
            df: CALAMR alignment DataFrame.

        Returns:
            {node_index: description}
        """
        descriptions = {}
        comp_labels = data.component_labels.numpy()

        # Extract unique descriptions from DataFrame
        source_descs = set()
        summary_descs = set()

        for _, row in df.iterrows():
            if row["name"] == "source":
                source_descs.add(row["s_descr"])
                source_descs.add(row["t_descr"])
            elif row["name"] == "summary":
                summary_descs.add(row["s_descr"])
                summary_descs.add(row["t_descr"])

        source_list = sorted(source_descs)
        summary_list = sorted(summary_descs)

        source_idx = 0
        summary_idx = 0
        for i in range(len(comp_labels)):
            if comp_labels[i] == 0 and source_idx < len(source_list):
                descriptions[i] = source_list[source_idx]
                source_idx += 1
            elif comp_labels[i] == 1 and summary_idx < len(summary_list):
                descriptions[i] = summary_list[summary_idx]
                summary_idx += 1
            else:
                descriptions[i] = f"node_{i}"

        return descriptions


class ExplanationGenerator:
    """Generate explanation cards combining graph viz, text, and statistics.

    Args:
        model: Trained GNN model.
        device: Torch device.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.visualizer = AlignmentGraphVisualizer()

    def generate_card(
        self,
        data: Data,
        pred_prob: float,
        label: int,
        edge_importance: Optional[torch.Tensor] = None,
        node_importance: Optional[torch.Tensor] = None,
        node_descriptions: Optional[Dict[int, str]] = None,
    ) -> Dict:
        """Generate an explanation card for a single prediction.

        Args:
            data: PyG Data object.
            pred_prob: Predicted probability.
            label: True label.
            edge_importance: Per-edge importance scores.
            node_importance: Per-node importance scores.
            node_descriptions: {node_idx: text_description}.

        Returns:
            Dict with prediction info, key statistics, top important nodes/edges,
            and visualization figure.
        """
        card = {
            "prediction": {
                "probability": pred_prob,
                "predicted_label": "Hallucination" if pred_prob >= 0.5 else "Truth",
                "true_label": "Hallucination" if label == 1 else "Truth",
                "correct": (pred_prob >= 0.5) == (label == 1),
            },
            "graph_stats": self._compute_graph_stats(data),
        }

        # Top important edges
        if edge_importance is not None:
            card["top_edges"] = self._top_important_edges(
                data, edge_importance, node_descriptions
            )

        # Top important nodes
        if node_importance is not None:
            card["top_nodes"] = self._top_important_nodes(
                data, node_importance, node_descriptions
            )

        # Visualization
        pred_label = card["prediction"]["predicted_label"]
        true_label = card["prediction"]["true_label"]
        category = "TP" if pred_label == true_label == "Hallucination" else \
                   "TN" if pred_label == true_label == "Truth" else \
                   "FP" if pred_label == "Hallucination" else "FN"
        title = f"{category}: Pred={pred_prob:.3f} | True={true_label}"

        card["figure"] = self.visualizer.plot_bipartite(
            data, edge_importance, node_importance, title=title
        )

        return card

    @staticmethod
    def _compute_graph_stats(data: Data) -> Dict[str, float]:
        """Compute summary statistics for explanation card."""
        edge_attr = data.edge_attr
        is_alignment = edge_attr[:, 3] == 1.0
        align_flows = edge_attr[is_alignment, 2]

        return {
            "n_nodes": int(data.x.shape[0]),
            "n_edges": int(data.edge_index.shape[1]),
            "n_alignment_edges": int(is_alignment.sum().item()),
            "n_source_nodes": int((data.component_labels == 0).sum().item()),
            "n_summary_nodes": int((data.component_labels == 1).sum().item()),
            "mean_alignment_flow": float(align_flows.mean().item()) if len(align_flows) > 0 else 0.0,
            "max_alignment_flow": float(align_flows.max().item()) if len(align_flows) > 0 else 0.0,
        }

    @staticmethod
    def _top_important_edges(
        data: Data,
        edge_importance: torch.Tensor,
        node_descriptions: Optional[Dict[int, str]] = None,
        top_k: int = 10,
    ) -> List[Dict]:
        """Get top-k most important edges."""
        imp = edge_importance
        if imp.dim() > 1:
            imp = imp.sum(dim=-1)

        top_indices = torch.argsort(imp, descending=True)[:top_k]
        edges = []
        for idx in top_indices:
            idx = int(idx.item())
            src = int(data.edge_index[0, idx].item())
            tgt = int(data.edge_index[1, idx].item())
            is_align = bool(data.edge_attr[idx, 3].item() == 1.0)
            flow = float(data.edge_attr[idx, 2].item())

            edge_info = {
                "source": src,
                "target": tgt,
                "importance": float(imp[idx].item()),
                "is_alignment": is_align,
                "flow": flow,
            }
            if node_descriptions:
                edge_info["source_desc"] = node_descriptions.get(src, f"node_{src}")
                edge_info["target_desc"] = node_descriptions.get(tgt, f"node_{tgt}")
            edges.append(edge_info)

        return edges

    @staticmethod
    def _top_important_nodes(
        data: Data,
        node_importance: torch.Tensor,
        node_descriptions: Optional[Dict[int, str]] = None,
        top_k: int = 10,
    ) -> List[Dict]:
        """Get top-k most important nodes."""
        imp = node_importance
        if imp.dim() > 1:
            imp = imp.sum(dim=-1)

        top_indices = torch.argsort(imp, descending=True)[:top_k]
        nodes = []
        for idx in top_indices:
            idx = int(idx.item())
            component = "source" if data.component_labels[idx] == 0 else "summary"
            node_info = {
                "index": idx,
                "importance": float(imp[idx].item()),
                "component": component,
                "node_type": float(data.x[idx, 0].item()),
            }
            if node_descriptions:
                node_info["description"] = node_descriptions.get(idx, f"node_{idx}")
            nodes.append(node_info)

        return nodes
