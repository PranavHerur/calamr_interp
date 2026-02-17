"""Phase 4: Node & Edge Attribution.

Multiple attribution methods for understanding which nodes, edges, and features
the GNN relies on for hallucination detection.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class GradientSaliency:
    """Gradient-based saliency: |dL/dx| per node, |dL/d(edge_attr)| per edge.

    Args:
        model: Trained GNN model.
        device: Torch device.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def attribute(self, data: Data) -> Dict[str, torch.Tensor]:
        """Compute gradient saliency for a single graph.

        Args:
            data: PyG Data object.

        Returns:
            Dict with 'node_saliency' [n_nodes, n_features],
            'edge_saliency' [n_edges, edge_dim] (if edge_attr exists).
        """
        data = data.clone().to(self.device)
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)
        data.x.requires_grad_(True)
        if data.edge_attr is not None:
            data.edge_attr.requires_grad_(True)

        # Forward pass
        self.model.zero_grad()
        out = self.model(data)
        target = torch.sigmoid(out)

        # Backward pass (use predicted score as target)
        target.backward()

        result = {
            "node_saliency": data.x.grad.abs().detach().cpu(),
        }
        if data.edge_attr is not None and data.edge_attr.grad is not None:
            result["edge_saliency"] = data.edge_attr.grad.abs().detach().cpu()

        return result


class IntegratedGradientsAttribution:
    """Integrated Gradients for more robust attribution.

    Uses zero-feature baseline with same graph topology.

    Args:
        model: Trained GNN model.
        n_steps: Number of interpolation steps.
        device: Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 50,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.n_steps = n_steps
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def attribute(self, data: Data) -> Dict[str, torch.Tensor]:
        """Compute Integrated Gradients for a single graph.

        Args:
            data: PyG Data object.

        Returns:
            Dict with 'node_ig' [n_nodes, n_features],
            'edge_ig' [n_edges, edge_dim] if edge_attr exists.
        """
        data = data.clone().to(self.device)
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)

        # Baselines: zero features, same topology
        baseline_x = torch.zeros_like(data.x)
        baseline_edge_attr = torch.zeros_like(data.edge_attr) if data.edge_attr is not None else None

        # Accumulate gradients along interpolation path
        node_grads = torch.zeros_like(data.x)
        edge_grads = torch.zeros_like(data.edge_attr) if data.edge_attr is not None else None

        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps
            interp_data = data.clone()

            # Interpolate node features
            interp_x = baseline_x + alpha * (data.x - baseline_x)
            interp_data.x = interp_x.clone().detach().requires_grad_(True)

            # Interpolate edge features
            if data.edge_attr is not None and baseline_edge_attr is not None:
                interp_edge = baseline_edge_attr + alpha * (data.edge_attr - baseline_edge_attr)
                interp_data.edge_attr = interp_edge.clone().detach().requires_grad_(True)

            self.model.zero_grad()
            out = self.model(interp_data)
            target = torch.sigmoid(out)
            target.backward()

            if interp_data.x.grad is not None:
                node_grads += interp_data.x.grad.detach()
            if data.edge_attr is not None and interp_data.edge_attr is not None and interp_data.edge_attr.grad is not None:
                edge_grads += interp_data.edge_attr.grad.detach()

        # Scale by input difference and average
        node_ig = (data.x.detach() - baseline_x) * node_grads / (self.n_steps + 1)
        result = {"node_ig": node_ig.abs().cpu()}

        if edge_grads is not None:
            edge_ig = (data.edge_attr.detach() - baseline_edge_attr) * edge_grads / (self.n_steps + 1)
            result["edge_ig"] = edge_ig.abs().cpu()

        return result


class GNNExplainerWrapper:
    """Wrapper around PyG's Explainer framework for graph-level explanation.

    Args:
        model: Trained GNN model.
        n_runs: Number of runs to average (reduces noise).
        device: Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        n_runs: int = 3,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.n_runs = n_runs
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def explain(self, data: Data, epochs: int = 200) -> Dict[str, torch.Tensor]:
        """Generate explanation for a single graph.

        Args:
            data: PyG Data object.
            epochs: Training epochs for GNNExplainer.

        Returns:
            Dict with 'node_mask' [n_nodes, n_features],
            'edge_mask' [n_edges].
        """
        from torch_geometric.explain import Explainer, GNNExplainer

        data = data.clone().to(self.device)

        node_masks = []
        edge_masks = []

        for _ in range(self.n_runs):
            explainer = Explainer(
                model=self.model,
                algorithm=GNNExplainer(epochs=epochs),
                explanation_type="model",
                node_mask_type="attributes",
                edge_mask_type="object",
                model_config=dict(
                    mode="binary_classification",
                    task_level="graph",
                    return_type="raw",
                ),
            )

            explanation = explainer(
                data.x, data.edge_index,
                edge_attr=data.edge_attr,
                batch=torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device),
            )

            if explanation.node_mask is not None:
                node_masks.append(explanation.node_mask.detach().cpu())
            if explanation.edge_mask is not None:
                edge_masks.append(explanation.edge_mask.detach().cpu())

        result = {}
        if node_masks:
            result["node_mask"] = torch.stack(node_masks).mean(dim=0)
        if edge_masks:
            result["edge_mask"] = torch.stack(edge_masks).mean(dim=0)

        return result


class AttentionAnalyzer:
    """Analyze attention weights from attention-based GNN models.

    Works with EdgeAwareGAT's get_attention_weights() method.

    Args:
        model: Trained attention-based GNN model.
        device: Torch device.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def get_attention_weights(self, data: Data) -> List[torch.Tensor]:
        """Extract attention weights per layer.

        Args:
            data: PyG Data object.

        Returns:
            List of attention weight tensors per layer.
        """
        data = data.clone().to(self.device)
        if hasattr(self.model, "get_attention_weights"):
            with torch.no_grad():
                weights = self.model.get_attention_weights(data)
            return [w.detach().cpu() for w in weights]
        else:
            raise AttributeError(
                f"Model {type(self.model).__name__} does not have get_attention_weights()"
            )

    def attention_by_edge_type(
        self, data: Data
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate attention weights by edge type.

        Args:
            data: PyG Data object.

        Returns:
            {layer_idx: {edge_type: mean_attention}}
        """
        weights = self.get_attention_weights(data)
        edge_attr = data.edge_attr

        # Edge type classification based on edge_attr
        is_alignment = edge_attr[:, 3] == 1.0
        edge_types = torch.zeros(edge_attr.shape[0], dtype=torch.long)
        # edge_attr col 0: 1=role/internal, 2=alignment type
        edge_types[edge_attr[:, 0] == 1] = 0  # role/internal
        edge_types[is_alignment] = 2  # alignment
        edge_types[(~is_alignment) & (edge_attr[:, 0] == 1)] = 0  # role
        # internal edges: not alignment and edge_type != role... both are 1
        # Let's use is_alignment as the primary discriminator
        type_names = {0: "internal", 2: "alignment"}

        results = {}
        for layer_idx, alpha in enumerate(weights):
            # alpha shape varies by model, average across heads
            if alpha.dim() > 1:
                alpha_mean = alpha.mean(dim=-1) if alpha.shape[-1] > 1 else alpha.squeeze(-1)
            else:
                alpha_mean = alpha

            layer_results = {}
            for type_id, type_name in type_names.items():
                mask = edge_types == type_id
                if mask.any():
                    layer_results[type_name] = float(alpha_mean[mask[:len(alpha_mean)]].mean().item())
            results[f"layer_{layer_idx}"] = layer_results

        return results


class EdgeTypeImportanceAggregator:
    """Aggregate edge importance scores by edge type across methods.

    Combines results from gradient saliency, integrated gradients,
    GNNExplainer, and attention to produce a unified importance analysis.
    """

    @staticmethod
    def classify_edges(data: Data) -> torch.Tensor:
        """Classify edges into types: 0=internal/role, 1=alignment.

        Args:
            data: PyG Data object.

        Returns:
            LongTensor of edge type labels.
        """
        is_alignment = data.edge_attr[:, 3] == 1.0
        return is_alignment.long()

    @staticmethod
    def aggregate(
        edge_importance: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> Dict[str, float]:
        """Aggregate edge importance by type.

        Args:
            edge_importance: Per-edge importance scores.
            edge_types: Per-edge type labels (0=internal, 1=alignment).

        Returns:
            {type_name: mean_importance}
        """
        type_names = {0: "internal/role", 1: "alignment"}
        result = {}
        for type_id, name in type_names.items():
            mask = edge_types == type_id
            if mask.any():
                result[name] = float(edge_importance[mask].mean().item())
            else:
                result[name] = 0.0
        return result

    @staticmethod
    def batch_aggregate(
        dataset: List[Data],
        attribution_fn,
        importance_key: str = "edge_saliency",
    ) -> pd.DataFrame:
        """Run attribution on a dataset and aggregate by edge type.

        Args:
            dataset: List of PyG Data objects.
            attribution_fn: Function that takes Data and returns dict with importance_key.
            importance_key: Key in attribution result containing edge importance tensor.

        Returns:
            DataFrame with per-sample importance by edge type.
        """
        records = []
        for data in dataset:
            attr_result = attribution_fn(data)
            if importance_key in attr_result:
                edge_imp = attr_result[importance_key]
                # Sum across feature dims if multi-dimensional
                if edge_imp.dim() > 1:
                    edge_imp = edge_imp.sum(dim=-1)
                edge_types = EdgeTypeImportanceAggregator.classify_edges(data)
                agg = EdgeTypeImportanceAggregator.aggregate(edge_imp, edge_types)
                agg["label"] = data.y.item()
                records.append(agg)

        return pd.DataFrame(records)

    @staticmethod
    def node_feature_saliency_by_category(
        node_saliency: torch.Tensor,
    ) -> Dict[str, float]:
        """Aggregate node feature saliency by feature category.

        Categories:
        - node_type: dim 0
        - component_type: dim 1
        - sbert: dims 2:770
        - metadata: dim 770

        Args:
            node_saliency: [n_nodes, 771] saliency tensor.

        Returns:
            {category: mean_saliency}
        """
        return {
            "node_type": float(node_saliency[:, 0].mean().item()),
            "component_type": float(node_saliency[:, 1].mean().item()),
            "sbert": float(node_saliency[:, 2:770].mean().item()),
            "metadata": float(node_saliency[:, 770].mean().item()) if node_saliency.shape[1] > 770 else 0.0,
        }
