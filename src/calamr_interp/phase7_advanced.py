"""Phase 7: Advanced GNN Interpretability.

Five complementary techniques for deeper model understanding:
1. Per-neuron concept alignment (which neurons encode which graph concepts)
2. PGExplainer (amortized edge explanations)
3. GraphMaskExplainer (layer-wise edge masking)
4. CaptumExplainer suite (DeepLIFT, GradCAM, SHAP)
5. Counterfactual edge explanations (minimal edits to flip predictions)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as sp_stats
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from calamr_interp.phase1_baselines import GraphFeatureExtractor
from calamr_interp.phase2_structural import StructuralAnalyzer
from calamr_interp.phase4_attribution import EdgeTypeImportanceAggregator
from calamr_interp.phase5_embeddings import LayerEmbeddingExtractor
from calamr_interp.utils.model_loading import ExplainerModelWrapper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Per-Neuron Concept Alignment
# ---------------------------------------------------------------------------


class NeuronConceptAnalyzer:
    """Correlate individual neurons with graph-level concepts.

    For each neuron in each GNN layer, computes point-biserial correlation
    with the binary label and Pearson correlation with continuous graph-level
    concepts (alignment_edge_ratio, graph_density, etc.).

    Args:
        model: Trained GNN model.
        device: Torch device.
    """

    # Concepts to correlate with (subset most relevant from Phase 1 & 2)
    CONCEPT_NAMES = [
        "alignment_edge_ratio",
        "graph_density",
        "mean_degree",
        "summary_node_count",
        "avg_alignment_flow",
        "modularity",
        "mean_alignment_degree",
        "source_summary_ratio",
        "flow_std",
        "nonzero_flow_ratio",
    ]

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.extractor = LayerEmbeddingExtractor(model, device=self.device)
        self.graph_feat_ext = GraphFeatureExtractor()
        self.struct_analyzer = StructuralAnalyzer()

    def _extract_concepts(self, dataset: List[Data]) -> pd.DataFrame:
        """Extract graph-level concept values for each sample."""
        records = []
        for data in dataset:
            feats = {}
            feats.update(self.graph_feat_ext.extract(data))
            try:
                feats.update(self.struct_analyzer.extract_all(data))
            except Exception:
                pass
            records.append(feats)
        return pd.DataFrame(records)

    def analyze(
        self,
        dataset: List[Data],
        top_k: int = 20,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run neuron-concept correlation analysis.

        Args:
            dataset: List of PyG Data objects.
            top_k: Number of top class-selective neurons to report per layer.

        Returns:
            (correlations_df, selective_neurons_df)
            - correlations_df: (layer, neuron_idx, concept, correlation_r, p_value)
            - selective_neurons_df: top_k neurons per layer with highest
              activation difference between hallu and truth.
        """
        logger.info("Extracting layer embeddings for neuron analysis...")
        self.extractor.register_hooks()
        embeddings = self.extractor.extract_graph_embeddings(dataset)
        self.extractor.clear_hooks()

        labels = np.array([d.y.item() for d in dataset])

        logger.info("Extracting graph-level concepts...")
        concepts_df = self._extract_concepts(dataset)

        # Correlations
        corr_records = []
        selective_records = []

        for layer_name, emb in embeddings.items():
            n_neurons = emb.shape[1]

            # Binary label correlation (point-biserial)
            for ni in range(n_neurons):
                neuron_vals = emb[:, ni]
                r, p = sp_stats.pointbiserialr(labels, neuron_vals)
                corr_records.append({
                    "layer": layer_name,
                    "neuron_idx": ni,
                    "concept": "label",
                    "correlation_r": float(r),
                    "p_value": float(p),
                })

            # Continuous concept correlations
            for concept in self.CONCEPT_NAMES:
                if concept not in concepts_df.columns:
                    continue
                concept_vals = concepts_df[concept].values
                if np.std(concept_vals) < 1e-10:
                    continue
                for ni in range(n_neurons):
                    neuron_vals = emb[:, ni]
                    if np.std(neuron_vals) < 1e-10:
                        continue
                    r, p = sp_stats.pearsonr(neuron_vals, concept_vals)
                    corr_records.append({
                        "layer": layer_name,
                        "neuron_idx": ni,
                        "concept": concept,
                        "correlation_r": float(r),
                        "p_value": float(p),
                    })

            # Class-selective neurons
            truth_mask = labels == 0
            hallu_mask = labels == 1
            mean_truth = emb[truth_mask].mean(axis=0)
            mean_hallu = emb[hallu_mask].mean(axis=0)
            activation_diff = np.abs(mean_hallu - mean_truth)
            top_indices = np.argsort(activation_diff)[-top_k:][::-1]

            for rank, ni in enumerate(top_indices):
                selective_records.append({
                    "layer": layer_name,
                    "neuron_idx": int(ni),
                    "rank": rank,
                    "activation_diff": float(activation_diff[ni]),
                    "mean_truth": float(mean_truth[ni]),
                    "mean_hallu": float(mean_hallu[ni]),
                })

        corr_df = pd.DataFrame(corr_records)
        selective_df = pd.DataFrame(selective_records)

        logger.info(
            "Neuron analysis: %d correlations, %d significant (p<0.05)",
            len(corr_df),
            (corr_df["p_value"] < 0.05).sum() if len(corr_df) > 0 else 0,
        )
        return corr_df, selective_df


# ---------------------------------------------------------------------------
# 2. PGExplainer (amortized edge explanations)
# ---------------------------------------------------------------------------


class PGExplainerAnalysis:
    """PGExplainer: train an amortized edge mask predictor, then explain graphs.

    Args:
        model: Trained GNN model.
        device: Torch device.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.wrapper = ExplainerModelWrapper(model).to(self.device)
        self.wrapper.eval()

    def run(
        self,
        train_dataset: List[Data],
        test_dataset: List[Data],
        epochs: int = 30,
        lr: float = 0.003,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Train PGExplainer and generate edge masks for test set.

        Args:
            train_dataset: Training graphs (for fitting the explainer).
            test_dataset: Test graphs (for generating explanations).
            epochs: Training epochs for PGExplainer.
            lr: Learning rate for PGExplainer training.

        Returns:
            (per_sample_df, aggregated_dict)
            - per_sample_df: per-graph edge importance by type and label
            - aggregated_dict: global mean importance by edge type
        """
        from torch_geometric.explain import Explainer, PGExplainer

        algorithm = PGExplainer(epochs=epochs, lr=lr)
        explainer = Explainer(
            model=self.wrapper,
            algorithm=algorithm,
            explanation_type="phenomenon",
            edge_mask_type="object",
            model_config=dict(
                mode="binary_classification",
                task_level="graph",
                return_type="raw",
            ),
        )

        # Training phase: iterate over epochs, each epoch over all train graphs
        logger.info("Training PGExplainer on %d graphs for %d epochs...", len(train_dataset), epochs)
        for epoch in range(epochs):
            total_loss = 0.0
            for data in train_dataset:
                data = data.clone().to(self.device)
                batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)
                target = data.y.long().to(self.device)
                loss = explainer.algorithm.train(
                    epoch=epoch,
                    model=self.wrapper,
                    x=data.x,
                    edge_index=data.edge_index,
                    target=target,
                    batch=batch,
                    edge_attr=data.edge_attr,
                )
                total_loss += loss
            if (epoch + 1) % 10 == 0:
                logger.info("  PGExplainer epoch %d/%d, avg loss: %.4f", epoch + 1, epochs, total_loss / len(train_dataset))

        # Inference phase
        logger.info("Generating PGExplainer masks for %d test graphs...", len(test_dataset))
        records = []
        all_edge_masks = []

        for data in test_dataset:
            data = data.clone().to(self.device)
            batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)

            explanation = explainer(
                data.x, data.edge_index,
                edge_attr=data.edge_attr,
                batch=batch,
                target=data.y.long().to(self.device),
            )

            if explanation.edge_mask is not None:
                edge_mask = explanation.edge_mask.detach().cpu()
                edge_types = EdgeTypeImportanceAggregator.classify_edges(data.cpu())
                agg = EdgeTypeImportanceAggregator.aggregate(edge_mask, edge_types)
                agg["label"] = data.y.item()
                records.append(agg)
                all_edge_masks.append(edge_mask)

        per_sample_df = pd.DataFrame(records)
        aggregated = {}
        if len(per_sample_df) > 0:
            for col in ["internal/role", "alignment"]:
                if col in per_sample_df.columns:
                    aggregated[col] = float(per_sample_df[col].mean())

        logger.info("PGExplainer: %d graphs explained", len(records))
        return per_sample_df, aggregated


# ---------------------------------------------------------------------------
# 3. GraphMaskExplainer (layer-wise edge masking)
# ---------------------------------------------------------------------------


class GraphMaskAnalysis:
    """GraphMaskExplainer: learn layer-wise edge masks.

    Args:
        model: Trained GNN model.
        num_layers: Number of message-passing layers in the model.
        device: Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        num_layers: int = 3,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        self.wrapper = ExplainerModelWrapper(model).to(self.device)
        self.wrapper.eval()
        self.num_layers = num_layers

    def run(
        self,
        dataset: List[Data],
        epochs: int = 100,
        lr: float = 0.01,
    ) -> pd.DataFrame:
        """Generate layer-wise edge importance for a dataset.

        Args:
            dataset: List of PyG Data objects.
            epochs: Training epochs per graph.
            lr: Learning rate.

        Returns:
            DataFrame with (layer, edge_type, mean_importance, sample_idx, label).
        """
        from torch_geometric.explain import Explainer, GraphMaskExplainer

        explainer = Explainer(
            model=self.wrapper,
            algorithm=GraphMaskExplainer(num_layers=self.num_layers, epochs=epochs, lr=lr),
            explanation_type="model",
            edge_mask_type="object",
            model_config=dict(
                mode="binary_classification",
                task_level="graph",
                return_type="raw",
            ),
        )

        records = []
        for idx, data in enumerate(dataset):
            data = data.clone().to(self.device)
            batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)

            try:
                explanation = explainer(
                    data.x, data.edge_index,
                    edge_attr=data.edge_attr,
                    batch=batch,
                )
            except Exception as e:
                logger.warning("GraphMask failed for graph %d: %s", idx, e)
                continue

            edge_types = EdgeTypeImportanceAggregator.classify_edges(data.cpu())

            # GraphMaskExplainer may provide per-layer masks or a single mask
            if explanation.edge_mask is not None:
                edge_mask = explanation.edge_mask.detach().cpu()
                agg = EdgeTypeImportanceAggregator.aggregate(edge_mask, edge_types)
                for etype, importance in agg.items():
                    records.append({
                        "layer": "all",
                        "edge_type": etype,
                        "mean_importance": importance,
                        "sample_idx": idx,
                        "label": data.y.item(),
                    })

        df = pd.DataFrame(records)
        logger.info("GraphMask: %d records from %d graphs", len(df), len(dataset))
        return df


# ---------------------------------------------------------------------------
# 4. CaptumExplainer Suite (DeepLIFT, GradCAM, SHAP)
# ---------------------------------------------------------------------------


class _ProbsWrapper(nn.Module):
    """Wrapper that applies sigmoid to convert logits to probabilities."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch=None, edge_attr=None, **kwargs):
        data = Data(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        logit = self.model(data)
        return torch.sigmoid(logit)


class CaptumAttributionSuite:
    """Run multiple Captum-based attribution methods via PyG's CaptumExplainer.

    Methods: IntegratedGradients, ShapleyValueSampling.

    Args:
        model: Trained GNN model.
        device: Torch device.
    """

    METHODS = ["IntegratedGradients", "Saliency"]

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.wrapper = _ProbsWrapper(model).to(self.device)
        self.wrapper.eval()

    def _run_method(
        self,
        method_name: str,
        dataset: List[Data],
    ) -> List[Dict[str, float]]:
        """Run a single Captum method on the dataset."""
        from torch_geometric.explain import Explainer, CaptumExplainer

        explainer = Explainer(
            model=self.wrapper,
            algorithm=CaptumExplainer(method_name),
            explanation_type="model",
            node_mask_type="attributes",
            model_config=dict(
                mode="binary_classification",
                task_level="graph",
                return_type="probs",
            ),
        )

        records = []
        for data in dataset:
            data = data.clone().to(self.device)
            batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)

            try:
                explanation = explainer(
                    data.x, data.edge_index,
                    batch=batch,
                )
            except Exception as e:
                logger.warning("%s failed for a graph: %s", method_name, e)
                continue

            if explanation.node_mask is not None:
                saliency = explanation.node_mask.abs().detach().cpu()
                cats = EdgeTypeImportanceAggregator.node_feature_saliency_by_category(saliency)
                cats["label"] = data.y.item()
                records.append(cats)

        return records

    def run(self, dataset: List[Data]) -> pd.DataFrame:
        """Run all Captum methods and return comparison table.

        Args:
            dataset: List of PyG Data objects.

        Returns:
            DataFrame with (method, node_type, component_type, sbert, metadata)
            aggregated means.
        """
        summary_records = []
        for method_name in self.METHODS:
            logger.info("Running Captum %s on %d graphs...", method_name, len(dataset))
            records = self._run_method(method_name, dataset)
            if records:
                method_df = pd.DataFrame(records)
                agg = {
                    "method": method_name,
                    "n_graphs": len(records),
                }
                for cat in ["node_type", "component_type", "sbert", "metadata"]:
                    if cat in method_df.columns:
                        agg[f"{cat}_mean"] = float(method_df[cat].mean())
                        agg[f"{cat}_std"] = float(method_df[cat].std())
                summary_records.append(agg)

        df = pd.DataFrame(summary_records)
        logger.info("Captum suite: %d methods completed", len(df))
        return df


# ---------------------------------------------------------------------------
# 5. Counterfactual Edge Explainer
# ---------------------------------------------------------------------------


class CounterfactualExplainer:
    """Greedy counterfactual: find minimal edge removals to flip predictions.

    Uses gradient saliency to rank edges, then greedily removes them until
    the model's prediction flips. This avoids differentiability issues with
    discrete edge removal through GCNConv.

    Args:
        model: Trained GNN model (takes Data objects directly).
        device: Torch device.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def _get_edge_importance(self, data: Data) -> torch.Tensor:
        """Compute gradient-based edge importance for ranking."""
        data = data.clone().to(self.device)
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)

        data.x.requires_grad_(True)
        self.model.zero_grad()
        out = self.model(data)
        torch.sigmoid(out).backward()

        if data.x.grad is None:
            return torch.ones(data.edge_index.shape[1])

        # Edge importance = sum of absolute gradient of source + target nodes
        src, dst = data.edge_index
        node_importance = data.x.grad.abs().sum(dim=-1).detach()
        edge_importance = node_importance[src] + node_importance[dst]
        return edge_importance.cpu()

    def _explain_single(
        self,
        data: Data,
        max_removal_frac: float = 0.5,
    ) -> Optional[Dict]:
        """Find counterfactual edge removal for a single graph."""
        data = data.clone().to(self.device)
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)

        n_edges = data.edge_index.shape[1]
        original_label = data.y.item()

        with torch.no_grad():
            original_logit = self.model(data)
            original_pred = (torch.sigmoid(original_logit) > 0.5).float().item()

        # Rank edges by importance (most important first)
        edge_importance = self._get_edge_importance(data)
        sorted_indices = torch.argsort(edge_importance, descending=True)

        max_removals = int(n_edges * max_removal_frac)
        if max_removals < 1:
            max_removals = 1

        # Greedily remove edges until prediction flips
        removed_set = set()
        flipped = False

        for step in range(min(max_removals, n_edges - 1)):
            edge_idx = sorted_indices[step].item()
            removed_set.add(edge_idx)

            keep_mask = torch.ones(n_edges, dtype=torch.bool)
            for ri in removed_set:
                keep_mask[ri] = False

            masked_data = data.clone()
            masked_data.edge_index = data.edge_index[:, keep_mask]
            if data.edge_attr is not None:
                masked_data.edge_attr = data.edge_attr[keep_mask]

            with torch.no_grad():
                new_logit = self.model(masked_data)
                new_pred = (torch.sigmoid(new_logit) > 0.5).float().item()

            if new_pred != original_pred:
                flipped = True
                break

        n_removed = len(removed_set)
        removed_indices = torch.tensor(list(removed_set), dtype=torch.long)

        # Classify removed edges by type
        edge_types = EdgeTypeImportanceAggregator.classify_edges(data.cpu())
        n_removed_alignment = int((edge_types[removed_indices] == 1).sum().item()) if n_removed > 0 else 0
        n_removed_internal = n_removed - n_removed_alignment

        return {
            "original_label": original_label,
            "original_pred": original_pred,
            "new_pred": new_pred,
            "flipped": flipped,
            "n_edges_total": n_edges,
            "n_edges_removed": n_removed,
            "removal_fraction": n_removed / n_edges if n_edges > 0 else 0.0,
            "n_removed_alignment": n_removed_alignment,
            "n_removed_internal": n_removed_internal,
        }

    def run(
        self,
        dataset: List[Data],
        max_removal_frac: float = 0.5,
    ) -> pd.DataFrame:
        """Run counterfactual explanation on a dataset.

        Args:
            dataset: List of PyG Data objects.
            max_removal_frac: Maximum fraction of edges to try removing.

        Returns:
            DataFrame with per-graph counterfactual results.
        """
        records = []
        for idx, data in enumerate(dataset):
            result = self._explain_single(data, max_removal_frac=max_removal_frac)
            if result is not None:
                result["sample_idx"] = idx
                records.append(result)
            if (idx + 1) % 10 == 0:
                logger.info("Counterfactual: %d/%d graphs processed", idx + 1, len(dataset))

        df = pd.DataFrame(records)
        if len(df) > 0:
            flip_rate = df["flipped"].mean()
            logger.info(
                "Counterfactual: %d graphs, %.1f%% flipped, avg %.1f%% edges removed",
                len(df), flip_rate * 100, df["removal_fraction"].mean() * 100,
            )
        return df


# ---------------------------------------------------------------------------
# Runner: execute all analyses and collect results
# ---------------------------------------------------------------------------


def run_all(
    model: nn.Module,
    dataset: List[Data],
    train_dataset: List[Data],
    test_dataset: List[Data],
    device: Optional[torch.device] = None,
    num_layers: int = 3,
) -> Dict:
    """Run all Phase 7 analyses and return consolidated results.

    Args:
        model: Trained GNN model (takes Data objects).
        dataset: Full dataset (for neuron analysis).
        train_dataset: Training split (for PGExplainer fitting).
        test_dataset: Test split (for explanations and counterfactuals).
        device: Torch device.
        num_layers: Number of GNN message-passing layers.

    Returns:
        Dict with results from each technique.
    """
    device = device or torch.device("cpu")
    results = {}

    # 1. Neuron Concept Alignment
    logger.info("=" * 60)
    logger.info("Phase 7.1: Neuron Concept Alignment")
    logger.info("=" * 60)
    neuron_analyzer = NeuronConceptAnalyzer(model, device=device)
    corr_df, selective_df = neuron_analyzer.analyze(dataset)
    sig_corr = corr_df[corr_df["p_value"] < 0.05] if len(corr_df) > 0 else corr_df
    results["neuron_concept"] = {
        "n_total_correlations": len(corr_df),
        "n_significant": len(sig_corr),
        "top_label_neurons": (
            sig_corr[sig_corr["concept"] == "label"]
            .nlargest(10, "correlation_r", keep="first")
            [["layer", "neuron_idx", "correlation_r", "p_value"]]
            .to_dict("records")
            if len(sig_corr) > 0 else []
        ),
        "top_selective_neurons": (
            selective_df.head(10).to_dict("records") if len(selective_df) > 0 else []
        ),
    }

    # 2. PGExplainer
    logger.info("=" * 60)
    logger.info("Phase 7.2: PGExplainer")
    logger.info("=" * 60)
    try:
        pg_analyzer = PGExplainerAnalysis(model, device=device)
        pg_df, pg_agg = pg_analyzer.run(train_dataset, test_dataset)
        results["pgexplainer"] = {
            "aggregated_importance": pg_agg,
            "n_explained": len(pg_df),
            "by_label": (
                pg_df.groupby("label")[["internal/role", "alignment"]]
                .mean().to_dict("index")
                if len(pg_df) > 0 else {}
            ),
        }
    except Exception as e:
        logger.error("PGExplainer failed: %s", e)
        results["pgexplainer"] = {"error": str(e)}

    # 3. GraphMaskExplainer
    logger.info("=" * 60)
    logger.info("Phase 7.3: GraphMaskExplainer")
    logger.info("=" * 60)
    try:
        gm_analyzer = GraphMaskAnalysis(model, num_layers=num_layers, device=device)
        gm_df = gm_analyzer.run(test_dataset[:20], epochs=50)  # Subset for speed
        results["graphmask"] = {
            "n_records": len(gm_df),
            "summary": (
                gm_df.groupby(["layer", "edge_type"])["mean_importance"]
                .mean().to_dict()
                if len(gm_df) > 0 else {}
            ),
        }
    except Exception as e:
        logger.error("GraphMaskExplainer failed: %s", e)
        results["graphmask"] = {"error": str(e)}

    # 4. Captum Attribution Suite
    logger.info("=" * 60)
    logger.info("Phase 7.4: Captum Attribution Suite")
    logger.info("=" * 60)
    try:
        captum_suite = CaptumAttributionSuite(model, device=device)
        captum_df = captum_suite.run(test_dataset[:30])  # Subset for speed
        results["captum"] = captum_df.to_dict("records") if len(captum_df) > 0 else []
    except Exception as e:
        logger.error("Captum suite failed: %s", e)
        results["captum"] = {"error": str(e)}

    # 5. Counterfactual Explainer
    logger.info("=" * 60)
    logger.info("Phase 7.5: Counterfactual Edge Explainer")
    logger.info("=" * 60)
    cf_explainer = CounterfactualExplainer(model, device=device)
    cf_df = cf_explainer.run(test_dataset[:30])  # Subset for speed
    if len(cf_df) > 0:
        results["counterfactual"] = {
            "n_graphs": len(cf_df),
            "flip_rate": float(cf_df["flipped"].mean()),
            "avg_removal_fraction": float(cf_df["removal_fraction"].mean()),
            "avg_removed_alignment": float(cf_df["n_removed_alignment"].mean()),
            "avg_removed_internal": float(cf_df["n_removed_internal"].mean()),
            "flipped_summary": (
                cf_df[cf_df["flipped"]]
                [["n_edges_total", "n_edges_removed", "removal_fraction",
                  "n_removed_alignment", "n_removed_internal"]]
                .describe().to_dict()
                if cf_df["flipped"].any() else {}
            ),
        }
    else:
        results["counterfactual"] = {"n_graphs": 0, "flip_rate": 0.0}

    return results
