"""Phase 1: Statistical Baselines & Feature Distributions.

Extracts graph-level features, runs distribution analysis, and fits
simple classifiers to quantify how much signal is in aggregate statistics
vs graph structure.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    make_scorer,
)

from calamr_interp.utils.statistics import (
    mann_whitney_test,
    cohens_d,
    mutual_information,
    bootstrap_ci,
)


class GraphFeatureExtractor:
    """Extract graph-level statistics from PyG Data objects.

    Features extracted:
    - Graph metadata: mean_flow, aligned_portion, tot_aligned, tot_alignable
    - Computed: node_count, edge_count, alignment_edge_count, alignment_edge_ratio,
      avg_alignment_flow, source_node_count, summary_node_count,
      graph_density, mean_degree, max_degree, flow_std, flow_skew, flow_kurtosis
    """

    FEATURE_NAMES = [
        "mean_flow",
        "aligned_portion",
        "tot_aligned",
        "tot_alignable",
        "node_count",
        "edge_count",
        "alignment_edge_count",
        "non_alignment_edge_count",
        "alignment_edge_ratio",
        "avg_alignment_flow",
        "max_alignment_flow",
        "source_node_count",
        "summary_node_count",
        "source_summary_ratio",
        "graph_density",
        "mean_degree",
        "max_degree",
        "flow_std",
        "flow_skew",
        "flow_kurtosis",
        "nonzero_flow_ratio",
    ]

    def extract(self, data: Data) -> Dict[str, float]:
        """Extract features from a single graph.

        Args:
            data: PyG Data object.

        Returns:
            Dict mapping feature name to value.
        """
        features = {}

        # Graph-level metadata (stored by pyg_export.py)
        features["mean_flow"] = self._safe_tensor_val(data, "mean_flow", 0.0)
        features["aligned_portion"] = self._safe_tensor_val(data, "aligned_portion", 0.0)
        features["tot_aligned"] = self._safe_tensor_val(data, "tot_aligned", 0.0)
        features["tot_alignable"] = self._safe_tensor_val(data, "tot_alignable", 0.0)

        # Basic counts
        n_nodes = data.x.shape[0]
        n_edges = data.edge_index.shape[1]
        features["node_count"] = float(n_nodes)
        features["edge_count"] = float(n_edges)

        # Edge type analysis (edge_attr: [edge_type, capacity, flow, is_alignment])
        edge_attr = data.edge_attr
        is_alignment = edge_attr[:, 3] == 1.0
        n_align = int(is_alignment.sum().item())
        features["alignment_edge_count"] = float(n_align)
        features["non_alignment_edge_count"] = float(n_edges - n_align)
        features["alignment_edge_ratio"] = float(n_align / n_edges) if n_edges > 0 else 0.0

        # Alignment flow statistics
        alignment_flows = edge_attr[is_alignment, 2]  # flow column
        if len(alignment_flows) > 0:
            flows_np = alignment_flows.numpy()
            features["avg_alignment_flow"] = float(flows_np.mean())
            features["max_alignment_flow"] = float(flows_np.max())
            features["flow_std"] = float(flows_np.std())
            features["nonzero_flow_ratio"] = float((flows_np > 0).mean())
            if len(flows_np) > 2:
                features["flow_skew"] = float(sp_stats.skew(flows_np))
                features["flow_kurtosis"] = float(sp_stats.kurtosis(flows_np))
            else:
                features["flow_skew"] = 0.0
                features["flow_kurtosis"] = 0.0
        else:
            features["avg_alignment_flow"] = 0.0
            features["max_alignment_flow"] = 0.0
            features["flow_std"] = 0.0
            features["flow_skew"] = 0.0
            features["flow_kurtosis"] = 0.0
            features["nonzero_flow_ratio"] = 0.0

        # Component analysis (component_labels: 0=source, 1=summary)
        comp_labels = data.component_labels
        n_source = int((comp_labels == 0).sum().item())
        n_summary = int((comp_labels == 1).sum().item())
        features["source_node_count"] = float(n_source)
        features["summary_node_count"] = float(n_summary)
        features["source_summary_ratio"] = float(n_source / n_summary) if n_summary > 0 else 0.0

        # Graph density
        max_edges = n_nodes * (n_nodes - 1)
        features["graph_density"] = float(n_edges / max_edges) if max_edges > 0 else 0.0

        # Degree statistics
        if n_edges > 0:
            degrees = torch.zeros(n_nodes, dtype=torch.long)
            degrees.scatter_add_(0, data.edge_index[0], torch.ones(n_edges, dtype=torch.long))
            degrees.scatter_add_(0, data.edge_index[1], torch.ones(n_edges, dtype=torch.long))
            features["mean_degree"] = float(degrees.float().mean().item())
            features["max_degree"] = float(degrees.max().item())
        else:
            features["mean_degree"] = 0.0
            features["max_degree"] = 0.0

        return features

    def extract_batch(self, dataset: List[Data]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract features from a list of graphs.

        Args:
            dataset: List of PyG Data objects.

        Returns:
            (features_df, labels) where features_df has one row per graph.
        """
        records = []
        labels = []
        for data in dataset:
            features = self.extract(data)
            records.append(features)
            labels.append(data.y.item())

        df = pd.DataFrame(records)
        return df, np.array(labels)

    @staticmethod
    def _safe_tensor_val(data: Data, attr: str, default: float = 0.0) -> float:
        """Safely get a scalar value from a Data attribute."""
        val = getattr(data, attr, None)
        if val is None:
            return default
        if isinstance(val, torch.Tensor):
            return float(val.item())
        return float(val)


class StatisticalBaseline:
    """Logistic regression and random forest baselines on extracted features.

    Uses 5-fold stratified CV with fixed seed for reproducible comparisons.
    """

    def __init__(self, seed: int = 42, n_folds: int = 5):
        self.seed = seed
        self.n_folds = n_folds
        self.models = {
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000, random_state=seed, class_weight="balanced"
                )),
            ]),
            "RandomForest": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=100, random_state=seed, class_weight="balanced"
                )),
            ]),
        }

    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Run 5-fold stratified CV for all baseline models.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Labels (n_samples,).

        Returns:
            {model_name: {metric: value}} with F1, accuracy, AUC.
        """
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        scoring = {
            "f1": make_scorer(f1_score),
            "accuracy": make_scorer(accuracy_score),
            "auc": "roc_auc",
        }

        results = {}
        for name, model in self.models.items():
            cv_results = cross_validate(
                model, X, y, cv=cv, scoring=scoring, return_train_score=True
            )
            results[name] = {
                "test_f1_mean": float(cv_results["test_f1"].mean()),
                "test_f1_std": float(cv_results["test_f1"].std()),
                "test_accuracy_mean": float(cv_results["test_accuracy"].mean()),
                "test_accuracy_std": float(cv_results["test_accuracy"].std()),
                "test_auc_mean": float(cv_results["test_auc"].mean()),
                "test_auc_std": float(cv_results["test_auc"].std()),
                "train_f1_mean": float(cv_results["train_f1"].mean()),
                "train_accuracy_mean": float(cv_results["train_accuracy"].mean()),
            }

        return results

    def get_roc_curves(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Get ROC curve data for each model (using last fold's test set).

        Args:
            X: Feature matrix.
            y: Labels.

        Returns:
            {model_name: {"fpr": array, "tpr": array, "auc": float}}
        """
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        curves = {}

        for name, model in self.models.items():
            # Use all data with cross-val predictions
            from sklearn.model_selection import cross_val_predict
            y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            auc = roc_auc_score(y, y_prob)
            curves[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}

        return curves

    def feature_importance(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> pd.DataFrame:
        """Get feature importances from Random Forest.

        Args:
            X: Feature matrix.
            y: Labels.
            feature_names: Feature names.

        Returns:
            DataFrame with feature importances sorted by importance.
        """
        rf = self.models["RandomForest"]
        rf.fit(X, y)
        importances = rf.named_steps["clf"].feature_importances_
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        return df


def distribution_analysis(
    features_df: pd.DataFrame, labels: np.ndarray
) -> pd.DataFrame:
    """Run distribution analysis: Mann-Whitney U, Cohen's d, mutual information.

    Args:
        features_df: DataFrame with features (one row per sample).
        labels: Binary labels array.

    Returns:
        DataFrame with test results for each feature.
    """
    results = []
    truth_mask = labels == 0
    hallu_mask = labels == 1

    for col in features_df.columns:
        truth_vals = features_df.loc[truth_mask, col].values
        hallu_vals = features_df.loc[hallu_mask, col].values

        mw = mann_whitney_test(truth_vals, hallu_vals)
        d = cohens_d(hallu_vals, truth_vals)
        mi = mutual_information(features_df[col].values, labels)

        truth_mean, truth_lo, truth_hi = bootstrap_ci(truth_vals)
        hallu_mean, hallu_lo, hallu_hi = bootstrap_ci(hallu_vals)

        results.append({
            "feature": col,
            "truth_mean": truth_mean,
            "truth_ci": f"[{truth_lo:.4f}, {truth_hi:.4f}]",
            "hallu_mean": hallu_mean,
            "hallu_ci": f"[{hallu_lo:.4f}, {hallu_hi:.4f}]",
            "mann_whitney_p": mw["p_value"],
            "effect_size_r": mw["effect_size_r"],
            "cohens_d": d,
            "mutual_info": mi,
        })

    df = pd.DataFrame(results)
    df = df.sort_values("mann_whitney_p")
    return df
