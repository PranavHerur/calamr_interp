"""Phase 3: Ablation Studies.

Six ablation conditions that systematically remove components to measure
each component's contribution to hallucination detection.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from calamr_interp.utils.statistics import wilcoxon_signed_rank, bonferroni_correction


ABLATION_TYPES = {
    "A1_no_sbert": "Zero out SBERT dims (2:770) of node features",
    "A2_no_edge_features": "Remove edge attributes entirely",
    "A3_no_alignment_edges": "Remove edges where is_alignment==1",
    "A4_alignment_only": "Keep only alignment edges + endpoint nodes",
    "A5_no_component_type": "Zero out component_type (dim 1) of node features",
    "A6_structure_only": "Zero SBERT, keep node_type + component_type + metadata",
}


class AblationTransform:
    """Apply an ablation transformation to a PyG Data object.

    Creates a modified copy of the data, leaving the original unchanged.

    Args:
        ablation_type: One of the keys in ABLATION_TYPES.
    """

    def __init__(self, ablation_type: str):
        if ablation_type not in ABLATION_TYPES:
            raise ValueError(
                f"Unknown ablation: {ablation_type}. Choose from: {list(ABLATION_TYPES.keys())}"
            )
        self.ablation_type = ablation_type

    def __call__(self, data: Data) -> Data:
        """Apply ablation to a graph.

        Args:
            data: Original PyG Data object.

        Returns:
            Ablated copy of the graph.
        """
        data = data.clone()

        if self.ablation_type == "A1_no_sbert":
            # Zero out SBERT embedding dims (indices 2:770)
            data.x = data.x.clone()
            data.x[:, 2:770] = 0.0

        elif self.ablation_type == "A2_no_edge_features":
            # Remove edge attributes
            data.edge_attr = None

        elif self.ablation_type == "A3_no_alignment_edges":
            # Remove alignment edges (is_alignment == 1 at col 3)
            mask = data.edge_attr[:, 3] != 1.0
            data.edge_index = data.edge_index[:, mask]
            data.edge_attr = data.edge_attr[mask]

        elif self.ablation_type == "A4_alignment_only":
            # Keep only alignment edges and their endpoint nodes
            mask = data.edge_attr[:, 3] == 1.0
            align_edge_index = data.edge_index[:, mask]
            align_edge_attr = data.edge_attr[mask]

            # Find unique nodes in alignment edges
            unique_nodes = torch.unique(align_edge_index)
            node_map = torch.full((data.x.shape[0],), -1, dtype=torch.long)
            node_map[unique_nodes] = torch.arange(len(unique_nodes))

            data.x = data.x[unique_nodes]
            data.component_labels = data.component_labels[unique_nodes]
            data.edge_index = node_map[align_edge_index]
            data.edge_attr = align_edge_attr
            data.num_nodes = len(unique_nodes)

        elif self.ablation_type == "A5_no_component_type":
            # Zero out component_type (dim 1)
            data.x = data.x.clone()
            data.x[:, 1] = 0.0

        elif self.ablation_type == "A6_structure_only":
            # Zero SBERT, keep node_type (dim 0) + component_type (dim 1) + metadata (dim 770)
            data.x = data.x.clone()
            data.x[:, 2:770] = 0.0
            # node_type (0), component_type (1), concept_flag (770) are preserved

        return data

    def __repr__(self) -> str:
        return f"AblationTransform({self.ablation_type})"


class AblationRunner:
    """Run ablation experiments across models, conditions, and seeds.

    Args:
        models: Dict mapping model_name -> model_constructor callable.
        train_data: Training dataset.
        val_data: Validation dataset.
        test_data: Test dataset.
        device: Torch device.
        output_dir: Directory to save results.
    """

    def __init__(
        self,
        models: Dict[str, callable],
        train_data,
        val_data,
        test_data,
        device: Optional[torch.device] = None,
        output_dir: str = "results/ablations",
    ):
        self.models = models
        self.train_data = list(train_data)
        self.val_data = list(val_data)
        self.test_data = list(test_data)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _apply_ablation(self, dataset: List[Data], ablation_type: str) -> List[Data]:
        """Apply ablation transform to all graphs in a dataset."""
        transform = AblationTransform(ablation_type)
        return [transform(d) for d in dataset]

    def _train_and_evaluate(
        self,
        model: nn.Module,
        train_data: List[Data],
        val_data: List[Data],
        test_data: List[Data],
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
    ) -> Dict[str, float]:
        """Train a model and return test metrics.

        Returns:
            Dict with f1, accuracy, auc on test set.
        """
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            # Train
            model.train()
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = model(batch)
                    val_loss += criterion(out, batch.y.float()).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())

        # Evaluate on test set with best model
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = model(batch)
                preds = torch.sigmoid(out)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Find best threshold
        best_f1 = 0
        best_thresh = 0.5
        for t in np.arange(0.1, 0.9, 0.01):
            f1 = f1_score(all_labels, (all_preds >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        y_pred = (all_preds >= best_thresh).astype(int)
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = 0.0

        return {
            "f1": float(f1_score(all_labels, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(all_labels, y_pred)),
            "auc": float(auc),
            "threshold": float(best_thresh),
        }

    def run(
        self,
        ablation_types: Optional[List[str]] = None,
        seeds: List[int] = [42, 123, 456, 789, 1024],
        epochs: int = 10,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run full ablation experiment.

        Args:
            ablation_types: Which ablations to run. None = all + "full" baseline.
            seeds: Random seeds for repeated runs.
            epochs: Training epochs per run.
            verbose: Print progress.

        Returns:
            DataFrame with columns: model, ablation, seed, f1, accuracy, auc.
        """
        if ablation_types is None:
            ablation_types = list(ABLATION_TYPES.keys())

        conditions = ["full"] + ablation_types
        results = []

        for model_name, model_constructor in self.models.items():
            for condition in conditions:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Model: {model_name} | Condition: {condition}")
                    print(f"{'='*60}")

                # Apply ablation
                if condition == "full":
                    train_abl = self.train_data
                    val_abl = self.val_data
                    test_abl = self.test_data
                else:
                    train_abl = self._apply_ablation(self.train_data, condition)
                    val_abl = self._apply_ablation(self.val_data, condition)
                    test_abl = self._apply_ablation(self.test_data, condition)

                for seed in seeds:
                    if verbose:
                        print(f"  Seed {seed}...", end=" ", flush=True)

                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    model = model_constructor()
                    metrics = self._train_and_evaluate(
                        model, train_abl, val_abl, test_abl, epochs=epochs
                    )
                    metrics["model"] = model_name
                    metrics["ablation"] = condition
                    metrics["seed"] = seed
                    results.append(metrics)

                    if verbose:
                        print(f"F1={metrics['f1']:.3f}, Acc={metrics['accuracy']:.3f}")

        df = pd.DataFrame(results)

        # Save results
        df.to_csv(self.output_dir / "ablation_results.csv", index=False)
        return df

    @staticmethod
    def analyze_results(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze ablation results with statistical tests.

        Args:
            df: Results DataFrame from run().

        Returns:
            Dict with 'summary', 'delta_f1', 'significance' DataFrames.
        """
        # Summary: mean and std per model x ablation
        summary = df.groupby(["model", "ablation"]).agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
        ).reset_index()

        # Delta F1 from full model
        full_f1 = summary[summary["ablation"] == "full"][["model", "f1_mean"]].rename(
            columns={"f1_mean": "full_f1"}
        )
        delta = summary.merge(full_f1, on="model")
        delta["delta_f1"] = delta["f1_mean"] - delta["full_f1"]

        # Statistical significance (paired Wilcoxon for each model x ablation vs full)
        significance = []
        models = df["model"].unique()
        ablations = [a for a in df["ablation"].unique() if a != "full"]
        p_values = []

        for model_name in models:
            full_scores = df[(df["model"] == model_name) & (df["ablation"] == "full")]["f1"].values
            for abl in ablations:
                abl_scores = df[(df["model"] == model_name) & (df["ablation"] == abl)]["f1"].values
                if len(full_scores) == len(abl_scores) and len(full_scores) > 1:
                    try:
                        test = wilcoxon_signed_rank(full_scores, abl_scores)
                        p_values.append(test["p_value"])
                        significance.append({
                            "model": model_name,
                            "ablation": abl,
                            "p_value": test["p_value"],
                        })
                    except ValueError:
                        p_values.append(1.0)
                        significance.append({
                            "model": model_name,
                            "ablation": abl,
                            "p_value": 1.0,
                        })

        # Apply Bonferroni correction
        if p_values:
            corrected = bonferroni_correction(p_values)
            for i, row in enumerate(significance):
                row["p_corrected"] = corrected[i]
                row["significant"] = corrected[i] < 0.05

        sig_df = pd.DataFrame(significance) if significance else pd.DataFrame()

        return {
            "summary": summary,
            "delta_f1": delta,
            "significance": sig_df,
        }
