"""Phase 5: Embedding Space Analysis.

Hooks into GNN layers to extract intermediate representations, visualize
embedding spaces, and measure where class separation emerges.
"""

from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score


class LayerEmbeddingExtractor:
    """Extract intermediate graph-level embeddings from GNN layers via forward hooks.

    Hooks into conv layers and pooling to capture representations at each depth.

    Args:
        model: Trained GNN model.
        device: Torch device.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self._activations = {}
        self._hooks = []

    def _find_conv_layers(self) -> List[Tuple[str, nn.Module]]:
        """Find convolutional/transformer layers in the model."""
        conv_layers = []
        for name, module in self.model.named_modules():
            # Match common GNN layer patterns
            module_type = type(module).__name__
            if any(pattern in module_type for pattern in [
                "Conv", "GINE", "GPS", "GAT", "Transformer",
                "GatedGraph", "SAGEConv", "GINConv",
            ]):
                conv_layers.append((name, module))
        return conv_layers

    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """Register forward hooks on conv layers.

        Args:
            layer_names: Specific layer names to hook. None = all conv layers.
        """
        self.clear_hooks()

        if layer_names is None:
            target_layers = self._find_conv_layers()
        else:
            target_layers = [
                (name, module) for name, module in self.model.named_modules()
                if name in layer_names
            ]

        for name, module in target_layers:
            def hook_fn(module, input, output, name=name):
                # Handle different output formats
                if isinstance(output, tuple):
                    self._activations[name] = output[0].detach().cpu()
                else:
                    self._activations[name] = output.detach().cpu()

            hook = module.register_forward_hook(hook_fn)
            self._hooks.append(hook)

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._activations = {}

    def extract(self, data: Data) -> Dict[str, torch.Tensor]:
        """Extract layer activations for a single graph.

        Args:
            data: PyG Data object.

        Returns:
            {layer_name: activation_tensor}
        """
        self._activations = {}
        data = data.clone().to(self.device)

        # Ensure batch attribute exists for single-graph inference
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)

        with torch.no_grad():
            self.model(data)

        return dict(self._activations)

    def extract_graph_embeddings(
        self,
        dataset: List[Data],
        pool_fn: Callable = None,
    ) -> Dict[str, np.ndarray]:
        """Extract graph-level embeddings at each layer for the whole dataset.

        Uses mean pooling by default to convert node embeddings to graph embeddings.

        Args:
            dataset: List of PyG Data objects.
            pool_fn: Custom pooling function (node_emb, batch) -> graph_emb.

        Returns:
            {layer_name: array of shape (n_graphs, embed_dim)}
        """
        from torch_geometric.nn import global_mean_pool

        all_embeddings = {}  # layer -> list of graph embeddings

        for data in dataset:
            activations = self.extract(data)

            for layer_name, node_emb in activations.items():
                n_nodes = node_emb.shape[0]
                batch = torch.zeros(n_nodes, dtype=torch.long)

                if pool_fn is not None:
                    graph_emb = pool_fn(node_emb, batch)
                else:
                    graph_emb = global_mean_pool(node_emb, batch)

                graph_emb_np = graph_emb.squeeze(0).numpy()
                # Skip scalar or 1D outputs (e.g., final classification head)
                if graph_emb_np.ndim == 0 or graph_emb_np.shape[0] < 2:
                    continue

                if layer_name not in all_embeddings:
                    all_embeddings[layer_name] = []
                all_embeddings[layer_name].append(graph_emb_np)

        return {k: np.stack(v) for k, v in all_embeddings.items()}


class EmbeddingVisualizer:
    """Visualize embedding spaces with t-SNE and UMAP.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def tsne(
        self, embeddings: np.ndarray, perplexity: int = 30
    ) -> np.ndarray:
        """Compute t-SNE projection.

        Args:
            embeddings: (n_samples, n_features).
            perplexity: t-SNE perplexity.

        Returns:
            (n_samples, 2) projected coordinates.
        """
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings) - 1),
                     random_state=self.seed)
        return tsne.fit_transform(embeddings)

    def umap(
        self, embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1
    ) -> np.ndarray:
        """Compute UMAP projection.

        Args:
            embeddings: (n_samples, n_features).
            n_neighbors: UMAP n_neighbors.
            min_dist: UMAP min_dist.

        Returns:
            (n_samples, 2) projected coordinates.
        """
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                            min_dist=min_dist, random_state=self.seed)
        return reducer.fit_transform(embeddings)

    def plot_scatter(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        title: str = "",
        method: str = "t-SNE",
        ax=None,
    ):
        """Plot 2D scatter colored by label.

        Args:
            coords: (n_samples, 2) coordinates.
            labels: Binary labels.
            title: Plot title.
            method: Dimensionality reduction method name.
            ax: Matplotlib axes. If None, creates new figure.
        """
        import matplotlib.pyplot as plt
        from calamr_interp.utils.visualization import COLORS

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        for label, name, color in [(0, "Truth", COLORS["truth"]), (1, "Hallucination", COLORS["hallu"])]:
            mask = labels == label
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=name, alpha=0.6, s=30)

        ax.set_title(title or f"{method} Embedding Space")
        ax.set_xlabel(f"{method}-1")
        ax.set_ylabel(f"{method}-2")
        ax.legend()


class ProbingClassifier:
    """Linear probing at each GNN layer to measure where separability emerges.

    Trains a logistic regression on each layer's embeddings using 5-fold CV.

    Args:
        seed: Random seed.
        n_folds: Number of CV folds.
    """

    def __init__(self, seed: int = 42, n_folds: int = 5):
        self.seed = seed
        self.n_folds = n_folds

    def probe(
        self,
        layer_embeddings: Dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> pd.DataFrame:
        """Run probing classifiers at each layer.

        Args:
            layer_embeddings: {layer_name: (n_samples, embed_dim)}.
            labels: Binary labels.

        Returns:
            DataFrame with layer, accuracy, f1 columns.
        """
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        results = []

        for layer_name, embeddings in layer_embeddings.items():
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=self.seed)),
            ])

            accuracy_scores = cross_val_score(clf, embeddings, labels, cv=cv, scoring="accuracy")
            f1_scores = cross_val_score(clf, embeddings, labels, cv=cv, scoring="f1")

            results.append({
                "layer": layer_name,
                "accuracy_mean": float(accuracy_scores.mean()),
                "accuracy_std": float(accuracy_scores.std()),
                "f1_mean": float(f1_scores.mean()),
                "f1_std": float(f1_scores.std()),
            })

        return pd.DataFrame(results)


class CKAAnalysis:
    """Centered Kernel Alignment for comparing representations across layers/models.

    CKA measures similarity between two sets of representations in a way
    that is invariant to orthogonal transformations and isotropic scaling.
    """

    @staticmethod
    def linear_kernel(X: np.ndarray) -> np.ndarray:
        """Compute linear kernel matrix."""
        return X @ X.T

    @staticmethod
    def centering_matrix(n: int) -> np.ndarray:
        """Compute centering matrix H = I - 1/n * 11^T."""
        return np.eye(n) - np.ones((n, n)) / n

    @staticmethod
    def hsic(K: np.ndarray, L: np.ndarray) -> float:
        """Compute Hilbert-Schmidt Independence Criterion."""
        n = K.shape[0]
        H = CKAAnalysis.centering_matrix(n)
        return float(np.trace(K @ H @ L @ H) / ((n - 1) ** 2))

    @staticmethod
    def cka(X: np.ndarray, Y: np.ndarray) -> float:
        """Compute linear CKA between two representation matrices.

        Args:
            X: (n_samples, dim_1)
            Y: (n_samples, dim_2)

        Returns:
            CKA similarity score in [0, 1].
        """
        K = CKAAnalysis.linear_kernel(X)
        L = CKAAnalysis.linear_kernel(Y)

        hsic_kl = CKAAnalysis.hsic(K, L)
        hsic_kk = CKAAnalysis.hsic(K, K)
        hsic_ll = CKAAnalysis.hsic(L, L)

        denom = np.sqrt(hsic_kk * hsic_ll)
        if denom < 1e-10:
            return 0.0
        return float(hsic_kl / denom)

    @staticmethod
    def compute_cka_matrix(
        layer_embeddings: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute pairwise CKA between all layers.

        Args:
            layer_embeddings: {layer_name: (n_samples, embed_dim)}.

        Returns:
            (cka_matrix, layer_names) where cka_matrix[i,j] = CKA(layer_i, layer_j).
        """
        names = list(layer_embeddings.keys())
        n = len(names)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                score = CKAAnalysis.cka(layer_embeddings[names[i]], layer_embeddings[names[j]])
                matrix[i, j] = score
                matrix[j, i] = score

        return matrix, names

    @staticmethod
    def cross_model_cka(
        model1_embeddings: Dict[str, np.ndarray],
        model2_embeddings: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Compute CKA between layers of two different models.

        Args:
            model1_embeddings: {layer_name: embeddings} for model 1.
            model2_embeddings: {layer_name: embeddings} for model 2.

        Returns:
            (cka_matrix, model1_names, model2_names)
        """
        names1 = list(model1_embeddings.keys())
        names2 = list(model2_embeddings.keys())
        matrix = np.zeros((len(names1), len(names2)))

        for i, n1 in enumerate(names1):
            for j, n2 in enumerate(names2):
                matrix[i, j] = CKAAnalysis.cka(
                    model1_embeddings[n1], model2_embeddings[n2]
                )

        return matrix, names1, names2


def cosine_similarity_analysis(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute within-class and between-class cosine similarity.

    Args:
        embeddings: (n_samples, embed_dim).
        labels: Binary labels.

    Returns:
        Dict with within_class_sim, between_class_sim, separation_ratio.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(embeddings)

    truth_mask = labels == 0
    hallu_mask = labels == 1

    # Within-class similarity (upper triangle only to avoid duplicates)
    truth_sims = sim_matrix[np.ix_(truth_mask, truth_mask)]
    hallu_sims = sim_matrix[np.ix_(hallu_mask, hallu_mask)]

    within_truth = truth_sims[np.triu_indices_from(truth_sims, k=1)].mean()
    within_hallu = hallu_sims[np.triu_indices_from(hallu_sims, k=1)].mean()
    within_class = (within_truth + within_hallu) / 2

    # Between-class similarity
    between = sim_matrix[np.ix_(truth_mask, hallu_mask)].mean()

    return {
        "within_class_sim": float(within_class),
        "between_class_sim": float(between),
        "separation_ratio": float(within_class / (between + 1e-8)),
        "within_truth_sim": float(within_truth),
        "within_hallu_sim": float(within_hallu),
    }
