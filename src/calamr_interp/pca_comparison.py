"""PCA Comparison: Prove the GNN learns discriminative representations.

Compares raw input features vs GNN embeddings via UMAP, LDA density,
cumulative Fisher discriminant analysis, and separability metrics.
Produces a publication-quality 5-panel figure.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import umap

from calamr_interp.phase5_embeddings import ProbingClassifier
from calamr_interp.utils.visualization import COLORS, LABEL_COLORS, setup_style


class RawEmbeddingExtractor:
    """Extract graph-level embeddings from raw node features via mean pooling."""

    def extract(self, dataset: List[Data]) -> np.ndarray:
        """Mean-pool raw 771-dim node features per graph.

        Args:
            dataset: List of PyG Data objects with x of shape (N, 771).

        Returns:
            Array of shape (n_graphs, 771).
        """
        embeddings = []
        for data in dataset:
            x = data.x.numpy() if hasattr(data.x, "numpy") else np.array(data.x)
            graph_emb = x.mean(axis=0)
            embeddings.append(graph_emb)
        return np.stack(embeddings)


class SeparabilityMetrics:
    """Compute separability metrics comparing representation spaces."""

    @staticmethod
    def fisher_discriminant_ratio(
        pca_coords: np.ndarray, labels: np.ndarray, n_pcs: int = 10,
    ) -> np.ndarray:
        """Fisher Discriminant Ratio per principal component.

        FDA_k = (mu_hallu_k - mu_truth_k)^2 / (var_hallu_k + var_truth_k)

        Args:
            pca_coords: PCA-transformed coordinates, shape (n, d) where d >= n_pcs.
            labels: Binary labels (0=truth, 1=hallucination).
            n_pcs: Number of PCs to compute.

        Returns:
            Array of shape (n_pcs,) with FDA value per PC.
        """
        n_pcs = min(n_pcs, pca_coords.shape[1])
        truth = pca_coords[labels == 0]
        hallu = pca_coords[labels == 1]

        fda = np.zeros(n_pcs)
        for k in range(n_pcs):
            mu_diff_sq = (hallu[:, k].mean() - truth[:, k].mean()) ** 2
            var_sum = hallu[:, k].var() + truth[:, k].var()
            fda[k] = mu_diff_sq / var_sum if var_sum > 1e-10 else 0.0
        return fda

    @staticmethod
    def silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Silhouette score on standardized full-dimensional embeddings.

        Args:
            embeddings: (n_samples, n_features).
            labels: Binary labels.

        Returns:
            Silhouette score in [-1, 1].
        """
        scaled = StandardScaler().fit_transform(embeddings)
        return float(silhouette_score(scaled, labels))

    @staticmethod
    def mahalanobis_distance(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Mahalanobis distance between class centroids.

        Multivariate generalization of Cohen's d.

        Args:
            embeddings: (n_samples, n_features).
            labels: Binary labels.

        Returns:
            Mahalanobis distance (scalar >= 0).
        """
        scaled = StandardScaler().fit_transform(embeddings)
        truth = scaled[labels == 0]
        hallu = scaled[labels == 1]

        mu_diff = hallu.mean(axis=0) - truth.mean(axis=0)
        # Pooled covariance
        cov_t = np.cov(truth, rowvar=False)
        cov_h = np.cov(hallu, rowvar=False)
        n_t, n_h = len(truth), len(hallu)
        pooled_cov = ((n_t - 1) * cov_t + (n_h - 1) * cov_h) / (n_t + n_h - 2)

        # Regularize for numerical stability
        pooled_cov += np.eye(pooled_cov.shape[0]) * 1e-6

        try:
            inv_cov = np.linalg.inv(pooled_cov)
            dist = float(np.sqrt(mu_diff @ inv_cov @ mu_diff))
        except np.linalg.LinAlgError:
            # Fallback: use pseudoinverse
            inv_cov = np.linalg.pinv(pooled_cov)
            dist = float(np.sqrt(mu_diff @ inv_cov @ mu_diff))

        return dist

    @staticmethod
    def lda_cohens_d(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Cohen's d along the optimal LDA discriminant direction.

        Projects onto the 1D LDA axis and computes Cohen's d there.

        Args:
            embeddings: (n_samples, n_features).
            labels: Binary labels.

        Returns:
            Cohen's d (scalar, can be negative).
        """
        scaled = StandardScaler().fit_transform(embeddings)
        lda = LinearDiscriminantAnalysis(n_components=1)
        projected = lda.fit_transform(scaled, labels).ravel()

        truth_proj = projected[labels == 0]
        hallu_proj = projected[labels == 1]

        mu_diff = hallu_proj.mean() - truth_proj.mean()
        pooled_std = np.sqrt(
            ((len(truth_proj) - 1) * truth_proj.var()
             + (len(hallu_proj) - 1) * hallu_proj.var())
            / (len(truth_proj) + len(hallu_proj) - 2)
        )
        return float(mu_diff / pooled_std) if pooled_std > 1e-10 else 0.0

    @staticmethod
    def linear_probe_f1(
        embeddings: np.ndarray, labels: np.ndarray,
    ) -> Tuple[float, float]:
        """Linear probe F1 via 5-fold CV (reuses ProbingClassifier).

        Args:
            embeddings: (n_samples, n_features).
            labels: Binary labels.

        Returns:
            (f1_mean, f1_std)
        """
        prober = ProbingClassifier(seed=42, n_folds=5)
        df = prober.probe({"probe": embeddings}, labels)
        row = df.iloc[0]
        return float(row["f1_mean"]), float(row["f1_std"])

    @staticmethod
    def lda_projection(
        embeddings: np.ndarray, labels: np.ndarray,
    ) -> np.ndarray:
        """Project embeddings onto the 1D LDA discriminant axis.

        Args:
            embeddings: (n_samples, n_features).
            labels: Binary labels.

        Returns:
            1D array of projected values.
        """
        scaled = StandardScaler().fit_transform(embeddings)
        lda = LinearDiscriminantAnalysis(n_components=1)
        return lda.fit_transform(scaled, labels).ravel()

    @staticmethod
    def compute_all(
        embeddings: np.ndarray, labels: np.ndarray, n_pcs: int = 10,
    ) -> dict:
        """Compute all separability metrics for a representation.

        Args:
            embeddings: (n_samples, n_features).
            labels: Binary labels.
            n_pcs: Number of PCs for FDA.

        Returns:
            Dict with all metric values.
        """
        m = SeparabilityMetrics

        # PCA for FDA
        scaled = StandardScaler().fit_transform(embeddings)
        pca = PCA(n_components=min(n_pcs, scaled.shape[1]))
        pca_coords = pca.fit_transform(scaled)

        fda = m.fisher_discriminant_ratio(pca_coords, labels, n_pcs)
        silh = m.silhouette(embeddings, labels)
        mahal = m.mahalanobis_distance(embeddings, labels)
        lda_d = m.lda_cohens_d(embeddings, labels)
        f1_mean, f1_std = m.linear_probe_f1(embeddings, labels)

        # LDA projection for density plots
        lda_proj = m.lda_projection(embeddings, labels)

        return {
            "fda_per_pc": fda.tolist(),
            "fda_sum_top10": float(fda.sum()),
            "silhouette": silh,
            "mahalanobis": mahal,
            "lda_cohens_d": lda_d,
            "linear_probe_f1_mean": f1_mean,
            "linear_probe_f1_std": f1_std,
            "pca_coords": pca_coords,
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
            "lda_projection": lda_proj,
        }


class PCAComparisonVisualizer:
    """Build a publication-quality 5-panel representation comparison figure.

    Layout (2 rows):
        Top:    (a) UMAP Raw Input  |  (b) UMAP GNN  |  (c) LDA Density
        Bottom: (d) Cumulative FDA  |  (e) Metric Table
    """

    # Publication-quality settings
    _SCATTER_KW = dict(s=12, alpha=0.35, edgecolors="none", rasterized=True)
    _CONTOUR_ALPHA = 0.6
    _CONTOUR_LEVELS = 6
    _LABEL_FS = 11
    _TITLE_FS = 12
    _TICK_FS = 9
    _LEGEND_FS = 9

    def plot(
        self,
        representations: Dict[str, np.ndarray],
        labels: np.ndarray,
        save_path: Optional[str] = None,
        n_pcs: int = 10,
        projection: str = "umap",
        seed: int = 42,
        pair_indices: Optional[List[Tuple[int, int]]] = None,
        highlight_pairs: Optional[List[int]] = None,
        n_highlight: int = 5,
    ) -> plt.Figure:
        """Create the 5-panel representation transformation figure.

        Args:
            representations: {"Raw Input": (n, 771), "GNN Layer 2": (n, 256)}.
            labels: Binary labels.
            save_path: If set, save figure to this path (also saves alternate format).
            n_pcs: Number of PCs for FDA curves.
            projection: "umap" or "tsne" for panels (a)/(b).
            seed: Random seed for projection reproducibility.
            pair_indices: List of (hallu_idx, truth_idx) tuples for all pairs.
                If None, pairs are inferred from consecutive even/odd indices.
            highlight_pairs: Specific pair indices (into pair_indices) to label.
                If None, auto-selects n_highlight pairs with largest displacement.
            n_highlight: Number of pairs to label when highlight_pairs is None.

        Returns:
            matplotlib Figure.
        """
        from sklearn.manifold import TSNE

        setup_style()
        plt.rcParams.update({
            "axes.titlesize": self._TITLE_FS,
            "axes.labelsize": self._LABEL_FS,
            "xtick.labelsize": self._TICK_FS,
            "ytick.labelsize": self._TICK_FS,
            "savefig.dpi": 300,
        })

        rep_names = list(representations.keys())
        assert len(rep_names) == 2, "Expected exactly 2 representations"

        # Compute metrics for both representations
        all_metrics = {}
        for name, emb in representations.items():
            all_metrics[name] = SeparabilityMetrics.compute_all(emb, labels, n_pcs)

        # Compute 2D projections
        proj_label = projection.upper()
        proj_coords = {}
        for name, emb in representations.items():
            scaled = StandardScaler().fit_transform(emb)
            if projection == "tsne":
                reducer = TSNE(
                    n_components=2, perplexity=30, random_state=seed,
                    learning_rate="auto", init="pca",
                )
                proj_coords[name] = reducer.fit_transform(scaled)
            else:
                reducer = umap.UMAP(
                    n_components=2, n_neighbors=30, min_dist=0.3,
                    random_state=seed, metric="euclidean",
                )
                proj_coords[name] = reducer.fit_transform(scaled)

        # Infer pairs if not provided: consecutive (hallu, truth) at even/odd indices
        if pair_indices is None:
            pair_indices = []
            for i in range(0, len(labels) - 1, 2):
                if labels[i] == 1 and labels[i + 1] == 0:
                    pair_indices.append((i, i + 1))

        # Auto-select highlight pairs: largest displacement in GNN space
        if highlight_pairs is None and pair_indices:
            gnn_coords = proj_coords[rep_names[1]]
            displacements = []
            for pi, (hi, ti) in enumerate(pair_indices):
                dist = np.linalg.norm(gnn_coords[hi] - gnn_coords[ti])
                displacements.append((dist, pi))
            displacements.sort(reverse=True)
            highlight_pairs = [pi for _, pi in displacements[:n_highlight]]

        # --- Build figure ---
        fig = plt.figure(figsize=(16, 8.5))
        gs = gridspec.GridSpec(
            2, 6, figure=fig,
            height_ratios=[1, 1],
            hspace=0.38, wspace=0.45,
        )

        ax_a = fig.add_subplot(gs[0, 0:2])  # UMAP Raw
        ax_b = fig.add_subplot(gs[0, 2:4])  # UMAP GNN
        ax_c = fig.add_subplot(gs[0, 4:6])  # LDA density
        ax_d = fig.add_subplot(gs[1, 0:3])  # Cumulative FDA
        ax_e = fig.add_subplot(gs[1, 3:6])  # Metric table

        # Panel (a): Projection on Raw Input
        self._plot_scatter_density(
            ax_a, proj_coords[rep_names[0]], labels,
            title=f"(a) {rep_names[0]}  ({proj_label})",
            axis_label=proj_label,
            pair_indices=pair_indices,
            highlight_pairs=highlight_pairs,
        )

        # Panel (b): Projection on GNN Embeddings
        self._plot_scatter_density(
            ax_b, proj_coords[rep_names[1]], labels,
            title=f"(b) {rep_names[1]}  ({proj_label})",
            axis_label=proj_label,
            pair_indices=pair_indices,
            highlight_pairs=highlight_pairs,
        )

        # Panel (c): LDA density comparison
        self._plot_lda_density(
            ax_c, all_metrics, rep_names, labels,
        )

        # Panel (d): Cumulative FDA
        self._plot_cumulative_fda(
            ax_d, all_metrics, rep_names, n_pcs,
        )

        # Panel (e): Metric summary table
        self._plot_metric_table(ax_e, all_metrics, rep_names)

        fig.suptitle(
            "Representation Transformation: Raw Input  \u2192  GNN Embeddings",
            fontsize=15, fontweight="bold", y=0.98,
        )

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            alt = save_path.replace(".pdf", ".png") if save_path.endswith(".pdf") \
                else save_path.replace(".png", ".pdf")
            fig.savefig(alt, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Panel helpers
    # ------------------------------------------------------------------

    def _plot_scatter_density(
        self, ax: plt.Axes, coords: np.ndarray, labels: np.ndarray,
        title: str, axis_label: str = "UMAP",
        pair_indices: Optional[List[Tuple[int, int]]] = None,
        highlight_pairs: Optional[List[int]] = None,
    ):
        """2D scatter with KDE density contours and paired-sample links."""
        # Draw faint lines for ALL pairs first (background layer)
        if pair_indices:
            for hi, ti in pair_indices:
                ax.plot(
                    [coords[hi, 0], coords[ti, 0]],
                    [coords[hi, 1], coords[ti, 1]],
                    color="#cccccc", linewidth=0.3, alpha=0.25, zorder=1,
                )

        # Scatter + contours
        for label, name, color in [
            (0, "Truth", COLORS["truth"]),
            (1, "Hallucination", COLORS["hallu"]),
        ]:
            mask = labels == label
            x, y = coords[mask, 0], coords[mask, 1]

            ax.scatter(x, y, c=color, label=name, zorder=2, **self._SCATTER_KW)

            try:
                kde = gaussian_kde(np.vstack([x, y]), bw_method=0.3)
                xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
                ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
                pad = 0.05 * max(xmax - xmin, ymax - ymin)
                xx, yy = np.mgrid[
                    xmin - pad : xmax + pad : 100j,
                    ymin - pad : ymax + pad : 100j,
                ]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(
                    xx, yy, zz, levels=self._CONTOUR_LEVELS,
                    colors=[color], alpha=self._CONTOUR_ALPHA, linewidths=1.0,
                )
            except np.linalg.LinAlgError:
                pass

        # Highlight specific pairs with bold lines + labels
        if pair_indices and highlight_pairs:
            for rank, pi in enumerate(highlight_pairs):
                hi, ti = pair_indices[pi]
                # Bold connecting line
                ax.plot(
                    [coords[hi, 0], coords[ti, 0]],
                    [coords[hi, 1], coords[ti, 1]],
                    color="#333333", linewidth=1.8, alpha=0.8, zorder=4,
                )
                # Endpoints
                ax.scatter(
                    coords[hi, 0], coords[hi, 1],
                    c=COLORS["hallu"], s=50, edgecolors="#333333",
                    linewidths=1.2, zorder=5,
                )
                ax.scatter(
                    coords[ti, 0], coords[ti, 1],
                    c=COLORS["truth"], s=50, edgecolors="#333333",
                    linewidths=1.2, zorder=5,
                )
                # Label at midpoint
                mx = (coords[hi, 0] + coords[ti, 0]) / 2
                my = (coords[hi, 1] + coords[ti, 1]) / 2
                ax.annotate(
                    f"P{rank + 1}", (mx, my),
                    fontsize=7, fontweight="bold", color="#333333",
                    ha="center", va="bottom", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="#999999", alpha=0.85, linewidth=0.5),
                )

        ax.set_xlabel(f"{axis_label}-1")
        ax.set_ylabel(f"{axis_label}-2")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=self._LEGEND_FS, loc="best", framealpha=0.8)
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def _plot_lda_density(
        self,
        ax: plt.Axes,
        all_metrics: dict,
        rep_names: List[str],
        labels: np.ndarray,
    ):
        """Overlapping density curves on the LDA discriminant axis."""
        raw_proj = all_metrics[rep_names[0]]["lda_projection"]
        gnn_proj = all_metrics[rep_names[1]]["lda_projection"]
        raw_d = all_metrics[rep_names[0]]["lda_cohens_d"]
        gnn_d = all_metrics[rep_names[1]]["lda_cohens_d"]

        for proj, rep_name, linestyle, d_val in [
            (raw_proj, rep_names[0], "--", raw_d),
            (gnn_proj, rep_names[1], "-", gnn_d),
        ]:
            truth_vals = proj[labels == 0]
            hallu_vals = proj[labels == 1]

            # KDE for smooth densities
            x_range = np.linspace(proj.min() - 0.5, proj.max() + 0.5, 300)

            kde_t = gaussian_kde(truth_vals, bw_method=0.3)
            kde_h = gaussian_kde(hallu_vals, bw_method=0.3)

            alpha = 0.3 if linestyle == "--" else 0.5
            lw = 1.5 if linestyle == "--" else 2.5

            ax.plot(x_range, kde_t(x_range), color=COLORS["truth"],
                    linestyle=linestyle, linewidth=lw)
            ax.fill_between(x_range, kde_t(x_range), color=COLORS["truth"],
                            alpha=alpha * 0.5)

            ax.plot(x_range, kde_h(x_range), color=COLORS["hallu"],
                    linestyle=linestyle, linewidth=lw)
            ax.fill_between(x_range, kde_h(x_range), color=COLORS["hallu"],
                            alpha=alpha * 0.5)

        # Custom legend
        legend_elements = [
            Line2D([0], [0], color=COLORS["truth"], lw=2, label="Truth"),
            Line2D([0], [0], color=COLORS["hallu"], lw=2, label="Hallucination"),
            Line2D([0], [0], color="gray", lw=1.5, ls="--", label=f"{rep_names[0]} (d={raw_d:.1f})"),
            Line2D([0], [0], color="gray", lw=2.5, ls="-", label=f"{rep_names[1]} (d={gnn_d:.1f})"),
        ]
        ax.legend(handles=legend_elements, fontsize=self._LEGEND_FS - 1,
                  loc="upper right", framealpha=0.8)

        ax.set_xlabel("LDA Projection")
        ax.set_ylabel("Density")
        ax.set_title("(c) LDA Discriminant Axis", fontweight="bold")
        ax.set_yticks([])

    @staticmethod
    def _plot_cumulative_fda(
        ax: plt.Axes,
        all_metrics: dict,
        rep_names: List[str],
        n_pcs: int,
    ):
        """Cumulative FDA curve showing signal accumulation across PCs."""
        fda_raw = np.array(all_metrics[rep_names[0]]["fda_per_pc"])
        fda_gnn = np.array(all_metrics[rep_names[1]]["fda_per_pc"])

        n = min(n_pcs, len(fda_raw), len(fda_gnn))
        x = np.arange(1, n + 1)

        cum_raw = np.cumsum(fda_raw[:n])
        cum_gnn = np.cumsum(fda_gnn[:n])

        ax.plot(x, cum_raw, "o--", color=COLORS["neutral"], linewidth=2,
                markersize=6, label=rep_names[0], zorder=3)
        ax.fill_between(x, 0, cum_raw, color=COLORS["neutral"], alpha=0.15)

        ax.plot(x, cum_gnn, "s-", color=COLORS["primary"], linewidth=2.5,
                markersize=6, label=rep_names[1], zorder=3)
        ax.fill_between(x, 0, cum_gnn, color=COLORS["primary"], alpha=0.15)

        # Annotate final values
        ax.annotate(
            f"{cum_raw[-1]:.3f}",
            xy=(n, cum_raw[-1]), xytext=(n - 1.5, cum_raw[-1] + 0.02),
            fontsize=9, color=COLORS["neutral"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["neutral"], lw=1),
        )
        ax.annotate(
            f"{cum_gnn[-1]:.3f}",
            xy=(n, cum_gnn[-1]), xytext=(n - 1.5, cum_gnn[-1] * 0.85),
            fontsize=9, color=COLORS["primary"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["primary"], lw=1),
        )

        ratio = cum_gnn[-1] / cum_raw[-1] if cum_raw[-1] > 1e-10 else float("inf")
        ax.text(
            0.03, 0.95, f"{ratio:.0f}\u00d7 more\ndiscriminative signal",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", color=COLORS["primary"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=COLORS["primary"], alpha=0.8),
        )

        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Cumulative FDA")
        ax.set_title("(d) Cumulative Fisher Discriminant Ratio", fontweight="bold")
        ax.set_xticks(x)
        ax.legend(fontsize=9, loc="center right", framealpha=0.8)
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(bottom=0)

    @staticmethod
    def _plot_metric_table(
        ax: plt.Axes,
        all_metrics: dict,
        rep_names: List[str],
    ):
        """Render metric summary as a publication-quality table."""
        ax.axis("off")
        ax.set_title("(e) Separability Metrics", fontweight="bold")

        rows = [
            ("Silhouette Score", "silhouette", ".3f"),
            ("Linear Probe F1", "linear_probe_f1_mean", ".3f"),
            ("Mahalanobis Dist.", "mahalanobis", ".2f"),
            ("LDA Cohen\u2019s d", "lda_cohens_d", ".2f"),
            ("\u03a3 FDA (top 10 PCs)", "fda_sum_top10", ".3f"),
        ]

        cell_text = []
        row_labels = []
        for label, key, fmt in rows:
            row_labels.append(label)
            vals = []
            for name in rep_names:
                v = all_metrics[name][key]
                vals.append(f"{v:{fmt}}")
            cell_text.append(vals)

        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=rep_names,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.8)

        # Style header row
        for j in range(len(rep_names)):
            table[(0, j)].set_facecolor("#d5e8f0")
            table[(0, j)].set_text_props(fontweight="bold", fontsize=10)

        # Style row labels
        for i in range(len(rows)):
            table[(i + 1, -1)].set_text_props(fontsize=9)

        # Highlight better values (bold + light green bg)
        for i, (_, key, _) in enumerate(rows):
            v0 = all_metrics[rep_names[0]][key]
            v1 = all_metrics[rep_names[1]][key]
            better_col = 1 if abs(v1) >= abs(v0) else 0
            table[(i + 1, better_col)].set_text_props(fontweight="bold")
            table[(i + 1, better_col)].set_facecolor("#e8f5e9")

        # White background for non-highlighted cells
        for i in range(len(rows)):
            for j in range(len(rep_names)):
                v0 = all_metrics[rep_names[0]][rows[i][1]]
                v1 = all_metrics[rep_names[1]][rows[i][1]]
                worse_col = 0 if abs(v1) >= abs(v0) else 1
                if j == worse_col:
                    table[(i + 1, j)].set_facecolor("white")


# ======================================================================
# Supplementary visualizations
# ======================================================================


class SupplementaryVisualizer:
    """Additional publication figures for representation analysis."""

    _SCATTER_KW = dict(s=14, alpha=0.45, edgecolors="none", rasterized=True)

    # ------------------------------------------------------------------
    # 1. Pair displacement histogram
    # ------------------------------------------------------------------

    @staticmethod
    def plot_pair_displacement(
        raw_emb: np.ndarray,
        gnn_emb: np.ndarray,
        labels: np.ndarray,
        pair_indices: List[Tuple[int, int]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Histogram of Euclidean pair displacement in raw vs GNN space.

        Args:
            raw_emb: (n, d_raw) raw embeddings.
            gnn_emb: (n, d_gnn) GNN embeddings.
            labels: Binary labels.
            pair_indices: List of (hallu_idx, truth_idx).
            save_path: Optional save path.
        """
        from scipy.stats import ttest_rel

        setup_style()

        # Standardize both for fair comparison
        raw_s = StandardScaler().fit_transform(raw_emb)
        gnn_s = StandardScaler().fit_transform(gnn_emb)

        raw_dists = np.array([
            np.linalg.norm(raw_s[hi] - raw_s[ti]) for hi, ti in pair_indices
        ])
        gnn_dists = np.array([
            np.linalg.norm(gnn_s[hi] - gnn_s[ti]) for hi, ti in pair_indices
        ])

        t_stat, p_val = ttest_rel(gnn_dists, raw_dists)

        fig, ax = plt.subplots(figsize=(8, 5))

        bins = np.linspace(0, max(raw_dists.max(), gnn_dists.max()) * 1.05, 35)
        ax.hist(raw_dists, bins=bins, alpha=0.5, color=COLORS["neutral"],
                label=f"Raw Input (mean={raw_dists.mean():.2f})", edgecolor="white")
        ax.hist(gnn_dists, bins=bins, alpha=0.5, color=COLORS["primary"],
                label=f"GNN Layer 2 (mean={gnn_dists.mean():.2f})", edgecolor="white")

        # Mean lines
        ax.axvline(raw_dists.mean(), color=COLORS["neutral"], linestyle="--", linewidth=2)
        ax.axvline(gnn_dists.mean(), color=COLORS["primary"], linestyle="--", linewidth=2)

        # Stats annotation
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        ax.text(
            0.97, 0.95,
            f"Paired t-test\nt = {t_stat:.2f}, p = {p_val:.1e} {sig}\nn = {len(pair_indices)} pairs",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
        )

        ax.set_xlabel("Euclidean Distance Between Truth/Hallu Pair (standardized)")
        ax.set_ylabel("Count")
        ax.set_title("Pair Displacement: Raw Input vs GNN Embeddings", fontweight="bold")
        ax.legend(fontsize=10, loc="upper left")
        fig.tight_layout()

        if save_path:
            _save_both(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # 2. Layer-by-layer filmstrip
    # ------------------------------------------------------------------

    @staticmethod
    def plot_layer_filmstrip(
        layer_embeddings: Dict[str, np.ndarray],
        labels: np.ndarray,
        projection: str = "tsne",
        seed: int = 42,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """t-SNE/UMAP at each GNN layer, showing separation emerging.

        Args:
            layer_embeddings: {layer_name: (n, dim)} from all layers.
            labels: Binary labels.
            projection: "tsne" or "umap".
            seed: Random seed.
            save_path: Optional save path.
        """
        from sklearn.manifold import TSNE

        setup_style()
        n_layers = len(layer_embeddings)
        fig, axes = plt.subplots(1, n_layers, figsize=(5.5 * n_layers, 5))
        if n_layers == 1:
            axes = [axes]

        proj_label = projection.upper()

        for idx, (layer_name, emb) in enumerate(layer_embeddings.items()):
            ax = axes[idx]
            scaled = StandardScaler().fit_transform(emb)

            if projection == "tsne":
                reducer = TSNE(n_components=2, perplexity=30, random_state=seed,
                               learning_rate="auto", init="pca")
                coords = reducer.fit_transform(scaled)
            else:
                reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                                    random_state=seed)
                coords = reducer.fit_transform(scaled)

            for label, name, color in [
                (0, "Truth", COLORS["truth"]),
                (1, "Hallucination", COLORS["hallu"]),
            ]:
                mask = labels == label
                ax.scatter(coords[mask, 0], coords[mask, 1], c=color,
                           label=name, s=14, alpha=0.45, edgecolors="none",
                           rasterized=True)

            # Silhouette as quality measure
            sil = silhouette_score(scaled, labels)

            short_name = layer_name.replace("gcn_layers.", "Layer ")
            ax.set_title(f"{short_name}\n(silhouette = {sil:.3f})",
                         fontweight="bold", fontsize=12)
            ax.set_xlabel(f"{proj_label}-1")
            if idx == 0:
                ax.set_ylabel(f"{proj_label}-2")
            ax.tick_params(axis="both", which="both", length=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if idx == n_layers - 1:
                ax.legend(fontsize=9, loc="best", framealpha=0.8)

        fig.suptitle(
            f"Separation Across GNN Layers ({proj_label})",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        if save_path:
            _save_both(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # 3. Decision boundary overlay
    # ------------------------------------------------------------------

    @staticmethod
    def plot_decision_boundary(
        raw_emb: np.ndarray,
        gnn_emb: np.ndarray,
        labels: np.ndarray,
        raw_name: str = "Raw Input",
        gnn_name: str = "GNN Layer 2",
        projection: str = "tsne",
        seed: int = 42,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """2D projection with logistic regression decision boundary overlay.

        Args:
            raw_emb: Raw embeddings.
            gnn_emb: GNN embeddings.
            labels: Binary labels.
            raw_name: Display name for raw.
            gnn_name: Display name for GNN.
            projection: "tsne" or "umap".
            seed: Random seed.
            save_path: Optional save path.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.manifold import TSNE
        from matplotlib.colors import ListedColormap

        setup_style()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        proj_label = projection.upper()

        for ax, emb, name in [(axes[0], raw_emb, raw_name), (axes[1], gnn_emb, gnn_name)]:
            scaled = StandardScaler().fit_transform(emb)

            if projection == "tsne":
                reducer = TSNE(n_components=2, perplexity=30, random_state=seed,
                               learning_rate="auto", init="pca")
                coords = reducer.fit_transform(scaled)
            else:
                reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                                    random_state=seed)
                coords = reducer.fit_transform(scaled)

            # Fit logistic regression on 2D coords
            lr = LogisticRegression(random_state=seed, max_iter=1000)
            lr.fit(coords, labels)
            acc = lr.score(coords, labels)

            # Decision boundary mesh
            pad = 1.5
            x_min, x_max = coords[:, 0].min() - pad, coords[:, 0].max() + pad
            y_min, y_max = coords[:, 1].min() - pad, coords[:, 1].max() + pad
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 300),
                np.linspace(y_min, y_max, 300),
            )
            Z = lr.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)

            # Shaded regions
            cmap_bg = ListedColormap([COLORS["truth"] + "15", COLORS["hallu"] + "15"])
            ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=cmap_bg, alpha=0.4)
            ax.contour(xx, yy, Z, levels=[0.5], colors=["#333333"],
                       linewidths=2, linestyles="--")

            # Points
            for label, lname, color in [
                (0, "Truth", COLORS["truth"]),
                (1, "Hallucination", COLORS["hallu"]),
            ]:
                mask = labels == label
                ax.scatter(coords[mask, 0], coords[mask, 1], c=color,
                           label=lname, s=14, alpha=0.5, edgecolors="none",
                           rasterized=True)

            ax.set_title(f"{name}\n(2D boundary acc = {acc:.1%})",
                         fontweight="bold", fontsize=12)
            ax.set_xlabel(f"{proj_label}-1")
            ax.set_ylabel(f"{proj_label}-2")
            ax.legend(fontsize=9, loc="best", framealpha=0.8)
            ax.tick_params(axis="both", which="both", length=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        fig.suptitle(
            f"Decision Boundary on {proj_label} Projection",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        if save_path:
            _save_both(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # 4. Confidence vs. displacement
    # ------------------------------------------------------------------

    @staticmethod
    def plot_confidence_vs_displacement(
        gnn_emb: np.ndarray,
        labels: np.ndarray,
        pred_probs: np.ndarray,
        pair_indices: List[Tuple[int, int]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Scatter of model confidence vs pair displacement distance.

        Args:
            gnn_emb: GNN embeddings.
            labels: Binary labels.
            pred_probs: Model prediction probabilities, shape (n,).
            pair_indices: List of (hallu_idx, truth_idx).
            save_path: Optional save path.
        """
        setup_style()

        gnn_s = StandardScaler().fit_transform(gnn_emb)
        pair_dists = []
        pair_confidences = []
        pair_correct = []

        for hi, ti in pair_indices:
            dist = np.linalg.norm(gnn_s[hi] - gnn_s[ti])
            pair_dists.append(dist)

            # Average confidence of both samples in the pair
            conf_h = pred_probs[hi] if labels[hi] == 1 else 1 - pred_probs[hi]
            conf_t = 1 - pred_probs[ti] if labels[ti] == 0 else pred_probs[ti]
            pair_confidences.append((conf_h + conf_t) / 2)

            # Both correct?
            pred_h = int(pred_probs[hi] >= 0.5)
            pred_t = int(pred_probs[ti] >= 0.5)
            pair_correct.append(pred_h == labels[hi] and pred_t == labels[ti])

        pair_dists = np.array(pair_dists)
        pair_confidences = np.array(pair_confidences)
        pair_correct = np.array(pair_correct)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Color by both-correct vs at-least-one-wrong
        ax.scatter(
            pair_confidences[pair_correct], pair_dists[pair_correct],
            c=COLORS["truth"], s=30, alpha=0.5, label="Both correct",
            edgecolors="none", rasterized=True,
        )
        ax.scatter(
            pair_confidences[~pair_correct], pair_dists[~pair_correct],
            c=COLORS["hallu"], s=40, alpha=0.7, label="Error in pair",
            edgecolors="#333333", linewidths=0.5, rasterized=True, marker="X",
        )

        # Correlation
        from scipy.stats import pearsonr
        r, p = pearsonr(pair_confidences, pair_dists)
        ax.text(
            0.03, 0.97,
            f"r = {r:.3f}, p = {p:.1e}\nn = {len(pair_indices)} pairs",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
        )

        ax.set_xlabel("Mean Confidence (pair average)")
        ax.set_ylabel("Pair Displacement (standardized Euclidean)")
        ax.set_title("Confidence vs. Pair Displacement in GNN Space", fontweight="bold")
        ax.legend(fontsize=10, loc="lower right", framealpha=0.8)
        fig.tight_layout()

        if save_path:
            _save_both(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # 5. Misclassification anatomy
    # ------------------------------------------------------------------

    @staticmethod
    def plot_misclassification_anatomy(
        raw_emb: np.ndarray,
        gnn_emb: np.ndarray,
        labels: np.ndarray,
        pred_probs: np.ndarray,
        raw_name: str = "Raw Input",
        gnn_name: str = "GNN Layer 2",
        projection: str = "tsne",
        seed: int = 42,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Scatter with misclassified samples highlighted.

        Args:
            raw_emb: Raw embeddings.
            gnn_emb: GNN embeddings.
            labels: Binary labels.
            pred_probs: Model prediction probabilities.
            raw_name: Display name for raw.
            gnn_name: Display name for GNN.
            projection: "tsne" or "umap".
            seed: Random seed.
            save_path: Optional save path.
        """
        from sklearn.manifold import TSNE

        setup_style()
        preds = (pred_probs >= 0.5).astype(int)
        correct = preds == labels
        n_wrong = (~correct).sum()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        proj_label = projection.upper()

        for ax, emb, name in [(axes[0], raw_emb, raw_name), (axes[1], gnn_emb, gnn_name)]:
            scaled = StandardScaler().fit_transform(emb)

            if projection == "tsne":
                reducer = TSNE(n_components=2, perplexity=30, random_state=seed,
                               learning_rate="auto", init="pca")
                coords = reducer.fit_transform(scaled)
            else:
                reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                                    random_state=seed)
                coords = reducer.fit_transform(scaled)

            # Correct points (faint)
            for label, lname, color in [
                (0, "Truth", COLORS["truth"]),
                (1, "Hallu", COLORS["hallu"]),
            ]:
                mask = (labels == label) & correct
                ax.scatter(coords[mask, 0], coords[mask, 1], c=color,
                           s=12, alpha=0.25, edgecolors="none", rasterized=True)

            # Misclassified points (bold with X marker)
            for label, lname, color in [
                (0, "Truth (error)", COLORS["truth"]),
                (1, "Hallu (error)", COLORS["hallu"]),
            ]:
                mask = (labels == label) & ~correct
                if mask.any():
                    ax.scatter(coords[mask, 0], coords[mask, 1], c=color,
                               s=60, alpha=0.9, edgecolors="#333333",
                               linewidths=1.0, marker="X", zorder=5,
                               label=f"{lname} ({mask.sum()})")

            ax.set_title(f"{name}", fontweight="bold", fontsize=12)
            ax.set_xlabel(f"{proj_label}-1")
            ax.set_ylabel(f"{proj_label}-2")
            ax.legend(fontsize=9, loc="best", framealpha=0.8)
            ax.tick_params(axis="both", which="both", length=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        fig.suptitle(
            f"Misclassification Anatomy ({n_wrong}/{len(labels)} errors, "
            f"acc = {correct.mean():.1%})",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        if save_path:
            _save_both(fig, save_path)
        return fig


def _save_both(fig: plt.Figure, path: str):
    """Save figure as both PDF and PNG."""
    fig.savefig(path, bbox_inches="tight")
    if path.endswith(".pdf"):
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    elif path.endswith(".png"):
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
