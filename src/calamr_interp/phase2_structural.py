"""Phase 2: Structural Pattern Analysis.

Analyzes topological properties that distinguish hallucination graphs from
truthful graphs using networkx conversions, spectral analysis, and
community detection.
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from scipy import stats as sp_stats
import networkx as nx


class StructuralAnalyzer:
    """Analyzes structural properties of PyG alignment graphs.

    Methods cover: topology, alignment-specific patterns, spectral features,
    and community structure.
    """

    # --- Conversion ---

    @staticmethod
    def to_nx(data: Data, directed: bool = False) -> nx.Graph:
        """Convert PyG Data to NetworkX graph.

        Args:
            data: PyG Data object.
            directed: If True, return DiGraph.

        Returns:
            NetworkX graph with edge attributes.
        """
        G = nx.DiGraph() if directed else nx.Graph()
        n_nodes = data.x.shape[0]
        G.add_nodes_from(range(n_nodes))

        edge_index = data.edge_index.numpy()
        edge_attr = data.edge_attr.numpy() if data.edge_attr is not None else None

        for i in range(edge_index.shape[1]):
            src, tgt = int(edge_index[0, i]), int(edge_index[1, i])
            attrs = {}
            if edge_attr is not None:
                attrs["edge_type"] = edge_attr[i, 0]
                attrs["capacity"] = edge_attr[i, 1]
                attrs["flow"] = edge_attr[i, 2]
                attrs["is_alignment"] = edge_attr[i, 3]
            G.add_edge(src, tgt, **attrs)

        # Store node metadata
        comp_labels = data.component_labels.numpy()
        for i in range(n_nodes):
            G.nodes[i]["component"] = int(comp_labels[i])  # 0=source, 1=summary
            G.nodes[i]["node_type"] = float(data.x[i, 0].item())

        return G

    # --- Topology ---

    def topology_features(self, data: Data) -> Dict[str, float]:
        """Compute topological features of a graph.

        Features: diameter (approx), avg_shortest_path (approx),
        clustering_coefficient, n_connected_components, density,
        avg_degree, degree_std.

        Args:
            data: PyG Data object.

        Returns:
            Dict of topology features.
        """
        G = self.to_nx(data)
        features = {}

        features["density"] = nx.density(G)
        features["n_connected_components"] = nx.number_connected_components(G)

        # Clustering coefficient
        features["avg_clustering"] = nx.average_clustering(G)

        # Degree stats
        degrees = [d for _, d in G.degree()]
        features["avg_degree"] = float(np.mean(degrees))
        features["degree_std"] = float(np.std(degrees))
        features["max_degree"] = float(max(degrees)) if degrees else 0.0

        # Diameter and avg shortest path on largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc).copy()
        if len(subG) > 1:
            try:
                features["diameter"] = float(nx.diameter(subG))
            except nx.NetworkXError:
                features["diameter"] = float("inf")
            try:
                features["avg_shortest_path"] = nx.average_shortest_path_length(subG)
            except nx.NetworkXError:
                features["avg_shortest_path"] = float("inf")
        else:
            features["diameter"] = 0.0
            features["avg_shortest_path"] = 0.0

        return features

    # --- Alignment-specific patterns ---

    def alignment_patterns(self, data: Data) -> Dict[str, float]:
        """Compute alignment-specific structural patterns.

        Features: fraction of summary nodes that are aligned,
        alignment degree distribution stats, flow value distribution stats.

        Args:
            data: PyG Data object.

        Returns:
            Dict of alignment pattern features.
        """
        features = {}
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        comp_labels = data.component_labels

        is_alignment = edge_attr[:, 3] == 1.0
        align_edges = edge_index[:, is_alignment]

        # Summary nodes that participate in at least one alignment edge
        summary_mask = comp_labels == 1
        summary_indices = set(torch.where(summary_mask)[0].numpy().tolist())
        n_summary = len(summary_indices)

        if align_edges.shape[1] > 0:
            aligned_nodes = set(align_edges[0].numpy().tolist()) | set(align_edges[1].numpy().tolist())
            aligned_summary = aligned_nodes & summary_indices
            features["summary_aligned_fraction"] = len(aligned_summary) / n_summary if n_summary > 0 else 0.0

            # Alignment degree per node (how many alignment edges each node has)
            n_nodes = data.x.shape[0]
            align_degree = torch.zeros(n_nodes, dtype=torch.long)
            align_degree.scatter_add_(0, align_edges[0], torch.ones(align_edges.shape[1], dtype=torch.long))
            align_degree.scatter_add_(0, align_edges[1], torch.ones(align_edges.shape[1], dtype=torch.long))

            # Stats on alignment degree (only for nodes with degree > 0)
            active = align_degree[align_degree > 0].float().numpy()
            features["mean_alignment_degree"] = float(active.mean()) if len(active) > 0 else 0.0
            features["std_alignment_degree"] = float(active.std()) if len(active) > 0 else 0.0
            features["max_alignment_degree"] = float(active.max()) if len(active) > 0 else 0.0

            # Flow value distribution
            flows = edge_attr[is_alignment, 2].numpy()
            features["flow_mean"] = float(flows.mean())
            features["flow_std"] = float(flows.std())
            features["flow_median"] = float(np.median(flows))
            if len(flows) > 2:
                features["flow_skew"] = float(sp_stats.skew(flows))
                features["flow_kurtosis"] = float(sp_stats.kurtosis(flows))
            else:
                features["flow_skew"] = 0.0
                features["flow_kurtosis"] = 0.0
            features["flow_nonzero_fraction"] = float((flows > 0).mean())
        else:
            features["summary_aligned_fraction"] = 0.0
            features["mean_alignment_degree"] = 0.0
            features["std_alignment_degree"] = 0.0
            features["max_alignment_degree"] = 0.0
            features["flow_mean"] = 0.0
            features["flow_std"] = 0.0
            features["flow_median"] = 0.0
            features["flow_skew"] = 0.0
            features["flow_kurtosis"] = 0.0
            features["flow_nonzero_fraction"] = 0.0

        return features

    # --- Spectral analysis ---

    def spectral_features(self, data: Data) -> Dict[str, float]:
        """Compute spectral features from graph Laplacian.

        Features: spectral_gap, algebraic_connectivity,
        largest_eigenvalue, eigenvalue_entropy.

        Args:
            data: PyG Data object.

        Returns:
            Dict of spectral features.
        """
        G = self.to_nx(data)
        features = {}

        # Normalized Laplacian eigenvalues
        try:
            eigenvalues = np.sort(nx.normalized_laplacian_spectrum(G))

            features["spectral_gap"] = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
            features["algebraic_connectivity"] = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
            features["largest_eigenvalue"] = float(eigenvalues[-1])

            # Eigenvalue entropy (normalized)
            positive_eigs = eigenvalues[eigenvalues > 1e-10]
            if len(positive_eigs) > 0:
                normalized = positive_eigs / positive_eigs.sum()
                entropy = -np.sum(normalized * np.log(normalized + 1e-10))
                features["eigenvalue_entropy"] = float(entropy)
            else:
                features["eigenvalue_entropy"] = 0.0
        except Exception:
            features["spectral_gap"] = 0.0
            features["algebraic_connectivity"] = 0.0
            features["largest_eigenvalue"] = 0.0
            features["eigenvalue_entropy"] = 0.0

        return features

    # --- Community detection ---

    def community_features(self, data: Data) -> Dict[str, float]:
        """Run community detection and analyze structure.

        Uses Louvain method. Features: n_communities, modularity,
        community_size_std, cross_community_alignment_fraction.

        Args:
            data: PyG Data object.

        Returns:
            Dict of community features.
        """
        G = self.to_nx(data)
        features = {}

        try:
            communities = nx.community.louvain_communities(G, seed=42)
            n_communities = len(communities)
            features["n_communities"] = float(n_communities)

            # Modularity
            features["modularity"] = nx.community.modularity(G, communities)

            # Community size statistics
            sizes = [len(c) for c in communities]
            features["community_size_mean"] = float(np.mean(sizes))
            features["community_size_std"] = float(np.std(sizes))

            # Fraction of alignment edges that cross communities
            node_to_community = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_community[node] = i

            edge_attr = data.edge_attr
            edge_index = data.edge_index
            is_alignment = edge_attr[:, 3] == 1.0
            align_edges = edge_index[:, is_alignment]

            if align_edges.shape[1] > 0:
                cross_count = 0
                for i in range(align_edges.shape[1]):
                    src = int(align_edges[0, i].item())
                    tgt = int(align_edges[1, i].item())
                    if node_to_community.get(src, -1) != node_to_community.get(tgt, -2):
                        cross_count += 1
                features["cross_community_alignment_fraction"] = cross_count / align_edges.shape[1]
            else:
                features["cross_community_alignment_fraction"] = 0.0
        except Exception:
            features["n_communities"] = 1.0
            features["modularity"] = 0.0
            features["community_size_mean"] = float(data.x.shape[0])
            features["community_size_std"] = 0.0
            features["cross_community_alignment_fraction"] = 0.0

        return features

    # --- Combined extraction ---

    def extract_all(self, data: Data) -> Dict[str, float]:
        """Extract all structural features from a graph.

        Args:
            data: PyG Data object.

        Returns:
            Combined dict of all structural features.
        """
        features = {}
        features.update(self.topology_features(data))
        features.update(self.alignment_patterns(data))
        features.update(self.spectral_features(data))
        features.update(self.community_features(data))
        return features

    def extract_batch(
        self, dataset: List[Data], verbose: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract all structural features from a dataset.

        Args:
            dataset: List of PyG Data objects.
            verbose: Print progress.

        Returns:
            (features_df, labels)
        """
        from tqdm import tqdm

        records = []
        labels = []
        iterator = tqdm(dataset, desc="Extracting structural features") if verbose else dataset

        for data in iterator:
            features = self.extract_all(data)
            records.append(features)
            labels.append(data.y.item())

        return pd.DataFrame(records), np.array(labels)
