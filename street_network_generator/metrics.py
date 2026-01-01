"""
Metrics computation for urban morphology and space syntax.
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple, List
from collections import Counter
from scipy.stats import wasserstein_distance
from .utils import calculate_bearing, compute_orientation_histogram, compute_entropy


class MorphologyMetrics:
    """Compute urban morphology metrics."""

    @staticmethod
    def compute_node_density(graph: nx.Graph, window_size_m: float) -> float:
        """
        Compute node density (nodes per km²).

        Args:
            graph: NetworkX graph
            window_size_m: Window size in meters

        Returns:
            Node density
        """
        area_km2 = (window_size_m / 1000.0) ** 2
        return graph.number_of_nodes() / area_km2

    @staticmethod
    def compute_degree_distribution(graph: nx.Graph) -> Dict[int, int]:
        """
        Compute node degree distribution.

        Args:
            graph: NetworkX graph

        Returns:
            Dict mapping degree -> count
        """
        degrees = [d for _, d in graph.degree()]
        return dict(Counter(degrees))

    @staticmethod
    def compute_segment_lengths(graph: nx.Graph, pos: dict) -> List[float]:
        """
        Compute all edge lengths.

        Args:
            graph: NetworkX graph
            pos: Node positions {node_id: (x, y)}

        Returns:
            List of edge lengths in meters
        """
        lengths = []
        for u, v in graph.edges():
            u_pos = np.array(pos[u])
            v_pos = np.array(pos[v])
            length = np.linalg.norm(u_pos - v_pos)
            lengths.append(length)
        return lengths

    @staticmethod
    def compute_dead_end_ratio(graph: nx.Graph) -> float:
        """
        Compute ratio of dead-end nodes (degree 1).

        Args:
            graph: NetworkX graph

        Returns:
            Ratio [0, 1]
        """
        if graph.number_of_nodes() == 0:
            return 0.0

        dead_ends = sum(1 for _, d in graph.degree() if d == 1)
        return dead_ends / graph.number_of_nodes()

    @staticmethod
    def compute_all_morphology(
        graph: nx.Graph,
        pos: dict,
        window_size_m: float,
        num_orientation_bins: int = 18
    ) -> Dict:
        """
        Compute all morphology metrics.

        Args:
            graph: NetworkX graph
            pos: Node positions
            window_size_m: Window size
            num_orientation_bins: Number of orientation bins

        Returns:
            Dict with all morphology metrics
        """
        # Node metrics
        node_density = MorphologyMetrics.compute_node_density(graph, window_size_m)
        degree_dist = MorphologyMetrics.compute_degree_distribution(graph)
        dead_end_ratio = MorphologyMetrics.compute_dead_end_ratio(graph)

        # Edge metrics
        segment_lengths = MorphologyMetrics.compute_segment_lengths(graph, pos)

        # Orientation
        bin_edges, orientation_counts = compute_orientation_histogram(
            graph, pos, num_orientation_bins
        )
        orientation_entropy = compute_entropy(orientation_counts)

        return {
            "node_density": node_density,
            "degree_distribution": degree_dist,
            "dead_end_ratio": dead_end_ratio,
            "segment_lengths": segment_lengths,
            "orientation_histogram": {
                "bin_edges": bin_edges.tolist(),
                "counts": orientation_counts.tolist(),
                "entropy": orientation_entropy,
            },
        }


class SpaceSyntaxMetrics:
    """Compute space syntax metrics (node-based for Phase 1)."""

    @staticmethod
    def compute_mean_depth(graph: nx.Graph) -> float:
        """
        Compute mean depth (average shortest path length).

        Args:
            graph: NetworkX graph

        Returns:
            Mean depth value
        """
        if not nx.is_connected(graph):
            # Use largest connected component
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc).copy()

        if graph.number_of_nodes() < 2:
            return 0.0

        # Compute all pairs shortest path lengths
        total_depth = 0
        count = 0

        for source in graph.nodes():
            lengths = nx.single_source_shortest_path_length(graph, source)
            total_depth += sum(lengths.values())
            count += len(lengths)

        return total_depth / count if count > 0 else 0.0

    @staticmethod
    def compute_local_integration(
        graph: nx.Graph,
        radius: int = 3
    ) -> Dict[int, float]:
        """
        Compute local integration (closeness centrality with radius).

        Args:
            graph: NetworkX graph
            radius: Topological radius

        Returns:
            Dict mapping node_id -> local integration value
        """
        if not nx.is_connected(graph):
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc).copy()

        integration = {}

        for node in graph.nodes():
            # Compute shortest paths up to radius
            lengths = nx.single_source_shortest_path_length(graph, node, cutoff=radius)

            if len(lengths) <= 1:
                integration[node] = 0.0
                continue

            # Integration is inverse of mean depth within radius
            total_depth = sum(lengths.values())
            n_reachable = len(lengths) - 1  # Exclude source node

            if total_depth > 0:
                integration[node] = n_reachable / total_depth
            else:
                integration[node] = 0.0

        return integration

    @staticmethod
    def compute_choice(graph: nx.Graph, normalized: bool = True) -> Dict[int, float]:
        """
        Compute choice (betweenness centrality).

        Args:
            graph: NetworkX graph
            normalized: Whether to normalize values

        Returns:
            Dict mapping node_id -> choice value
        """
        if not nx.is_connected(graph):
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc).copy()

        return nx.betweenness_centrality(graph, normalized=normalized)

    @staticmethod
    def compute_intelligibility(
        graph: nx.Graph,
        local_integration: Dict[int, float]
    ) -> float:
        """
        Compute intelligibility (correlation between degree and local integration).

        Args:
            graph: NetworkX graph
            local_integration: Pre-computed local integration values

        Returns:
            Pearson correlation coefficient
        """
        if len(local_integration) < 2:
            return 0.0

        degrees = []
        integrations = []

        for node in local_integration.keys():
            degrees.append(graph.degree(node))
            integrations.append(local_integration[node])

        # Compute Pearson correlation
        degrees_array = np.array(degrees)
        integrations_array = np.array(integrations)

        if np.std(degrees_array) == 0 or np.std(integrations_array) == 0:
            return 0.0

        corr_matrix = np.corrcoef(degrees_array, integrations_array)
        return corr_matrix[0, 1]

    @staticmethod
    def compute_all_syntax(graph: nx.Graph, radius: int = 3) -> Dict:
        """
        Compute all space syntax metrics.

        Args:
            graph: NetworkX graph
            radius: Radius for local integration

        Returns:
            Dict with all syntax metrics
        """
        # Mean depth
        mean_depth = SpaceSyntaxMetrics.compute_mean_depth(graph)

        # Local integration
        local_integration = SpaceSyntaxMetrics.compute_local_integration(graph, radius)

        # Choice
        choice = SpaceSyntaxMetrics.compute_choice(graph, normalized=True)

        # Intelligibility
        intelligibility = SpaceSyntaxMetrics.compute_intelligibility(
            graph, local_integration
        )

        return {
            "mean_depth": mean_depth,
            "local_integration": local_integration,
            "choice": choice,
            "intelligibility": intelligibility,
        }


def compute_histogram_distance(
    hist1_counts: np.ndarray,
    hist2_counts: np.ndarray,
    bin_edges: np.ndarray,
    method: str = "wasserstein"
) -> float:
    """
    Compute distance between two histograms.

    Args:
        hist1_counts: First histogram counts
        hist2_counts: Second histogram counts
        bin_edges: Histogram bin edges
        method: Distance method ('wasserstein' or 'jensen_shannon')

    Returns:
        Distance value
    """
    if method == "wasserstein":
        # Use bin centers as values
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return wasserstein_distance(
            bin_centers, bin_centers,
            hist1_counts, hist2_counts
        )
    elif method == "jensen_shannon":
        # Normalize to probabilities
        p = hist1_counts / (np.sum(hist1_counts) + 1e-10)
        q = hist2_counts / (np.sum(hist2_counts) + 1e-10)

        # Jensen-Shannon divergence
        m = (p + q) / 2
        js_div = 0.5 * np.sum(p * np.log2(p / (m + 1e-10) + 1e-10)) + \
                 0.5 * np.sum(q * np.log2(q / (m + 1e-10) + 1e-10))
        return js_div
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_distribution_histogram(
    values: List[float],
    num_bins: int = 20,
    range_min: float = None,
    range_max: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram from value list.

    Args:
        values: List of values
        num_bins: Number of bins
        range_min: Minimum value (auto if None)
        range_max: Maximum value (auto if None)

    Returns:
        (bin_edges, counts) arrays
    """
    if not values:
        return np.linspace(0, 1, num_bins + 1), np.zeros(num_bins)

    if range_min is None:
        range_min = min(values)
    if range_max is None:
        range_max = max(values)

    counts, bin_edges = np.histogram(
        values,
        bins=num_bins,
        range=(range_min, range_max)
    )

    return bin_edges, counts
