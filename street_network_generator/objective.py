"""
Module C: Objective function for network generation.
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple
from collections import Counter

from .config import GeneratorConfig
from .reference import ReferenceData
from .metrics import (
    MorphologyMetrics,
    SpaceSyntaxMetrics,
    compute_histogram_distance,
    compute_distribution_histogram,
)


class ObjectiveFunction:
    """Compute objective score for generated network."""

    def __init__(
        self,
        reference: ReferenceData,
        config: GeneratorConfig,
        window_size_m: float = 500.0
    ):
        """
        Initialize objective function.

        Args:
            reference: Reference data target
            config: Generator configuration
            window_size_m: Window size in meters
        """
        self.reference = reference
        self.config = config
        self.window_size_m = window_size_m

        # Cache for expensive metrics
        self.cached_syntax_score = 0.0
        self.cached_syntax_iteration = 0

    def compute_cheap_score(
        self,
        graph: nx.Graph,
        pos: Dict,
        progress: float
    ) -> Tuple[float, Dict]:
        """
        Compute cheap score (morphology only) for candidate evaluation.

        Args:
            graph: Current graph
            pos: Node positions
            progress: Generation progress [0, 1]

        Returns:
            (total_score, breakdown_dict) tuple
        """
        breakdown = {}

        # Get current weights
        w_morph = self.config.get_weight_at_progress(progress, "morph")
        w_syntax = self.config.get_weight_at_progress(progress, "syntax")

        # Compute morphology divergences
        morph_score = self._compute_morphology_divergence(graph, pos, breakdown)

        # Use cached syntax score
        syntax_score = self.cached_syntax_score

        # Compute penalties
        penalty_score = self._compute_penalties(graph, pos, breakdown)

        # Total score
        total_score = (
            w_morph * morph_score +
            w_syntax * syntax_score +
            penalty_score
        )

        breakdown["w_morph"] = w_morph
        breakdown["w_syntax"] = w_syntax
        breakdown["morph_score"] = morph_score
        breakdown["syntax_score"] = syntax_score
        breakdown["penalty_score"] = penalty_score
        breakdown["total_score"] = total_score

        return total_score, breakdown

    def compute_full_score(
        self,
        graph: nx.Graph,
        pos: Dict,
        progress: float,
        iteration: int
    ) -> Tuple[float, Dict]:
        """
        Compute full score including expensive syntax metrics.

        Args:
            graph: Current graph
            pos: Node positions
            progress: Generation progress [0, 1]
            iteration: Current iteration number

        Returns:
            (total_score, breakdown_dict) tuple
        """
        breakdown = {}

        # Get current weights
        w_morph = self.config.get_weight_at_progress(progress, "morph")
        w_syntax = self.config.get_weight_at_progress(progress, "syntax")

        # Compute morphology
        morph_score = self._compute_morphology_divergence(graph, pos, breakdown)

        # Compute syntax (expensive)
        syntax_score = self._compute_syntax_divergence(graph, breakdown)

        # Cache syntax score
        self.cached_syntax_score = syntax_score
        self.cached_syntax_iteration = iteration

        # Compute penalties
        penalty_score = self._compute_penalties(graph, pos, breakdown)

        # Total score
        total_score = (
            w_morph * morph_score +
            w_syntax * syntax_score +
            penalty_score
        )

        breakdown["w_morph"] = w_morph
        breakdown["w_syntax"] = w_syntax
        breakdown["morph_score"] = morph_score
        breakdown["syntax_score"] = syntax_score
        breakdown["penalty_score"] = penalty_score
        breakdown["total_score"] = total_score

        return total_score, breakdown

    def _compute_morphology_divergence(
        self,
        graph: nx.Graph,
        pos: Dict,
        breakdown: Dict
    ) -> float:
        """Compute morphology divergence from reference."""
        total_divergence = 0.0
        weights = self.config.metric_weights

        # Node density
        current_density = MorphologyMetrics.compute_node_density(
            graph, self.window_size_m
        )
        density_error = abs(current_density - self.reference.node_density) / (
            self.reference.node_density + 1e-6
        )
        total_divergence += weights["density"] * density_error
        breakdown["density_error"] = density_error

        # Degree distribution
        current_degree_dist = MorphologyMetrics.compute_degree_distribution(graph)
        degree_divergence = self._compare_degree_distributions(
            current_degree_dist,
            self.reference.degree_distribution
        )
        total_divergence += weights["degree_dist"] * degree_divergence
        breakdown["degree_divergence"] = degree_divergence

        # Segment lengths
        current_lengths = MorphologyMetrics.compute_segment_lengths(graph, pos)
        if current_lengths and len(self.reference.segment_length_hist[0]) > 1:
            # Create histogram with same bins as reference
            ref_bins = self.reference.segment_length_hist[0]
            current_counts, _ = np.histogram(current_lengths, bins=ref_bins)

            length_divergence = compute_histogram_distance(
                current_counts,
                self.reference.segment_length_hist[1],
                ref_bins,
                method="wasserstein"
            )
            # Normalize by bin range
            bin_range = ref_bins[-1] - ref_bins[0]
            length_divergence = length_divergence / (bin_range + 1e-6)
        else:
            length_divergence = 1.0

        total_divergence += weights["segment_length"] * length_divergence
        breakdown["length_divergence"] = length_divergence

        # Orientation
        if graph.number_of_edges() > 0 and len(self.reference.orientation_hist[0]) > 1:
            ref_bins = self.reference.orientation_hist[0]
            orientations = []
            for u, v in graph.edges():
                u_pos = pos[u]
                v_pos = pos[v]
                dx = v_pos[0] - u_pos[0]
                dy = v_pos[1] - u_pos[1]
                bearing = np.degrees(np.arctan2(dy, dx))
                if bearing < 0:
                    bearing += 180
                if bearing >= 180:
                    bearing -= 180
                orientations.append(bearing)

            current_counts, _ = np.histogram(orientations, bins=ref_bins)

            orientation_divergence = compute_histogram_distance(
                current_counts,
                self.reference.orientation_hist[1],
                ref_bins,
                method="wasserstein"
            )
            # Normalize
            orientation_divergence = orientation_divergence / 180.0
        else:
            orientation_divergence = 0.0

        total_divergence += weights["orientation"] * orientation_divergence
        breakdown["orientation_divergence"] = orientation_divergence

        # Dead-end ratio
        current_dead_end = MorphologyMetrics.compute_dead_end_ratio(graph)
        dead_end_error = abs(current_dead_end - self.reference.dead_end_ratio)
        total_divergence += weights["dead_end_ratio"] * dead_end_error
        breakdown["dead_end_error"] = dead_end_error

        return total_divergence

    def _compute_syntax_divergence(self, graph: nx.Graph, breakdown: Dict) -> float:
        """Compute space syntax divergence from reference."""
        if graph.number_of_nodes() < 2:
            return 1.0

        total_divergence = 0.0
        weights = self.config.syntax_weights

        # Compute syntax metrics
        syntax = SpaceSyntaxMetrics.compute_all_syntax(graph, radius=3)

        # Mean depth
        mean_depth_error = abs(syntax["mean_depth"] - self.reference.mean_depth) / (
            self.reference.mean_depth + 1e-6
        )
        total_divergence += weights["mean_depth"] * mean_depth_error
        breakdown["mean_depth_error"] = mean_depth_error

        # Local integration
        local_int_values = list(syntax["local_integration"].values())
        if local_int_values and len(self.reference.local_integration_hist[0]) > 1:
            ref_bins = self.reference.local_integration_hist[0]
            current_counts, _ = np.histogram(local_int_values, bins=ref_bins)

            local_int_divergence = compute_histogram_distance(
                current_counts,
                self.reference.local_integration_hist[1],
                ref_bins,
                method="wasserstein"
            )
            # Normalize
            bin_range = ref_bins[-1] - ref_bins[0]
            local_int_divergence = local_int_divergence / (bin_range + 1e-6)
        else:
            local_int_divergence = 0.0

        total_divergence += weights["local_integration"] * local_int_divergence
        breakdown["local_int_divergence"] = local_int_divergence

        # Choice
        choice_values = list(syntax["choice"].values())
        if choice_values and len(self.reference.choice_hist[0]) > 1:
            ref_bins = self.reference.choice_hist[0]
            current_counts, _ = np.histogram(choice_values, bins=ref_bins)

            choice_divergence = compute_histogram_distance(
                current_counts,
                self.reference.choice_hist[1],
                ref_bins,
                method="wasserstein"
            )
            # Normalize
            bin_range = ref_bins[-1] - ref_bins[0]
            choice_divergence = choice_divergence / (bin_range + 1e-6)
        else:
            choice_divergence = 0.0

        total_divergence += weights["choice"] * choice_divergence
        breakdown["choice_divergence"] = choice_divergence

        # Intelligibility
        intelligibility_error = abs(
            syntax["intelligibility"] - self.reference.intelligibility
        )
        total_divergence += weights["intelligibility"] * intelligibility_error
        breakdown["intelligibility_error"] = intelligibility_error

        return total_divergence

    def _compute_penalties(
        self,
        graph: nx.Graph,
        pos: Dict,
        breakdown: Dict
    ) -> float:
        """Compute penalty terms."""
        total_penalty = 0.0
        weights = self.config.penalty_weights

        # Disconnected graph
        if not nx.is_connected(graph):
            n_components = nx.number_connected_components(graph)
            total_penalty += weights["disconnected"] * (n_components - 1)
            breakdown["disconnected_penalty"] = weights["disconnected"] * (n_components - 1)

            # Small component penalty
            components = list(nx.connected_components(graph))
            largest_size = max(len(c) for c in components)
            total_nodes = graph.number_of_nodes()
            if total_nodes > 0:
                giant_ratio = largest_size / total_nodes
                if giant_ratio < 0.95:
                    total_penalty += weights["small_component"] * (0.95 - giant_ratio)
                    breakdown["small_component_penalty"] = weights["small_component"] * (0.95 - giant_ratio)

        # Boundary dead-ends
        boundary_dist = self.config.boundary_dead_end_distance_m
        boundary_dead_ends = 0

        for node, degree in graph.degree():
            if degree == 1:  # Dead-end
                node_pos = pos[node]
                x, y = node_pos

                # Check if near boundary
                if (x < boundary_dist or x > self.window_size_m - boundary_dist or
                    y < boundary_dist or y > self.window_size_m - boundary_dist):
                    boundary_dead_ends += 1

        if boundary_dead_ends > 0:
            total_penalty += weights["boundary_dead_end"] * boundary_dead_ends
            breakdown["boundary_dead_end_penalty"] = weights["boundary_dead_end"] * boundary_dead_ends

        return total_penalty

    def _compare_degree_distributions(
        self,
        current: Dict[int, int],
        reference: Dict[int, int]
    ) -> float:
        """
        Compare two degree distributions.

        Args:
            current: Current degree distribution
            reference: Reference degree distribution

        Returns:
            Divergence score
        """
        # Get all degrees
        all_degrees = set(current.keys()) | set(reference.keys())

        # Normalize to probabilities
        total_current = sum(current.values()) or 1
        total_reference = sum(reference.values()) or 1

        divergence = 0.0
        for degree in all_degrees:
            p_current = current.get(degree, 0) / total_current
            p_reference = reference.get(degree, 0) / total_reference
            divergence += abs(p_current - p_reference)

        # Total variation distance
        return divergence / 2.0
