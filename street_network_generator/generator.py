"""
Module B: Street network generator with planar growth.
"""

import random
import math
import numpy as np
import networkx as nx
from typing import Dict, Tuple, List, Optional, Set
from shapely.geometry import Point, LineString

from .config import GeneratorConfig
from .reference import ReferenceData
from .objective import ObjectiveFunction
from .utils import (
    create_window_polygon,
    snap_to_node,
    SpatialIndex,
    calculate_bearing,
)


class StreetNetworkGenerator:
    """Generate planar street network matching reference statistics."""

    def __init__(
        self,
        reference: ReferenceData,
        config: GeneratorConfig,
        seed: Optional[int] = None
    ):
        """
        Initialize generator.

        Args:
            reference: Reference data to match
            config: Generator configuration
            seed: Random seed (uses config.seed if None)
        """
        self.reference = reference
        self.config = config
        self.window_size_m = config.window_size_m

        # Set random seed
        if seed is None:
            seed = config.seed
        random.seed(seed)
        np.random.seed(seed)

        # Initialize graph
        self.graph = nx.Graph()
        self.pos: Dict[int, Tuple[float, float]] = {}
        self.next_node_id = 0

        # Spatial index
        self.spatial_index = SpatialIndex()

        # Window boundary
        self.window = create_window_polygon(self.window_size_m)

        # Objective function
        self.objective = ObjectiveFunction(reference, config, self.window_size_m)

        # Generation state
        self.accepted_edges = 0
        self.iteration = 0
        self.best_score = float('inf')
        self.no_improvement_count = 0

        # Audit history
        self.audit_scores = []

    def generate(self) -> Tuple[nx.Graph, Dict, Dict]:
        """
        Generate network.

        Returns:
            (graph, positions, metadata) tuple
        """
        print("Initializing seed skeleton...")
        self._create_seed_skeleton()

        print(f"Starting iterative growth (max {self.config.max_iterations} iterations)...")
        self._iterative_growth()

        print("Generation complete!")

        metadata = {
            "iterations": self.iteration,
            "accepted_edges": self.accepted_edges,
            "final_score": self.best_score,
            "audit_history": self.audit_scores,
        }

        return self.graph, self.pos, metadata

    def _create_seed_skeleton(self):
        """Create initial skeleton with boundary-to-boundary spines."""
        n_spines = random.randint(
            self.config.min_boundary_spines,
            self.config.max_boundary_spines
        )

        # Sample orientations from reference
        if len(self.reference.orientation_hist[1]) > 0:
            # Use reference orientation distribution
            bin_edges = self.reference.orientation_hist[0]
            counts = self.reference.orientation_hist[1]

            # Sample weighted by counts
            total_count = np.sum(counts)
            if total_count > 0:
                probs = counts / total_count
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                orientations = np.random.choice(
                    bin_centers,
                    size=n_spines,
                    p=probs,
                    replace=True
                )
            else:
                orientations = np.random.uniform(0, 180, n_spines)
        else:
            # Random orientations
            orientations = np.random.uniform(0, 180, n_spines)

        # Create spines
        for orientation in orientations:
            self._create_spine(orientation)

        print(f"Created {n_spines} boundary spines with {self.graph.number_of_nodes()} nodes")

    def _create_spine(self, orientation_deg: float):
        """
        Create a boundary-to-boundary spine at given orientation.

        Args:
            orientation_deg: Bearing in degrees [0, 180)
        """
        # Convert to radians
        angle = math.radians(orientation_deg)
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Find entry point on boundary
        # Try from a random side
        side = random.choice(['left', 'right', 'top', 'bottom'])

        if side == 'left':
            start_x = 0
            start_y = random.uniform(0, self.window_size_m)
        elif side == 'right':
            start_x = self.window_size_m
            start_y = random.uniform(0, self.window_size_m)
        elif side == 'bottom':
            start_x = random.uniform(0, self.window_size_m)
            start_y = 0
        else:  # top
            start_x = random.uniform(0, self.window_size_m)
            start_y = self.window_size_m

        # Trace line across window
        start_point = Point(start_x, start_y)

        # Find exit point
        # Extend line far enough to cross window
        line_length = 2 * self.window_size_m
        end_x = start_x + line_length * dx
        end_y = start_y + line_length * dy

        spine_line = LineString([(start_x, start_y), (end_x, end_y)])

        # Clip to window
        clipped = spine_line.intersection(self.window)

        if clipped.is_empty or clipped.geom_type != 'LineString':
            return

        # Create nodes along spine
        coords = list(clipped.coords)
        step_size = random.uniform(30, 60)  # meters between nodes

        # Interpolate points
        total_length = clipped.length
        n_segments = max(2, int(total_length / step_size))

        prev_node = None
        for i in range(n_segments + 1):
            t = i / n_segments
            point = clipped.interpolate(t, normalized=True)

            # Add node
            node_id = self._add_node(point.x, point.y)

            # Connect to previous
            if prev_node is not None:
                self._add_edge(prev_node, node_id)

            prev_node = node_id

    def _iterative_growth(self):
        """Main iterative growth loop with simulated annealing."""
        while self.iteration < self.config.max_iterations:
            self.iteration += 1

            # Compute progress
            progress = self.accepted_edges / max(
                self.reference.graph.number_of_edges(),
                1
            )
            progress = min(1.0, progress)

            # Temperature for SA
            temperature = self.config.initial_temp * (
                self.config.cooling_rate ** self.iteration
            )

            # Choose frontier node
            frontier_node = self._choose_frontier_node()

            if frontier_node is None:
                # No valid frontier, try to add random boundary connection
                frontier_node = random.choice(list(self.graph.nodes()))

            # Generate candidates
            candidates = self._generate_candidates(frontier_node)

            if not candidates:
                continue

            # Score candidates
            best_candidate = None
            best_score = float('inf')
            best_breakdown = None

            current_score, _ = self.objective.compute_cheap_score(
                self.graph, self.pos, progress
            )

            for candidate in candidates:
                # Try adding candidate
                temp_graph = self.graph.copy()
                temp_pos = self.pos.copy()

                # Add candidate edge
                u, v, new_node = candidate
                if new_node:
                    temp_pos[v] = new_node
                    temp_graph.add_node(v)

                temp_graph.add_edge(u, v)

                # Score
                new_score, breakdown = self.objective.compute_cheap_score(
                    temp_graph, temp_pos, progress
                )

                # Track best
                if new_score < best_score:
                    best_score = new_score
                    best_candidate = candidate
                    best_breakdown = breakdown

            # Simulated annealing acceptance
            if best_candidate is not None:
                delta_score = best_score - current_score

                # Accept if improves OR with probability based on temperature
                if delta_score < 0:
                    accept_prob = 1.0
                else:
                    accept_prob = math.exp(-delta_score / (temperature + 1e-10))

                if random.random() < accept_prob:
                    # Accept candidate
                    u, v, new_node = best_candidate

                    if new_node:
                        self._add_node_at_position(v, new_node)

                    self._add_edge(u, v)
                    self.accepted_edges += 1

                    # Track best score
                    if best_score < self.best_score:
                        self.best_score = best_score
                        self.no_improvement_count = 0
                    else:
                        self.no_improvement_count += 1

            # Periodic full audit
            if self.iteration % self.config.syntax_recompute_interval == 0:
                full_score, breakdown = self.objective.compute_full_score(
                    self.graph, self.pos, progress, self.iteration
                )
                self.audit_scores.append({
                    "iteration": self.iteration,
                    "score": full_score,
                    "breakdown": breakdown
                })

                print(f"Iter {self.iteration}: "
                      f"Nodes={self.graph.number_of_nodes()}, "
                      f"Edges={self.graph.number_of_edges()}, "
                      f"Score={full_score:.4f}, "
                      f"Temp={temperature:.4f}")

            # Check stopping conditions
            if self._should_stop(progress):
                break

    def _choose_frontier_node(self) -> Optional[int]:
        """
        Choose a frontier node for growth.

        Returns:
            Node ID or None
        """
        if self.graph.number_of_nodes() == 0:
            return None

        # Prefer nodes with low degree
        target_mean_degree = np.mean(list(self.reference.degree_distribution.keys()))

        candidates = []
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            if degree < target_mean_degree:
                # Weight by inverse degree (prefer lower degree)
                weight = 1.0 / (degree + 1)
                candidates.append((node, weight))

        if not candidates:
            # All nodes saturated, pick random
            return random.choice(list(self.graph.nodes()))

        # Weighted random choice
        nodes, weights = zip(*candidates)
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        return np.random.choice(nodes, p=weights)

    def _generate_candidates(
        self,
        from_node: int
    ) -> List[Tuple[int, int, Optional[Tuple[float, float]]]]:
        """
        Generate candidate edges from a node.

        Args:
            from_node: Source node ID

        Returns:
            List of (u, v, new_position) tuples
            new_position is None if connecting to existing node
        """
        candidates = []
        from_pos = self.pos[from_node]

        for _ in range(self.config.candidate_per_step):
            # Sample length from reference
            lengths = [l for l in self.reference.segment_length_hist[1] if l > 0]
            if lengths:
                bin_edges = self.reference.segment_length_hist[0]
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                counts = self.reference.segment_length_hist[1]

                # Sample weighted by counts
                total = np.sum(counts)
                if total > 0:
                    probs = counts / total
                    length = np.random.choice(bin_centers, p=probs)
                else:
                    length = random.uniform(
                        self.config.min_seg_len_m,
                        self.config.max_seg_len_m
                    )
            else:
                length = random.uniform(
                    self.config.min_seg_len_m,
                    self.config.max_seg_len_m
                )

            # Sample bearing from reference
            if len(self.reference.orientation_hist[1]) > 0:
                bin_edges = self.reference.orientation_hist[0]
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                counts = self.reference.orientation_hist[1]

                total = np.sum(counts)
                if total > 0:
                    probs = counts / total
                    bearing = np.random.choice(bin_centers, p=probs)
                else:
                    bearing = random.uniform(0, 180)
            else:
                bearing = random.uniform(0, 180)

            # Add noise
            bearing += random.gauss(0, 15)  # 15 degree std
            bearing = bearing % 180

            # Compute endpoint
            angle = math.radians(bearing)
            end_x = from_pos[0] + length * math.cos(angle)
            end_y = from_pos[1] + length * math.sin(angle)

            # Validate candidate
            candidate = self._validate_candidate(
                from_node,
                from_pos,
                (end_x, end_y)
            )

            if candidate:
                candidates.append(candidate)

        return candidates

    def _validate_candidate(
        self,
        from_node: int,
        from_pos: Tuple[float, float],
        to_pos: Tuple[float, float]
    ) -> Optional[Tuple[int, int, Optional[Tuple[float, float]]]]:
        """
        Validate candidate edge.

        Args:
            from_node: Source node ID
            from_pos: Source position
            to_pos: Target position

        Returns:
            (u, v, new_position) or None if invalid
        """
        # Check if endpoint inside window
        to_point = Point(to_pos)
        if not self.window.contains(to_point) and not self.window.touches(to_point):
            return None

        # Check if can snap to existing node
        existing_nodes = [self.pos[n] for n in self.graph.nodes() if n != from_node]
        snapped = snap_to_node(to_pos, existing_nodes, self.config.snap_tolerance_m)

        if snapped:
            # Find node ID
            to_node = None
            for n, p in self.pos.items():
                if p == snapped:
                    to_node = n
                    break

            if to_node is None:
                return None

            # Check if edge already exists
            if self.graph.has_edge(from_node, to_node):
                return None

            # Check for crossings
            candidate_line = LineString([from_pos, snapped])
            ignore_nodes = {from_node, to_node}

            if self.spatial_index.check_intersection(candidate_line, ignore_nodes):
                return None

            return (from_node, to_node, None)

        else:
            # New node
            to_node = self.next_node_id
            candidate_line = LineString([from_pos, to_pos])

            # Check for crossings
            ignore_nodes = {from_node}

            if self.spatial_index.check_intersection(candidate_line, ignore_nodes):
                return None

            return (from_node, to_node, to_pos)

    def _add_node(self, x: float, y: float) -> int:
        """Add node and return ID."""
        node_id = self.next_node_id
        self.next_node_id += 1
        self.graph.add_node(node_id, x=x, y=y)
        self.pos[node_id] = (x, y)
        return node_id

    def _add_node_at_position(self, node_id: int, pos: Tuple[float, float]):
        """Add node with specific ID."""
        self.graph.add_node(node_id, x=pos[0], y=pos[1])
        self.pos[node_id] = pos
        if node_id >= self.next_node_id:
            self.next_node_id = node_id + 1

    def _add_edge(self, u: int, v: int):
        """Add edge and update spatial index."""
        self.graph.add_edge(u, v)
        self.spatial_index.add_edge(self.pos[u], self.pos[v], u, v)

    def _should_stop(self, progress: float) -> bool:
        """Check stopping conditions."""
        # Minimum iterations
        if self.iteration < self.config.min_iterations:
            return False

        # Check morphology divergence threshold
        if len(self.audit_scores) > 0:
            last_audit = self.audit_scores[-1]
            breakdown = last_audit["breakdown"]

            # Check if morphology divergences are below threshold
            morph_good = all(
                breakdown.get(key, 1.0) < self.config.morph_divergence_threshold
                for key in ["density_error", "degree_divergence",
                           "length_divergence", "orientation_divergence",
                           "dead_end_error"]
            )

            if morph_good and len(self.audit_scores) >= self.config.no_improvement_audits:
                # Check if no improvement
                recent_scores = [a["score"] for a in self.audit_scores[-self.config.no_improvement_audits:]]
                if max(recent_scores) - min(recent_scores) < 0.01:
                    print("Converged: morphology within threshold and no improvement")
                    return True

        return False
