"""
Utility functions for geometry operations and spatial indexing.
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Set
from shapely.geometry import Point, LineString, Polygon, box
from shapely.strtree import STRtree
import networkx as nx


def calculate_bearing(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate bearing (0-180°) of line segment from p1 to p2.

    Args:
        p1: Start point (x, y)
        p2: End point (x, y)

    Returns:
        Bearing in degrees [0, 180)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # Calculate angle in radians
    angle = math.atan2(dy, dx)

    # Convert to degrees
    bearing = math.degrees(angle)

    # Normalize to [0, 180) - we don't care about direction
    if bearing < 0:
        bearing += 180
    if bearing >= 180:
        bearing -= 180

    return bearing


def point_to_line_distance(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float]
) -> float:
    """
    Calculate perpendicular distance from point to line segment.

    Args:
        point: Query point (x, y)
        line_start: Line start (x, y)
        line_end: Line end (x, y)

    Returns:
        Distance in meters
    """
    p = Point(point)
    line = LineString([line_start, line_end])
    return p.distance(line)


def create_window_polygon(size_m: float) -> Polygon:
    """
    Create square window polygon [0, size_m] x [0, size_m].

    Args:
        size_m: Window size in meters

    Returns:
        Square polygon
    """
    return box(0, 0, size_m, size_m)


def snap_to_node(
    point: Tuple[float, float],
    nodes: List[Tuple[float, float]],
    tolerance: float
) -> Optional[Tuple[float, float]]:
    """
    Find nearest node within tolerance and return it, or None.

    Args:
        point: Query point (x, y)
        nodes: List of existing nodes
        tolerance: Maximum snap distance

    Returns:
        Nearest node if within tolerance, else None
    """
    if not nodes:
        return None

    nodes_array = np.array(nodes)
    point_array = np.array(point)

    # Vectorized distance calculation
    distances = np.sqrt(np.sum((nodes_array - point_array) ** 2, axis=1))

    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]

    if min_dist <= tolerance:
        return tuple(nodes[min_idx])
    return None


class SpatialIndex:
    """
    Spatial index for fast edge intersection queries.
    """

    def __init__(self):
        """Initialize empty spatial index."""
        self.edges: List[LineString] = []
        self.edge_data: List[Tuple[int, int]] = []  # (u, v) node pairs
        self.tree: Optional[STRtree] = None

    def add_edge(self, u_pos: Tuple[float, float], v_pos: Tuple[float, float], u: int, v: int):
        """
        Add edge to spatial index.

        Args:
            u_pos: Start node position
            v_pos: End node position
            u: Start node ID
            v: End node ID
        """
        line = LineString([u_pos, v_pos])
        self.edges.append(line)
        self.edge_data.append((u, v))
        # Rebuild tree (will be lazy rebuilt on next query)
        self.tree = None

    def _ensure_tree(self):
        """Rebuild spatial index tree if needed."""
        if self.tree is None and self.edges:
            self.tree = STRtree(self.edges)

    def check_intersection(
        self,
        candidate_line: LineString,
        ignore_nodes: Set[int] = None
    ) -> bool:
        """
        Check if candidate line intersects any existing edges (invalid crossing).

        Args:
            candidate_line: Proposed new edge
            ignore_nodes: Node IDs to ignore (endpoints of candidate)

        Returns:
            True if invalid intersection found, False otherwise
        """
        self._ensure_tree()

        if not self.edges:
            return False

        if ignore_nodes is None:
            ignore_nodes = set()

        # Query potential intersections
        potential_hits = self.tree.query(candidate_line)

        for idx in potential_hits:
            edge = self.edges[idx]
            u, v = self.edge_data[idx]

            # Skip if edge shares a node with candidate (valid connection)
            if u in ignore_nodes or v in ignore_nodes:
                continue

            # Check for actual intersection (not just touch at endpoints)
            if candidate_line.crosses(edge):
                return True

            # Check for overlap (collinear segments)
            if candidate_line.overlaps(edge):
                return True

        return False

    def find_nearby_nodes(
        self,
        point: Tuple[float, float],
        radius: float
    ) -> List[Tuple[int, float]]:
        """
        Find all nodes within radius of point.

        Args:
            point: Query point (x, y)
            radius: Search radius

        Returns:
            List of (node_id, distance) tuples
        """
        results = []
        p = Point(point)

        for i, (u, v) in enumerate(self.edge_data):
            edge = self.edges[i]

            # Check both endpoints
            for node_id, node_geom in [(u, Point(edge.coords[0])), (v, Point(edge.coords[1]))]:
                dist = p.distance(node_geom)
                if dist <= radius:
                    results.append((node_id, dist))

        # Remove duplicates and sort by distance
        unique_results = {}
        for node_id, dist in results:
            if node_id not in unique_results or dist < unique_results[node_id]:
                unique_results[node_id] = dist

        return sorted(unique_results.items(), key=lambda x: x[1])


def compute_orientation_histogram(
    graph: nx.Graph,
    pos: dict,
    num_bins: int = 18
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute orientation histogram for edges.

    Args:
        graph: NetworkX graph
        pos: Node positions dict {node_id: (x, y)}
        num_bins: Number of bins for [0, 180)

    Returns:
        (bin_edges, counts) arrays
    """
    bearings = []

    for u, v in graph.edges():
        u_pos = pos[u]
        v_pos = pos[v]
        bearing = calculate_bearing(u_pos, v_pos)
        bearings.append(bearing)

    if not bearings:
        return np.linspace(0, 180, num_bins + 1), np.zeros(num_bins)

    counts, bin_edges = np.histogram(bearings, bins=num_bins, range=(0, 180))
    return bin_edges, counts


def compute_entropy(histogram_counts: np.ndarray) -> float:
    """
    Compute Shannon entropy of histogram.

    Args:
        histogram_counts: Array of bin counts

    Returns:
        Entropy value
    """
    # Normalize to probabilities
    total = np.sum(histogram_counts)
    if total == 0:
        return 0.0

    probs = histogram_counts / total
    # Remove zeros to avoid log(0)
    probs = probs[probs > 0]

    return -np.sum(probs * np.log2(probs))
