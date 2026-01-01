"""
Module A: Reference extraction from real city districts.
"""

import json
import numpy as np
import networkx as nx
import geopandas as gpd
from typing import Dict, Tuple, List, Optional
from pathlib import Path
from shapely.geometry import box

from .metrics import (
    MorphologyMetrics,
    SpaceSyntaxMetrics,
    compute_distribution_histogram,
)
from .utils import create_window_polygon


class ReferenceData:
    """Container for reference district data."""

    def __init__(self, city_name: str):
        """
        Initialize reference data container.

        Args:
            city_name: Name of reference city
        """
        self.city_name = city_name
        self.graph: Optional[nx.Graph] = None
        self.pos: Optional[Dict] = None

        # Morphology metrics
        self.node_density: float = 0.0
        self.degree_distribution: Dict[int, int] = {}
        self.segment_length_hist: Tuple[np.ndarray, np.ndarray] = (
            np.array([]), np.array([])
        )
        self.orientation_hist: Tuple[np.ndarray, np.ndarray] = (
            np.array([]), np.array([])
        )
        self.dead_end_ratio: float = 0.0

        # Syntax metrics
        self.mean_depth: float = 0.0
        self.local_integration_hist: Tuple[np.ndarray, np.ndarray] = (
            np.array([]), np.array([])
        )
        self.choice_hist: Tuple[np.ndarray, np.ndarray] = (
            np.array([]), np.array([])
        )
        self.intelligibility: float = 0.0


class ReferenceExtractor:
    """Extract reference metrics from real city districts."""

    def __init__(self, data_dir: str = "inv_city/outputs"):
        """
        Initialize reference extractor.

        Args:
            data_dir: Directory containing reference GeoJSON files
        """
        self.data_dir = Path(data_dir)

    def load_from_geojson(
        self,
        city_name: str,
        window_size_m: float = 500.0,
    ) -> ReferenceData:
        """
        Load reference data from GeoJSON files.

        Args:
            city_name: City name (e.g., 'london', 'berlin')
            window_size_m: Window size in meters

        Returns:
            ReferenceData object with computed metrics
        """
        ref = ReferenceData(city_name)

        # Load edges and nodes
        edges_file = self.data_dir / "geojson" / f"{city_name}_edges.geojson"
        nodes_file = self.data_dir / "geojson" / f"{city_name}_nodes.geojson"

        if not edges_file.exists() or not nodes_file.exists():
            raise FileNotFoundError(
                f"Reference data not found for {city_name} in {self.data_dir}"
            )

        # Load GeoDataFrames
        edges_gdf = gpd.read_file(edges_file)
        nodes_gdf = gpd.read_file(nodes_file)

        # Build NetworkX graph
        ref.graph, ref.pos = self._build_graph_from_geodata(edges_gdf, nodes_gdf)

        # Compute morphology metrics
        morph = MorphologyMetrics.compute_all_morphology(
            ref.graph, ref.pos, window_size_m
        )

        ref.node_density = morph["node_density"]
        ref.degree_distribution = morph["degree_distribution"]
        ref.dead_end_ratio = morph["dead_end_ratio"]

        # Segment length histogram
        segment_lengths = morph["segment_lengths"]
        bin_edges, counts = compute_distribution_histogram(
            segment_lengths,
            num_bins=20,
            range_min=0,
            range_max=max(segment_lengths) if segment_lengths else 100
        )
        ref.segment_length_hist = (bin_edges, counts)

        # Orientation histogram
        orientation_data = morph["orientation_histogram"]
        ref.orientation_hist = (
            np.array(orientation_data["bin_edges"]),
            np.array(orientation_data["counts"])
        )

        # Compute syntax metrics
        syntax = SpaceSyntaxMetrics.compute_all_syntax(ref.graph, radius=3)

        ref.mean_depth = syntax["mean_depth"]
        ref.intelligibility = syntax["intelligibility"]

        # Local integration histogram
        local_int_values = list(syntax["local_integration"].values())
        if local_int_values:
            bin_edges, counts = compute_distribution_histogram(
                local_int_values,
                num_bins=15,
                range_min=0,
                range_max=max(local_int_values)
            )
            ref.local_integration_hist = (bin_edges, counts)

        # Choice histogram
        choice_values = list(syntax["choice"].values())
        if choice_values:
            bin_edges, counts = compute_distribution_histogram(
                choice_values,
                num_bins=15,
                range_min=0,
                range_max=max(choice_values)
            )
            ref.choice_hist = (bin_edges, counts)

        return ref

    def load_from_metrics_json(self, city_name: str) -> ReferenceData:
        """
        Load reference data from pre-computed metrics JSON.

        Args:
            city_name: City name

        Returns:
            ReferenceData object
        """
        metrics_file = self.data_dir / "metrics" / "urban_metrics.json"

        with open(metrics_file, 'r') as f:
            data = json.load(f)

        city_data = data["urban_metrics"][city_name]

        ref = ReferenceData(city_name)

        # Extract morphology from JSON
        ref.node_density = city_data["nodes"]["total_count"] / 0.25  # 500m x 500m = 0.25 km²

        # Degree distribution
        degree_dist_str = city_data["nodes"]["degree_distribution"]
        ref.degree_distribution = {int(k): v for k, v in degree_dist_str.items()}

        # Segment lengths
        seg_data = city_data["edges"]["segment_length_distribution"]
        ref.segment_length_hist = (
            np.array(seg_data["bins"]),
            np.array(seg_data["counts"])
        )

        # Dead-end ratio (compute from degree distribution)
        total_nodes = sum(ref.degree_distribution.values())
        dead_ends = ref.degree_distribution.get(1, 0)
        ref.dead_end_ratio = dead_ends / total_nodes if total_nodes > 0 else 0.0

        # Note: Orientation and syntax metrics not in JSON, need to compute from GeoJSON
        # For now, set placeholder values
        ref.orientation_hist = (
            np.linspace(0, 180, 19),
            np.zeros(18)
        )

        return ref

    def _build_graph_from_geodata(
        self,
        edges_gdf: gpd.GeoDataFrame,
        nodes_gdf: gpd.GeoDataFrame
    ) -> Tuple[nx.Graph, Dict]:
        """
        Build NetworkX graph from GeoDataFrames.

        Args:
            edges_gdf: Edges GeoDataFrame
            nodes_gdf: Nodes GeoDataFrame

        Returns:
            (graph, pos) tuple
        """
        G = nx.Graph()

        # Add nodes with positions
        pos = {}
        for idx, row in nodes_gdf.iterrows():
            node_id = idx
            geom = row.geometry
            pos[node_id] = (geom.x, geom.y)
            G.add_node(node_id, x=geom.x, y=geom.y)

        # Add edges
        # Try to infer connectivity from geometry
        for idx, row in edges_gdf.iterrows():
            geom = row.geometry

            # Get start and end points
            coords = list(geom.coords)
            start_point = coords[0]
            end_point = coords[-1]

            # Find matching nodes
            u = self._find_nearest_node(start_point, pos)
            v = self._find_nearest_node(end_point, pos)

            if u is not None and v is not None and u != v:
                G.add_edge(u, v, length=geom.length)

        return G, pos

    def _find_nearest_node(
        self,
        point: Tuple[float, float],
        pos: Dict,
        tolerance: float = 2.0
    ) -> Optional[int]:
        """
        Find nearest node to point within tolerance.

        Args:
            point: Query point (x, y)
            pos: Node positions
            tolerance: Maximum distance

        Returns:
            Node ID or None
        """
        min_dist = float('inf')
        nearest_node = None

        for node_id, node_pos in pos.items():
            dist = np.linalg.norm(np.array(point) - np.array(node_pos))
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id

        if min_dist <= tolerance:
            return nearest_node
        return None

    def blend_references(
        self,
        references: List[ReferenceData],
        weights: Optional[List[float]] = None
    ) -> ReferenceData:
        """
        Blend multiple reference datasets into weighted average.

        Args:
            references: List of ReferenceData objects
            weights: Optional weights (default: uniform)

        Returns:
            Blended ReferenceData
        """
        if not references:
            raise ValueError("No references provided")

        if weights is None:
            weights = [1.0 / len(references)] * len(references)

        if len(weights) != len(references):
            raise ValueError("Number of weights must match number of references")

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Create blended reference
        blended = ReferenceData("blended")

        # Blend scalar values
        blended.node_density = sum(
            ref.node_density * w for ref, w in zip(references, weights)
        )
        blended.dead_end_ratio = sum(
            ref.dead_end_ratio * w for ref, w in zip(references, weights)
        )
        blended.mean_depth = sum(
            ref.mean_depth * w for ref, w in zip(references, weights)
        )
        blended.intelligibility = sum(
            ref.intelligibility * w for ref, w in zip(references, weights)
        )

        # Blend degree distribution
        all_degrees = set()
        for ref in references:
            all_degrees.update(ref.degree_distribution.keys())

        blended.degree_distribution = {}
        for degree in all_degrees:
            blended.degree_distribution[degree] = int(sum(
                ref.degree_distribution.get(degree, 0) * w
                for ref, w in zip(references, weights)
            ))

        # Blend histograms (use first reference's bin structure)
        ref0 = references[0]

        # Segment lengths
        blended_counts = sum(
            ref.segment_length_hist[1] * w
            for ref, w in zip(references, weights)
        )
        blended.segment_length_hist = (ref0.segment_length_hist[0], blended_counts)

        # Orientation
        blended_counts = sum(
            ref.orientation_hist[1] * w
            for ref, w in zip(references, weights)
        )
        blended.orientation_hist = (ref0.orientation_hist[0], blended_counts)

        # Local integration
        if ref0.local_integration_hist[0].size > 0:
            blended_counts = sum(
                ref.local_integration_hist[1] * w
                for ref, w in zip(references, weights)
            )
            blended.local_integration_hist = (
                ref0.local_integration_hist[0],
                blended_counts
            )

        # Choice
        if ref0.choice_hist[0].size > 0:
            blended_counts = sum(
                ref.choice_hist[1] * w
                for ref, w in zip(references, weights)
            )
            blended.choice_hist = (ref0.choice_hist[0], blended_counts)

        return blended
