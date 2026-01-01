"""
Module F: Validation and reporting.
"""

import json
import numpy as np
import networkx as nx
import geopandas as gpd
from pathlib import Path
from typing import Dict, Tuple
from shapely.geometry import Point, LineString

from .reference import ReferenceData
from .metrics import MorphologyMetrics, SpaceSyntaxMetrics
from .config import GeneratorConfig


class NetworkValidator:
    """Validate and report on generated networks."""

    def __init__(self, config: GeneratorConfig):
        """
        Initialize validator.

        Args:
            config: Generator configuration
        """
        self.config = config

    def validate_and_export(
        self,
        graph: nx.Graph,
        pos: Dict,
        reference: ReferenceData,
        metadata: Dict,
        output_dir: str,
        prefix: str = "generated"
    ):
        """
        Validate network and export all outputs.

        Args:
            graph: Generated graph
            pos: Node positions
            reference: Reference data
            metadata: Generation metadata
            output_dir: Output directory
            prefix: Filename prefix
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Computing final metrics...")

        # Compute all metrics
        morph = MorphologyMetrics.compute_all_morphology(
            graph, pos, self.config.window_size_m
        )

        syntax = SpaceSyntaxMetrics.compute_all_syntax(graph, radius=3)

        # Compute divergences
        divergences = self._compute_divergences(
            graph, pos, morph, syntax, reference
        )

        # Export GeoJSON
        print("Exporting GeoJSON...")
        self._export_geojson(graph, pos, output_path, prefix)

        # Export graph
        print("Exporting NetworkX graph...")
        graph_file = output_path / f"{prefix}_graph.gpickle"
        nx.write_gpickle(graph, graph_file)

        # Export metrics JSON
        print("Exporting metrics...")
        metrics_data = {
            "metadata": metadata,
            "morphology": {
                "node_density": morph["node_density"],
                "degree_distribution": morph["degree_distribution"],
                "dead_end_ratio": morph["dead_end_ratio"],
                "segment_length_stats": {
                    "mean": float(np.mean(morph["segment_lengths"])) if morph["segment_lengths"] else 0,
                    "median": float(np.median(morph["segment_lengths"])) if morph["segment_lengths"] else 0,
                    "std": float(np.std(morph["segment_lengths"])) if morph["segment_lengths"] else 0,
                },
                "orientation_entropy": morph["orientation_histogram"]["entropy"],
            },
            "syntax": {
                "mean_depth": syntax["mean_depth"],
                "intelligibility": syntax["intelligibility"],
                "local_integration_stats": {
                    "mean": float(np.mean(list(syntax["local_integration"].values()))) if syntax["local_integration"] else 0,
                    "std": float(np.std(list(syntax["local_integration"].values()))) if syntax["local_integration"] else 0,
                },
                "choice_stats": {
                    "mean": float(np.mean(list(syntax["choice"].values()))) if syntax["choice"] else 0,
                    "std": float(np.std(list(syntax["choice"].values()))) if syntax["choice"] else 0,
                },
            },
            "divergences": divergences,
            "reference_city": reference.city_name,
        }

        metrics_file = output_path / f"{prefix}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        # Generate report
        print("Generating report...")
        self._generate_report(
            graph, pos, morph, syntax, reference, divergences, metadata,
            output_path, prefix
        )

        print(f"Validation complete. Results in {output_dir}/")

    def _export_geojson(
        self,
        graph: nx.Graph,
        pos: Dict,
        output_path: Path,
        prefix: str
    ):
        """Export graph to GeoJSON."""
        # Export nodes
        node_features = []
        for node_id in graph.nodes():
            x, y = pos[node_id]
            degree = graph.degree(node_id)

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [x, y]
                },
                "properties": {
                    "node_id": int(node_id),
                    "degree": degree,
                }
            }
            node_features.append(feature)

        nodes_geojson = {
            "type": "FeatureCollection",
            "features": node_features
        }

        nodes_file = output_path / f"{prefix}_nodes.geojson"
        with open(nodes_file, 'w') as f:
            json.dump(nodes_geojson, f)

        # Export edges
        edge_features = []
        for u, v in graph.edges():
            u_pos = pos[u]
            v_pos = pos[v]

            length = np.linalg.norm(np.array(u_pos) - np.array(v_pos))

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[u_pos[0], u_pos[1]], [v_pos[0], v_pos[1]]]
                },
                "properties": {
                    "u": int(u),
                    "v": int(v),
                    "length": float(length),
                }
            }
            edge_features.append(feature)

        edges_geojson = {
            "type": "FeatureCollection",
            "features": edge_features
        }

        edges_file = output_path / f"{prefix}_edges.geojson"
        with open(edges_file, 'w') as f:
            json.dump(edges_geojson, f)

    def _compute_divergences(
        self,
        graph: nx.Graph,
        pos: Dict,
        morph: Dict,
        syntax: Dict,
        reference: ReferenceData
    ) -> Dict:
        """Compute all divergences from reference."""
        divergences = {}

        # Node density
        ref_density = reference.node_density
        gen_density = morph["node_density"]
        divergences["node_density"] = {
            "reference": float(ref_density),
            "generated": float(gen_density),
            "relative_error": float(abs(gen_density - ref_density) / (ref_density + 1e-6))
        }

        # Dead-end ratio
        ref_dead_end = reference.dead_end_ratio
        gen_dead_end = morph["dead_end_ratio"]
        divergences["dead_end_ratio"] = {
            "reference": float(ref_dead_end),
            "generated": float(gen_dead_end),
            "absolute_error": float(abs(gen_dead_end - ref_dead_end))
        }

        # Mean depth
        ref_mean_depth = reference.mean_depth
        gen_mean_depth = syntax["mean_depth"]
        divergences["mean_depth"] = {
            "reference": float(ref_mean_depth),
            "generated": float(gen_mean_depth),
            "relative_error": float(abs(gen_mean_depth - ref_mean_depth) / (ref_mean_depth + 1e-6))
        }

        # Intelligibility
        ref_intelligibility = reference.intelligibility
        gen_intelligibility = syntax["intelligibility"]
        divergences["intelligibility"] = {
            "reference": float(ref_intelligibility),
            "generated": float(gen_intelligibility),
            "absolute_error": float(abs(gen_intelligibility - ref_intelligibility))
        }

        return divergences

    def _generate_report(
        self,
        graph: nx.Graph,
        pos: Dict,
        morph: Dict,
        syntax: Dict,
        reference: ReferenceData,
        divergences: Dict,
        metadata: Dict,
        output_path: Path,
        prefix: str
    ):
        """Generate markdown report."""
        report = []

        report.append(f"# Street Network Generation Report\n")
        report.append(f"\n## Configuration\n")
        report.append(f"- Reference City: **{reference.city_name}**\n")
        report.append(f"- Window Size: {self.config.window_size_m}m × {self.config.window_size_m}m\n")
        report.append(f"- Seed: {self.config.seed}\n")

        report.append(f"\n## Generation Statistics\n")
        report.append(f"- Iterations: {metadata['iterations']}\n")
        report.append(f"- Accepted Edges: {metadata['accepted_edges']}\n")
        report.append(f"- Final Score: {metadata['final_score']:.4f}\n")
        report.append(f"- Total Nodes: {graph.number_of_nodes()}\n")
        report.append(f"- Total Edges: {graph.number_of_edges()}\n")

        report.append(f"\n## Morphology Comparison\n")
        report.append(f"\n### Node Density (nodes/km²)\n")
        report.append(f"- Reference: {divergences['node_density']['reference']:.2f}\n")
        report.append(f"- Generated: {divergences['node_density']['generated']:.2f}\n")
        report.append(f"- Relative Error: {divergences['node_density']['relative_error']:.2%}\n")

        report.append(f"\n### Dead-End Ratio\n")
        report.append(f"- Reference: {divergences['dead_end_ratio']['reference']:.3f}\n")
        report.append(f"- Generated: {divergences['dead_end_ratio']['generated']:.3f}\n")
        report.append(f"- Absolute Error: {divergences['dead_end_ratio']['absolute_error']:.3f}\n")

        report.append(f"\n### Degree Distribution\n")
        report.append(f"```\n")
        report.append(f"Degree | Reference | Generated\n")
        report.append(f"-------|-----------|----------\n")

        all_degrees = set(reference.degree_distribution.keys()) | set(morph["degree_distribution"].keys())
        for deg in sorted(all_degrees):
            ref_count = reference.degree_distribution.get(deg, 0)
            gen_count = morph["degree_distribution"].get(deg, 0)
            report.append(f"  {deg}    |   {ref_count:4d}    |   {gen_count:4d}\n")
        report.append(f"```\n")

        report.append(f"\n## Space Syntax Comparison\n")
        report.append(f"\n### Mean Depth\n")
        report.append(f"- Reference: {divergences['mean_depth']['reference']:.3f}\n")
        report.append(f"- Generated: {divergences['mean_depth']['generated']:.3f}\n")
        report.append(f"- Relative Error: {divergences['mean_depth']['relative_error']:.2%}\n")

        report.append(f"\n### Intelligibility (Correlation: Degree ↔ Local Integration)\n")
        report.append(f"- Reference: {divergences['intelligibility']['reference']:.3f}\n")
        report.append(f"- Generated: {divergences['intelligibility']['generated']:.3f}\n")
        report.append(f"- Absolute Error: {divergences['intelligibility']['absolute_error']:.3f}\n")

        report.append(f"\n## Audit History\n")
        report.append(f"\n```\n")
        report.append(f"Iteration | Score  | Morph Score | Syntax Score\n")
        report.append(f"----------|--------|-------------|-------------\n")
        for audit in metadata["audit_history"]:
            report.append(
                f"  {audit['iteration']:5d}   | {audit['score']:.4f} | "
                f"  {audit['breakdown'].get('morph_score', 0):.4f}    | "
                f"   {audit['breakdown'].get('syntax_score', 0):.4f}\n"
            )
        report.append(f"```\n")

        # Write report
        report_file = output_path / f"{prefix}_report.md"
        with open(report_file, 'w') as f:
            f.writelines(report)
