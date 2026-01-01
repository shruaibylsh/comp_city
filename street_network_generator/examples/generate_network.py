#!/usr/bin/env python3
"""
Example script for generating street networks.

Usage:
    python generate_network.py --city london --output ./output_london
    python generate_network.py --city berlin --config custom_config.json
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from street_network_generator import (
    GeneratorConfig,
    ReferenceExtractor,
    StreetNetworkGenerator,
)
from street_network_generator.validation import NetworkValidator


def main():
    parser = argparse.ArgumentParser(
        description="Generate calibrated street networks"
    )
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        choices=["london", "berlin", "belgrade", "torino"],
        help="Reference city to match"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="inv_city/outputs",
        help="Directory with reference data"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        print(f"Loading config from {args.config}")
        config = GeneratorConfig.from_json(args.config)
    else:
        print("Using default configuration")
        # Use example config
        example_config = Path(__file__).parent / "config.json"
        if example_config.exists():
            config = GeneratorConfig.from_json(str(example_config))
        else:
            config = GeneratorConfig()

    print(f"\n{'='*60}")
    print(f"Street Network Generator - Phase 1 MVP")
    print(f"{'='*60}")
    print(f"Reference City: {args.city}")
    print(f"Window Size: {config.window_size_m}m")
    print(f"Max Iterations: {config.max_iterations}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    # Extract reference data
    print("Step 1: Loading reference data...")
    extractor = ReferenceExtractor(data_dir=args.data_dir)

    try:
        reference = extractor.load_from_geojson(
            args.city,
            window_size_m=config.window_size_m
        )
        print(f"✓ Loaded reference data for {args.city}")
        print(f"  - Nodes: {reference.graph.number_of_nodes()}")
        print(f"  - Edges: {reference.graph.number_of_edges()}")
        print(f"  - Node density: {reference.node_density:.2f} nodes/km²")
        print(f"  - Mean depth: {reference.mean_depth:.3f}")
        print()
    except Exception as e:
        print(f"✗ Error loading reference data: {e}")
        return 1

    # Generate network
    print("Step 2: Generating network...")
    print("-" * 60)

    generator = StreetNetworkGenerator(reference, config)
    graph, pos, metadata = generator.generate()

    print("-" * 60)
    print(f"✓ Generation complete!")
    print(f"  - Iterations: {metadata['iterations']}")
    print(f"  - Nodes: {graph.number_of_nodes()}")
    print(f"  - Edges: {graph.number_of_edges()}")
    print(f"  - Final score: {metadata['final_score']:.4f}")
    print()

    # Validate and export
    print("Step 3: Validating and exporting results...")
    validator = NetworkValidator(config)

    output_dir = Path(args.output)
    validator.validate_and_export(
        graph,
        pos,
        reference,
        metadata,
        str(output_dir),
        prefix=f"{args.city}_generated"
    )

    print(f"\n{'='*60}")
    print("✓ All done!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - {args.city}_generated_nodes.geojson")
    print(f"  - {args.city}_generated_edges.geojson")
    print(f"  - {args.city}_generated_graph.gpickle")
    print(f"  - {args.city}_generated_metrics.json")
    print(f"  - {args.city}_generated_report.md")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
