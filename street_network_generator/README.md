# Street Network Generator

A Python system for generating planar street networks calibrated to real-world urban morphology and space syntax patterns.

## Overview

This generator creates synthetic 500×500m street networks that match statistical distributions from reference city districts. It combines:

- **Urban morphology metrics**: node density, degree distribution, segment lengths, orientation
- **Space syntax metrics**: mean depth, local integration, choice, intelligibility
- **Planar growth algorithm** with simulated annealing
- **Progressive weight scheduling** (morphology first, then syntax)

## Current Status: Phase 1 (MVP)

✓ **Implemented**:
- Morphology-based generation (node density, degrees, segment lengths, orientation)
- Node-based space syntax (mean depth, local integration, choice, intelligibility)
- Simulated annealing with temperature scheduling
- Reference extraction from GeoJSON
- Validation and reporting
- GeoJSON/NetworkX export

⏳ **Planned**:
- Phase 2: Repair pass, multi-city blending
- Phase 3: Segment-based syntax, block areas, angular metrics

## Quick Start

### Installation

```bash
# From comp_city root directory
cd street_network_generator
pip install -r requirements.txt
```

### Generate a Network

```bash
cd examples
python generate_network.py --city london --output ./output_london
```

### Available Reference Cities

- `london` - London, UK
- `berlin` - Berlin, Germany
- `belgrade` - Belgrade, Serbia
- `torino` - Torino, Italy

### Using Custom Configuration

```bash
python generate_network.py --city berlin --config my_config.json --output ./outputs
```

See `examples/config.json` for all available parameters.

## Output Files

For each generation, you get:

- `{city}_generated_nodes.geojson` - Network nodes (Point features)
- `{city}_generated_edges.geojson` - Network edges (LineString features)
- `{city}_generated_graph.gpickle` - NetworkX graph (Python pickle)
- `{city}_generated_metrics.json` - All computed metrics + divergences
- `{city}_generated_report.md` - Human-readable comparison report

## Architecture

```
street_network_generator/
├── __init__.py           # Package exports
├── config.py             # GeneratorConfig dataclass
├── reference.py          # Module A: Reference extraction
├── generator.py          # Module B: Network growth engine
├── objective.py          # Module C: Objective function
├── metrics.py            # Morphology + syntax computation
├── utils.py              # Geometry helpers, spatial index
└── validation.py         # Module F: Validation & export
```

## Key Concepts

### Two-Tier Objective Function

**Cheap metrics** (computed every iteration):
- Degree distribution
- Segment length distribution
- Orientation histogram
- Node density
- Connectivity penalties

**Expensive metrics** (computed every N=80 edges):
- Mean depth
- Local integration (R=3)
- Choice (betweenness)
- Intelligibility

This design keeps generation fast (~1-5 min per network).

### Weight Scheduling

Weights automatically adjust during generation:

- **First 60%**: morphology only (w_morph=1.0, w_syntax=0.0)
- **Next 20%**: ramp syntax to 0.2
- **Final 20%**: ramp syntax to 0.3

This prevents premature optimization and ensures basic topology is correct first.

### Simulated Annealing

Temperature: `T = T_initial × (cooling_rate ^ iteration)`

Acceptance: `P(accept) = exp(-Δscore / T)` if Δscore > 0, else always accept

Default: `T_initial=5.0`, `cooling_rate=0.997`

## Configuration Parameters

See `examples/config.json` for full configuration with comments.

Key parameters:

```json
{
  "seed": 42,
  "window_size_m": 500,
  "min_iterations": 250,
  "max_iterations": 2500,
  "syntax_recompute_interval": 80,
  "candidate_per_step": 12,
  "initial_temp": 5.0,
  "cooling_rate": 0.997
}
```

## Performance

**Phase 1 targets**: 1-5 minutes per network on a laptop

Typical generation:
- 250-1000 iterations
- 200-400 nodes
- 400-800 edges

## Stopping Conditions

Generation stops when:

1. All morphology divergences < 10% threshold, AND
2. No score improvement for last 5 audits, AND
3. Iteration ≥ min_iterations

Or hard stop at max_iterations.

## Development Phases

**Phase 1** (Current - MVP):
- ✓ Morphology matching
- ✓ Node-based syntax
- ✓ Basic validation

**Phase 2** (Next):
- Repair pass (snap nodes, split edges, add T-junctions)
- Multi-city blending
- Enhanced reporting with plots

**Phase 3** (Future):
- Segment-based space syntax
- Angular metrics
- Block area distributions
- Performance optimization

## Example Usage in Code

```python
from street_network_generator import (
    GeneratorConfig,
    ReferenceExtractor,
    StreetNetworkGenerator,
)

# Load reference
extractor = ReferenceExtractor(data_dir="inv_city/outputs")
reference = extractor.load_from_geojson("london", window_size_m=500)

# Configure
config = GeneratorConfig(seed=42, max_iterations=1000)

# Generate
generator = StreetNetworkGenerator(reference, config)
graph, pos, metadata = generator.generate()

print(f"Generated {graph.number_of_nodes()} nodes, "
      f"{graph.number_of_edges()} edges")
```

## Reference Data

The generator expects reference data in `inv_city/outputs/geojson/`:

- `{city}_edges.geojson` - Street edges
- `{city}_nodes.geojson` - Street nodes

These files should already exist from previous analysis.

## Citation

If you use this generator, please cite:

```
Calibrated Urban Street Network Generator
Phase 1: Morphology + Node-based Space Syntax
2024
```

## License

MIT License (pending)

## Contact

For questions or issues, please open a GitHub issue.
