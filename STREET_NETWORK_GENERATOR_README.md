# Calibrated Street Network Generator - Phase 1

**A Python system for generating synthetic 500×500m street networks that match real-world urban morphology and space syntax patterns.**

---

## 🎯 Project Overview

This generator creates planar street networks calibrated to reference city districts using:

- **Urban Morphology**: node density, degree distribution, segment lengths, orientation
- **Space Syntax**: mean depth, local integration, choice, intelligibility
- **Simulated Annealing**: temperature-based optimization with progressive weight scheduling
- **Histogram Matching**: distributions, not just averages

### Cities Supported

| City | Coordinates | Window | Characteristics |
|------|-------------|--------|-----------------|
| **London, UK** | 51.511°N, 0.130°W | 500×500m | Dense, mixed grid/irregular |
| **Berlin, Germany** | 52.528°N, 13.403°E | 500×500m | Grid-like, regular blocks |
| **Belgrade, Serbia** | 44.816°N, 20.462°E | 500×500m | Irregular, organic growth |
| **Torino, Italy** | 45.069°N, 7.682°E | 500×500m | Strong orthogonal grid |

---

## 📊 Workflow: Interactive Jupyter Notebooks

### **Notebook 01: Reference Data Analysis**
`01_reference_data_analysis.ipynb`

**Purpose**: Load and visualize all 4 reference cities

**Outputs**:
- Network visualizations for all 4 cities
- Morphology comparison tables
- Degree distribution plots
- Segment length histograms
- Orientation (bearing) distributions
- Space syntax scalar metrics

**Key Visualizations**:
- 2×2 grid of reference networks colored by node degree
- Comparative histograms across cities
- Summary statistics table

**What You'll Learn**:
- Which cities have grid patterns vs. irregular?
- Distribution shapes for each metric
- Space syntax correlation patterns

---

### **Notebook 02: Generate Single Network**
`02_generate_single_network.ipynb`

**Purpose**: Generate ONE network for a target city with full tracking

**Features**:
- Configurable parameters (seed, iterations, temperature)
- Real-time progress tracking via audit history
- Full comparison: reference vs. generated
- Export to GeoJSON + metrics JSON

**Visualizations**:
- Generation progress plot (score convergence)
- Side-by-side network comparison (reference | generated)
- Histogram overlays:
  - Degree distribution
  - Segment lengths
  - Orientation
  - Local integration
  - Choice (betweenness)
- Summary statistics table

**Configuration Example**:
```python
TARGET_CITY = 'london'
SEED = 42
MAX_ITERATIONS = 1500
```

**Outputs**:
- `{city}_generated_nodes.geojson`
- `{city}_generated_edges.geojson`
- `{city}_generated_graph.gpickle`
- `{city}_generated_metrics.json`
- `{city}_generated_report.md`

---

### **Notebook 03: Batch Generation (10 Networks)**
`03_batch_generation_10_networks.ipynb`

**Purpose**: Generate **10 networks** with different seeds to test robustness

**Features**:
- Loop over 10 seeds (100-109)
- Automated batch processing with progress bars
- Statistical analysis across all runs
- Best/worst network identification

**Visualizations**:
- **5×2 grid** showing all 10 generated networks
- Boxplots: distribution of metrics across runs
- Best vs. worst comparison
- Aggregated histograms (all networks combined)

**Statistics Computed**:
- Mean ± Std for all metrics
- Relative error vs. reference
- Score variability
- Consistency checks

**Outputs**:
- `{city}_batch_10_results.pkl` (all 10 networks)
- Best network exported to GeoJSON

---

## 🚀 Quick Start

### Installation

```bash
cd /path/to/comp_city

# Install dependencies
pip install numpy scipy networkx geopandas shapely matplotlib

# Verify reference data exists
ls inv_city/outputs/geojson/
# Should show: london_edges.geojson, london_nodes.geojson, etc.
```

### Run Notebooks in Order

```bash
# Start Jupyter
jupyter notebook

# Open and run:
1. 01_reference_data_analysis.ipynb       # Loads all 4 cities
2. 02_generate_single_network.ipynb       # Generate 1 network
3. 03_batch_generation_10_networks.ipynb  # Generate 10 networks
```

---

## 📁 Project Structure

```
comp_city/
├── street_network_generator/           # Core Python package
│   ├── __init__.py
│   ├── config.py                       # GeneratorConfig
│   ├── reference.py                    # Module A: Reference extraction
│   ├── generator.py                    # Module B: Network growth
│   ├── objective.py                    # Module C: Scoring function
│   ├── metrics.py                      # Morphology + syntax
│   ├── utils.py                        # Geometry helpers
│   ├── validation.py                   # Module F: Export & validation
│   ├── visualization.py                # Plotting utilities
│   ├── examples/
│   │   ├── generate_network.py         # CLI script
│   │   └── config.json                 # Example config
│   └── README.md                       # Package docs
│
├── 01_reference_data_analysis.ipynb    # Step 1: Load references
├── 02_generate_single_network.ipynb    # Step 2: Generate 1 network
├── 03_batch_generation_10_networks.ipynb  # Step 3: Generate 10 networks
│
├── inv_city/outputs/                   # Reference data (existing)
│   ├── geojson/
│   │   ├── london_edges.geojson
│   │   ├── london_nodes.geojson
│   │   └── ... (berlin, belgrade, torino)
│   └── metrics/
│       └── urban_metrics.json
│
└── outputs_generated/                  # Generated networks output
    ├── london_single/
    ├── london_batch_best/
    └── ...
```

---

## 🔬 Technical Details

### Phase 1 Features (Current)

✅ **Morphology Metrics**:
- Node density (nodes/km²)
- Degree distribution
- Segment length distribution
- Orientation histogram (0-180°)
- Dead-end ratio

✅ **Space Syntax Metrics** (Node-based):
- Mean depth
- Local integration (R=3)
- Choice (betweenness centrality)
- Intelligibility (degree ↔ local integration correlation)

✅ **Generation Algorithm**:
- Seed skeleton: 2-4 boundary-to-boundary spines
- Iterative growth: sample from reference distributions
- Simulated annealing: temperature-based acceptance
- Planarity enforcement: no edge crossings
- Spatial indexing: fast intersection checks

✅ **Progressive Weight Scheduling**:
- First 60%: morphology only (w_morph=1.0, w_syntax=0.0)
- Next 20%: ramp syntax → 0.2
- Final 20%: ramp syntax → 0.3

✅ **Two-Tier Scoring**:
- **Cheap metrics** (every iteration): morphology + connectivity
- **Expensive metrics** (every 60-80 edges): space syntax

### Performance

- **Target**: 1-5 minutes per network
- **Typical**: 250-1000 iterations
- **Output size**: 200-400 nodes, 400-800 edges

### Stopping Conditions

Generation stops when:
1. Morphology divergences < 10% threshold, AND
2. No improvement for 5 audits, AND
3. Iteration ≥ min_iterations

Or hard stop at `max_iterations`.

---

## 📊 Example Results

### London Generation (Seed 42)

| Metric | Reference | Generated | Error |
|--------|-----------|-----------|-------|
| Nodes | 525 | 512 | 2.5% |
| Edges | 1431 | 1389 | 2.9% |
| Node Density | 2100 | 2048 nodes/km² | 2.5% |
| Dead-End Ratio | 0.072 | 0.076 | 0.004 |
| Mean Depth | 4.32 | 4.41 | 2.1% |
| Intelligibility | 0.68 | 0.64 | 5.9% |

*(Results vary by seed)*

---

## 🎨 Visualization Examples

### What You'll See in Notebooks

1. **Network Comparison**:
   - Reference network (left) vs. Generated (right)
   - Nodes colored by degree (blue=low, red=high)
   - 500×500m window with boundary

2. **Histogram Overlays**:
   - Blue bars = Reference distribution
   - Red bars = Generated distribution
   - Clear visualization of distribution matching

3. **Progress Plots**:
   - Total score over iterations
   - Morphology vs. syntax score split
   - Convergence tracking

4. **Batch Grid** (10 networks):
   - 5×2 grid layout
   - Each network labeled with seed + stats
   - Visual comparison of variability

---

## 🔧 Configuration Options

Key parameters in `GeneratorConfig`:

```python
config = GeneratorConfig(
    seed=42,                              # Random seed
    window_size_m=500,                    # Window size
    max_iterations=2500,                  # Max iterations
    min_iterations=250,                   # Min before early stop
    syntax_recompute_interval=80,         # Audit frequency
    candidate_per_step=12,                # Candidates per iteration
    initial_temp=5.0,                     # SA temperature
    cooling_rate=0.997,                   # SA cooling
    snap_tolerance_m=1.5,                 # Node snapping distance
    min_seg_len_m=12.0,                   # Min edge length
    max_seg_len_m=90.0,                   # Max edge length
)
```

### Weight Tuning

Adjust metric importance:

```python
config.metric_weights = {
    "degree_dist": 0.3,
    "segment_length": 0.3,
    "orientation": 0.15,
    "density": 0.15,
    "dead_end_ratio": 0.1,
}

config.syntax_weights = {
    "mean_depth": 0.4,
    "local_integration": 0.3,
    "choice": 0.2,
    "intelligibility": 0.1,
}
```

---

## 🐍 Python API Usage

```python
from street_network_generator import (
    ReferenceExtractor,
    GeneratorConfig,
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

# Results
print(f"Generated {graph.number_of_nodes()} nodes")
print(f"Final score: {metadata['final_score']:.4f}")
```

---

## 🗺️ Roadmap

### Phase 2 (Planned)
- ⏳ Repair pass (snap nodes, split edges, T-junctions)
- ⏳ Multi-city blending (weighted average of multiple references)
- ⏳ Enhanced visualization (matplotlib + folium maps)

### Phase 3 (Future)
- ⏳ Segment-based space syntax (more accurate)
- ⏳ Angular metrics (turn-angle weighting)
- ⏳ Block area distributions (polygonize planar graph)
- ⏳ Performance optimization (caching, parallel)

---

## 📝 Citation

If you use this generator in research:

```bibtex
@software{street_network_generator_2024,
  title={Calibrated Urban Street Network Generator},
  author={...},
  year={2024},
  version={0.1.0-phase1},
  note={Phase 1: Morphology + Node-based Space Syntax}
}
```

---

## 🙋 FAQ

**Q: Why do my networks look different from the reference?**
A: The generator matches *statistical distributions*, not exact topology. Try multiple seeds (see Notebook 03).

**Q: How do I improve matching accuracy?**
A: Increase `max_iterations`, adjust `metric_weights`, or tune SA temperature.

**Q: Can I use my own reference city?**
A: Yes! Add GeoJSON files to `inv_city/outputs/geojson/` with format: `{city}_edges.geojson`, `{city}_nodes.geojson`.

**Q: Generation is too slow?**
A: Reduce `max_iterations`, increase `syntax_recompute_interval`, or decrease `candidate_per_step`.

**Q: Networks have disconnected components?**
A: Increase `penalty_weights['disconnected']` in config to penalize fragmentation.

---

## 📬 Support

- **Issues**: Open GitHub issue
- **Docs**: See `street_network_generator/README.md`
- **Examples**: Check `street_network_generator/examples/`

---

## ✅ Validation Checklist

Before running notebooks:

- [ ] Python 3.7+ installed
- [ ] Dependencies installed (`pip install numpy scipy networkx geopandas shapely matplotlib`)
- [ ] Reference data exists in `inv_city/outputs/geojson/`
- [ ] Jupyter notebook environment ready

---

**Happy Generating! 🏙️**
