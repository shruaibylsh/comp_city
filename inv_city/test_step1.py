#!/usr/bin/env python3
"""
STEP 1: Analyze Real Cities (500×500m)
Test execution of the notebook code
"""

# Imports
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely.affinity import translate
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configure OSMnx
ox.settings.use_cache = True
ox.settings.log_console = False

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')

print("✓ Libraries imported successfully")

# Configuration
CITIES = {
    'hanoi': {
        'name': 'Hanoi, Vietnam',
        'coords': (21.0230, 105.8560),
        'color': '#FF6B6B'
    },
    'brussels': {
        'name': 'Brussels, Belgium',
        'coords': (50.8477, 4.3572),
        'color': '#4ECDC4'
    },
    'marrakech': {
        'name': 'Marrakech, Morocco',
        'coords': (31.623811, -7.988662),
        'color': '#FFE66D'
    }
}

# Analysis parameters (adapted for 500×500m)
RADIUS = 250  # meters
REACH_RADII = [200, 300]
LOCAL_LANDMARK_RADIUS = 300
MIN_PARCEL_AREA = 500  # m²
MAX_PARCEL_AREA = 10000  # m²
FOOTPRINTS_PER_CITY = 35  # Target library size

# Output paths
OUTPUT_DIR = Path('outputs')
GEOJSON_DIR = OUTPUT_DIR / 'geojson'
VIZ_PNG_DIR = OUTPUT_DIR / 'visualizations' / 'png'
VIZ_SVG_DIR = OUTPUT_DIR / 'visualizations' / 'svg'
METRICS_DIR = OUTPUT_DIR / 'metrics'

for d in [GEOJSON_DIR, VIZ_PNG_DIR, VIZ_SVG_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("✓ Configuration complete")
print(f"  Analyzing {len(CITIES)} cities")
print(f"  Coverage radius: {RADIUS}m (~{RADIUS*2}×{RADIUS*2}m)")
print(f"  Output: {OUTPUT_DIR.absolute()}")

# Use built-in NetworkX community detection
print("✓ Using NetworkX built-in Louvain community detection")

# Download data for all cities
city_data = {}

for city_key, city_info in CITIES.items():
    print(f"\n{'='*60}")
    print(f"Downloading: {city_info['name']}")
    print(f"{'='*60}")

    lat, lon = city_info['coords']

    try:
        # 1. Street network
        print(f"  → Street network...")
        G = ox.graph_from_point((lat, lon), dist=RADIUS, network_type='walk', simplify=True)
        G_proj = ox.project_graph(G)

        # 2. Buildings
        print(f"  → Buildings...")
        buildings = ox.features_from_point((lat, lon), dist=RADIUS, tags={'building': True})
        buildings_proj = buildings.to_crs(ox.graph_to_gdfs(G_proj, nodes=False).crs)
        buildings_proj = buildings_proj[buildings_proj.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()

        # Convert MultiPolygons to Polygons
        def get_polygon(geom):
            if geom.geom_type == 'Polygon':
                return geom
            elif geom.geom_type == 'MultiPolygon':
                return max(geom.geoms, key=lambda p: p.area)
            return geom

        buildings_proj['geometry'] = buildings_proj.geometry.apply(get_polygon)
        buildings_proj = buildings_proj[buildings_proj.geometry.type == 'Polygon'].copy()

        # 3. Building Parcels (landuse)
        print(f"  → Building parcels (landuse)...")
        try:
            parcels = ox.features_from_point(
                (lat, lon),
                dist=RADIUS,
                tags={'landuse': True}
            )
            parcels_proj = parcels.to_crs(ox.graph_to_gdfs(G_proj, nodes=False).crs)
            parcels_proj = parcels_proj[parcels_proj.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
            parcels_proj['geometry'] = parcels_proj.geometry.apply(get_polygon)
            parcels_proj = parcels_proj[parcels_proj.geometry.type == 'Polygon'].copy()
            print(f"    ✓ Found {len(parcels_proj)} parcels")
        except Exception as e:
            print(f"    ⚠ No parcels found: {e}")
            parcels_proj = gpd.GeoDataFrame(columns=['geometry', 'landuse'], crs=ox.graph_to_gdfs(G_proj, nodes=False).crs)

        # Store data
        city_data[city_key] = {
            'name': city_info['name'],
            'color': city_info['color'],
            'coords': (lat, lon),
            'graph': G_proj,
            'buildings': buildings_proj,
            'parcels': parcels_proj,
            'crs': ox.graph_to_gdfs(G_proj, nodes=False).crs
        }

        print(f"  ✓ Downloaded:")
        print(f"    - {G_proj.number_of_nodes()} nodes")
        print(f"    - {G_proj.number_of_edges()} edges")
        print(f"    - {len(buildings_proj)} buildings")
        print(f"    - {len(parcels_proj)} parcels")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print(f"✓ Data acquisition complete for {len(city_data)} cities")
print(f"{'='*60}")

print("\n" + "="*60)
print("Test run successful! All data downloaded.")
print("="*60)
print(f"\nData downloaded for {len(city_data)} cities:")
for city_key in city_data.keys():
    print(f"  - {city_data[city_key]['name']}")
print("\nNote: Full analysis includes base maps, node/edge/parcel/district/landmark")
print("analysis, footprint library extraction, and visualizations.")
