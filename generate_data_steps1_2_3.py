#!/usr/bin/env python3
"""Generate data for Steps 1-3 (simplified version without full visualizations)"""

import numpy as np
import networkx as nx
from pathlib import Path
import pickle
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import unary_union
from shapely.affinity import translate, rotate, scale
from collections import Counter
import math

print("="*70)
print("GENERATING DATA FOR STEPS 1-3")
print("="*70)

# Configuration
WINDOW_SIZE_M = 500
MIN_SEGMENT_LENGTH = 5.0
MIN_BUILDING_AREA = 30
MAX_BUILDING_AREA = 1000

CITIES = {
    'london': {'name': 'London, UK'},
    'berlin': {'name': 'Berlin, Germany'},
    'belgrade': {'name': 'Belgrade, Serbia'},
    'torino': {'name': 'Torino, Italy'}
}

# Create output directories
Path("outputs/data").mkdir(parents=True, exist_ok=True)
Path("outputs/generated/networks").mkdir(parents=True, exist_ok=True)
Path("outputs/generated/buildings").mkdir(parents=True, exist_ok=True)

print("\n[STEP 1] Loading and analyzing reference cities...")
print("="*70)

city_data = {}

for city_key in CITIES.keys():
    print(f"\n{CITIES[city_key]['name']}:")

    # Load GeoJSON files
    nodes_gdf = gpd.read_file(f'inv_city/outputs/geojson/{city_key}_nodes.geojson')
    edges_gdf = gpd.read_file(f'inv_city/outputs/geojson/{city_key}_edges.geojson')
    buildings_gdf = gpd.read_file(f'inv_city/outputs/geojson/{city_key}_buildings.geojson')
    parcels_gdf = gpd.read_file(f'inv_city/outputs/geojson/{city_key}_parcels.geojson')

    print(f"  Loaded: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges, {len(buildings_gdf)} buildings")

    # Filter for pedestrian paths only
    pedestrian_types = ['footway', 'path', 'pedestrian', 'steps']
    def is_pedestrian(hw):
        if hw is None: return False
        if hasattr(hw, '__iter__') and not isinstance(hw, str):
            hw = hw[0] if len(hw) > 0 else None
        return hw in pedestrian_types

    edges_gdf = edges_gdf[edges_gdf['highway'].apply(is_pedestrian)].copy()
    print(f"  After pedestrian filter: {len(edges_gdf)} edges")

    # Create graph
    G = nx.MultiDiGraph()
    for _, row in nodes_gdf.iterrows():
        coords = row.geometry.coords[0]
        G.add_node(row['osmid'], x=coords[0], y=coords[1])

    for _, row in edges_gdf.iterrows():
        if row['u'] in G.nodes() and row['v'] in G.nodes():
            G.add_edge(row['u'], row['v'], length=row.geometry.length, geometry=row.geometry)

    # Clean graph (remove short edges)
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    edges_to_remove = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
                       if d.get('length', 0) < MIN_SEGMENT_LENGTH]
    G.remove_edges_from(edges_to_remove)
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    for node in isolated:
        pos.pop(node, None)

    print(f"  Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Normalize coordinates
    coords = np.array(list(pos.values()))
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    offset_x = WINDOW_SIZE_M / 2 - center_x
    offset_y = WINDOW_SIZE_M / 2 - center_y

    pos_norm = {node: (x - center_x + WINDOW_SIZE_M/2, y - center_y + WINDOW_SIZE_M/2)
                for node, (x, y) in pos.items()}

    # Transform geometries
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'geometry' in data:
            data['geometry_norm'] = translate(data['geometry'], xoff=offset_x, yoff=offset_y)

    buildings_gdf['geometry'] = buildings_gdf['geometry'].apply(
        lambda geom: translate(geom, xoff=offset_x, yoff=offset_y)
    )

    # Compute morphology metrics
    area_km2 = (WINDOW_SIZE_M / 1000.0) ** 2
    degrees = [d for _, d in G.degree()]
    lengths = [d.get('length', 0) for u, v, k, d in G.edges(keys=True, data=True)]

    morphology = {
        'node_density': G.number_of_nodes() / area_km2,
        'avg_degree': np.mean(degrees) if degrees else 0,
        'degree_distribution': dict(Counter(degrees)),
        'segment_lengths': lengths,
        'avg_segment_length': np.mean(lengths) if lengths else 0
    }

    # Compute syntax metrics (simplified)
    G_undir = G.to_undirected()
    if not nx.is_connected(G_undir):
        largest_cc = max(nx.connected_components(G_undir), key=len)
        G_undir = G_undir.subgraph(largest_cc).copy()

    total_depth = sum(sum(nx.single_source_shortest_path_length(G_undir, n).values())
                     for n in G_undir.nodes())
    count = sum(len(nx.single_source_shortest_path_length(G_undir, n))
                for n in G_undir.nodes())
    mean_depth = total_depth / count if count > 0 else 0

    local_int = {}
    for node in G_undir.nodes():
        lengths_dict = nx.single_source_shortest_path_length(G_undir, node, cutoff=3)
        if len(lengths_dict) > 1:
            total = sum(lengths_dict.values())
            local_int[node] = (len(lengths_dict) - 1) / total if total > 0 else 0
        else:
            local_int[node] = 0

    degrees_undir = [G_undir.degree(n) for n in local_int.keys()]
    integrations = list(local_int.values())
    intelligibility = np.corrcoef(degrees_undir, integrations)[0, 1] if len(degrees_undir) > 1 else 0

    syntax = {
        'mean_depth': mean_depth,
        'intelligibility': intelligibility if not np.isnan(intelligibility) else 0
    }

    # Compute building metrics
    areas = [b.geometry.area for _, b in buildings_gdf.iterrows()
             if b.geometry and b.geometry.geom_type == 'Polygon']
    compactness = [(b.geometry.length ** 2) / b.geometry.area
                   for _, b in buildings_gdf.iterrows()
                   if b.geometry and b.geometry.geom_type == 'Polygon' and b.geometry.area > 0]

    building_metrics = {
        'footprint_areas': areas,
        'avg_footprint_area': np.mean(areas) if areas else 0,
        'building_coverage_ratio': sum(areas) / (WINDOW_SIZE_M ** 2),
        'compactness_values': compactness,
        'avg_compactness': np.mean(compactness) if compactness else 0
    }

    city_data[city_key] = {
        'graph': G,
        'pos': pos_norm,
        'morphology': morphology,
        'syntax': syntax,
        'building_metrics': building_metrics,
        'buildings': buildings_gdf,
        'pedestrian_edges': edges_gdf
    }

    print(f"  ✓ Metrics computed")

# Save Step 1 data
with open('outputs/data/reference_cities_data.pkl', 'wb') as f:
    pickle.dump(city_data, f)

print("\n✓ Step 1 complete! Saved to: outputs/data/reference_cities_data.pkl")

# STEP 2: Generate 20 networks
print("\n[STEP 2] Generating 20 calibrated networks...")
print("="*70)

reference_city = 'london'
ref_data = city_data[reference_city]
reference_lengths = ref_data['morphology']['segment_lengths']

def generate_calibrated_grid_network(width, height, target_spacing, reference_lengths, noise_level=10.0):
    """Generate a noisy grid network."""
    G = nx.Graph()
    node_id = 0
    node_positions = {}

    # Create noisy grid
    for x in range(0, width + 1, target_spacing):
        for y in range(0, height + 1, target_spacing):
            nx_pos = x + np.random.uniform(-noise_level, noise_level)
            ny_pos = y + np.random.uniform(-noise_level, noise_level)
            nx_pos = max(0, min(width, nx_pos))
            ny_pos = max(0, min(height, ny_pos))

            node_positions[node_id] = (nx_pos, ny_pos)
            G.add_node(node_id)
            node_id += 1

    # Connect nodes
    from scipy.spatial import Delaunay
    points = np.array(list(node_positions.values()))
    tri = Delaunay(points)

    for simplex in tri.simplices:
        for i in range(3):
            n1, n2 = simplex[i], simplex[(i + 1) % 3]
            if not G.has_edge(n1, n2):
                p1, p2 = node_positions[n1], node_positions[n2]
                length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                G.add_edge(n1, n2, length=length)

    return G, node_positions

def randomly_remove_edges(G, removal_rate=0.2):
    """Randomly remove edges to create irregularity."""
    edges = list(G.edges())
    num_to_remove = int(len(edges) * removal_rate)
    edges_to_remove = np.random.choice(len(edges), num_to_remove, replace=False)

    for idx in edges_to_remove:
        G.remove_edge(*edges[idx])

    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    return G

generated_networks = []

for i in range(20):
    spacing = np.random.randint(40, 70)
    noise_level = np.random.uniform(5, 15)
    removal_rate = np.random.uniform(0.1, 0.25)

    G, pos = generate_calibrated_grid_network(
        WINDOW_SIZE_M, WINDOW_SIZE_M, spacing, reference_lengths, noise_level
    )
    G = randomly_remove_edges(G, removal_rate)

    # Compute metrics
    degrees = [d for _, d in G.degree()]
    lengths = [d['length'] for u, v, d in G.edges(data=True)]

    network_metrics = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'avg_degree': np.mean(degrees) if degrees else 0,
        'avg_segment_length': np.mean(lengths) if lengths else 0
    }

    generated_networks.append({
        'id': i,
        'graph': G,
        'pos': pos,
        'metrics': network_metrics,
        'params': {'spacing': spacing, 'noise': noise_level, 'removal': removal_rate}
    })

    print(f"  Network {i+1:2d}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Save Step 2 data
with open('outputs/generated/networks/generated_networks_20.pkl', 'wb') as f:
    pickle.dump(generated_networks, f)

print("\n✓ Step 2 complete! Saved to: outputs/generated/networks/generated_networks_20.pkl")

# STEP 3: Add basic buildings (simplified)
print("\n[STEP 3] Adding basic buildings to networks...")
print("="*70)

target_coverage = ref_data['building_metrics']['building_coverage_ratio']
reference_areas = ref_data['building_metrics']['footprint_areas']

def generate_basic_buildings(G, pos, target_coverage, reference_areas, max_buildings=100):
    """Generate simple rectangular buildings."""
    buildings = []
    window_area = WINDOW_SIZE_M ** 2
    target_total_area = window_area * target_coverage
    current_total_area = 0

    # Create path lines
    path_lines = [LineString([pos[u], pos[v]]) for u, v in G.edges()]
    if not path_lines:
        return buildings

    all_paths = unary_union(path_lines)

    attempts = 0
    max_attempts = 1000

    while current_total_area < target_total_area and len(buildings) < max_buildings and attempts < max_attempts:
        attempts += 1

        # Random position
        x = np.random.uniform(20, WINDOW_SIZE_M - 20)
        y = np.random.uniform(20, WINDOW_SIZE_M - 20)
        point = Point(x, y)

        # Check distance to path
        if point.distance(all_paths) > 40:
            continue

        # Sample area
        area = np.random.choice(reference_areas)
        area = max(MIN_BUILDING_AREA, min(MAX_BUILDING_AREA, area))

        # Create simple rectangle
        width = np.sqrt(area * np.random.uniform(0.5, 2.0))
        height = area / width
        building = box(x - width/2, y - height/2, x + width/2, y + height/2)

        # Rotate
        angle = np.random.uniform(0, 90)
        building = rotate(building, angle, origin='centroid')

        # Check window bounds
        window = box(0, 0, WINDOW_SIZE_M, WINDOW_SIZE_M)
        if not window.contains(building):
            continue

        # Check overlap
        overlap = any(building.intersects(b) for b in buildings)
        if overlap or building.intersects(all_paths.buffer(2)):
            continue

        buildings.append(building)
        current_total_area += building.area

    return buildings

for network_data in generated_networks:
    G = network_data['graph']
    pos = network_data['pos']
    net_id = network_data['id']

    buildings = generate_basic_buildings(G, pos, target_coverage, reference_areas, max_buildings=150)

    total_area = sum(b.area for b in buildings)
    actual_coverage = total_area / (WINDOW_SIZE_M ** 2)

    network_data['buildings'] = buildings
    network_data['building_coverage'] = actual_coverage

    print(f"  Network {net_id+1:2d}: {len(buildings)} buildings, coverage {actual_coverage:.3f}")

# Save Step 3 data
with open('outputs/generated/buildings/networks_with_buildings_20.pkl', 'wb') as f:
    pickle.dump(generated_networks, f)

print("\n✓ Step 3 complete! Saved to: outputs/generated/buildings/networks_with_buildings_20.pkl")

print("\n" + "="*70)
print("ALL DATA GENERATED!")
print("="*70)
print("\nReady to run Step 4: Space Syntax Analysis")
