#!/usr/bin/env python3
"""Step 4: Space Syntax Analysis"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import pickle
from collections import defaultdict
import math

print('✓ Libraries loaded')

# Configuration
WINDOW_SIZE_M = 500
Path('outputs/generated/visualizations').mkdir(parents=True, exist_ok=True)
Path('outputs/generated/syntax').mkdir(parents=True, exist_ok=True)
print(f'Window size: {WINDOW_SIZE_M}m × {WINDOW_SIZE_M}m')
print('✓ Output directories created')

# Load data
with open('outputs/data/reference_cities_data.pkl', 'rb') as f:
    reference_data = pickle.load(f)

with open('outputs/generated/buildings/networks_with_buildings_20.pkl', 'rb') as f:
    generated_networks = pickle.load(f)

print('✓ Loaded reference data from Step 1')
print(f'✓ Loaded {len(generated_networks)} networks with buildings from Step 3')

reference_cities = ['london', 'berlin', 'belgrade', 'torino']
cities_str = ', '.join([c.upper() for c in reference_cities])
print(f'Reference cities: {cities_str}')
print()

# Space syntax functions
def compute_integration(G, node, radius=None):
    """Compute integration for a node."""
    if radius is None:
        lengths = nx.single_source_shortest_path_length(G, node)
    else:
        lengths = nx.single_source_dijkstra_path_length(G, node, cutoff=radius, weight='length')

    if len(lengths) <= 1:
        return 0.0

    total_depth = sum(lengths.values())
    n = len(lengths) - 1

    if n == 0:
        return 0.0

    mean_depth = total_depth / n

    if mean_depth > 0:
        integration = 1.0 / mean_depth
    else:
        integration = 0.0

    return integration

def compute_mean_depth(G, node):
    """Compute mean depth for a node."""
    lengths = nx.single_source_shortest_path_length(G, node)

    if len(lengths) <= 1:
        return 0.0

    total_depth = sum(lengths.values())
    n = len(lengths) - 1

    if n == 0:
        return 0.0

    return total_depth / n

def compute_space_syntax_metrics(G, local_radius=200):
    """Compute space syntax metrics for all nodes."""
    metrics = {
        'global_integration': {},
        'local_integration': {},
        'choice': {},
        'mean_depth': {}
    }

    if G.is_directed():
        G_undirected = G.to_undirected()
    else:
        G_undirected = G

    if not nx.is_connected(G_undirected):
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(largest_cc).copy()

    nodes = list(G_undirected.nodes())

    print('  Computing choice (betweenness)...')
    betweenness = nx.betweenness_centrality(G_undirected, weight='length', normalized=True)

    print('  Computing integration and mean depth...')
    for i, node in enumerate(nodes):
        metrics['global_integration'][node] = compute_integration(G_undirected, node, radius=None)
        metrics['local_integration'][node] = compute_integration(G_undirected, node, radius=local_radius)
        metrics['choice'][node] = betweenness.get(node, 0.0)
        metrics['mean_depth'][node] = compute_mean_depth(G_undirected, node)

        if (i + 1) % 20 == 0:
            print(f'    Processed {i+1}/{len(nodes)} nodes')

    return metrics

def compute_intelligibility(integration_values, choice_values):
    """Compute intelligibility (R² between integration and choice)."""
    if len(integration_values) < 2 or len(choice_values) < 2:
        return 0.0

    correlation = np.corrcoef(integration_values, choice_values)[0, 1]

    if np.isnan(correlation):
        return 0.0

    return correlation ** 2

print('✓ Space syntax functions defined')
print()

# Compute for all networks
print('Computing space syntax metrics for 20 networks...')
print('='*70)

for network_data in generated_networks:
    G = network_data['graph']
    net_id = network_data['id']

    print(f'\nNetwork {net_id+1}:')

    syntax_metrics = compute_space_syntax_metrics(G, local_radius=200)

    global_int_values = list(syntax_metrics['global_integration'].values())
    local_int_values = list(syntax_metrics['local_integration'].values())
    choice_values = list(syntax_metrics['choice'].values())
    mean_depth_values = list(syntax_metrics['mean_depth'].values())

    intelligibility = compute_intelligibility(global_int_values, choice_values)

    network_data['syntax_metrics'] = {
        'node_metrics': syntax_metrics,
        'avg_global_integration': np.mean(global_int_values) if global_int_values else 0,
        'avg_local_integration': np.mean(local_int_values) if local_int_values else 0,
        'avg_choice': np.mean(choice_values) if choice_values else 0,
        'avg_mean_depth': np.mean(mean_depth_values) if mean_depth_values else 0,
        'intelligibility': intelligibility
    }

    print(f'  Global integration: {network_data["syntax_metrics"]["avg_global_integration"]:.4f}')
    print(f'  Local integration:  {network_data["syntax_metrics"]["avg_local_integration"]:.4f}')
    print(f'  Choice:             {network_data["syntax_metrics"]["avg_choice"]:.4f}')
    print(f'  Mean depth:         {network_data["syntax_metrics"]["avg_mean_depth"]:.2f}')
    print(f'  Intelligibility:    {intelligibility:.4f}')

print('\n' + '='*70)
print('✓ Space syntax computed for all 20 networks')
print()

# Summary statistics
all_global_int = [net['syntax_metrics']['avg_global_integration'] for net in generated_networks]
all_local_int = [net['syntax_metrics']['avg_local_integration'] for net in generated_networks]
all_choice = [net['syntax_metrics']['avg_choice'] for net in generated_networks]
all_mean_depth = [net['syntax_metrics']['avg_mean_depth'] for net in generated_networks]
all_intelligibility = [net['syntax_metrics']['intelligibility'] for net in generated_networks]

print('='*70)
print('GENERATED NETWORKS - SPACE SYNTAX METRICS')
print('='*70)
print(f'\nGlobal Integration: {np.mean(all_global_int):.4f} ± {np.std(all_global_int):.4f}')
print(f'Local Integration:  {np.mean(all_local_int):.4f} ± {np.std(all_local_int):.4f}')
print(f'Choice:             {np.mean(all_choice):.4f} ± {np.std(all_choice):.4f}')
print(f'Mean Depth:         {np.mean(all_mean_depth):.2f} ± {np.std(all_mean_depth):.2f}')
print(f'Intelligibility:    {np.mean(all_intelligibility):.4f} ± {np.std(all_intelligibility):.4f}')

print('\n' + '='*70)
print('REFERENCE CITIES - SPACE SYNTAX METRICS')
print('='*70)

for city in reference_cities:
    ref_syntax = reference_data[city]['syntax']
    print(f'\n{city.upper()}:')
    print(f'  Intelligibility: {ref_syntax["intelligibility"]:.4f}')
    print(f'  Mean Depth:      {ref_syntax["mean_depth"]:.2f}')

print('='*70)
print()

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Global Integration
ax = axes[0, 0]
ax.hist(all_global_int, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Global Integration')
ax.set_ylabel('Frequency')
ax.set_title('Global Integration Distribution\n(20 networks)', fontweight='bold')
ax.axvline(np.mean(all_global_int), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_global_int):.4f}')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Local Integration
ax = axes[0, 1]
ax.hist(all_local_int, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Local Integration')
ax.set_ylabel('Frequency')
ax.set_title('Local Integration Distribution\n(20 networks)', fontweight='bold')
ax.axvline(np.mean(all_local_int), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_local_int):.4f}')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Choice
ax = axes[0, 2]
ax.hist(all_choice, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Choice (Betweenness)')
ax.set_ylabel('Frequency')
ax.set_title('Choice Distribution\n(20 networks)', fontweight='bold')
ax.axvline(np.mean(all_choice), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_choice):.4f}')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Mean Depth
ax = axes[1, 0]
ax.hist(all_mean_depth, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Mean Depth')
ax.set_ylabel('Frequency')
ax.set_title('Mean Depth Distribution\n(20 networks)', fontweight='bold')
ax.axvline(np.mean(all_mean_depth), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_mean_depth):.2f}')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Intelligibility
ax = axes[1, 1]
ax.hist(all_intelligibility, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Intelligibility (R²)')
ax.set_ylabel('Frequency')
ax.set_title('Intelligibility Distribution\n(20 networks)', fontweight='bold')
ax.axvline(np.mean(all_intelligibility), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_intelligibility):.4f}')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Comparison to reference cities
ax = axes[1, 2]
ref_intelligibility = [reference_data[city]['syntax']['intelligibility'] for city in reference_cities]
ref_cities_upper = [c.upper() for c in reference_cities]

x_pos = np.arange(len(reference_cities))
ax.bar(x_pos, ref_intelligibility, color='coral', alpha=0.7, edgecolor='black', label='Reference')
ax.axhline(np.mean(all_intelligibility), color='steelblue', linestyle='--', linewidth=2, label='Generated (mean)')
ax.set_xticks(x_pos)
ax.set_xticklabels(ref_cities_upper)
ax.set_ylabel('Intelligibility (R²)')
ax.set_title('Intelligibility Comparison\nReference vs Generated', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Space Syntax Analysis - 20 Generated Networks', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save
plt.savefig('outputs/generated/visualizations/D1_space_syntax_distributions.svg',
           format='svg', bbox_inches='tight', dpi=300)
print('Saved visualization: outputs/generated/visualizations/D1_space_syntax_distributions.svg')

# Save enriched networks
with open('outputs/generated/syntax/networks_with_syntax_20.pkl', 'wb') as f:
    pickle.dump(generated_networks, f)

print('✓ Saved networks to: outputs/generated/syntax/networks_with_syntax_20.pkl')
print('\nEach network now includes:')
print('  - NetworkX graph')
print('  - Node positions')
print('  - Network metrics (morphology)')
print('  - Building polygons')
print('  - Building metrics')
print('  - Space syntax metrics (NEW)')
print('  - Generation parameters')
print('\n✓ Step 4 complete!')
