"""
Visualization utilities for network generation and comparison.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, Tuple, List, Optional
from .reference import ReferenceData


def plot_network(
    graph: nx.Graph,
    pos: Dict,
    window_size_m: float = 500,
    ax: Optional[plt.Axes] = None,
    title: str = "Street Network",
    show_node_degrees: bool = False,
    node_size: float = 20,
    edge_width: float = 1.0,
    edge_color: str = '#2C3E50',
    node_color_map: Optional[Dict] = None
) -> plt.Axes:
    """
    Plot street network.

    Args:
        graph: NetworkX graph
        pos: Node positions
        window_size_m: Window size
        ax: Matplotlib axis (creates new if None)
        title: Plot title
        show_node_degrees: Color nodes by degree
        node_size: Node marker size
        edge_width: Edge line width
        edge_color: Edge color
        node_color_map: Optional dict mapping node_id -> color

    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Draw window boundary
    ax.add_patch(Rectangle(
        (0, 0), window_size_m, window_size_m,
        fill=False, edgecolor='gray', linestyle='--', linewidth=1
    ))

    # Draw edges
    for u, v in graph.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=edge_color, linewidth=edge_width, zorder=1)

    # Draw nodes
    if show_node_degrees:
        # Color by degree
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 1

        for node in graph.nodes():
            degree = degrees[node]
            # Color scale: blue (low) to red (high)
            color_val = degree / max_degree
            color = plt.cm.RdYlBu_r(color_val)

            ax.scatter(
                pos[node][0], pos[node][1],
                s=node_size, c=[color], zorder=2,
                edgecolors='black', linewidths=0.5
            )
    elif node_color_map:
        # Use provided color map
        for node in graph.nodes():
            color = node_color_map.get(node, 'gray')
            ax.scatter(
                pos[node][0], pos[node][1],
                s=node_size, c=color, zorder=2,
                edgecolors='black', linewidths=0.5
            )
    else:
        # Single color
        node_positions = np.array([pos[n] for n in graph.nodes()])
        if len(node_positions) > 0:
            ax.scatter(
                node_positions[:, 0], node_positions[:, 1],
                s=node_size, c='#E74C3C', zorder=2,
                edgecolors='black', linewidths=0.5
            )

    ax.set_xlim(-10, window_size_m + 10)
    ax.set_ylim(-10, window_size_m + 10)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True, alpha=0.3)

    return ax


def plot_histogram_comparison(
    reference_hist: Tuple[np.ndarray, np.ndarray],
    generated_hist: Tuple[np.ndarray, np.ndarray],
    ax: Optional[plt.Axes] = None,
    title: str = "Distribution Comparison",
    xlabel: str = "Value",
    normalize: bool = True
) -> plt.Axes:
    """
    Plot histogram comparison between reference and generated.

    Args:
        reference_hist: (bin_edges, counts) for reference
        generated_hist: (bin_edges, counts) for generated
        ax: Matplotlib axis
        title: Plot title
        xlabel: X-axis label
        normalize: Normalize to probabilities

    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ref_bins, ref_counts = reference_hist
    gen_bins, gen_counts = generated_hist

    # Normalize if requested
    if normalize:
        ref_total = np.sum(ref_counts)
        gen_total = np.sum(gen_counts)
        if ref_total > 0:
            ref_counts = ref_counts / ref_total
        if gen_total > 0:
            gen_counts = gen_counts / gen_total

    # Compute bin centers
    ref_centers = (ref_bins[:-1] + ref_bins[1:]) / 2
    gen_centers = (gen_bins[:-1] + gen_bins[1:]) / 2

    # Plot
    ax.bar(
        ref_centers, ref_counts,
        width=(ref_bins[1] - ref_bins[0]) * 0.8,
        alpha=0.6, color='#3498DB', label='Reference',
        edgecolor='black', linewidth=0.5
    )
    ax.bar(
        gen_centers, gen_counts,
        width=(gen_bins[1] - gen_bins[0]) * 0.8,
        alpha=0.6, color='#E74C3C', label='Generated',
        edgecolor='black', linewidth=0.5
    )

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability' if normalize else 'Count')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    return ax


def plot_degree_distribution_comparison(
    reference_degree_dist: Dict[int, int],
    generated_degree_dist: Dict[int, int],
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot degree distribution comparison."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    all_degrees = sorted(set(reference_degree_dist.keys()) | set(generated_degree_dist.keys()))

    # Normalize to probabilities
    ref_total = sum(reference_degree_dist.values())
    gen_total = sum(generated_degree_dist.values())

    ref_probs = [reference_degree_dist.get(d, 0) / ref_total for d in all_degrees]
    gen_probs = [generated_degree_dist.get(d, 0) / gen_total for d in all_degrees]

    x = np.arange(len(all_degrees))
    width = 0.35

    ax.bar(x - width/2, ref_probs, width, label='Reference', color='#3498DB', alpha=0.7)
    ax.bar(x + width/2, gen_probs, width, label='Generated', color='#E74C3C', alpha=0.7)

    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Probability')
    ax.set_title('Degree Distribution Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_degrees)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    return ax


def plot_generation_progress(
    audit_history: List[Dict],
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot score evolution during generation.

    Args:
        audit_history: List of audit dicts with iteration, score, breakdown
        ax: Matplotlib axis

    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    iterations = [a['iteration'] for a in audit_history]
    total_scores = [a['score'] for a in audit_history]
    morph_scores = [a['breakdown'].get('morph_score', 0) for a in audit_history]
    syntax_scores = [a['breakdown'].get('syntax_score', 0) for a in audit_history]

    ax.plot(iterations, total_scores, 'o-', label='Total Score', linewidth=2, markersize=4)
    ax.plot(iterations, morph_scores, 's--', label='Morphology', linewidth=1.5, markersize=3)
    ax.plot(iterations, syntax_scores, '^--', label='Syntax', linewidth=1.5, markersize=3)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Generation Progress', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_full_comparison(
    reference: ReferenceData,
    graph: nx.Graph,
    pos: Dict,
    morph: Dict,
    syntax: Dict,
    window_size_m: float = 500,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create comprehensive comparison figure.

    Args:
        reference: Reference data
        graph: Generated graph
        pos: Node positions
        morph: Morphology metrics dict
        syntax: Syntax metrics dict
        window_size_m: Window size
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Networks
    ax1 = fig.add_subplot(gs[0, 0])
    plot_network(
        reference.graph, reference.pos, window_size_m,
        ax=ax1, title=f"Reference ({reference.city_name})",
        show_node_degrees=True
    )

    ax2 = fig.add_subplot(gs[0, 1])
    plot_network(
        graph, pos, window_size_m,
        ax=ax2, title="Generated Network",
        show_node_degrees=True
    )

    ax3 = fig.add_subplot(gs[0, 2])
    plot_degree_distribution_comparison(
        reference.degree_distribution,
        morph['degree_distribution'],
        ax=ax3
    )

    # Row 2: Segment lengths and orientation
    ax4 = fig.add_subplot(gs[1, 0])
    # Segment length histogram
    gen_lengths = morph['segment_lengths']
    if gen_lengths:
        gen_hist = (
            reference.segment_length_hist[0],
            np.histogram(gen_lengths, bins=reference.segment_length_hist[0])[0]
        )
        plot_histogram_comparison(
            reference.segment_length_hist,
            gen_hist,
            ax=ax4,
            title="Segment Length Distribution",
            xlabel="Length (m)"
        )

    ax5 = fig.add_subplot(gs[1, 1])
    # Orientation histogram
    ref_orientation_hist = reference.orientation_hist
    gen_orientation = morph['orientation_histogram']
    gen_orientation_hist = (
        np.array(gen_orientation['bin_edges']),
        np.array(gen_orientation['counts'])
    )
    plot_histogram_comparison(
        ref_orientation_hist,
        gen_orientation_hist,
        ax=ax5,
        title="Orientation Distribution",
        xlabel="Bearing (degrees)"
    )

    # Row 3: Space syntax
    ax6 = fig.add_subplot(gs[1, 2])
    # Local integration
    ref_local_int = reference.local_integration_hist
    gen_local_int_values = list(syntax['local_integration'].values())
    if gen_local_int_values and len(ref_local_int[0]) > 1:
        gen_local_int_hist = (
            ref_local_int[0],
            np.histogram(gen_local_int_values, bins=ref_local_int[0])[0]
        )
        plot_histogram_comparison(
            ref_local_int,
            gen_local_int_hist,
            ax=ax6,
            title="Local Integration (R=3)",
            xlabel="Integration Value"
        )

    ax7 = fig.add_subplot(gs[2, 0])
    # Choice
    ref_choice = reference.choice_hist
    gen_choice_values = list(syntax['choice'].values())
    if gen_choice_values and len(ref_choice[0]) > 1:
        gen_choice_hist = (
            ref_choice[0],
            np.histogram(gen_choice_values, bins=ref_choice[0])[0]
        )
        plot_histogram_comparison(
            ref_choice,
            gen_choice_hist,
            ax=ax7,
            title="Choice (Betweenness)",
            xlabel="Choice Value"
        )

    # Summary stats table
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')

    summary_text = f"""
    SUMMARY STATISTICS

    Nodes:           Ref={reference.graph.number_of_nodes()}  |  Gen={graph.number_of_nodes()}
    Edges:           Ref={reference.graph.number_of_edges()}  |  Gen={graph.number_of_edges()}

    Node Density:    Ref={reference.node_density:.1f}  |  Gen={morph['node_density']:.1f} nodes/km²
    Dead-End Ratio:  Ref={reference.dead_end_ratio:.3f}  |  Gen={morph['dead_end_ratio']:.3f}

    Mean Depth:      Ref={reference.mean_depth:.3f}  |  Gen={syntax['mean_depth']:.3f}
    Intelligibility: Ref={reference.intelligibility:.3f}  |  Gen={syntax['intelligibility']:.3f}
    """

    ax8.text(
        0.1, 0.5, summary_text,
        fontsize=10, family='monospace',
        verticalalignment='center'
    )

    fig.suptitle(
        f"Street Network Generation: {reference.city_name}",
        fontsize=16, fontweight='bold', y=0.98
    )

    return fig


def plot_network_grid(
    networks: List[Tuple[nx.Graph, Dict, str]],
    window_size_m: float = 500,
    figsize: Tuple[int, int] = (16, 12),
    cols: int = 5
) -> plt.Figure:
    """
    Plot multiple networks in a grid.

    Args:
        networks: List of (graph, pos, title) tuples
        window_size_m: Window size
        figsize: Figure size
        cols: Number of columns

    Returns:
        Matplotlib figure
    """
    n = len(networks)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (graph, pos, title) in enumerate(networks):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        plot_network(
            graph, pos, window_size_m,
            ax=ax, title=title,
            node_size=15, edge_width=0.8
        )

    # Hide unused subplots
    for idx in range(n, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    fig.suptitle("Generated Street Networks", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig
