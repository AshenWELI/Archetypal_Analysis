"""
Archetypal Analysis Visualization with Radial Layout

This module provides a comprehensive visualization for Archetypal Analysis (AA),
including:
- Data points projected into 2D space
- Learned archetypes
- Convex hull structure
- Approximation (reconstruction) error lines from data points to their projections

Usage:
------
After running Archetypal Analysis, import and call:

    from archetypal_plot import plot_aa_archetypal_complete

    fig, ax = plot_aa_archetypal_complete(
        X=X_norm,                  # Original data matrix (N × M)
        S=aa_S,                    # Document–archetype weights (N × K)
        A=aa_A,                    # Archetype matrix (K × M)
        K=K,                       # Number of archetypes
        vocab=vocab,               # Vocabulary list (optional)
        method='tsne',             # Dimensionality reduction: 'tsne', 'pca', or '2d'
        output_file='aa_complete.png',
        n_sample_lines=15,         # Number of reconstruction error lines to display
        figsize=(14, 14)
    )

References:
-----------
Inspired by the visualization concepts presented in:
https://github.com/atmguille/archetypal-analysis/blob/main/Thesis/[English]%20Archetypal%20analysis%20and%20applications.pdf

Author:
-------
Ashen Weligalle

Date:
-----
2025-12-02

Intellectual Property Notice:
-----------------------------
This code was developed by Ashen Weligalle. Any reuse or redistribution
should include appropriate attribution to the original author.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def point_to_segment_distance(point, seg_start, seg_end):
    """
    Calculate the closest point on a line segment to a given point.
    Returns the closest point and the distance.
    """
    # Vector from seg_start to seg_end
    segment = seg_end - seg_start
    # Vector from seg_start to point
    point_vec = point - seg_start
    
    # Project point onto segment
    seg_length_sq = np.dot(segment, segment)
    if seg_length_sq == 0:
        return seg_start, np.linalg.norm(point - seg_start)
    
    t = max(0, min(1, np.dot(point_vec, segment) / seg_length_sq))
    closest_point = seg_start + t * segment
    distance = np.linalg.norm(point - closest_point)
    
    return closest_point, distance


def find_closest_point_on_hull(point, hull_vertices):
    """
    Find the closest point on the convex hull boundary to a given point.
    Returns the closest point on the hull.
    """
    min_distance = float('inf')
    closest_point = None
    
    n = len(hull_vertices)
    for i in range(n):
        seg_start = hull_vertices[i]
        seg_end = hull_vertices[(i + 1) % n]
        
        point_on_seg, distance = point_to_segment_distance(point, seg_start, seg_end)
        
        if distance < min_distance:
            min_distance = distance
            closest_point = point_on_seg
    
    return closest_point, min_distance


def plot_aa_archetypal_complete(X, S, A, K, vocab=None,
                                method='tsne',
                                output_file='aa_complete.png',
                                n_sample_lines=15,
                                figsize=(14, 14)):
    """
    Create a comprehensive archetypal analysis visualization showing:
    - Data points (X) with convex hull boundary (light green)
    - Archetypes (A) as red dots with convex hull (light blue)
    - Approximation error lines perpendicular to Conv(A) boundary
    
    Args:
        X: (N, M) original data matrix
        S: (N, K) document-archetype weights
        A: (K, M) archetype matrix
        K: number of archetypes
        vocab: vocabulary list (optional, for labeling)
        method: '2d' (if X is already 2D), 'tsne', or 'pca'
        output_file: path to save figure
        n_sample_lines: number of approximation error lines to show
        figsize: figure size tuple
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    
    print(f"\nCreating complete AA visualization (K={K})...")
    
    # Step 1: Reduce to 2D if needed
    if method == '2d' and X.shape[1] == 2:
        X_2d = X
        print("  Using existing 2D coordinates")
    elif method == 'tsne':
        print("  Reducing to 2D with t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0]-1))
        X_2d = tsne.fit_transform(X)
    elif method == 'pca':
        print("  Reducing to 2D with PCA...")
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
    else:
        raise ValueError("method must be '2d', 'tsne', or 'pca'")
    
    # Step 2: Project archetypes to 2D
    A_2d = np.zeros((K, 2))
    for k in range(K):
        weights = S[:, k]
        if weights.sum() > 0:
            A_2d[k] = np.average(X_2d, axis=0, weights=weights)
        else:
            top_docs = np.argsort(S[:, k])[-10:]
            A_2d[k] = X_2d[top_docs].mean(axis=0)
    
    print(f"  Data points: {X_2d.shape[0]}, Archetypes: {K}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal', adjustable='box')
    
    # Style axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Step 3: Draw convex hull of X (data points boundary - LIGHT GREEN)
    print("  Drawing convex hull of data points...")
    try:
        hull_X = ConvexHull(X_2d)
        hull_X_vertices = X_2d[hull_X.vertices]
        
        # Draw boundary
        ax.plot(np.append(hull_X_vertices[:, 0], hull_X_vertices[0, 0]),
               np.append(hull_X_vertices[:, 1], hull_X_vertices[0, 1]),
               'g-', linewidth=3, alpha=0.7, label='Conv(X) Boundary', zorder=2)
        
        # Fill with light green
        ax.fill(hull_X_vertices[:, 0], hull_X_vertices[:, 1],
               color='lightgreen', alpha=0.2, zorder=1)
        
        print(f"    ✓ Conv(X) has {len(hull_X.vertices)} vertices")
    except Exception as e:
        print(f"    Warning: Could not create Conv(X): {e}")
        hull_X_vertices = None
    
    # Step 4: Draw convex hull of A (archetypes - LIGHT BLUE)
    hull_A_vertices = None
    if K >= 3:
        print("  Drawing convex hull of archetypes...")
        try:
            hull_A = ConvexHull(A_2d)
            hull_A_vertices = A_2d[hull_A.vertices]
            
            # Draw boundary
            ax.plot(np.append(hull_A_vertices[:, 0], hull_A_vertices[0, 0]),
                   np.append(hull_A_vertices[:, 1], hull_A_vertices[0, 1]),
                   'b-', linewidth=3, alpha=0.8, label='Conv(A) Boundary', zorder=4)
            
            # Fill with light blue
            ax.fill(hull_A_vertices[:, 0], hull_A_vertices[:, 1],
                   color='lightblue', alpha=0.3, zorder=3)
            
            print(f"    ✓ Conv(A) has {len(hull_A.vertices)} vertices")
        except Exception as e:
            print(f"    Warning: Could not create Conv(A): {e}")
    elif K == 2:
        # For K=2, just draw line between archetypes
        hull_A_vertices = A_2d.copy()
        ax.plot(A_2d[:, 0], A_2d[:, 1], 'b-', linewidth=3, alpha=0.8,
               label='Conv(A) Boundary', zorder=4)
    
    # Step 5: Plot data points (X) colored by dominant archetype
    print("  Plotting data points colored by dominant archetype...")
    
    # Find dominant archetype for each data point
    dominant_archetype = np.argmax(S, axis=1)
    
    # Color palette for different archetypes
    if K <= 10:
        color_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                        '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5']
        colors = [color_palette[k % len(color_palette)] for k in range(K)]
    else:
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20')
        colors = [cmap(i / K) for i in range(K)]
    
    # Plot each group separately
    for k in range(K):
        mask = dominant_archetype == k
        if np.sum(mask) > 0:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=colors[k], s=60, alpha=0.6,
                      edgecolors='white', linewidths=0.5,
                      label=f'Dominant A{k+1}', zorder=5)
    
    # Step 6: Draw approximation error lines (perpendicular to Conv(A) boundary)
    if hull_A_vertices is not None:
        print(f"  Drawing {n_sample_lines} approximation error lines...")
        n_samples = min(n_sample_lines, X.shape[0])
        
        # Select random samples
        sample_indices = np.random.RandomState(42).choice(
            X.shape[0], 
            n_samples, 
            replace=False
        )
        
        for idx in sample_indices:
            x_point = X_2d[idx]
            
            # Find closest point on Conv(A) boundary
            closest_on_hull, distance = find_closest_point_on_hull(x_point, hull_A_vertices)
            
            # Draw perpendicular line from point to hull boundary
            ax.plot([x_point[0], closest_on_hull[0]],
                   [x_point[1], closest_on_hull[1]],
                   'k:', linewidth=1.5, alpha=0.4, zorder=6)
            
            # Mark the point on hull boundary
            ax.scatter([closest_on_hull[0]], [closest_on_hull[1]],
                      c='orange', s=40, alpha=0.6,
                      edgecolors='black', linewidths=0.5, zorder=7)
        
        # Add legend entry for approximation
        ax.scatter([], [], c='orange', s=40, alpha=0.6,
                  edgecolors='black', linewidths=0.5,
                  label='Projection on Conv(A)')
    
    # Step 7: Plot archetypes as RED DOTS (no black outline)
    print("  Plotting archetypes...")
    ax.scatter(A_2d[:, 0], A_2d[:, 1],
              marker='o', s=400, c='red',
              edgecolors='none',
              label='Archetypes (A)', zorder=10)
    
    # Step 8: Label archetypes with top word only
    for k in range(K):
        if vocab is not None:
            top_word_idx = np.argmax(A[k, :])
            label = vocab[top_word_idx]
        else:
            label = f"A{k+1}"
        
        # Position label directly on or near archetype
        offset_x = (X_2d[:, 0].max() - X_2d[:, 0].min()) * 0.04
        offset_y = (X_2d[:, 1].max() - X_2d[:, 1].min()) * 0.04
        
        ax.text(A_2d[k, 0] + offset_x, A_2d[k, 1] + offset_y,
               label,
               fontsize=12, fontweight='bold',
               ha='left', va='bottom',
               color='red',
               zorder=11)
    
    # Step 9: Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='g', linewidth=3, alpha=0.7,
              label='Conv(X) - Data Boundary'),
        Line2D([0], [0], color='b', linewidth=3, alpha=0.8,
              label='Conv(A) - Archetype Hull'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
              markersize=12, markeredgecolor='none',
              label='Archetypes (A)', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
              markersize=8, markeredgecolor='black', markeredgewidth=0.5,
              label='Projection on Conv(A)', linestyle='None'),
        Line2D([0], [0], color='black', linewidth=1.5, linestyle=':',
              label='Perpendicular Distance')
    ]
    
    # Add legend entries for each archetype's data points
    for k in range(K):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=colors[k],
                  markersize=10, markeredgecolor='white', markeredgewidth=0.5,
                  label=f'Points dominant in A{k+1}', linestyle='None')
        )
    
    ax.legend(handles=legend_elements, loc='upper right',
             fontsize=11, framealpha=0.95, edgecolor='black', 
             fancybox=True, shadow=True)
    
    # Step 10: Labels and title
    ax.set_xlabel('Dimension 1', fontsize=13, fontweight='bold')
    ax.set_ylabel('Dimension 2', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Archetypal Analysis Complete Visualization (K={K})\n'
        f'Data Points, Archetypes, Convex Hulls & Perpendicular Distances',
        fontsize=15, fontweight='bold', pad=20
    )
    
    # Step 11: Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.png', '.pdf'), dpi=300,
               bbox_inches='tight', transparent=True)
    
    print(f"\n  ✓ Saved: {output_file}")
    print(f"  ✓ Saved: {output_file.replace('.png', '.pdf')}")
    print(f"  ✓ Visualization complete!\n")
    
    return fig, ax


# ============================================================================
# Integration example
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\nUsage example:")
    print("="*80)
    print("""
# After running Archetypal Analysis:
from archetypal_plot import plot_aa_archetypal_complete

# Create complete visualization
fig, ax = plot_aa_archetypal_complete(
    X=X_norm,                  # Original data (N x M)
    S=aa_S,                    # Document-archetype weights (N x K)
    A=aa_A,                    # Archetype matrix (K x M)
    K=K,                       # Number of archetypes
    vocab=vocab,               # Vocabulary (optional)
    method='tsne',             # 'tsne', 'pca', or '2d'
    output_file='aa_complete.png',
    n_sample_lines=15,         # Number of error lines to show
    figsize=(14, 14)
)

plt.show()
    """)
    print("="*80)