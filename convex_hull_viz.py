"""
Standalone Archetypal Analysis Convex Hull Visualization
Creates publication-quality figure showing archetypes and their convex hull.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D


def plot_aa_convex_hull(X, S, A, K, vocab=None, 
                        method='tsne', 
                        output_file='aa_convex_hull.png',
                        show_sample_lines=True,
                        n_sample_lines=10):
    """
    Create a clean convex hull visualization for Archetypal Analysis.
    Similar to the example with Z points and convex hull.
    
    Args:
        X: (N, M) original data matrix
        S: (N, K) document-archetype weights
        A: (K, M) archetype matrix
        K: number of archetypes
        vocab: vocabulary list (optional, for labeling)
        method: '2d' (if X is already 2D), 'tsne', or 'pca'
        output_file: path to save figure
        show_sample_lines: whether to show approximation lines
        n_sample_lines: number of sample documents to show lines for
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    
    print(f"\nCreating AA convex hull visualization (K={K})...")
    
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
    # Since A = C @ X, we need to project A to 2D
    # We can approximate: A_2d ≈ S^+ @ X_2d (pseudo-inverse)
    # Or better: use the archetype composition C if available
    # For simplicity, we'll compute weighted average
    A_2d = np.zeros((K, 2))
    for k in range(K):
        # Weight documents by their contribution to this archetype
        weights = S[:, k]
        if weights.sum() > 0:
            A_2d[k] = np.average(X_2d, axis=0, weights=weights)
        else:
            # Fallback: use documents most associated with this archetype
            top_docs = np.argsort(S[:, k])[-10:]
            A_2d[k] = X_2d[top_docs].mean(axis=0)
    
    print(f"  Archetype positions in 2D: {A_2d.shape}")
    
    # Step 3: Create the visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Remove top and right spines for clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Optional: remove ticks for very clean look
    # ax.set_xticks([])
    # ax.set_yticks([])
    
    ax.set_aspect('equal', adjustable='box')
    
    # Color scheme
    if K <= 10:
        # Predefined colors for common K values (visually distinct)
        color_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                        '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5']
        colors = [color_palette[k] for k in range(K)]
    else:
        # For K > 10, generate colors using colormap
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20')  # Good for up to 20 distinct colors
        colors = [cmap(i / K) for i in range(K)]
    archetype_color = 'royalblue'
    
    # Step 4: Draw convex hull of archetypes
    if K >= 3:
        try:
            hull = ConvexHull(A_2d)
            hull_polygon = Polygon(A_2d[hull.vertices], 
                                  color=archetype_color, 
                                  alpha=0.15, 
                                  label='Convex Hull of Archetypes',
                                  zorder=1)
            ax.add_patch(hull_polygon)
            
            # Draw hull edges
            for simplex in hull.simplices:
                ax.plot(A_2d[simplex, 0], A_2d[simplex, 1], 
                       'k-', linewidth=2, alpha=0.3, zorder=2)
            
            print(f"  ✓ Drew convex hull with {len(hull.vertices)} vertices")
        except Exception as e:
            print(f"  Warning: Could not create convex hull: {e}")
    
    # Step 5: Plot documents colored by dominant archetype
    dominant_archetype = np.argmax(S, axis=1)
    
    for k in range(K):
        mask = dominant_archetype == k
        if np.sum(mask) > 0:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                      c=colors[k % len(colors)], 
                      s=50, alpha=0.6, 
                      edgecolors='black', linewidths=0.5,
                      label=f'Archetype {k+1} docs',
                      zorder=3)
    
    # Step 6: Draw sample approximation lines
    if show_sample_lines and n_sample_lines > 0:
        # Select sample documents
        n_samples = min(n_sample_lines, X.shape[0])
        sample_indices = np.random.RandomState(42).choice(X.shape[0], n_samples, replace=False)
        
        for idx in sample_indices:
            # True position
            x_true = X_2d[idx]
            
            # Approximated position: x ≈ S @ A
            # In 2D: x_approx ≈ S @ A_2d
            x_approx = S[idx] @ A_2d
            
            # Find closest archetype
            distances = np.sum((A_2d - x_true)**2, axis=1)
            closest_archetype = np.argmin(distances)
            
            # Draw true approximation line (solid)
            ax.plot([x_true[0], x_approx[0]], 
                   [x_true[1], x_approx[1]], 
                   'k-', alpha=0.4, linewidth=1, zorder=4)
            
            # Draw line to closest archetype (dashed)
            ax.plot([x_true[0], A_2d[closest_archetype, 0]], 
                   [x_true[1], A_2d[closest_archetype, 1]], 
                   'k--', alpha=0.3, linewidth=1, zorder=4)
    
    # Step 7: Plot archetypes as large gold stars
    ax.scatter(A_2d[:, 0], A_2d[:, 1], 
              marker='*', s=800, c='gold', 
              edgecolors='black', linewidths=2.5, 
              label='Archetypes', zorder=10)
    
    # Step 8: Label archetypes
    for k in range(K):
        # Get top words for this archetype if vocabulary provided
        if vocab is not None:
            top_word_indices = np.argsort(A[k, :])[-3:][::-1]
            top_words = [vocab[i] for i in top_word_indices]
            label = f"A{k+1}\n({', '.join(top_words)})"
        else:
            label = f"A{k+1}"
        
        # Position label slightly above archetype
        offset_y = (X_2d[:, 1].max() - X_2d[:, 1].min()) * 0.03
        ax.text(A_2d[k, 0], A_2d[k, 1] + offset_y, 
               label, fontsize=9, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='black', alpha=0.8),
               zorder=11)
    
    # Step 9: Create custom legend
    legend_elements = [
        Polygon([(0,0)], color=archetype_color, alpha=0.15, 
               label='Convex Hull of Archetypes'),
        Line2D([0], [0], color='k', linestyle='-', linewidth=1,
              label='True Distance (S @ A)'),
        Line2D([0], [0], color='k', linestyle='--', linewidth=1,
              label='Approximated Distance'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
              markersize=15, markeredgecolor='black', markeredgewidth=2,
              label='Archetypes', linestyle='None')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', 
             fontsize=10, framealpha=0.95)
    
    # Step 10: Labels and title
    ax.set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(f'Archetypal Analysis with K={K} Archetypes\nConvex Hull Representation', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Step 11: Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.png', '.pdf'), dpi=300, 
               bbox_inches='tight', transparent=True)
    
    print(f"  ✓ Saved: {output_file}")
    print(f"  ✓ Saved: {output_file.replace('.png', '.pdf')}")
    
    return fig, ax


# ============================================================================
# Integration with your main script
# ============================================================================

def add_to_main():
    """
    Example of how to integrate into your main script.
    Add this after running AA:
    """
    code_example = '''
# After running AA:
aa_S, aa_A, aa_rss, aa_time = run_aa(X_norm, K, random_state=random_state, verbose=True)

# Generate convex hull visualization
from aa_convex_hull_viz import plot_aa_convex_hull

fig, ax = plot_aa_convex_hull(
    X=X_norm,           # Original data
    S=aa_S,             # Document-archetype weights
    A=aa_A,             # Archetype matrix
    K=K,                # Number of archetypes
    vocab=vocab,        # Vocabulary (for labeling)
    method='tsne',      # Dimensionality reduction method
    output_file=f'aa_convex_hull_K{K}.png',
    show_sample_lines=True,
    n_sample_lines=10
)

plt.show()
'''
    return code_example


# ============================================================================
# Standalone usage example
# ============================================================================

if __name__ == "__main__":
    print("This is a module to be imported.")
    print("\nUsage in your main script:")
    print("="*80)
    print(add_to_main())
    print("="*80)