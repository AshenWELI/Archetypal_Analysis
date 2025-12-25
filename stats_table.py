"""
stat_table.py
----------------------
Efficient module for comprehensive method comparison statistics.
Computes sparsity, distinctiveness, and creates publication-quality visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


# ============================================================================
# Core Metrics
# ============================================================================

def compute_sparsity(S, threshold=1e-3):
    """
    Compute percentage of non-zero entries.
    
    Args:
        S: (N, K) document representation matrix
        threshold: Values below this are considered zero
    
    Returns:
        float: Percentage of non-zero entries
    """
    non_zero = np.sum(S > threshold)
    total = S.size
    return (non_zero / total) * 100


def compute_distinctiveness(components):
    """
    Compute average cosine angle between components.
    Higher angle = more distinct components.
    
    Args:
        components: (K, M) component/topic matrix
    
    Returns:
        float: Average cosine angle in degrees
    """
    K = components.shape[0]
    similarities = cosine_similarity(components)
    
    angles = []
    for i in range(K):
        for j in range(i+1, K):
            sim = similarities[i, j]
            angle = np.arccos(np.clip(sim, -1, 1)) * 180 / np.pi
            angles.append(angle)
    
    return np.mean(angles) if angles else 0.0


def compute_avg_active_components(S, threshold=1e-3):
    """
    Compute average number of active components per document.
    
    Args:
        S: (N, K) document representation matrix
        threshold: Threshold for considering component active
    
    Returns:
        float: Average number of active components
    """
    active = np.sum(S > threshold, axis=1)
    return np.mean(active)


def compute_reconstruction_error(X, reconstruction):
    """
    Compute Frobenius norm of reconstruction error.
    
    Args:
        X: (N, M) original data
        reconstruction: (N, M) reconstructed data
    
    Returns:
        float: Frobenius norm
    """
    return np.linalg.norm(X - reconstruction, 'fro')


# ============================================================================
# Comprehensive Statistics Function
# ============================================================================

def compute_comprehensive_stats(X, aa_results, nmf_results, km_results, lda_results):
    """
    Compute all statistics for all methods efficiently (single pass).
    
    Args:
        X: (N, M) original data matrix
        aa_results: dict with 'S' and 'A'
        nmf_results: dict with 'W' and 'H'
        km_results: dict with 'W' and 'centers'
        lda_results: dict with 'W' and 'H'
    
    Returns:
        dict: Comprehensive statistics for all methods
    """
    print("\nComputing comprehensive statistics...")
    
    stats = {}
    
    # ========== Archetypal Analysis ==========
    aa_S = aa_results['S']
    aa_A = aa_results['A']
    aa_reconstruction = aa_S @ aa_A
    
    stats['AA'] = {
        'sparsity': compute_sparsity(aa_S),
        'distinctiveness': compute_distinctiveness(aa_A),
        'avg_active': compute_avg_active_components(aa_S),
        'reconstruction_error': compute_reconstruction_error(X, aa_reconstruction)
    }
    
    # ========== NMF ==========
    nmf_W = nmf_results['W']
    nmf_H = nmf_results['H']
    nmf_W_norm = nmf_W / (nmf_W.sum(axis=1, keepdims=True) + 1e-10)
    nmf_reconstruction = nmf_W @ nmf_H
    
    stats['NMF'] = {
        'sparsity': compute_sparsity(nmf_W_norm),
        'distinctiveness': compute_distinctiveness(nmf_H),
        'avg_active': compute_avg_active_components(nmf_W_norm),
        'reconstruction_error': compute_reconstruction_error(X, nmf_reconstruction)
    }
    
    # ========== K-means ==========
    km_W = km_results['W']
    km_centers = km_results['centers']
    km_reconstruction = km_W @ km_centers
    
    stats['K-means'] = {
        'sparsity': compute_sparsity(km_W),  # Always 10% for hard clustering
        'distinctiveness': compute_distinctiveness(km_centers),
        'avg_active': compute_avg_active_components(km_W),  # Always 1.0
        'reconstruction_error': compute_reconstruction_error(X, km_reconstruction)
    }
    
    # ========== LDA ==========
    lda_W = lda_results['W']
    lda_H = lda_results['H']
    lda_reconstruction = lda_W @ lda_H
    
    stats['LDA'] = {
        'sparsity': compute_sparsity(lda_W),
        'distinctiveness': compute_distinctiveness(lda_H),
        'avg_active': compute_avg_active_components(lda_W),
        'reconstruction_error': compute_reconstruction_error(X, lda_reconstruction)
    }
    
    print("  ✓ Statistics computed for all methods")
    
    return stats


# ============================================================================
# Print Statistics Table
# ============================================================================

def print_comprehensive_table(stats):
    """
    Print publication-quality statistics table.
    
    Args:
        stats: Dictionary with statistics from compute_comprehensive_stats()
    """
    print("\n" + "="*100)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("="*100)
    
    # Header
    print(f"\n{'Metric':<45} {'NMF':<15} {'AA':<15} {'K-means':<15} {'LDA':<15}")
    print("-"*100)
    
    # 1. Sparsity
    print(f"{'% Non-zero entries in S':<45}", end="")
    for method in ['NMF', 'AA', 'K-means', 'LDA']:
        val = stats[method]['sparsity']
        print(f"{val:>6.1f}%       ", end="")
    print()
    
    # 2. Distinctiveness
    print(f"{'Avg. cosine angle (degrees)':<45}", end="")
    for method in ['NMF', 'AA', 'K-means', 'LDA']:
        val = stats[method]['distinctiveness']
        print(f"{val:>6.1f}°       ", end="")
    print()
    
    # 3. Reconstruction error
    print(f"{'Reconstruction error (Frobenius)':<45}", end="")
    for method in ['NMF', 'AA', 'K-means', 'LDA']:
        val = stats[method]['reconstruction_error']
        print(f"{val:>8.2f}     ", end="")
    print()
    
    # 4. Average active components
    print(f"{'Avg. active components/document':<45}", end="")
    for method in ['NMF', 'AA', 'K-means', 'LDA']:
        val = stats[method]['avg_active']
        print(f"{val:>6.2f}       ", end="")
    print()
    
    print("\n" + "="*100)
    print("INTERPRETATION")
    print("="*100)
    print("• Lower % non-zero = More sparse (easier to interpret)")
    print("• Higher cosine angle = More distinct components")
    print("• Lower reconstruction error = Better fit to data")
    print("• Lower active components = More focused representation")
    print("="*100)


# ============================================================================
# Save Statistics to CSV
# ============================================================================

def save_stats_to_csv(stats, output_file='comprehensive_statistics.csv'):
    """
    Save statistics to CSV file.
    
    Args:
        stats: Dictionary with statistics
        output_file: Output filename
    """
    data = []
    for method in ['NMF', 'AA', 'K-means', 'LDA']:
        data.append({
            'Method': method,
            'Sparsity_%': stats[method]['sparsity'],
            'Distinctiveness_deg': stats[method]['distinctiveness'],
            'Reconstruction_Error': stats[method]['reconstruction_error'],
            'Avg_Active_Components': stats[method]['avg_active']
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, float_format='%.2f')
    print(f"\n✓ Saved statistics to: {output_file}")
    
    return df


# ============================================================================
# Get Top Words
# ============================================================================

def get_top_words(components, vocabulary, top_n=10):
    """
    Get top N words for each component.
    
    Args:
        components: (K, M) component matrix
        vocabulary: List of words
        top_n: Number of top words
    
    Returns:
        list: List of (K) lists of (word, weight) tuples
    """
    K = components.shape[0]
    top_words_list = []
    
    for k in range(K):
        top_indices = np.argsort(components[k, :])[-top_n:][::-1]
        top_words = [(vocabulary[i], components[k, i]) for i in top_indices]
        top_words_list.append(top_words)
    
    return top_words_list


# ============================================================================
# Create Word Table Visualization
# ============================================================================

def create_word_table(aa_A, nmf_H, km_centers, lda_H, vocabulary, 
                     K, output_file='word_comparison_table.png'):
    """
    Create publication-quality table with top words and grayscale strengths.
    
    Args:
        aa_A, nmf_H, km_centers, lda_H: Component matrices
        vocabulary: List of vocabulary words
        K: Number of components
        output_file: Output filename
    """
    print(f"\nCreating word comparison table...")
    
    top_n = 10
    
    # Get top words for each method
    aa_words = get_top_words(aa_A, vocabulary, top_n)
    nmf_words = get_top_words(nmf_H, vocabulary, top_n)
    km_words = get_top_words(km_centers, vocabulary, top_n)
    lda_words = get_top_words(lda_H, vocabulary, top_n)
    
    # Create figure
    fig = plt.figure(figsize=(20, 14))
    
    methods = ['NMF', 'AA', 'K-means', 'LDA']
    method_words = [nmf_words, aa_words, km_words, lda_words]
    
    for method_idx, (method_name, words_list) in enumerate(zip(methods, method_words)):
        for comp_idx in range(K):
            ax = plt.subplot(4, K, method_idx * K + comp_idx + 1)
            ax.axis('off')
            
            words_weights = words_list[comp_idx]
            
            # Normalize weights for grayscale
            weights = np.array([w[1] for w in words_weights])
            if weights.max() > 0:
                normalized_weights = weights / weights.max()
            else:
                normalized_weights = weights
            
            # Display words with grayscale background
            y_pos = 0.9
            for (word, weight), norm_weight in zip(words_weights, normalized_weights):
                gray_intensity = 1 - norm_weight * 0.7
                
                rect = Rectangle((0, y_pos - 0.08), 1, 0.09, 
                                facecolor=(gray_intensity, gray_intensity, gray_intensity),
                                edgecolor='none')
                ax.add_patch(rect)
                
                ax.text(0.5, y_pos - 0.03, word, 
                       fontsize=8, ha='center', va='center',
                       fontweight='bold')
                
                y_pos -= 0.09
            
            # Component title
            if method_idx == 0:
                ax.set_title(f'C{comp_idx+1}', fontsize=10, fontweight='bold', pad=10)
            
            # Method label
            if comp_idx == 0:
                ax.text(-0.3, 0.5, method_name, 
                       fontsize=12, ha='right', va='center',
                       fontweight='bold', rotation=0,
                       transform=ax.transAxes)
    
    plt.suptitle(f'Top {top_n} Words per Component (Grayscale = Relative Strength)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.99])
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved word table: {output_file}")
    
    return fig


# ============================================================================
# Main Function - Call This in Your Main Script
# ============================================================================

def run_comprehensive_analysis(X, aa_results, nmf_results, km_results, lda_results, 
                               vocabulary, output_dir=''):
    """
    Run complete comprehensive analysis with one function call.
    
    Args:
        X: Original data matrix
        aa_results: dict with 'S', 'A'
        nmf_results: dict with 'W', 'H'
        km_results: dict with 'W', 'centers'
        lda_results: dict with 'W', 'H'
        vocabulary: List of vocabulary words
        output_dir: Directory to save files (optional)
    
    Returns:
        dict: Statistics dictionary
    """
    import os
    
    print("\n" + "="*100)
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print("="*100)
    
    # Compute statistics
    stats = compute_comprehensive_stats(X, aa_results, nmf_results, km_results, lda_results)
    
    # Print statistics table
    print_comprehensive_table(stats)
    
    # Save to CSV
    csv_file = os.path.join(output_dir, 'comprehensive_statistics.csv')
    save_stats_to_csv(stats, csv_file)
    
    # Create word table
    K = aa_results['A'].shape[0]
    table_file = os.path.join(output_dir, 'word_comparison_table.png')
    create_word_table(aa_results['A'], nmf_results['H'], 
                     km_results['centers'], lda_results['H'],
                     vocabulary, K, table_file)
    
    print("\n" + "="*100)
    print("✓ Comprehensive analysis complete!")
    print("="*100)
    print(f"\nGenerated files:")
    print(f"  • {csv_file}")
    print(f"  • {table_file}")
    
    return stats


# ============================================================================
# Standalone Test
# ============================================================================

if __name__ == "__main__":
    print("This is a module to be imported.")
    print("\nUsage in your main script:")
    print("="*80)
    print("""
from comprehensive_stats import run_comprehensive_analysis

# After running all methods, prepare results dictionaries
aa_results = {'S': aa_S, 'A': aa_A}
nmf_results = {'W': nmf_W, 'H': nmf_H}
km_results = {'W': km_W, 'centers': km_centers}
lda_results = {'W': lda_W, 'H': lda_H}

# Run comprehensive analysis (one function call!)
stats = run_comprehensive_analysis(
    X_norm, 
    aa_results, 
    nmf_results, 
    km_results, 
    lda_results,
    vocabulary=vocab
)
    """)
    print("="*80)