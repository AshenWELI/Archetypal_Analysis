import numpy as np
from AA import archetypal_analysis   # <--- import from AA.py
import matplotlib.pyplot as plt
from func_module import load_nips_data, normalize_documents, check_nips_data,print_summary_statistics,top_words_comparison, plot_comparison_results # type: ignore
from func_module import run_aa, run_nmf, run_kmeans, run_lda
from convex_hull_viz import plot_aa_convex_hull
from stats_table import run_comprehensive_analysis

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Set local data directory
    data_dir = r"D:\liu master\Statistics and ML\sem5\732A76 Research Project\project\Archetypal_Analysis\data\test_data"
    
    # Check if required files exist
    print("Checking for NIPS dataset files...")
    check_nips_data(data_dir)
    
    # Load dataset (use max_docs to limit size for testing)
    # For full dataset, remove max_docs parameter
    X, vocab = load_nips_data(data_dir, max_docs=500)
    
    # Normalize documents
    X_norm = normalize_documents(X)
    
    print(f"\nDataset shape: {X_norm.shape}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Run Archetypal Analysis
    K = 10  # Number of archetypes (topics)
    random_state = 42
    print(f"\nRunning Archetypal Analysis with K={K} archetypes...")
    
    # Run all methods
    aa_S, aa_A, aa_rss, aa_time = run_aa(X_norm, K, random_state=random_state, verbose=True)
    nmf_W, nmf_H, nmf_rss, nmf_time = run_nmf(X_norm, K, random_state=random_state, verbose=True)
    km_W, km_centers, km_rss, km_time, km_sil, km_cal = run_kmeans(X_norm, K, random_state=random_state, verbose=True)
    lda_W, lda_H, lda_rss, lda_time, lda_perp = run_lda(X_norm, K, random_state=random_state, verbose=True)
    
    # Print summary statistics
    print_summary_statistics(aa_S, nmf_W, km_W, lda_W, aa_rss, nmf_rss, km_rss, lda_rss,
                           aa_time, nmf_time, km_time, lda_time, km_sil, km_cal, lda_perp)
    
    # Display top words comparison
    top_words_comparison(vocab, aa_A, nmf_H, km_centers, lda_H, top_n=10)
    
    # Statistic comparison

    # Prepare results dictionaries
    aa_results = {'S': aa_S, 'A': aa_A}
    nmf_results = {'W': nmf_W, 'H': nmf_H}
    km_results = {'W': km_W, 'centers': km_centers}
    lda_results = {'W': lda_W, 'H': lda_H}
    
    # Run comprehensive analysis (computes all stats + creates word table)
    stats = run_comprehensive_analysis(
        X_norm, 
        aa_results, 
        nmf_results, 
        km_results, 
        lda_results,
        vocabulary=vocab
    )

    # Create comparison plots
    plot_comparison_results(aa_S, nmf_W, km_W, lda_W, aa_rss, nmf_rss, km_rss, lda_rss,
                          aa_time, nmf_time, km_time, lda_time)

    print(f"\n{'='*100}")
    print("CREATING AA CONVEX HULL VISUALIZATION")
    print(f"{'='*100}")
    
    fig_aa, ax_aa = plot_aa_convex_hull(
        X=X_norm,                           # Original normalized data
        S=aa_S,                             # Document-archetype weights
        A=aa_A,                             # Archetype matrix
        K=K,                                # Number of archetypes
        vocab=vocab,                        # Vocabulary for labeling
        method='tsne',                      # Use t-SNE for 2D reduction
        output_file=f'aa_convex_hull_K{K}.png',
        show_sample_lines=True,             # Show approximation lines
        n_sample_lines=10                   # Number of sample lines
    )
    
    plt.show()

    print("\n Key Findings:")
    print("-"*100)
    print(f"  Sparsity (lower = better):")
    print(f"    AA:      {stats['AA']['sparsity']:.1f}% {'✓ Best' if stats['AA']['sparsity'] < stats['NMF']['sparsity'] else ''}")
    print(f"    NMF:     {stats['NMF']['sparsity']:.1f}%")
    print(f"    K-means: {stats['K-means']['sparsity']:.1f}%")
    print(f"    LDA:     {stats['LDA']['sparsity']:.1f}%")
    
    print(f"\n  Reconstruction Error (lower = better):")
    print(f"    AA:      {stats['AA']['reconstruction_error']:.2f} {'✓ Best' if stats['AA']['reconstruction_error'] < min(stats['NMF']['reconstruction_error'], stats['LDA']['reconstruction_error']) else ''}")
    print(f"    NMF:     {stats['NMF']['reconstruction_error']:.2f}")
    print(f"    K-means: {stats['K-means']['reconstruction_error']:.2f}")
    print(f"    LDA:     {stats['LDA']['reconstruction_error']:.2f}")
    
    print(f"\n  Distinctiveness (higher = better):")
    print(f"    AA:      {stats['AA']['distinctiveness']:.1f}°")
    print(f"    NMF:     {stats['NMF']['distinctiveness']:.1f}°")
    print(f"    K-means: {stats['K-means']['distinctiveness']:.1f}°")
    print(f"    LDA:     {stats['LDA']['distinctiveness']:.1f}°")
    
    print("\n" + "="*100)
    print("Analysis complete!")
    print("="*100)