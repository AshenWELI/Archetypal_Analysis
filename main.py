import numpy as np
from AA import archetypal_analysis   # <--- import from AA.py
import matplotlib.pyplot as plt
from test_txt_AA import load_nips_data, normalize_documents, check_nips_data,print_summary_statistics,top_words_comparison, plot_comparison_results # type: ignore
from test_txt_AA import run_aa, run_nmf, run_kmeans, run_lda

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
    K = 5  # Number of archetypes (topics)
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
    
    # Create comparison plots
    plot_comparison_results(aa_S, nmf_W, km_W, lda_W, aa_rss, nmf_rss, km_rss, lda_rss,
                          aa_time, nmf_time, km_time, lda_time)
    
    plt.show()
    
    print("\n" + "="*100)
    print("Analysis complete!")
    print("="*100)