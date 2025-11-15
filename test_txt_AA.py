import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from AA import archetypal_analysis 
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# NIPS Dataset Loading and Preprocessing
# ============================================================================

def check_nips_data(data_dir):
    """Check if NIPS bag-of-words dataset exists in local directory."""
    required_files = ['docword.nips.txt.gz', 'vocab.nips.txt']
    
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Required file '{filename}' not found in {data_dir}\n"
                f"Please ensure both 'docword.nips.txt.gz' and 'vocab.nips.txt' are present."
            )
        print(f"Found: {filename}")
    
    return data_dir


def load_nips_data(data_dir, max_docs=None):
    """
    Load NIPS bag-of-words dataset.
    
    Returns:
        X: Document-term matrix (N documents × M vocabulary)
        vocab: List of vocabulary words
    """
    import gzip
    
    # Load vocabulary
    vocab_file = os.path.join(data_dir, 'vocab.nips.txt')
    with open(vocab_file, 'r') as f:
        vocab = [line.strip() for line in f]
    
    # Load document-word matrix
    docword_file = os.path.join(data_dir, 'docword.nips.txt.gz')
    
    print("Loading NIPS dataset...")
    with gzip.open(docword_file, 'rt') as f:
        # Read header
        num_docs = int(f.readline())
        num_words = int(f.readline())
        num_entries = int(f.readline())
        
        print(f"Dataset info: {num_docs} documents, {num_words} words, {num_entries} entries")
        
        # Read sparse entries
        docs = []
        words = []
        counts = []
        
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print(f"  Processed {i}/{num_entries} entries...")
            
            doc_id, word_id, count = map(int, line.split())
            docs.append(doc_id - 1)  # Convert to 0-indexed
            words.append(word_id - 1)
            counts.append(count)
    
    # Create sparse matrix
    X_sparse = csr_matrix((counts, (docs, words)), shape=(num_docs, num_words))
    
    # Optionally limit number of documents for faster testing
    if max_docs is not None and max_docs < num_docs:
        X_sparse = X_sparse[:max_docs, :]
        print(f"Using first {max_docs} documents")
    
    # Convert to dense (WARNING: This might be memory intensive!)
    # For large datasets, you might want to work with sparse matrices
    print("Converting to dense matrix...")
    X = X_sparse.toarray()
    
    print(f"Final matrix shape: {X.shape}")
    
    return X, vocab


def normalize_documents(X):
    """
    Normalize documents by their L2 norm (TF normalization).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return X / norms


def top_words_per_archetype(A, vocab, top_n=10):
    """
    Extract top words for each archetype.
    
    Args:
        A: (K, M) archetype matrix
        vocab: vocabulary list
        top_n: number of top words to show
    """
    K = A.shape[0]
    
    print("\n" + "="*80)
    print("TOP WORDS PER ARCHETYPE")
    print("="*80)
    
    for k in range(K):
        top_indices = np.argsort(A[k, :])[-top_n:][::-1]
        top_words = [(vocab[i], A[k, i]) for i in top_indices]
        
        print(f"\nArchetype {k+1}:")
        for word, weight in top_words:
            print(f"  {word:20s} {weight:.4f}")

# ============================================================================
# Model Comparison Functions
# ============================================================================

def run_nmf(X, K, random_state=0, verbose=False):
    """Run Non-negative Matrix Factorization."""
    if verbose:
        print(f"\nRunning NMF with K={K}...")
    
    start_time = time.time()
    model = NMF(n_components=K, init='random', random_state=random_state, 
                max_iter=200, verbose=0)
    W = model.fit_transform(X)  # Document-topic matrix (N x K)
    H = model.components_       # Topic-word matrix (K x M)
    
    # Normalize W to get probability distribution
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)
    
    elapsed = time.time() - start_time
    
    # Compute reconstruction error
    reconstruction = W @ H
    rss = np.linalg.norm(X - reconstruction, 'fro') ** 2
    
    if verbose:
        print(f"  Completed in {elapsed:.2f}s, RSS = {rss:.4e}")
    
    return W_norm, H, rss, elapsed


def run_kmeans(X, K, random_state=0, verbose=False):
    """Run K-means clustering."""
    if verbose:
        print(f"\nRunning K-means with K={K}...")
    
    start_time = time.time()
    model = KMeans(n_clusters=K, random_state=random_state, n_init=10, max_iter=300)
    labels = model.fit_predict(X)
    centers = model.cluster_centers_  # (K x M)
    
    # Create one-hot encoding for hard assignments
    W = np.zeros((X.shape[0], K))
    W[np.arange(X.shape[0]), labels] = 1.0
    
    elapsed = time.time() - start_time
    
    # Compute reconstruction error
    reconstruction = W @ centers
    rss = np.linalg.norm(X - reconstruction, 'fro') ** 2
    
    # Compute clustering metrics
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    
    if verbose:
        print(f"  Completed in {elapsed:.2f}s, RSS = {rss:.4e}")
        print(f"  Silhouette Score: {silhouette:.4f}, Calinski-Harabasz: {calinski:.2f}")
    
    return W, centers, rss, elapsed, silhouette, calinski


def run_lda(X, K, random_state=0, verbose=False):
    """Run Latent Dirichlet Allocation."""
    if verbose:
        print(f"\nRunning LDA with K={K}...")
    
    start_time = time.time()
    model = LatentDirichletAllocation(n_components=K, random_state=random_state,
                                     max_iter=20, learning_method='batch', verbose=0)
    W = model.fit_transform(X)  # Document-topic distribution (N x K)
    H = model.components_        # Topic-word distribution (K x M)
    
    elapsed = time.time() - start_time
    
    # Compute reconstruction error (approximate for LDA)
    reconstruction = W @ H
    rss = np.linalg.norm(X - reconstruction, 'fro') ** 2
    
    # Compute perplexity
    perplexity = model.perplexity(X)
    
    if verbose:
        print(f"  Completed in {elapsed:.2f}s, RSS = {rss:.4e}")
        print(f"  Perplexity: {perplexity:.2f}")
    
    return W, H, rss, elapsed, perplexity


def run_aa(X, K, random_state=0, verbose=False):
    """Run Archetypal Analysis."""
    if verbose:
        print(f"\nRunning Archetypal Analysis with K={K}...")
    
    start_time = time.time()
    S, C, A = archetypal_analysis(X, K, max_iter=50, random_state=random_state, verbose=verbose)
    elapsed = time.time() - start_time
    
    reconstruction = S @ A
    rss = np.linalg.norm(X - reconstruction, 'fro') ** 2
    
    if verbose:
        print(f"  Completed in {elapsed:.2f}s, RSS = {rss:.4e}")
    
    return S, A, rss, elapsed
# ============================================================================
# Analysis and Visualization Functions
# ============================================================================

def top_words_comparison(vocab, aa_archetypes, nmf_topics, kmeans_centers, lda_topics, top_n=10):
    """Display top words for each method side by side."""
    K = aa_archetypes.shape[0]
    
    print("\n" + "="*120)
    print("TOP WORDS COMPARISON")
    print("="*120)
    
    for k in range(K):
        print(f"\n{'='*120}")
        print(f"Component/Topic/Cluster {k+1}:")
        print(f"{'='*120}")
        
        # Archetypal Analysis
        aa_idx = np.argsort(aa_archetypes[k, :])[-top_n:][::-1]
        aa_words = [f"{vocab[i]} ({aa_archetypes[k, i]:.3f})" for i in aa_idx]
        
        # NMF
        nmf_idx = np.argsort(nmf_topics[k, :])[-top_n:][::-1]
        nmf_words = [f"{vocab[i]} ({nmf_topics[k, i]:.3f})" for i in nmf_idx]
        
        # K-means
        km_idx = np.argsort(kmeans_centers[k, :])[-top_n:][::-1]
        km_words = [f"{vocab[i]} ({kmeans_centers[k, i]:.3f})" for i in km_idx]
        
        # LDA
        lda_idx = np.argsort(lda_topics[k, :])[-top_n:][::-1]
        lda_words = [f"{vocab[i]} ({lda_topics[k, i]:.3f})" for i in lda_idx]
        
        # Print in columns
        print(f"{'AA':<30} {'NMF':<30} {'K-means':<30} {'LDA':<30}")
        print(f"{'-'*30} {'-'*30} {'-'*30} {'-'*30}")
        for i in range(top_n):
            print(f"{aa_words[i]:<30} {nmf_words[i]:<30} {km_words[i]:<30} {lda_words[i]:<30}")


def plot_comparison_results(aa_S, nmf_W, kmeans_W, lda_W, aa_rss, nmf_rss, km_rss, lda_rss,
                           aa_time, nmf_time, km_time, lda_time):
    """Create comprehensive comparison plots."""
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Document representations (first 50 docs)
    methods = ['AA', 'NMF', 'K-means', 'LDA']
    representations = [aa_S[:50, :].T, nmf_W[:50, :].T, kmeans_W[:50, :].T, lda_W[:50, :].T]
    
    for i, (method, rep) in enumerate(zip(methods, representations)):
        plt.subplot(3, 4, i+1)
        plt.imshow(rep, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Weight')
        plt.xlabel('Document')
        plt.ylabel('Component/Topic')
        plt.title(f'{method}: Doc Representations')
    
    # Plot 2: Sparsity patterns
    for i, (method, rep) in enumerate(zip(methods, representations)):
        plt.subplot(3, 4, i+5)
        sparsity_per_doc = np.sum(rep.T > 0.01, axis=1)
        plt.hist(sparsity_per_doc, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Active Components')
        plt.ylabel('Frequency')
        plt.title(f'{method}: Component Sparsity')
        plt.axvline(np.mean(sparsity_per_doc), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(sparsity_per_doc):.2f}')
        plt.legend()
    
    # Plot 3: Reconstruction error comparison
    plt.subplot(3, 4, 9)
    rss_values = [aa_rss, nmf_rss, km_rss, lda_rss]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(methods, rss_values, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Reconstruction Error (RSS)')
    plt.title('Reconstruction Error Comparison')
    plt.yscale('log')
    for bar, val in zip(bars, rss_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Computation time comparison
    plt.subplot(3, 4, 10)
    time_values = [aa_time, nmf_time, km_time, lda_time]
    bars = plt.bar(methods, time_values, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Time (seconds)')
    plt.title('Computation Time Comparison')
    for bar, val in zip(bars, time_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
    
    # Plot 5: Normalized RSS per sample
    plt.subplot(3, 4, 11)
    N = aa_S.shape[0]
    rss_per_sample = [r/N for r in rss_values]
    bars = plt.bar(methods, rss_per_sample, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('RSS per Document')
    plt.title('Normalized Reconstruction Error')
    for bar, val in zip(bars, rss_per_sample):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9)
    
    # Plot 6: Average component usage
    plt.subplot(3, 4, 12)
    avg_usage = []
    for rep in [aa_S, nmf_W, kmeans_W, lda_W]:
        avg_usage.append(np.mean(np.sum(rep > 0.01, axis=1)))
    bars = plt.bar(methods, avg_usage, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Avg Components per Document')
    plt.title('Average Component Usage')
    for bar, val in zip(bars, avg_usage):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('method_comparison_results.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved as 'method_comparison_results.png'")


def print_summary_statistics(aa_S, nmf_W, kmeans_W, lda_W, aa_rss, nmf_rss, km_rss, lda_rss,
                            aa_time, nmf_time, km_time, lda_time, km_sil, km_cal, lda_perp):
    """Print summary statistics table."""
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)

    # Show top words for each archetype
    #top_words_per_archetype(A, vocab, top_n=15)
    
    print(f"\n{'Metric':<40} {'AA':<15} {'NMF':<15} {'K-means':<15} {'LDA':<15}")
    print("-"*100)
    
    # Reconstruction error
    print(f"{'Reconstruction Error (RSS)':<40} {aa_rss:<15.4e} {nmf_rss:<15.4e} {km_rss:<15.4e} {lda_rss:<15.4e}")
    
    # Normalized RSS
    N = aa_S.shape[0]
    print(f"{'RSS per Document':<40} {aa_rss/N:<15.4e} {nmf_rss/N:<15.4e} {km_rss/N:<15.4e} {lda_rss/N:<15.4e}")
    
    # Computation time
    print(f"{'Computation Time (s)':<40} {aa_time:<15.2f} {nmf_time:<15.2f} {km_time:<15.2f} {lda_time:<15.2f}")
    
    # Sparsity
    aa_sparse = np.mean(np.sum(aa_S > 0.01, axis=1))
    nmf_sparse = np.mean(np.sum(nmf_W > 0.01, axis=1))
    km_sparse = np.mean(np.sum(kmeans_W > 0.01, axis=1))
    lda_sparse = np.mean(np.sum(lda_W > 0.01, axis=1))
    print(f"{'Avg Active Components/Doc':<40} {aa_sparse:<15.2f} {nmf_sparse:<15.2f} {km_sparse:<15.2f} {lda_sparse:<15.2f}")
    
    # Method-specific metrics
    print(f"{'Silhouette Score (K-means)':<40} {'-':<15} {'-':<15} {km_sil:<15.4f} {'-':<15}")
    print(f"{'Calinski-Harabasz (K-means)':<40} {'-':<15} {'-':<15} {km_cal:<15.2f} {'-':<15}")
    print(f"{'Perplexity (LDA)':<40} {'-':<15} {'-':<15} {'-':<15} {lda_perp:<15.2f}")
    
    print("="*100)