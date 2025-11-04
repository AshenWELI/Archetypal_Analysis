import numpy as np
import os
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from AA import archetypal_analysis 


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
    print(f"\nRunning Archetypal Analysis with K={K} archetypes...")
    
    S, C, A = archetypal_analysis(X_norm, K=K, max_iter=30, verbose=True)
    
    # Display results
    print(f"\nResults:")
    print(f"  S shape (documents × archetypes): {S.shape}")
    print(f"  C shape (archetypes × documents): {C.shape}")
    print(f"  A shape (archetypes × vocabulary): {A.shape}")
    
    # Show top words for each archetype
    top_words_per_archetype(A, vocab, top_n=15)
    
    # Compute final reconstruction error
    reconstruction = S @ A
    rss = np.linalg.norm(X_norm - reconstruction, 'fro') ** 2
    print(f"\nFinal RSS (Residual Sum of Squares): {rss:.4e}")
    
    # Show statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Mean archetype sparsity (C): {np.mean(np.sum(C > 1e-3, axis=1)):.2f} documents per archetype")
    print(f"Mean document representation (S): {np.mean(np.sum(S > 1e-3, axis=1)):.2f} archetypes per document")
    
    # Plot archetype weights for first few documents
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(S[:50, :].T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Weight')
    plt.xlabel('Document')
    plt.ylabel('Archetype')
    plt.title('Archetype Weights for First 50 Documents')
    
    plt.subplot(1, 2, 2)
    plt.imshow(C, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Weight')
    plt.xlabel('Document (all)')
    plt.ylabel('Archetype')
    plt.title('Document Weights for Each Archetype')
    
    plt.tight_layout()
    plt.savefig('archetypal_analysis_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'archetypal_analysis_results.png'")
    
    plt.show()