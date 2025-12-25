import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Archetypal Analysis Implementation (Standalone)
# ============================================================================

def project_to_simplex(v):
    """Project a vector v onto the probability simplex."""
    v = np.asarray(v, dtype=float)
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


def solve_simplex_least_squares(B, y, max_iter=500, tol=1e-6):
    """Solve: min_s || B^T s - y ||^2 subject to s >= 0, sum(s) = 1"""
    K = B.shape[0]
    s = np.ones(K) / K
    step_size = 1.0 / (np.linalg.norm(B, 2) ** 2 + 1e-12)

    prev_val = np.inf
    for _ in range(max_iter):
        grad = B @ (B.T @ s - y)
        s = s - step_size * grad
        s = project_to_simplex(s)

        val = 0.5 * np.linalg.norm(B.T @ s - y) ** 2
        if abs(prev_val - val) < tol * (1 + prev_val):
            break
        prev_val = val
    return s


def archetypal_analysis(X, K, max_iter=50, tol=1e-5, random_state=0, verbose=True):
    """Archetypal Analysis via alternating optimization."""
    rng = np.random.default_rng(random_state)
    N, M = X.shape

    idx = rng.choice(N, size=K, replace=False)
    A = X[idx, :].copy()

    S = np.ones((N, K)) / K
    C = np.ones((K, N)) / N
    prev_obj = np.inf

    for iteration in range(max_iter):
        for n in range(N):
            S[n, :] = solve_simplex_least_squares(A, X[n, :])

        StS = S.T @ S + 1e-10 * np.eye(K)
        A = np.linalg.solve(StS, S.T @ X)

        for k in range(K):
            C[k, :] = solve_simplex_least_squares(X, A[k, :])

        A = C @ X

        obj = 0.5 * np.linalg.norm(X - S @ A, "fro") ** 2
        
        if verbose and iteration % 10 == 0:
            print(f"  Iteration {iteration:3d}, RSS = {obj:.4e}")
        
        if abs(prev_obj - obj) < tol * (1 + prev_obj):
            if verbose:
                print(f"  Converged at iteration {iteration}, RSS = {obj:.4e}")
            break
        prev_obj = obj

    return S, C, A


# ============================================================================
# NIPS Dataset Loading
# ============================================================================

def check_nips_data(data_dir):
    """Check if NIPS bag-of-words dataset exists in local directory."""
    required_files = ['docword.nips.txt.gz', 'vocab.nips.txt']
    
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Required file '{filename}' not found in {data_dir}\n"
                f"Please ensure both files are present."
            )
        print(f"Found: {filename}")
    
    return data_dir


def load_nips_data(data_dir, max_docs=None):
    """Load NIPS bag-of-words dataset."""
    import gzip
    from scipy.sparse import csr_matrix
    
    vocab_file = os.path.join(data_dir, 'vocab.nips.txt')
    with open(vocab_file, 'r', encoding='utf-8', errors='ignore') as f:
        vocab = [line.strip() for line in f]
    
    docword_file = os.path.join(data_dir, 'docword.nips.txt.gz')
    
    print("Loading NIPS dataset...")
    with gzip.open(docword_file, 'rt') as f:
        num_docs = int(f.readline())
        num_words = int(f.readline())
        num_entries = int(f.readline())
        
        print(f"Dataset: {num_docs} documents, {num_words} words")
        
        docs = []
        words = []
        counts = []
        
        for line in f:
            doc_id, word_id, count = map(int, line.split())
            docs.append(doc_id - 1)
            words.append(word_id - 1)
            counts.append(count)
    
    X_sparse = csr_matrix((counts, (docs, words)), shape=(num_docs, num_words))
    
    if max_docs is not None and max_docs < num_docs:
        X_sparse = X_sparse[:max_docs, :]
        print(f"Using first {max_docs} documents")
    
    print("Converting to dense matrix...")
    X = X_sparse.toarray()
    
    print(f"Final matrix shape: {X.shape}")
    
    return X, vocab


def normalize_documents(X):
    """Normalize documents by their L2 norm."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def reduce_to_2d(X, method='tsne', random_state=42):
    """Reduce high-dimensional data to 2D for visualization."""
    print(f"\nReducing to 2D using {method.upper()}...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=random_state, 
                      perplexity=30, n_iter=1000, verbose=0)
        X_2d = reducer.fit_transform(X)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        X_2d = reducer.fit_transform(X)
    
    print(f"Reduced data shape: {X_2d.shape}")
    return X_2d


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_archetypal_analysis(X_2d, S, A_2d, output_path='aa_visualization.png'):
    """Visualize Archetypal Analysis."""
    K = A_2d.shape[0]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # Draw convex hull of archetypes
    if K >= 3:
        try:
            hull = ConvexHull(A_2d)
            ax.add_artist(Polygon(A_2d[hull.vertices], color='royalblue', 
                                 alpha=0.1, label='Convex Hull of Archetypes'))
        except:
            pass
    
    # Plot documents colored by dominant archetype
    dominant_archetype = np.argmax(S, axis=1)
    for k in range(K):
        mask = dominant_archetype == k
        if np.sum(mask) > 0:
            ax.plot(X_2d[mask, 0], X_2d[mask, 1], 'o', ms=4, 
                   c=colors[k % len(colors)], alpha=0.5)
    
    # Draw approximation lines for sample points
    n_samples = min(10, X_2d.shape[0])
    sample_indices = np.random.RandomState(42).choice(X_2d.shape[0], n_samples, replace=False)
    
    for idx in sample_indices:
        X_approx = S[idx] @ A_2d
        distances = np.sum((A_2d - X_2d[idx])**2, axis=1)
        closest_archetype = np.argmin(distances)
        
        ax.plot([X_2d[idx, 0], X_approx[0]], 
               [X_2d[idx, 1], X_approx[1]], 
               'k-', alpha=0.5, linewidth=1)
        ax.plot([X_2d[idx, 0], A_2d[closest_archetype, 0]], 
               [X_2d[idx, 1], A_2d[closest_archetype, 1]], 
               'k--', alpha=0.5, linewidth=1)
    
    # Plot archetypes
    ax.plot(A_2d[:, 0], A_2d[:, 1], '*', ms=15, c='gold', 
           markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    
    legend_elements = [
        Polygon([(0,0)], color='royalblue', alpha=0.1, label='Convex Hull'),
        Line2D([0], [0], color='k', linestyle='-', label='True Distance'),
        Line2D([0], [0], color='k', linestyle='--', label='Approx. Distance'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
               markersize=12, label='Archetypes')
    ]
    
    ax.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=8)
    ax.set_title('Archetypal Analysis', fontsize=12, fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, 
               transparent=True, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_kmeans(X_2d, labels, centers_2d, output_path='kmeans_visualization.png'):
    """Visualize K-means clustering."""
    K = centers_2d.shape[0]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # Draw convex hull of centers
    if K >= 3:
        try:
            hull = ConvexHull(centers_2d)
            ax.add_artist(Polygon(centers_2d[hull.vertices], color='royalblue', 
                                 alpha=0.1))
        except:
            pass
    
    # Plot clusters
    for k in range(K):
        cluster_points = X_2d[labels == k]
        if len(cluster_points) >= 3:
            try:
                hull_cluster = ConvexHull(cluster_points)
                ax.add_artist(Polygon(cluster_points[hull_cluster.vertices], 
                                     color=colors[k % len(colors)], alpha=0.15))
            except:
                pass
        ax.plot(cluster_points[:, 0], cluster_points[:, 1], 'o', 
               ms=4, c=colors[k % len(colors)], alpha=0.6)
    
    # Draw sample lines
    n_samples = min(10, X_2d.shape[0])
    sample_indices = np.random.RandomState(42).choice(X_2d.shape[0], n_samples, replace=False)
    
    for idx in sample_indices:
        cluster = labels[idx]
        center = centers_2d[cluster]
        ax.plot([X_2d[idx, 0], center[0]], 
               [X_2d[idx, 1], center[1]], 
               'k--', alpha=0.4, linewidth=1)
    
    # Plot centers
    ax.plot(centers_2d[:, 0], centers_2d[:, 1], 'X', ms=12, c='black', 
           markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    
    ax.set_title('K-means Clustering', fontsize=12, fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, 
               transparent=True, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_nmf(X_2d, W, output_path='nmf_visualization.png'):
    """Visualize NMF."""
    K = W.shape[1]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)
    
    # Compute topic centers
    topic_centers = np.zeros((K, 2))
    for k in range(K):
        weights = W_norm[:, k]
        topic_centers[k] = np.average(X_2d, axis=0, weights=weights)
    
    # Draw hull of centers
    if K >= 3:
        try:
            hull = ConvexHull(topic_centers)
            ax.add_artist(Polygon(topic_centers[hull.vertices], color='royalblue', 
                                 alpha=0.1))
        except:
            pass
    
    # Plot documents
    dominant_topic = np.argmax(W_norm, axis=1)
    for k in range(K):
        mask = dominant_topic == k
        cluster_points = X_2d[mask]
        if len(cluster_points) >= 3:
            try:
                hull_cluster = ConvexHull(cluster_points)
                ax.add_artist(Polygon(cluster_points[hull_cluster.vertices], 
                                     color=colors[k % len(colors)], alpha=0.15))
            except:
                pass
        if np.sum(mask) > 0:
            ax.plot(cluster_points[:, 0], cluster_points[:, 1], 'o', 
                   ms=4, c=colors[k % len(colors)], alpha=0.6)
    
    # Draw sample lines
    n_samples = min(10, X_2d.shape[0])
    sample_indices = np.random.RandomState(42).choice(X_2d.shape[0], n_samples, replace=False)
    
    for idx in sample_indices:
        X_approx = W_norm[idx] @ topic_centers
        ax.plot([X_2d[idx, 0], X_approx[0]], 
               [X_2d[idx, 1], X_approx[1]], 
               'k-', alpha=0.5, linewidth=1)
    
    # Plot centers
    ax.plot(topic_centers[:, 0], topic_centers[:, 1], 's', ms=10, 
           c='gold', markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    
    ax.set_title('Non-negative Matrix Factorization', fontsize=12, fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, 
               transparent=True, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_lda(X_2d, W_lda, output_path='lda_visualization.png'):
    """Visualize LDA."""
    K = W_lda.shape[1]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # Compute topic centers
    topic_centers = np.zeros((K, 2))
    for k in range(K):
        weights = W_lda[:, k]
        topic_centers[k] = np.average(X_2d, axis=0, weights=weights)
    
    # Draw hull
    if K >= 3:
        try:
            hull = ConvexHull(topic_centers)
            ax.add_artist(Polygon(topic_centers[hull.vertices], color='royalblue', 
                                 alpha=0.1))
        except:
            pass
    
    # Plot documents
    dominant_topic = np.argmax(W_lda, axis=1)
    for k in range(K):
        mask = dominant_topic == k
        cluster_points = X_2d[mask]
        if len(cluster_points) >= 3:
            try:
                hull_cluster = ConvexHull(cluster_points)
                ax.add_artist(Polygon(cluster_points[hull_cluster.vertices], 
                                     color=colors[k % len(colors)], alpha=0.15))
            except:
                pass
        if np.sum(mask) > 0:
            ax.plot(cluster_points[:, 0], cluster_points[:, 1], 'o', 
                   ms=4, c=colors[k % len(colors)], alpha=0.6)
    
    # Draw sample lines
    n_samples = min(10, X_2d.shape[0])
    sample_indices = np.random.RandomState(42).choice(X_2d.shape[0], n_samples, replace=False)
    
    for idx in sample_indices:
        X_approx = W_lda[idx] @ topic_centers
        ax.plot([X_2d[idx, 0], X_approx[0]], 
               [X_2d[idx, 1], X_approx[1]], 
               'k-', alpha=0.5, linewidth=1)
    
    # Plot centers
    ax.plot(topic_centers[:, 0], topic_centers[:, 1], 'd', ms=10, 
           c='gold', markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    
    ax.set_title('Latent Dirichlet Allocation', fontsize=12, fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, 
               transparent=True, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    try:
        # Set local data directory
        data_dir = r"D:\liu master\Statistics and ML\sem5\732A76 Research Project\project\Archetypal_Analysis\data\test_data"
        
        print("="*80)
        print("CREATING VISUALIZATIONS FOR AA, K-MEANS, NMF, AND LDA")
        print("="*80)
        
        # Check and load data
        check_nips_data(data_dir)
        X, vocab = load_nips_data(data_dir, max_docs=300)
        X_norm = normalize_documents(X)
        
        print(f"\nDataset shape: {X_norm.shape}")
        
        # Reduce to 2D
        X_2d = reduce_to_2d(X_norm, method='tsne', random_state=42)
        
        # ===== CONTROL NUMBER OF TOPICS/CLUSTERS HERE =====
        K = 3  # Change this to 3, 5, 10, etc. to control number of archetypes/topics/clusters
        random_state = 42
        # ==================================================
        
        # AA
        print("\n" + "="*80)
        print("Running Archetypal Analysis...")
        S_aa, C_aa, A_aa = archetypal_analysis(X_norm, K=K, max_iter=30, 
                                              random_state=random_state, verbose=False)
        A_2d = C_aa @ X_2d
        plot_archetypal_analysis(X_2d, S_aa, A_2d)
        
        # K-means
        print("\n" + "="*80)
        print("Running K-means...")
        kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=10)
        labels_km = kmeans.fit_predict(X_norm)
        centers_km_2d = np.array([X_2d[labels_km == k].mean(axis=0) for k in range(K)])
        plot_kmeans(X_2d, labels_km, centers_km_2d)
        
        # NMF
        print("\n" + "="*80)
        print("Running NMF...")
        X_nonneg = X_norm - X_norm.min() + 0.1
        nmf = NMF(n_components=K, init='random', random_state=random_state, max_iter=200)
        W_nmf = nmf.fit_transform(X_nonneg)
        plot_nmf(X_2d, W_nmf)
        
        # LDA
        print("\n" + "="*80)
        print("Running LDA...")
        lda = LatentDirichletAllocation(n_components=K, random_state=random_state, max_iter=10)
        W_lda = lda.fit_transform(X_norm)
        plot_lda(X_2d, W_lda)
        
        print("\n" + "="*80)
        print("ALL VISUALIZATIONS COMPLETE!")
        print("="*80)
        print("\nGenerated files:")
        print("  - aa_visualization.png (and .pdf)")
        print("  - kmeans_visualization.png (and .pdf)")
        print("  - nmf_visualization.png (and .pdf)")
        print("  - lda_visualization.png (and .pdf)")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()