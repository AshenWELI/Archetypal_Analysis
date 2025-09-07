import numpy as np

def project_to_simplex(v):
    """
    Project a vector v onto the probability simplex:
        {w | w >= 0, sum(w) = 1}.
    """
    v = np.asarray(v, dtype=float)
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


def solve_simplex_least_squares(B, y, max_iter=500, tol=1e-6):
    """
    Solve:  min_s  || B s - y ||^2   subject to s >= 0, sum(s) = 1
    using projected gradient descent (PGD).
    """
    K = B.shape[1]
    s = np.ones(K) / K  # start uniform
    step_size = 1.0 / (np.linalg.norm(B, 2) ** 2 + 1e-12)

    prev_val = np.inf
    for _ in range(max_iter):
        grad = B.T @ (B @ s - y)
        s = s - step_size * grad
        s = project_to_simplex(s)

        val = 0.5 * np.linalg.norm(B @ s - y) ** 2
        if abs(prev_val - val) < tol * (1 + prev_val):
            break
        prev_val = val
    return s


def archetypal_analysis(X, K, max_iter=50, tol=1e-5, random_state=0):
    """
    Archetypal Analysis (AA) via alternating optimization.
    Args:
        X: (N, M) data matrix
        K: number of archetypes
    Returns:
        S: (N, K) coefficients for data points in archetype space
        C: (K, N) coefficients for archetypes in data space
        A: (K, M) archetypes
    """
    rng = np.random.default_rng(random_state)
    N, M = X.shape

    # --- Initialize A with random rows of X ---
    idx = rng.choice(N, size=K, replace=False)
    A = X[idx, :].copy()

    S = np.ones((N, K)) / K
    C = np.ones((K, N)) / N
    prev_obj = np.inf

    for _ in range(max_iter):
        # --- Update S row by row ---
        for n in range(N):
            S[n, :] = solve_simplex_least_squares(A.T, X[n, :])

        # --- Update A via least squares ---
        StS = S.T @ S + 1e-10 * np.eye(K)  # stability
        A = np.linalg.solve(StS, S.T @ X)

        # --- Update C row by row ---
        for k in range(K):
            C[k, :] = solve_simplex_least_squares(X.T, A[k, :])

        # --- Update A using C ---
        A = C @ X

        # --- Check convergence ---
        obj = 0.5 * np.linalg.norm(X - S @ A, "fro") ** 2
        if abs(prev_obj - obj) < tol * (1 + prev_obj):
            break
        prev_obj = obj

    return S, C, A
