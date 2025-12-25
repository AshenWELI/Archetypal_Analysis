
"""
Archetypal Analysis (AA) via Alternating Optimization.

This implementation follows Algorithm 1 (References) and solves the optimization problem:

    min ||X − S C X||_F^2  ≡  min ||X − S A||_F^2,

where A = C X denotes the archetypes.

The method alternates between optimizing S and C under simplex constraints.

Parameters:
-----------
X : ndarray of shape (N, M)
    Data matrix with N samples and M features.
K : int
    Number of archetypes.
max_iter : int
    Maximum number of alternating optimization iterations.
tol : float
    Convergence tolerance.
random_state : int or None
    Random seed for reproducibility.

Returns:
--------
S : ndarray of shape (N, K)
    Coefficient matrix representing data points in the archetype space,
    where S ≥ 0 and each row sums to 1.
C : ndarray of shape (K, N)
    Coefficient matrix representing archetypes in the data space,
    where C ≥ 0 and each row sums to 1.
A : ndarray of shape (K, M)
    Archetype matrix, computed as A = C X.

References:
-----------
Inspired by the methodology presented in:
https://arxiv.org/pdf/2504.12392

Author:
-------
Ashen Weligalle

Date:
-----
2025-11-05

Intellectual Property Notice:
-----------------------------
This code was developed by Ashen Weligalle. Any reuse or redistribution
should include appropriate attribution to the original author.
"""

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

# Reference from https://apxml.com/courses/optimization-techniques-ml/chapter-7-advanced-specialized-optimization/practice-projected-gradient-descent .See more for example uses.
def solve_sls(B, y, max_iter=500, tol=1e-6):
    """
    Solve:  min_s  || B^T s - y ||^2   subject to s >= 0, sum(s) = 1
    using projected gradient descent (PGD).
    
    Note: B should be passed as the matrix (not transposed), and this
    function will compute B^T @ (B^T @ s - y) for the gradient.
    """
    K = B.shape[0]  # B is (M, K) so we want K archetypes
    s = np.ones(K) / K  # start uniform
    
    # For min ||B^T s - y||^2, gradient is: B @ (B^T @ s - y)
    # step_size = 1e-2
    # Step size based on Lipschitz constant
    step_size = 1.0 / (np.linalg.norm(B, 2) ** 2 + 1e-12)

    prev_val = np.inf
    for _ in range(max_iter):
        grad = B @ (B.T @ s - y)  # Gradient for ||B^T s - y||^2
        s = s - step_size * grad
        # s = np.argmin(s)
        s = project_to_simplex(s)

        val = 0.5 * np.linalg.norm(B.T @ s - y) ** 2
        if abs(prev_val - val) < tol * (1 + prev_val):
            break
        prev_val = val
    return s


def archetypal_analysis(X, K, max_iter=50, tol=1e-5, random_state=0, verbose=True):
    """
    Archetypal Analysis (AA) via alternating optimization.
    
    Follows Algorithm 1: min ||X - S*C*X||_F^2 = min ||X - S*A||_F^2
    where A = C*X are the archetypes.
    
    Args:
        X: (N, M) data matrix (N samples, M features)
        K: number of archetypes
        max_iter: maximum iterations
        tol: convergence tolerance
        random_state: random seed
        
    Returns:
        S: (N, K) coefficients for data points in archetype space (s_n >= 0, sum=1)
        C: (K, N) coefficients for archetypes in data space (c_k >= 0, sum=1)
        A: (K, M) archetypes where A = C*X
    """
    rng = np.random.default_rng(random_state)
    N, M = X.shape

    # --- Initialize A with random rows of X ---
    idx = rng.choice(N, size=K, replace=False)
    A = X[idx, :].copy()

    S = np.ones((N, K)) / K
    C = np.ones((K, N)) / N
    prev_obj = np.inf

    for iteration in range(max_iter):
        # --- Step 1: Update S row by row ---
        # For each data point n, solve: min ||A^T s_n - x_n||^2 s.t. s_n >= 0, sum(s_n) = 1
        for n in range(N):
            S[n, :] = solve_sls(A, X[n, :])

        # --- Step 2: Update A via least squares ---
        # A = (S^T S)^{-1} S^T X
        StS = S.T @ S + 1e-10 * np.eye(K)  # Add small regularization for stability
        A = np.linalg.solve(StS, S.T @ X)

        # --- Step 3: Update C row by row ---
        # For each archetype k, solve: min ||X^T c_k - a_k||^2 s.t. c_k >= 0, sum(c_k) = 1
        for k in range(K):
            C[k, :] = solve_sls(X, A[k, :])

        # --- Step 4: Update A using C ---
        # A = C * X (archetypes are convex combinations of data points)
        A = C @ X

        # --- Check convergence ---
        # RSS = ||X - S*A||_F^2 = ||X - S*C*X||_F^2
        obj = 0.5 * np.linalg.norm(X - S @ A, "fro") ** 2

        if verbose and iteration % 5 == 0:
            print(f"Iteration {iteration:3d}, RSS = {obj:.4e}")
        if abs(prev_obj - obj) < tol * (1 + prev_obj):
            if verbose:
                print(f"Converged at iteration {iteration}, RSS = {obj:.4e}")
            break
        prev_obj = obj

    return S, C, A