import numpy as np
from AA import archetypal_analysis   # <--- import from AA.py

# Example usage
X = np.random.rand(100, 5)
S, C, A = archetypal_analysis(X, K=3)


print("S shape:", S.shape)  # (100, 3)
print("C shape:", C.shape)  # (3, 100)
print("A shape:", A.shape)  # (3, 5)