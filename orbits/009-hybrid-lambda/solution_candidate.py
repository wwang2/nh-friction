"""
Candidate solution: a=0.72, b=2.8, c=0.10
Found via proxy grid search in orbit 009.
"""
import numpy as np

_a = 0.72
_b = 2.8
_c = 0.10

def friction_function(xi: np.ndarray) -> np.ndarray:
    xi2 = xi * xi
    return xi * (_a + _b * xi2) / (1.0 + _c * xi2)

def friction_derivative(xi: np.ndarray) -> np.ndarray:
    xi2 = xi * xi
    N = _a + _b * xi2
    D = 1.0 + _c * xi2
    D2 = D * D
    return N / D + 2.0 * xi2 * (_b * D - N * _c) / D2

def setup(seed: int = 42) -> None:
    pass
