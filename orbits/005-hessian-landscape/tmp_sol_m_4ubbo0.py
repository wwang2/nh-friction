"""Auto-generated Pade friction for Hessian sweep. a=0.7, b=3.0, c=0.06"""
import numpy as np

_a = 0.7
_b = 3.0
_c = 0.06

def friction_function(xi: np.ndarray) -> np.ndarray:
    xi2 = xi * xi
    num = _a + _b * xi2
    den = 1.0 + _c * xi2
    return xi * num / den

def friction_derivative(xi: np.ndarray) -> np.ndarray:
    xi2 = xi * xi
    u = xi2
    N = _a + _b * u
    D = 1.0 + _c * u
    D2 = D * D
    return N / D + 2.0 * u * (_b * D - N * _c) / D2

def setup(seed: int = 42) -> None:
    pass
