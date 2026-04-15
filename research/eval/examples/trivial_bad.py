"""Trivial bad solution — g(xi) = xi (standard Nosé-Hoover, known non-ergodic on 1D HO)."""
import numpy as np

def friction_function(xi: np.ndarray) -> np.ndarray:
    return np.asarray(xi, dtype=float)

def friction_derivative(xi: np.ndarray) -> np.ndarray:
    return np.ones_like(np.asarray(xi, dtype=float))
