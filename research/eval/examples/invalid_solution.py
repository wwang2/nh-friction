"""Invalid solution — g(xi) is not odd (even function)."""
import numpy as np

def friction_function(xi: np.ndarray) -> np.ndarray:
    return np.ones_like(np.asarray(xi, dtype=float))  # constant, not odd

def friction_derivative(xi: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(xi, dtype=float))
