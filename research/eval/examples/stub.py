"""Stub solution — Bulgac-Kusnezov friction function g(xi) = 2xi/(1+xi^2).
Known to improve ergodicity over standard NH on some potentials.
"""
import numpy as np

def friction_function(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=float)
    return 2.0 * xi / (1.0 + xi**2)

def friction_derivative(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=float)
    return (2.0 - 2.0 * xi**2) / (1.0 + xi**2)**2
