"""
Pade rational friction function for Nose-Hoover thermostat.
orbit/005-hessian-landscape

g(xi) = xi * (a + b*xi^2) / (1 + c*xi^2)

Parameters: starting from parent orbit 003's optimum (a=0.7, b=3.0, c=0.06).
This orbit performs Hessian landscape analysis to identify soft directions
and hill-climb along them.
"""

import numpy as np

# ── Optimized parameters ─────────────────────────────────────────────────
_a = 0.7
_b = 3.0
_c = 0.06


def friction_function(xi: np.ndarray) -> np.ndarray:
    """g(xi) = xi * (a + b*xi^2) / (1 + c*xi^2)

    Behavior:
      Near zero: g(xi) ~ a*xi = 0.7*xi (linear coupling)
      Large |xi|: g(xi) ~ (b/c)*xi = 50*xi (strong restoring)
    """
    xi2 = xi * xi
    num = _a + _b * xi2
    den = 1.0 + _c * xi2
    return xi * num / den


def friction_derivative(xi: np.ndarray) -> np.ndarray:
    """Analytical derivative g'(xi) via quotient rule.

    g(xi) = xi * N(u) / D(u) where u = xi^2
    N(u) = a + b*u, D(u) = 1 + c*u
    g'(xi) = N/D + 2*u * (b*D - N*c) / D^2
    """
    xi2 = xi * xi
    u = xi2
    N = _a + _b * u
    D = 1.0 + _c * u
    D2 = D * D
    return N / D + 2.0 * u * (_b * D - N * _c) / D2


def setup(seed: int = 42) -> None:
    """No-op: parameters pre-optimized."""
    pass
