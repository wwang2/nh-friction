"""
Extended rational Padé friction function — orbit/002-pade-cmaes-refine.

g(ξ) = ξ · (a + b·ξ² + e·ξ⁴) / (1 + c·ξ² + d·ξ⁴ + f·ξ⁶)

Parent orbit 001 found (a=0.5, b=3.0, c=0.05, d=0.0) → metric=97.9.
This orbit refines by exploring the extended 6-parameter space around
the parent's optimum.

The function is automatically odd: g(-ξ) = -g(ξ) since it factors as
ξ times an even function of ξ².
"""

import numpy as np

# ── Parameters (tuned offline, hardcoded for deterministic eval) ──────────
# Starting from parent's best: a=0.5, b=3.0, c=0.05, d=0, e=0, f=0
_a = 0.7
_b = 3.0
_c = 0.06
_d = 0.0
_e = 0.0
_f = 0.0


def friction_function(xi: np.ndarray) -> np.ndarray:
    """g(ξ) = ξ · (a + b·ξ² + e·ξ⁴) / (1 + c·ξ² + d·ξ⁴ + f·ξ⁶)"""
    xi2 = xi * xi
    xi4 = xi2 * xi2
    num = _a + _b * xi2 + _e * xi4
    den = 1.0 + _c * xi2 + _d * xi4 + _f * xi4 * xi2
    return xi * num / den


def friction_derivative(xi: np.ndarray) -> np.ndarray:
    """Analytical derivative g'(ξ) via quotient rule.

    g(ξ) = ξ · N(u) / D(u) where u = ξ², N(u) = a + b·u + e·u²,
    D(u) = 1 + c·u + d·u² + f·u³.

    g'(ξ) = N/D + 2·ξ² · (N'·D - N·D') / D²
    where N'(u) = b + 2e·u, D'(u) = c + 2d·u + 3f·u².
    """
    xi2 = xi * xi
    u = xi2
    u2 = u * u
    N = _a + _b * u + _e * u2
    D = 1.0 + _c * u + _d * u2 + _f * u2 * u
    D2 = D * D
    Np = _b + 2.0 * _e * u
    Dp = _c + 2.0 * _d * u + 3.0 * _f * u2
    return N / D + 2.0 * u * (Np * D - N * Dp) / D2


def setup(seed: int = 42) -> None:
    """No-op: parameters are pre-tuned offline."""
    pass
