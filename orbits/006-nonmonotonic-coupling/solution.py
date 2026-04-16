"""
Non-monotonic effective coupling for Nose-Hoover thermostat.
orbit/006-nonmonotonic-coupling

g(xi) = pade(xi) * h(xi^2)
where:
  pade(xi) = xi * (a + b*xi^2) / (1 + c*xi^2)   [parent orbit's optimal]
  h(u) = 1 - alpha * u * exp(-beta * u)           [modulation dip]

The modulation h has these properties:
  h(0) = 1  (no change at origin -> preserves KAM tori breaking)
  h(inf) = 1  (no change at large xi -> preserves thermostat restoring)
  h_min = 1 - alpha/(beta*e) at |xi| = 1/sqrt(beta)  (friction dip)

The dip creates a reduced-friction window at intermediate |xi|, letting
kinetic energy accumulate for barrier crossing in multimodal potentials.

Parent: orbit/003-cmaes-3pot with (a=0.7, b=3.0, c=0.06) -> metric=84.14
"""

import numpy as np

# ── Parent Pade parameters (frozen from orbit 003) ────────────────────────
_a = 0.7
_b = 3.0
_c = 0.06

# ── Modulation parameters ────────────────────────────────────────────────
# alpha controls dip depth: h_min = 1 - alpha/(beta*e)
# beta controls dip location: dip at |xi| = 1/sqrt(beta)
_alpha = 0.0598   # 5% dip
_beta = 0.44      # dip at |xi| ~ 1.5


def friction_function(xi: np.ndarray) -> np.ndarray:
    """g(xi) = pade(xi) * h(xi^2)"""
    xi2 = xi * xi
    # Parent Pade
    num = _a + _b * xi2
    den = 1.0 + _c * xi2
    pade = xi * num / den
    # Modulation
    h = 1.0 - _alpha * xi2 * np.exp(-_beta * xi2)
    return pade * h


def friction_derivative(xi: np.ndarray) -> np.ndarray:
    """Analytical g'(xi) via product rule: g = pade * h.

    pade(xi) = xi * N(u)/D(u), u = xi^2
    h(xi) = 1 - alpha * u * exp(-beta * u)

    g'(xi) = pade'(xi) * h(u) + pade(xi) * dh/dxi
    dh/dxi = dh/du * du/dxi = dh/du * 2*xi
    dh/du = -alpha * exp(-beta*u) * (1 - beta*u)
    """
    xi2 = xi * xi
    u = xi2

    # Parent Pade and its derivative
    N = _a + _b * u
    D = 1.0 + _c * u
    D2 = D * D
    pade = xi * N / D
    pade_prime = N / D + 2.0 * u * (_b * D - N * _c) / D2

    # Modulation and its derivative
    exp_term = np.exp(-_beta * u)
    h = 1.0 - _alpha * u * exp_term
    dh_du = -_alpha * exp_term * (1.0 - _beta * u)
    dh_dxi = dh_du * 2.0 * xi

    return pade_prime * h + pade * dh_dxi


def setup(seed: int = 42) -> None:
    """No-op: parameters pre-optimized."""
    pass
