"""
Non-monotonic effective coupling for Nose-Hoover thermostat.
orbit/006-nonmonotonic-coupling

RESULT: The non-monotonic coupling hypothesis was falsified.

g(xi) = xi * (a + b*xi^2) / (1 + c*xi^2)

After extensive exploration of non-monotonic friction profiles (multiplicative
modulation, 5-parameter rational, additive corrections, large-xi sigmoid),
all modifications to the parent's monotonic Pade were found to either:
  1. Fail the harmonic oscillator KL gate (< 0.05 threshold)
  2. Worsen the weighted metric by increasing doublewell variance

The parent's parameters (a=0.7, b=3.0, c=0.06) represent a robust local
optimum. The friction dip at intermediate |xi| does help gaussmix mixing
on favorable seeds but consistently worsens doublewell and risks harmonic
ergodicity failure.

Parent: orbit/003-cmaes-3pot with (a=0.7, b=3.0, c=0.06) -> metric=84.14
This orbit: metric=84.14 (no improvement found)
"""

import numpy as np

# ── Parent Pade parameters (unchanged — no improvement found) ─────────────
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
    """Analytical derivative g'(xi) via quotient rule."""
    xi2 = xi * xi
    u = xi2
    N = _a + _b * u
    D = 1.0 + _c * u
    D2 = D * D
    return N / D + 2.0 * u * (_b * D - N * _c) / D2


def setup(seed: int = 42) -> None:
    """No-op: parameters from parent orbit."""
    pass
