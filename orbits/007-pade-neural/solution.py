"""
Pade friction function for Nose-Hoover thermostat.
orbit/007-pade-neural

g(xi) = xi * (a + b*xi^2) / (1 + c*xi^2)

This orbit attempted to enhance the parent Pade form with neural network
components (Gaussian bumps, sigmoid transitions, double rationals, power-law
tails). The key finding: the KL gate on the 1D harmonic oscillator is
extremely constraining. Any modification to g(xi) in the dynamically
relevant range |xi| < 1.6 either fails KL (harmonic ergodicity broken)
or worsens the weighted tau_int.

The thermostat variable xi stays in [-1.6, 1.6] for all benchmark potentials,
meaning the large-xi behavior of g(xi) is completely irrelevant. The function
shape in this narrow range is essentially determined by 3 Taylor coefficients,
which the Pade captures optimally.

After extensive search (15+ parameter configurations tested with full 1M-step
evaluator, 100+ configurations tested in proxy), the original Pade optimum
from orbit/003-cmaes-3pot remains the best KL-passing solution.

Parameters: a=0.7, b=3.0, c=0.06 (unchanged from parent)
Metric: 84.14

Key structural properties:
  - Odd by construction: g(-xi) = -g(xi) since xi * f(xi^2)
  - g'(0) = a = 0.7 (critical for breaking KAM tori on harmonic oscillator)
  - g(xi)/xi at xi=1: 3.49 (transition regime)
  - Large |xi|: g(xi) ~ (b/c)*xi = 50*xi (never visited in practice)
"""

import numpy as np

# ── Optimized parameters ─────────────────────────────────────────────────
_a = 0.7
_b = 3.0
_c = 0.06


def friction_function(xi: np.ndarray) -> np.ndarray:
    """g(xi) = xi * (a + b*xi^2) / (1 + c*xi^2)

    Behavior in dynamically relevant range |xi| < 1.6:
      xi=0:   g/xi = 0.70
      xi=0.5: g/xi = 1.38
      xi=1.0: g/xi = 3.49
      xi=1.5: g/xi = 6.56
    """
    xi = np.asarray(xi, dtype=np.float64)
    xi2 = xi * xi
    return xi * (_a + _b * xi2) / (1.0 + _c * xi2)


def friction_derivative(xi: np.ndarray) -> np.ndarray:
    """Analytical derivative g'(xi).

    g(xi) = xi * N(u)/D(u) where u = xi^2, N = a+b*u, D = 1+c*u
    g'(xi) = N/D + 2u*(b*D - N*c)/D^2
    """
    xi = np.asarray(xi, dtype=np.float64)
    xi2 = xi * xi
    u = xi2
    N = _a + _b * u
    D = 1.0 + _c * u
    D2 = D * D
    return N / D + 2.0 * u * (_b * D - N * _c) / D2


def setup(seed: int = 42) -> None:
    """No-op: parameters pre-optimized via offline search."""
    pass
