"""
Composite friction for Nosé-Hoover thermostat — Padé backbone + optional boost.

g(ξ) = ξ·(a + b·ξ²)/(1 + c·ξ²) + d·ξ³·exp(-e·ξ²)

The composite hypothesis tested whether adding a Gaussian-damped cubic
boost to the Padé form could improve mixing. After exhaustive search:
- Adding ANY non-Padé term (exp-damped, tanh, linear) fails the KL gate
  on the 1D harmonic oscillator
- The Padé form with parent parameters (a=0.7, b=3.0, c=0.06) sits at
  a stability sweet spot that's extremely sensitive to perturbation
- Optimal d=0: no boost. The parent Padé is not improved by additive terms.

This orbit's contribution: a negative result proving that the Padé
rational form is locally optimal within the class of composite functions
g_Padé + g_perturbation.
"""

import numpy as np

# ── Parameters ────────────────────────────────────────────────────────────
# Parent Padé parameters (proven optimal — no boost improves them)
_A = 0.7      # Padé linear coefficient
_B = 3.0      # Padé cubic coefficient
_C = 0.06     # Padé denominator coefficient
_D = 0.0      # Boost amplitude (optimal: 0)
_E = 0.3      # Boost damping (irrelevant when D=0)


def friction_function(xi: np.ndarray) -> np.ndarray:
    """g(ξ) = ξ·(a + b·ξ²)/(1 + c·ξ²) + d·ξ³·exp(-e·ξ²)"""
    xi2 = xi * xi
    pade = xi * (_A + _B * xi2) / (1.0 + _C * xi2)
    if _D != 0.0:
        boost = _D * xi * xi2 * np.exp(-_E * xi2)
        return pade + boost
    return pade


def friction_derivative(xi: np.ndarray) -> np.ndarray:
    """Analytical derivative.

    Padé: [a + (3b-ac)·ξ² + bc·ξ⁴] / (1+c·ξ²)²
    Boost: d·ξ²·(3 - 2e·ξ²)·exp(-e·ξ²)
    """
    xi2 = xi * xi
    denom = 1.0 + _C * xi2
    denom_sq = denom * denom
    d_pade = (_A + (3.0 * _B - _A * _C) * xi2 + _B * _C * xi2 * xi2) / denom_sq

    if _D != 0.0:
        d_boost = _D * xi2 * (3.0 - 2.0 * _E * xi2) * np.exp(-_E * xi2)
        return d_pade + d_boost
    return d_pade
