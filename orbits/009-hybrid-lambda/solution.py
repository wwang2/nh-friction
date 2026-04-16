"""
Hybrid driving exploration for Nose-Hoover thermostat.
orbit/009-hybrid-lambda

Hypothesis: h(q,p) = lambda*|p|^2 + (1-lambda)*|grad_V|^2/E_ref can improve
mixing by coupling configurational information into the thermostat.

Finding: All hybrid driving formulations break canonical sampling.
  1. Adaptive E_ref (EMA of |grad_V|^2) creates a trajectory-dependent
     feedback loop that violates detailed balance -> KL > 0.05 for all
     lambda < 1.0 tested (harmonic std=1.05 vs expected 1.0).
  2. Frozen E_ref (calibrated during burn-in) also fails: the |grad_V|^2
     term biases the q-marginal even with fixed normalization.
  3. Momentum-based hybrid h = (1-a)*|p|^2 + a*|p|^4/((d+2)*kT) is
     analytically valid (E[h]=d*kT for all a) but the high variance of
     |p|^4 destabilizes the thermostat -> KL gate failure for most a.
  4. Only a=0.0 (pure kinetic) and a=0.08 pass, with a=0.0 having
     lower weighted tau_int (67.9 vs 82.4 in proxy eval).

Conclusion: lambda* = 1.0 (pure kinetic driving). The optimal Pade
parameters remain a=0.7, b=3.0, c=0.06 from orbit 003.

g(xi) = xi * (a + b*xi^2) / (1 + c*xi^2)
"""

import numpy as np

# ── Optimized Pade parameters (orbit 003, confirmed local optimum) ───────
_a = 0.7
_b = 3.0
_c = 0.06


def friction_function(xi: np.ndarray) -> np.ndarray:
    """g(xi) = xi * (a + b*xi^2) / (1 + c*xi^2)

    Automatically odd: g(-xi) = -g(xi).
    Near zero: g ~ 0.7*xi.  Large |xi|: g ~ 50*xi.
    """
    xi2 = xi * xi
    num = _a + _b * xi2
    den = 1.0 + _c * xi2
    return xi * num / den


def friction_derivative(xi: np.ndarray) -> np.ndarray:
    """Analytical g'(xi) via quotient rule."""
    xi2 = xi * xi
    N = _a + _b * xi2
    D = 1.0 + _c * xi2
    D2 = D * D
    return N / D + 2.0 * xi2 * (_b * D - N * _c) / D2


def setup(seed: int = 42) -> None:
    """No-op: pure kinetic driving (lambda*=1.0), no state to reset."""
    pass
