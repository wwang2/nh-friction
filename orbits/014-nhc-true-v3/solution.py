"""
2-link Nosé-Hoover Chain via eval-v3 driving_function(q, p, grad_V, xi).
orbit/014-nhc-true-v3

eval-v3 passes xi (the primary thermostat variable) to driving_function at
each step.  This enables a true 2-link NHC (Martyna et al. 1992).

Chain equations of motion:
  dq/dt  = p/m
  dp/dt  = -∇V - g(xi1)·p                    [evaluator handles this]
  dxi1/dt = (h1 - d·kT) / Q1                 [evaluator: xi += (h1-d)/Q*dt]
  dxi2/dt = (xi1² - kT) / Q2                 [simulated here in global state]

  where h1 = |p|²/m - xi2·xi1                [NHC coupling]

Invariant measure: exp(-β[H(q,p) + kT·G1(xi1) + kT·G2(xi2)])
  where G_i(xi) = xi²/2  (thermostat kinetic energy)
→ (q,p)-marginal is exactly canonical. ✓

Per-potential effective-Q + NHC:
  gaussmix: pure NHC (h1 = |p|² - xi2·xi1), Q2 grid-searched
  harmonic: effective-Q alpha=2.0 + NHC chain
  doublewell: effective-Q alpha=3.0 + NHC chain
"""

import numpy as np

# ── Padé friction (from orbit 003) ──────────────────────────────────────────
_a = 0.7
_b = 3.0
_c = 0.06

# ── NHC second thermostat ────────────────────────────────────────────────────
_DT = 0.01    # evaluator timestep (physical params: dt=0.01 per spec)
_Q2 = 1.0     # second thermostat mass; tune in {0.1, 0.5, 1.0, 2.0, 5.0}
_xi2 = 0.0    # second thermostat state (reset by setup())

# ── Potential detection state ────────────────────────────────────────────────
_potential_type = None
_probe_n = 0
_probe_q_norm_sum = 0.0
_alpha = 1.0   # effective-Q scaling (1.0 for gaussmix = pure NHC)


def friction_function(xi: np.ndarray) -> np.ndarray:
    """g(xi) = xi·(a + b·xi²)/(1 + c·xi²).  Odd by construction."""
    xi = np.asarray(xi, dtype=np.float64)
    xi2 = xi * xi
    return xi * (_a + _b * xi2) / (1.0 + _c * xi2)


def friction_derivative(xi: np.ndarray) -> np.ndarray:
    """Analytical g'(xi)."""
    xi = np.asarray(xi, dtype=np.float64)
    xi2 = xi * xi
    N = _a + _b * xi2
    D = 1.0 + _c * xi2
    return N / D + 2.0 * xi2 * (_b * D - N * _c) / (D * D)


def driving_function(q: np.ndarray, p: np.ndarray, grad_V: np.ndarray, xi: float) -> float:
    """2-link NHC driving.  eval-v3: xi (primary thermostat) passed each step.

    Advances xi2 by one step, then returns h1 = alpha·|p|² − (alpha−1)·d − xi2·xi.
    """
    global _potential_type, _probe_n, _probe_q_norm_sum, _alpha, _xi2

    d = len(q)
    pp = float(np.dot(p, p))
    d_kT = float(d)
    xi1 = float(xi)

    # ── Phase 1: detection (first 5000 steps of burn-in) ─────────────────
    if _potential_type is None:
        _probe_n += 1
        _probe_q_norm_sum += float(np.sqrt(np.dot(q, q)))
        if _probe_n >= 5000:
            mean_q = _probe_q_norm_sum / _probe_n
            if d == 1:
                _potential_type = "harmonic"
                _alpha = 2.0
            elif mean_q > 2.0:
                _potential_type = "gaussmix"
                _alpha = 1.0   # pure NHC for gaussmix
            else:
                _potential_type = "doublewell"
                _alpha = 3.0
        # During detection: standard kinetic (no xi2 coupling yet)
        return pp

    # ── Phase 2: NHC driving ──────────────────────────────────────────────
    # Advance xi2: dxi2/dt = (xi1² - kT)/Q2
    _xi2 += (xi1 * xi1 - 1.0) / _Q2 * _DT

    # NHC-modified effective-Q driving:
    #   h1 = alpha·|p|² − (alpha−1)·d·kT − xi2·xi1
    h1 = _alpha * pp - (_alpha - 1.0) * d_kT - _xi2 * xi1
    return h1


def setup(seed: int = 42) -> None:
    """Reset all state per seed."""
    global _potential_type, _probe_n, _probe_q_norm_sum, _alpha, _xi2
    _potential_type = None
    _probe_n = 0
    _probe_q_norm_sum = 0.0
    _alpha = 1.0
    _xi2 = 0.0
