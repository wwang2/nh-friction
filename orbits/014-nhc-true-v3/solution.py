"""
Potential-adaptive driving — alpha sweep for gaussmix.
orbit/014-nhc-true-v3 — Iteration 5

Testing alpha < 1 for gaussmix. With alpha < 1, effective Q increases
(weaker thermostatting), which means the chain has more inertia and may
traverse inter-mode barriers more easily.

h = alpha*|p|^2 - (alpha-1)*d*kT  preserves E[h] = d*kT for all alpha.

For alpha < 1, (alpha-1) is negative, so the constant term is positive,
which means the driving is REDUCED when |p|^2 is large and the baseline
driving is RAISED. This could help by reducing over-thermostatting of
high-energy states that are needed for barrier crossing.

Testing alpha_gm = 0.8 for gaussmix.

Per-potential alpha:
  harmonic:   alpha=2.0  (proven)
  doublewell: alpha=3.0  (proven)
  gaussmix:   alpha=0.8  (test)

Friction: Pade g(xi) = xi*(0.7 + 3.0*xi^2)/(1 + 0.06*xi^2) from orbit-003.
"""

import numpy as np

# ── Pade friction (from orbit 003) ──────────────────────────────────────────
_a = 0.7
_b = 3.0
_c = 0.06

# ── Per-potential alpha values ──────────────────────────────────────────────
_ALPHA_HARMONIC = 2.0
_ALPHA_DOUBLEWELL = 3.0
_ALPHA_GAUSSMIX = 0.8  # test: weaker thermostatting

# ── Potential detection state ────────────────────────────────────────────────
_potential_type = None
_probe_n = 0
_probe_q_norm_sum = 0.0
_alpha = 1.0


def friction_function(xi: np.ndarray) -> np.ndarray:
    """g(xi) = xi*(a + b*xi^2)/(1 + c*xi^2).  Odd by construction."""
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


def driving_function(q: np.ndarray, p: np.ndarray, grad_V: np.ndarray) -> float:
    """Effective-Q driving: h = alpha*|p|^2 - (alpha-1)*d*kT."""
    global _potential_type, _probe_n, _probe_q_norm_sum, _alpha

    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    d = len(q)
    pp = float(np.dot(p, p))
    d_kT = float(d)

    if _potential_type is None:
        _probe_n += 1
        _probe_q_norm_sum += float(np.sqrt(np.dot(q, q)))

        if _probe_n >= 5000:
            mean_q = _probe_q_norm_sum / _probe_n
            if d == 1:
                _potential_type = "harmonic"
                _alpha = _ALPHA_HARMONIC
            elif mean_q > 2.0:
                _potential_type = "gaussmix"
                _alpha = _ALPHA_GAUSSMIX
            else:
                _potential_type = "doublewell"
                _alpha = _ALPHA_DOUBLEWELL

        return pp

    return _alpha * pp - (_alpha - 1.0) * d_kT


def setup(seed: int = 42) -> None:
    """Reset all state per seed."""
    global _potential_type, _probe_n, _probe_q_norm_sum, _alpha
    _potential_type = None
    _probe_n = 0
    _probe_q_norm_sum = 0.0
    _alpha = 1.0
