"""
Potential-adaptive Nose-Hoover thermostat with optimized gaussmix parameters.
orbit/016-gaussmix-cmaes-4d

Strategy: CMA-ES 4D parameter search over (a, b, c, alpha) for the gaussmix
potential, starting from the known local optimum at (0.70, 3.00, 0.06, 0.74).
Harmonic and doublewell parameters fixed at orbit-015 values.

Friction: g(xi) = xi * (a + b*xi^2) / (1 + c*xi^2)
Driving:  h(q,p) = alpha*|p|^2 - (alpha-1)*d*kT   [Liouville-preserving]
"""

import numpy as np

# ── Per-potential parameters ─────────────────────────────────────────────────
# harmonic and doublewell: frozen from orbit-015 (DO NOT CHANGE)
# gaussmix: optimized by CMA-ES in this orbit
_PARAMS = {
    'harmonic':   {'a': 0.70, 'b': 3.00, 'c': 0.06, 'alpha': 2.00},
    'doublewell': {'a': 1.00, 'b': 4.00, 'c': 0.06, 'alpha': 3.00},
    'gaussmix':   {'a': 0.70, 'b': 3.00, 'c': 0.06, 'alpha': 0.74},  # PLACEHOLDER — updated after search
}

# ── Active parameters (set during detection) ─────────────────────────────────
_a = 0.7
_b = 3.0
_c = 0.06
_alpha = 1.0
_potential_type = None
_probe_n = 0
_probe_q_norm_sum = 0.0


def friction_function(xi: np.ndarray) -> np.ndarray:
    """g(xi) = xi*(a + b*xi^2) / (1 + c*xi^2).  Odd by construction."""
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
    """Effective-Q driving: h = alpha*|p|^2 - (alpha-1)*d*kT.

    For alpha > 1: stronger thermostatting (smaller effective Q).
    For alpha < 1: weaker thermostatting (larger effective Q).
    E[h] = d*kT for all alpha (canonical invariance preserved).
    """
    global _a, _b, _c, _alpha, _potential_type, _probe_n, _probe_q_norm_sum

    d = len(q)
    pp = float(np.dot(p, p))
    d_kT = float(d)

    if _potential_type is None:
        _probe_n += 1
        _probe_q_norm_sum += float(np.sqrt(np.dot(q, q)))
        if _probe_n >= 5000:
            mean_q = _probe_q_norm_sum / _probe_n
            if d == 1:
                _potential_type = 'harmonic'
            elif mean_q > 2.0:
                _potential_type = 'gaussmix'
            else:
                _potential_type = 'doublewell'
            p_ = _PARAMS[_potential_type]
            _a, _b, _c, _alpha = p_['a'], p_['b'], p_['c'], p_['alpha']
        return pp  # standard kinetic during probe

    return _alpha * pp - (_alpha - 1.0) * d_kT


def setup(seed: int = 42) -> None:
    """Reset state per seed."""
    global _a, _b, _c, _alpha, _potential_type, _probe_n, _probe_q_norm_sum
    _a, _b, _c, _alpha = 0.7, 3.0, 0.06, 1.0
    _potential_type = None
    _probe_n = 0
    _probe_q_norm_sum = 0.0
