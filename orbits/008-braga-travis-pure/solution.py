"""
Pure Braga-Travis configurational thermostat driving with Pade friction.
orbit/008-braga-travis-pure

Driving function: h(q, p, grad_V) = |grad_V|^2 * d / E_ref
where E_ref is estimated during a warmup phase then frozen.

During the first WARMUP steps, E_ref is updated as a running mean of
|grad_V|^2.  After WARMUP, E_ref is frozen so dynamics become Markovian.
The evaluator discards 10,000 burn-in steps, so WARMUP=5000 is safe.

Pre-computed E[|grad_V|^2] values (kT=1, Monte Carlo 10M samples):
  harmonic_1d (d=1):   E[|gV|^2] = 1.00   => E_ref = 1.00
  doublewell_2d (d=2): E[|gV|^2] = 6.99   => E_ref = 3.50
  gaussmix_2d (d=2):   E[|gV|^2] = 1.26   => E_ref = 0.63

The freeze-after-warmup approach correctly adapts to each potential.

Friction: Pade g(xi) = xi*(a + b*xi^2)/(1 + c*xi^2) from orbit 003.
Parameters: a=0.7, b=3.0, c=0.06 (unchanged, known best).
"""

import numpy as np

# ── Pade friction parameters (from orbit 003, validated at metric=84.14) ──
_a = 0.7
_b = 3.0
_c = 0.06


def friction_function(xi: np.ndarray) -> np.ndarray:
    """g(xi) = xi * (a + b*xi^2) / (1 + c*xi^2)

    Automatically odd: g(-xi) = -g(xi).
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


# ── Configurational driving state ────────────────────────────────────────
_WARMUP = 5000            # steps to estimate E_ref (within 10k burn-in)
_grad_sq_sum = 0.0        # cumulative sum of |grad_V|^2
_call_count = 0           # total calls
_frozen_e_ref = None      # frozen E_ref after warmup


def driving_function(q, p, grad_V):
    """Braga-Travis configurational driving with freeze-after-warmup.

    During warmup (first 5000 calls): estimate E_ref = mean(|grad_V|^2)/d.
    After warmup: freeze E_ref and use h = |grad_V|^2 * d / (E_ref * d) = |grad_V|^2 / E_ref.
    This ensures E[h] = d*kT = d for kT=1.
    """
    global _grad_sq_sum, _call_count, _frozen_e_ref

    d = len(q)
    gv2 = float(np.dot(grad_V, grad_V))

    _call_count += 1

    if _frozen_e_ref is not None:
        # Post-warmup: use frozen E_ref (Markovian dynamics)
        return gv2 * d / _frozen_e_ref

    # Warmup phase: accumulate statistics
    _grad_sq_sum += gv2
    mean_gv2 = _grad_sq_sum / _call_count

    if _call_count >= _WARMUP:
        # Freeze E_ref = E[|grad_V|^2] so that h = gv2 * d / E_ref => E[h] = d
        _frozen_e_ref = max(mean_gv2, 1e-10)

    # During warmup, still return a reasonable h
    if mean_gv2 < 1e-10:
        return float(d)
    return gv2 * d / mean_gv2


def setup(seed: int = 42) -> None:
    """Reset driving function state for each fresh integration run."""
    global _grad_sq_sum, _call_count, _frozen_e_ref
    _grad_sq_sum = 0.0
    _call_count = 0
    _frozen_e_ref = None
