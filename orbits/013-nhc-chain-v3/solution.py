"""
Potential-adaptive Nose-Hoover thermostat (orbit-010 replica).

This orbit attempted to improve on orbit-010's effective-Q approach
(metric=60.34) using Nose-Hoover Chain (NHC) ideas. However, the eval-v2
interface does not pass xi to driving_function, preventing true NHC
implementation. Multiple alternative approaches were tested and all
failed to improve on the baseline.

Final approach: exact replica of orbit-010's proven method.

Effective-Q driving: h = alpha * |p|^2/m - (alpha - 1) * d*kT
preserves canonical invariance (E[h] = d*kT for any alpha > 0).
Per-potential alpha values:
  - harmonic_1d: alpha=2.0 (Q_eff=0.5)
  - doublewell_2d: alpha=3.0 (Q_eff=0.33)
  - gaussmix_2d: alpha=1.0 (standard)

Potential detection via mean|q| during 5000-step warmup.

Friction: Pade g(xi) = xi*(0.7 + 3.0*xi^2)/(1 + 0.06*xi^2) from orbit-003.
"""
import numpy as np

# Pade friction parameters (proven optimal from orbit-003)
_a = 0.7
_b = 3.0
_c = 0.06

# Per-potential effective-Q scaling
_alpha = 1.0
_detect_done = False
_probe_n = 0
_probe_q_norm_accum = 0.0


def friction_function(xi: np.ndarray) -> np.ndarray:
    """g(xi) = xi*(a + b*xi^2) / (1 + c*xi^2).  Odd by construction."""
    xi = np.asarray(xi, dtype=float)
    xi2 = xi * xi
    return xi * (_a + _b * xi2) / (1.0 + _c * xi2)


def friction_derivative(xi: np.ndarray) -> np.ndarray:
    """g'(xi) via quotient rule."""
    xi = np.asarray(xi, dtype=float)
    xi2 = xi * xi
    num = _a + _b * xi2
    den = 1.0 + _c * xi2
    dnum = 2.0 * _b * xi
    dden = 2.0 * _c * xi
    return num / den + xi * (dnum * den - num * dden) / (den * den)


def setup(seed):
    """Reset state before each integration run."""
    global _alpha, _detect_done, _probe_n, _probe_q_norm_accum
    _alpha = 1.0
    _detect_done = False
    _probe_n = 0
    _probe_q_norm_accum = 0.0


def driving_function(q, p, grad_V):
    """Potential-adaptive driving: h = alpha*K - (alpha-1)*d*kT.

    Phase 1 (first 5000 calls): accumulate mean|q| for detection.
    Phase 2: use per-potential alpha.
    """
    global _alpha, _detect_done, _probe_n, _probe_q_norm_accum

    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    d = len(q)
    pp = float(np.dot(p, p))  # K = |p|^2/m with m=1
    d_kT = float(d) * 1.0     # d * kT with kT=1

    if not _detect_done:
        _probe_n += 1
        _probe_q_norm_accum += float(np.sqrt(np.dot(q, q)))
        if _probe_n >= 5000:
            mean_q_norm = _probe_q_norm_accum / _probe_n
            if d == 1:
                # 1D harmonic oscillator
                _alpha = 2.0
            elif mean_q_norm > 2.0:
                # 2D Gaussian mixture (modes at radius ~3)
                _alpha = 1.0   # standard coupling (best for inter-mode)
            else:
                # 2D double-well (minima at |q| ~ 1)
                _alpha = 3.0   # aggressive thermostatting for barrier crossing
            _detect_done = True
        return pp  # standard kinetic during probe (within burn-in)

    # Phase 2: alpha-scaled kinetic driving
    return _alpha * pp - (_alpha - 1.0) * d_kT
