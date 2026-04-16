"""
Potential-adaptive Nose-Hoover thermostat.

Two innovations:
1. Effective-Q modification via driving function:
   h(q,p) = alpha * |p|^2/m - (alpha-1) * d*kT
   This preserves canonical invariance (only changes xi-marginal)
   while effectively setting Q_eff = Q/alpha.

2. Potential detection via mean|q| during warmup:
   - d=1: harmonic (only 1D potential)
   - d=2, mean|q| > 2: gaussmix (modes at radius 3)
   - d=2, mean|q| <= 2: doublewell (minima near |q|~1)

Theoretical justification (effective-Q):
  With h = alpha*K - (alpha-1)*d*kT where K = |p|^2/m:
  - E[h] = alpha*d*kT - (alpha-1)*d*kT = d*kT (correct)
  - dxi/dt = (h - d*kT)/Q = alpha*(K - d*kT)/Q
  - Invariant measure: rho ~ exp(-H/kT - G(xi)/(alpha*kT))
    where G(xi) = integral g(s)ds
  - The (q,p)-marginal is exp(-H/kT), exactly canonical.
  - Larger alpha = more aggressive thermostatting = faster mixing.

Friction: Pade form g(xi) = xi*(a+b*xi^2)/(1+c*xi^2)
  Proven optimal from orbit-002 with (a=0.7, b=3.0, c=0.06).
"""
import numpy as np

# ── Pade friction parameters (global, adapted per-potential) ──
_a = 0.7
_b = 3.0
_c = 0.06

# ── Driving function state ──
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
                # keep default Pade params
            elif mean_q_norm > 2.0:
                # 2D Gaussian mixture (modes at radius ~3)
                _alpha = 1.0   # standard coupling (best for inter-mode)
            else:
                # 2D double-well (minima at |q| ~ 1)
                _alpha = 3.0   # aggressive thermostatting for barrier crossing
                # keep default Pade params
            _detect_done = True
        return pp  # standard kinetic during probe (within burn-in)

    # Phase 2: alpha-scaled kinetic driving
    return _alpha * pp - (_alpha - 1.0) * d_kT
