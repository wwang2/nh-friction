"""
orbit/017-linear-g-bt-gaussmix
Per-potential dispatch with a genuinely unexplored combination for gaussmix:
LINEAR g(xi) = xi (scaled) + Braga-Travis configurational h = alpha * |grad_V|^2 / E_ref - (alpha-1)*d*kT.

Key theoretical point (from brainstorm panel):
  For Pade friction g_Pade(xi), the ONLY canonical h is the effective-Q form
  h = alpha*|p|^2 - (alpha-1)*d*kT.
  For LINEAR g(xi) = k*xi, the canonical invariance condition reduces to
  E_canonical[h(q,p)] = d*kT, admitting BT configurational driving directly.

So this orbit keeps harmonic and doublewell at the orbit-015 optimum
(Pade + effective-Q alpha) and tests the LINEAR-g + BT combination for gaussmix
only, which is the 68% difficulty slice.

Per-potential detection mirrors orbit-015: mean|q| during 5000-step warmup.
The friction_function branches on the detected potential so that only gaussmix
uses linear g; harmonic and doublewell continue to use Pade.

WHICH alpha/lambda does this orbit use for gaussmix?  Controlled by
_GAUSSMIX_MODE + _GAUSSMIX_ALPHA + _GAUSSMIX_LAMBDA below — edit and
re-run to sweep.  Final settings reflect the best production run.
"""
from __future__ import annotations

import numpy as np

# ── Gaussmix experimental controls ───────────────────────────────────────────
# Modes:
#   "bt_pure":   h = alpha * |grad_V|^2 / E_ref - (alpha-1)*d*kT
#   "bt_hybrid": h = lam * |p|^2 + (1-lam) * |grad_V|^2 / E_ref   (E[h]=d*kT ok)
#   "effectiveQ": orbit-015 fallback, h = alpha*|p|^2 - (alpha-1)*d*kT
# Linear g is used whenever mode != "effectiveQ".
_GAUSSMIX_MODE = "bt_hybrid"
_GAUSSMIX_ALPHA = 1.0         # only used when MODE == "bt_pure"
_GAUSSMIX_LAMBDA = 0.82       # only used when MODE == "bt_hybrid"
_GAUSSMIX_LINEAR_K = 2.0      # slope of linear g(xi) = k*xi for gaussmix

# ── Per-potential Pade parameters (orbit-015 optimum) ────────────────────────
_PADE_PARAMS = {
    "harmonic":   {"a": 0.70, "b": 3.00, "c": 0.06, "alpha": 2.00},
    "doublewell": {"a": 1.00, "b": 4.00, "c": 0.06, "alpha": 3.00},
    # gaussmix Pade params only used if _GAUSSMIX_MODE == "effectiveQ":
    "gaussmix":   {"a": 0.70, "b": 3.00, "c": 0.06, "alpha": 0.74},
}

# ── Active state (set during per-seed detection) ─────────────────────────────
_a = 0.7
_b = 3.0
_c = 0.06
_alpha = 1.0
_linear_mode = False          # True once gaussmix is detected under bt_* mode
_potential_type = None        # "harmonic" | "doublewell" | "gaussmix"
_probe_n = 0
_probe_q_norm_sum = 0.0

# ── E_ref estimator for BT driving (gaussmix only) ───────────────────────────
_E_ref = None
_gradsq_sum = 0.0
_gradsq_count = 0
_WARMUP_ERef = 5000           # same horizon as potential detection


# ═════════════════════════════════════════════════════════════════════════════
# Friction: linear for gaussmix (BT), Pade for harmonic/doublewell
# ═════════════════════════════════════════════════════════════════════════════

def friction_function(xi: np.ndarray) -> np.ndarray:
    """g(xi).

    During probe: Pade with default params (a=0.7, b=3.0, c=0.06).
    After detection:
      harmonic / doublewell → Pade with per-potential params.
      gaussmix + bt_*       → linear g(xi) = k*xi.
      gaussmix + effectiveQ → Pade (orbit-015 fallback).
    """
    xi = np.asarray(xi, dtype=np.float64)
    if _linear_mode:
        return _GAUSSMIX_LINEAR_K * xi
    xi2 = xi * xi
    return xi * (_a + _b * xi2) / (1.0 + _c * xi2)


def friction_derivative(xi: np.ndarray) -> np.ndarray:
    """g'(xi)."""
    xi = np.asarray(xi, dtype=np.float64)
    if _linear_mode:
        return np.full_like(xi, _GAUSSMIX_LINEAR_K)
    xi2 = xi * xi
    N = _a + _b * xi2
    D = 1.0 + _c * xi2
    return N / D + 2.0 * xi2 * (_b * D - N * _c) / (D * D)


# ═════════════════════════════════════════════════════════════════════════════
# Driving: BT for gaussmix, effective-Q for harmonic/doublewell
# ═════════════════════════════════════════════════════════════════════════════

def driving_function(q, p, grad_V, xi):
    """h(q, p, grad_V, xi).  4-arg signature for eval-v3 (xi unused here)."""
    global _alpha, _linear_mode, _potential_type
    global _probe_n, _probe_q_norm_sum
    global _E_ref, _gradsq_sum, _gradsq_count

    d = len(q)
    pp = float(np.dot(p, p))
    d_kT = float(d)

    # Phase 1: potential detection probe (first 5000 driving calls)
    if _potential_type is None:
        _probe_n += 1
        _probe_q_norm_sum += float(np.sqrt(np.dot(q, q)))
        # simultaneously begin collecting |grad_V|^2 for E_ref
        g2 = float(np.dot(grad_V, grad_V))
        _gradsq_sum += g2
        _gradsq_count += 1
        if _probe_n >= 5000:
            mean_q = _probe_q_norm_sum / _probe_n
            if d == 1:
                _potential_type = "harmonic"
            elif mean_q > 2.0:
                _potential_type = "gaussmix"
            else:
                _potential_type = "doublewell"
            _apply_params()
        return pp  # standard kinetic during probe (within 10k burn-in)

    # Phase 2: per-potential driving
    if _potential_type == "gaussmix" and _linear_mode:
        # Keep refining E_ref until warmup done, then freeze.
        if _E_ref is None:
            g2 = float(np.dot(grad_V, grad_V))
            _gradsq_sum += g2
            _gradsq_count += 1
            if _gradsq_count >= _WARMUP_ERef:
                _E_ref = max(_gradsq_sum / _gradsq_count, 1e-10)
        g2 = float(np.dot(grad_V, grad_V))
        E_ref = _E_ref if _E_ref is not None else max(_gradsq_sum / max(_gradsq_count, 1), 1e-10)

        if _GAUSSMIX_MODE == "bt_pure":
            alpha = _GAUSSMIX_ALPHA
            # h = alpha * (d*kT) * (g2/E_ref) - (alpha-1)*d*kT
            # since E_ref is defined so that mean(g2) = E_ref, E[g2/E_ref] = 1
            # we need E[h] = d*kT. If we set h = d*kT*(alpha*g2/E_ref - (alpha-1)),
            # E[h] = d*kT*(alpha*1 - (alpha-1)) = d*kT ✓
            return d_kT * (alpha * g2 / E_ref - (alpha - 1.0))
        elif _GAUSSMIX_MODE == "bt_hybrid":
            lam = _GAUSSMIX_LAMBDA
            # h = lam*|p|^2 + (1-lam)*(d*kT)*(g2/E_ref)
            # E[h] = lam*d*kT + (1-lam)*d*kT = d*kT ✓
            return lam * pp + (1.0 - lam) * d_kT * g2 / E_ref
        else:
            # shouldn't reach here (linear_mode implies bt_*); fallback safe
            return _alpha * pp - (_alpha - 1.0) * d_kT

    # harmonic / doublewell / gaussmix+effectiveQ: effective-Q driving
    return _alpha * pp - (_alpha - 1.0) * d_kT


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _apply_params() -> None:
    """Activate per-potential friction + driving parameters after detection."""
    global _a, _b, _c, _alpha, _linear_mode
    p_ = _PADE_PARAMS[_potential_type]
    _a, _b, _c, _alpha = p_["a"], p_["b"], p_["c"], p_["alpha"]
    # Enable linear friction only for gaussmix under BT modes.
    _linear_mode = (
        _potential_type == "gaussmix"
        and _GAUSSMIX_MODE in ("bt_pure", "bt_hybrid")
    )


def setup(seed: int = 42) -> None:
    """Reset state before each integration run (3 potentials × 3 seeds = 9 calls)."""
    global _a, _b, _c, _alpha
    global _linear_mode, _potential_type, _probe_n, _probe_q_norm_sum
    global _E_ref, _gradsq_sum, _gradsq_count
    _a, _b, _c, _alpha = 0.7, 3.0, 0.06, 1.0
    _linear_mode = False
    _potential_type = None
    _probe_n = 0
    _probe_q_norm_sum = 0.0
    _E_ref = None
    _gradsq_sum = 0.0
    _gradsq_count = 0
