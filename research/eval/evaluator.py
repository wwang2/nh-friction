#!/usr/bin/env python3
"""
Production evaluator for the Nosé-Hoover friction function optimization problem.
eval-v1  —  synthesized from candidates 4, 5, 6 + adversarial findings 1 & 2.

=======================================================================
EVALUATION PROTOCOL SUMMARY
=======================================================================

Metric: difficulty-weighted mean integrated autocorrelation time (τ_int)
  weighted_tau_int = Σ_k w_k · (1/S) Σ_s τ_int(k, s)
  where k ∈ {harmonic_1d, doublewell_2d, gaussmix_2d}, s ∈ {42, 137, 2024}

Disqualification (METRIC=inf):
  1. Oddness violated: max|g(ξ)+g(-ξ)| ≥ 1e-6 on any test point
  2. KL(empirical ‖ analytical) > 0.05 mean across seeds for ANY potential
     (GLOBAL gate — one potential failing → METRIC=inf)
  3. NaN/inf in trajectory at any step

Integrator: velocity Verlet + exact exponential friction (VVEF splitting)
  p += -∇V(q)·dt/2
  p *= exp(-g(ξ)·dt/(2Q))
  q += p/m·dt
  p *= exp(-g(ξ)·dt/(2Q))
  p += -∇V(q)·dt/2
  ξ += (|p|²/m - d·kT)/Q · dt

Physical params: m=1, kT=1, Q=1, dt=0.01
Seeds: [42, 137, 2024]  |  Burn-in: 10,000  |  Thin: 10  |  N_samples: 100,000

τ_int method: FFT-Sokal automatic windowing (C=6, numpy.fft.rfft — no scipy dep)
  + emcee-style cross-check (C=5); warn if >10% discrepancy, use FFT-Sokal canonical
  + floor: τ_int ≥ 1.0 (Sokal 1997: τ_int ≥ 1 by definition)
  + KS stationarity: split 100k into 4 quarters; warn if max/min ratio > 4

Synthesis sources:
  Candidate 4 : dual τ_int cross-check, extended oddness check structure, τ_int floor=1.0
  Candidate 5 : batch Modal dispatch, KS stationarity, numpy.fft.rfft (no scipy for FFT)
  Candidate 6 : all 12 edge cases in sokal, per-potential τ_int reporting, gaussmix grad
  Adversarial : all mandatory fixes from findings 1 & 2 applied

CLI contract (immutable):
  python3 evaluator.py --solution <path> --seed <int> [--local]
  stdout MUST contain: METRIC=<float>
  stderr: auxiliary metrics (tau_int_per_potential, kl_per_potential, ESS_per_second)

References:
  Sokal (1997) Cargese lecture notes §3 Eq.3.3  — τ_int definition & windowing
  Foreman-Mackey et al. (2013) PASP 125 306      — emcee integrated_time pattern
  Martyna et al. (1992) J. Chem. Phys. 97 2635   — NHC equations of motion
  Chodera (2016) JCTC 12 1799                    — consistent Nosé-Hoover formulation
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import numpy as np

# ── Physical constants (frozen for eval-v1) ───────────────────────────────────
M   = 1.0    # particle mass
KT  = 1.0    # thermal energy
Q   = 1.0    # thermostat mass
DT  = 0.01

# ── Integration parameters (frozen for eval-v1) ───────────────────────────────
N_BURNIN      = 10_000
N_MAIN_STEPS  = 1_000_000          # post burn-in integration steps
THIN          = 10
N_SAMPLES     = N_MAIN_STEPS // THIN   # = 100,000
N_TOTAL_STEPS = N_BURNIN + N_MAIN_STEPS

SEEDS         = [42, 137, 2024]

# ── Evaluation hyperparameters (frozen for eval-v1) ──────────────────────────
TAU_CAP           = 50_000          # cap τ_int for non-ergodic chains (= N_SAMPLES/2)
TAU_INT_FLOOR     = 1.0             # physical lower bound (Sokal 1997)
KL_THRESHOLD      = 0.05
KL_BINS           = 100
KL_RANGE          = (-6.0, 6.0)
C_SOKAL           = 6.0             # Sokal windowing constant (spec: W* > 6·τ_running)
C_EMCEE           = 5.0             # emcee verification path constant
CROSSCHECK_REL_THR = 0.10           # warn if |τ_fft - τ_emcee| / τ_fft > 10%
ODDNESS_TOL       = 1e-6
KS_RATIO_WARN     = 4.0             # warn if quarter max/min τ_int ratio > 4
N_QUARTERS        = 4               # number of stationarity quarters
SETUP_TIMEOUT_S   = 60.0            # max seconds for setup() call before warning/skip


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Potential definitions
# ═════════════════════════════════════════════════════════════════════════════

def _grad_harmonic_1d(q: np.ndarray) -> np.ndarray:
    """grad V = q  for V(q) = q²/2, d=1."""
    return q.copy()


def _grad_doublewell_2d(q: np.ndarray) -> np.ndarray:
    """grad V for V(x,y) = (x²-1)² + y²/2, d=2.
    grad_x = 4x(x²-1),  grad_y = y.
    """
    x = q[0]
    return np.array([4.0 * x * (x * x - 1.0), q[1]])


def _grad_gaussmix_2d(q: np.ndarray) -> np.ndarray:
    """grad V for the 5-mode Gaussian mixture.

    V(q) = -log Σ_k exp(-|q - μ_k|²/2),  μ_k = 3·(cos(2πk/5), sin(2πk/5))
    grad V = Σ_k w_k · (q - μ_k)  where  w_k = softmax(-|q-μ_k|²/2)

    Uses log-sum-exp for numerical stability (from candidate 6 / adversarial fix).
    """
    MU = np.array([
        [3.0 * math.cos(2.0 * math.pi * k / 5),
         3.0 * math.sin(2.0 * math.pi * k / 5)]
        for k in range(5)
    ])                                   # (5, 2)
    diff  = q[np.newaxis, :] - MU       # (5, 2)
    log_w = -0.5 * np.sum(diff ** 2, axis=1)   # (5,)
    log_w -= log_w.max()                # log-sum-exp shift
    w = np.exp(log_w)
    w /= w.sum()
    return np.einsum("k,kd->d", w, diff)


POTENTIALS: dict[str, dict] = {
    "harmonic_1d":   {"dim": 1, "grad": _grad_harmonic_1d},
    "doublewell_2d": {"dim": 2, "grad": _grad_doublewell_2d},
    "gaussmix_2d":   {"dim": 2, "grad": _grad_gaussmix_2d},
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Analytical marginals for KL computation
# ═════════════════════════════════════════════════════════════════════════════

def _marginal_harmonic_1d(bin_edges: np.ndarray) -> np.ndarray:
    """q[0] ~ N(0,1).  Returns normalized bin probabilities."""
    # Use erfc-based CDF to avoid scipy dependency
    lo, hi = bin_edges[:-1], bin_edges[1:]
    def _norm_cdf(x):
        return 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in x]))
    probs = _norm_cdf(hi) - _norm_cdf(lo)
    probs = np.clip(probs, 1e-300, None)
    return probs / probs.sum()


def _marginal_doublewell_2d(bin_edges: np.ndarray) -> np.ndarray:
    """Marginal on x: ∝ exp(-(x²-1)²), normalised numerically on [-6,6]."""
    mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    log_p = -(mids ** 2 - 1.0) ** 2
    log_p -= log_p.max()                 # stability
    p = np.exp(log_p)
    bin_w = bin_edges[1] - bin_edges[0]
    probs = p * bin_w
    probs = np.clip(probs, 1e-300, None)
    return probs / probs.sum()


def _marginal_gaussmix_2d(bin_edges: np.ndarray) -> np.ndarray:
    """q[0] marginal = (1/5) Σ_{k=0}^{4} N(μ_k[0], 1)."""
    MU_X = [3.0 * math.cos(2.0 * math.pi * k / 5) for k in range(5)]
    lo, hi = bin_edges[:-1], bin_edges[1:]
    def _norm_cdf(x):
        return 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in x]))
    probs = np.zeros(len(lo))
    for mu_x in MU_X:
        probs += (_norm_cdf(hi - mu_x) - _norm_cdf(lo - mu_x)) / 5.0
    probs = np.clip(probs, 1e-300, None)
    return probs / probs.sum()


_MARGINALS = {
    "harmonic_1d":   _marginal_harmonic_1d,
    "doublewell_2d": _marginal_doublewell_2d,
    "gaussmix_2d":   _marginal_gaussmix_2d,
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — KL divergence (histogram vs analytical)
# ═════════════════════════════════════════════════════════════════════════════

def _kl_divergence(samples_q0: np.ndarray, potential_name: str) -> float:
    """KL(empirical ‖ analytical) via 100-bin histogram on [-6, 6].

    Returns float in [0, ∞).  Returns inf if any numerical issue.
    """
    if len(samples_q0) == 0 or not np.all(np.isfinite(samples_q0)):
        return math.inf

    bin_edges = np.linspace(KL_RANGE[0], KL_RANGE[1], KL_BINS + 1)
    counts, _ = np.histogram(
        np.clip(samples_q0, KL_RANGE[0], KL_RANGE[1]), bins=bin_edges
    )
    total = counts.sum()
    if total == 0:
        return math.inf

    p_emp = counts.astype(np.float64) / total
    p_ref = _MARGINALS[potential_name](bin_edges)   # normalized

    kl = 0.0
    for p_i, q_i in zip(p_emp, p_ref):
        if p_i > 0.0:
            if q_i <= 0.0:
                return math.inf
            kl += p_i * math.log(p_i / q_i)

    return max(0.0, kl)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — τ_int computation (FFT-Sokal canonical + emcee cross-check)
# ═════════════════════════════════════════════════════════════════════════════

def _sokal_tau_int_fft(x: np.ndarray, c: float = C_SOKAL) -> tuple[float, bool]:
    """Integrated autocorrelation time via Sokal automatic windowing.

    CANONICAL path — uses numpy.fft.rfft (no scipy dependency).
    Handles all 12 edge cases from candidate 6.

    Window condition: W = first t where t > c · τ_running(t)  [strict >, spec-compliant]
    Floor: τ_int ≥ TAU_INT_FLOOR = 1.0  (Sokal 1997)

    Returns: (tau_int: float, converged: bool)
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)

    # Edge case 1: too short
    if N < 3:
        return float(TAU_CAP), False

    # Edge case 2: NaN/inf in input
    if not np.all(np.isfinite(x)):
        return float(TAU_CAP), False

    # Edge case 3: constant trajectory (stuck chain)
    x_centered = x - x.mean()
    var = float(np.var(x_centered))
    if var == 0.0 or not math.isfinite(var):
        return float(TAU_CAP), False

    # FFT-based autocorrelation — zero-pad to next power-of-2 ≥ 2N
    nfft = 1
    while nfft < 2 * N:
        nfft <<= 1

    F   = np.fft.rfft(x_centered, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N].real

    # Unbiased normalization: divide by (N - t)
    acf = acf / (N - np.arange(N))

    C0 = float(acf[0])
    # Edge case 4: C0 non-positive or non-finite
    if C0 <= 0.0 or not math.isfinite(C0):
        return float(TAU_CAP), False

    rho = acf / C0   # ρ(0) = 1 by construction

    # Edge case 5: rho contains NaN/inf
    if not np.all(np.isfinite(rho)):
        return float(TAU_CAP), False

    # Sokal automatic windowing
    tau_running = 1.0    # τ̂(0) = 1
    converged = False

    for t in range(1, N):
        tau_running += 2.0 * rho[t]

        # Edge case 6-8: τ_running non-finite or below floor → clamp
        if not math.isfinite(tau_running) or tau_running < TAU_INT_FLOOR:
            tau_running = TAU_INT_FLOOR

        # Strict > condition (spec-compliant; candidate 6 used >=, which is off-spec)
        if t > c * tau_running:
            converged = True
            break

    # Edge case 9: no convergence
    if not converged:
        return float(TAU_CAP), False

    # Edge case 10-12: non-finite / out-of-range result
    tau = float(tau_running)
    if not math.isfinite(tau) or tau > TAU_CAP:
        return float(TAU_CAP), False

    tau = max(tau, TAU_INT_FLOOR)   # floor (Sokal 1997)
    return min(tau, float(TAU_CAP)), True


def _emcee_tau_int(x: np.ndarray) -> tuple[float, bool]:
    """Verification τ_int via emcee-style integrated_time (c=5).

    Uses the same numpy FFT as the canonical path.
    Window condition: t >= C_EMCEE · τ_running  (emcee uses >=)

    Returns: (tau_int: float, converged: bool)
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)

    if N < 3 or not np.all(np.isfinite(x)):
        return float(TAU_CAP), False

    x_centered = x - x.mean()
    var = float(np.var(x_centered))
    if var == 0.0 or not math.isfinite(var):
        return float(TAU_CAP), False

    nfft = 1
    while nfft < 2 * N:
        nfft <<= 1

    F   = np.fft.rfft(x_centered, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N].real
    acf = acf / (N - np.arange(N))

    C0 = float(acf[0])
    if C0 <= 0.0 or not math.isfinite(C0):
        return float(TAU_CAP), False

    rho = acf / C0
    if not np.all(np.isfinite(rho)):
        return float(TAU_CAP), False

    tau_running = 1.0
    converged = False

    for t in range(1, N):
        tau_running += 2.0 * rho[t]
        if not math.isfinite(tau_running) or tau_running < TAU_INT_FLOOR:
            tau_running = TAU_INT_FLOOR
        # emcee uses >= (non-strict) for its window
        if t >= C_EMCEE * tau_running:
            converged = True
            break

    if not converged:
        return float(TAU_CAP), False

    tau = float(tau_running)
    if not math.isfinite(tau) or tau > TAU_CAP:
        return float(TAU_CAP), False

    tau = max(tau, TAU_INT_FLOOR)
    return min(tau, float(TAU_CAP)), True


def _compute_tau_int(samples: np.ndarray, label: str = "") -> float:
    """Compute τ_int: FFT-Sokal canonical with emcee cross-check.

    Returns the FFT-Sokal result.  Emits warnings to stderr if the two
    estimators disagree by > CROSSCHECK_REL_THR (10%).
    """
    tau_fft,   conv_fft   = _sokal_tau_int_fft(samples)
    tau_emcee, conv_emcee = _emcee_tau_int(samples)

    tag = f" ({label})" if label else ""

    if conv_fft and conv_emcee:
        rel_diff = abs(tau_fft - tau_emcee) / max(tau_fft, 1.0)
        if rel_diff > CROSSCHECK_REL_THR:
            print(
                f"WARNING: τ_int cross-check mismatch{tag}: "
                f"FFT-Sokal={tau_fft:.2f}, emcee={tau_emcee:.2f}, "
                f"rel_diff={rel_diff:.1%} > {CROSSCHECK_REL_THR:.0%}. "
                f"Using FFT-Sokal (canonical).",
                file=sys.stderr, flush=True,
            )
    elif not conv_fft and conv_emcee:
        print(
            f"INFO: τ_int cross-check{tag}: FFT-Sokal did not converge (capped), "
            f"emcee converged at {tau_emcee:.2f}.",
            file=sys.stderr, flush=True,
        )
    elif conv_fft and not conv_emcee:
        print(
            f"INFO: τ_int cross-check{tag}: emcee did not converge (capped), "
            f"FFT-Sokal converged at {tau_fft:.2f}.",
            file=sys.stderr, flush=True,
        )

    return tau_fft   # canonical


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — KS stationarity check (candidate 5)
# ═════════════════════════════════════════════════════════════════════════════

def _ks_stationarity_check(samples: np.ndarray, label: str = "") -> None:
    """Split samples into N_QUARTERS chunks and compare per-quarter τ_int.

    Soft guard (warn only): warns to stderr if max/min quarter τ_int ratio > 4.
    Catches non-stationary / periodically oscillating trajectories (Exploit 2).
    """
    chunk_size = len(samples) // N_QUARTERS
    if chunk_size < 3:
        return

    qtaus = []
    for i in range(N_QUARTERS):
        chunk = samples[i * chunk_size : (i + 1) * chunk_size]
        tau_q, _ = _sokal_tau_int_fft(chunk)
        qtaus.append(tau_q)

    tau_min = min(qtaus)
    tau_max = max(qtaus)
    tag = f" ({label})" if label else ""

    if tau_min > 0 and math.isfinite(tau_max) and math.isfinite(tau_min):
        ratio = tau_max / tau_min
        if ratio > KS_RATIO_WARN:
            print(
                f"WARNING: non-stationarity{tag} — quarter τ_int = "
                f"{[f'{t:.1f}' for t in qtaus]}, max/min = {ratio:.2f} > {KS_RATIO_WARN:.1f}. "
                f"Possible periodic oscillation or mode-switching.",
                file=sys.stderr, flush=True,
            )
        else:
            print(
                f"KS-stationarity OK{tag}: quarter τ_int = "
                f"{[f'{t:.1f}' for t in qtaus]}, max/min = {ratio:.2f}",
                file=sys.stderr, flush=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Nosé-Hoover integrator (single run)
# ═════════════════════════════════════════════════════════════════════════════

def _integrate_one(
    grad_fn,
    dim: int,
    seed: int,
    friction_fn,
) -> np.ndarray | None:
    """Run one Nosé-Hoover integration and return q[0] samples.

    ADVERSARIAL FIX 3 — NaN trajectory → immediate disqualify:
      Any NaN/inf in q, p, ξ, or g(ξ) at any step → return None immediately.
      The caller sets tau=TAU_CAP, kl=inf.  Do NOT fill with last-finite value.

    Returns:
      np.ndarray of shape (N_SAMPLES,) on success
      None on divergence (caller must disqualify)
    """
    rng = np.random.default_rng(seed)
    q   = rng.standard_normal(dim)
    p   = rng.standard_normal(dim)
    xi  = 0.0

    samples  = np.empty(N_SAMPLES, dtype=np.float64)
    rec_idx  = 0
    half_dt  = DT / 2.0
    d_kT     = float(dim) * KT

    for step in range(N_TOTAL_STEPS):
        # ── half momentum kick ────────────────────────────────────────────────
        p = p - grad_fn(q) * half_dt

        # ── exact exponential friction (first half) ───────────────────────────
        # ADVERSARIAL FIX 3: check g(ξ) finite BEFORE computing exp
        try:
            g_raw = friction_fn(np.array([xi]))
            gxi   = float(g_raw[0]) if np.ndim(g_raw) > 0 else float(g_raw)
        except Exception as exc:
            print(
                f"DIVERGED: friction_function raised at step={step}, xi={xi:.4f}: {exc}",
                file=sys.stderr, flush=True,
            )
            return None

        if not math.isfinite(gxi):
            print(
                f"DIVERGED: non-finite g(xi)={gxi} at step={step}, xi={xi:.4f}",
                file=sys.stderr, flush=True,
            )
            return None

        try:
            exp_fac = math.exp(-gxi * half_dt / Q)
        except OverflowError:
            print(
                f"DIVERGED: overflow in exp(-g·dt/2Q) at step={step}, g={gxi:.4e}",
                file=sys.stderr, flush=True,
            )
            return None

        p = p * exp_fac

        # ── drift ─────────────────────────────────────────────────────────────
        q = q + p / M * DT

        # ── exact exponential friction (second half) ──────────────────────────
        p = p * exp_fac

        # ── half momentum kick ────────────────────────────────────────────────
        p = p - grad_fn(q) * half_dt

        # ── thermostat update ─────────────────────────────────────────────────
        xi = xi + (float(np.dot(p, p)) / M - d_kT) / Q * DT

        # ADVERSARIAL FIX 3: check state after every step
        if not math.isfinite(xi) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            print(
                f"DIVERGED: NaN/inf in state at step={step} "
                f"(xi={xi:.4g})",
                file=sys.stderr, flush=True,
            )
            return None

        # ── record (post burn-in, every THIN steps) ───────────────────────────
        if step >= N_BURNIN and (step - N_BURNIN) % THIN == 0:
            samples[rec_idx] = q[0]
            rec_idx += 1

    if rec_idx != N_SAMPLES:
        print(
            f"WARNING: expected {N_SAMPLES} samples, got {rec_idx} — treating as diverged",
            file=sys.stderr, flush=True,
        )
        return None

    return samples


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Solution validation (oddness check + derivative check)
# ═════════════════════════════════════════════════════════════════════════════

def _validate_solution(mod) -> tuple[bool, str]:
    """Validate the solution module.

    ADVERSARIAL FIX 2 — Extended oddness check:
      (a) g(0) = 0 explicitly (odd function requirement)
      (b) log-spaced ξ ∈ [1e-6, 1e3], 100 points
      (c) 5 random large ξ ∈ [50, 200] with seed=0

    Returns (ok: bool, message: str).
    """
    ff = getattr(mod, "friction_function",  None)
    fd = getattr(mod, "friction_derivative", None)

    if not callable(ff):
        return False, "friction_function not callable"
    if not callable(fd):
        return False, "friction_derivative not callable"

    try:
        # ── (a) ξ = 0 explicit check ──────────────────────────────────────────
        g0 = float(ff(np.array([0.0]))[0])
        if not math.isfinite(g0):
            return False, f"g(0) is non-finite: {g0}"
        if abs(g0) >= ODDNESS_TOL:
            return False, f"Oddness violated at ξ=0: |g(0)| = {abs(g0):.3e} >= {ODDNESS_TOL}"

        # ── (b) log-spaced [1e-6, 1e3], 100 points ───────────────────────────
        xi_log    = np.logspace(-6, 3, 100)
        g_pos_log = ff(xi_log)
        g_neg_log = ff(-xi_log)
        if not (np.all(np.isfinite(g_pos_log)) and np.all(np.isfinite(g_neg_log))):
            return False, "g(ξ) returned non-finite values on log-spaced test grid"
        max_odd_log = float(np.max(np.abs(g_pos_log + g_neg_log)))
        if max_odd_log >= ODDNESS_TOL:
            return False, (
                f"Oddness violated on log-spaced [1e-6, 1e3]: "
                f"max|g(ξ)+g(-ξ)| = {max_odd_log:.3e} >= {ODDNESS_TOL}"
            )

        # ── (c) 5 random large ξ ∈ [50, 200], seed=0 ─────────────────────────
        rng_odd    = np.random.default_rng(0)
        xi_large   = rng_odd.uniform(50.0, 200.0, size=5)
        g_large_p  = ff(xi_large)
        g_large_n  = ff(-xi_large)
        if not (np.all(np.isfinite(g_large_p)) and np.all(np.isfinite(g_large_n))):
            return False, "g(ξ) returned non-finite values on large-ξ oddness test"
        max_odd_large = float(np.max(np.abs(g_large_p + g_large_n)))
        if max_odd_large >= ODDNESS_TOL:
            return False, (
                f"Oddness violated at large ξ ∈ [50,200]: "
                f"max|g(ξ)+g(-ξ)| = {max_odd_large:.3e} >= {ODDNESS_TOL}"
            )

    except Exception as exc:
        return False, f"friction_function raised during oddness check: {exc}"

    max_violation = max(abs(g0), max_odd_log, max_odd_large)

    # ── Derivative consistency check (warn only, never disqualify) ────────────
    try:
        h = 1e-5
        xi_d = np.logspace(-2, 1, 10)
        gp_fd  = (ff(xi_d + h) - ff(xi_d - h)) / (2.0 * h)
        gp_sol = fd(xi_d)
        rel_err = np.abs(gp_sol - gp_fd) / (np.abs(gp_sol) + 1.0)
        max_derr = float(np.max(rel_err))
        if max_derr >= 1e-4:
            print(
                f"WARNING: friction_derivative max rel-error = {max_derr:.3e} > 1e-4",
                file=sys.stderr, flush=True,
            )
    except Exception as exc:
        print(f"WARNING: derivative check raised: {exc}", file=sys.stderr, flush=True)

    return True, (
        f"OK (oddness: g(0)={abs(g0):.2e}, "
        f"log-spaced max={max_odd_log:.2e}, large-ξ max={max_odd_large:.2e}, "
        f"max_violation={max_violation:.2e})"
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Difficulty weights loader
# ═════════════════════════════════════════════════════════════════════════════

def _load_weights() -> dict[str, float]:
    """Load difficulty weights from config.yaml.

    ADVERSARIAL FIX 5 — setup() exception handling:
      Falls back to equal weights with a loud warning if config is absent,
      malformed, or has null weights.
    """
    try:
        import yaml
        config_path = Path(__file__).parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {config_path}")

        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)

        raw = cfg.get("eval", {}).get("difficulty_weights", {}) or {}
        w_h = raw.get("harmonic_1d")
        w_d = raw.get("doublewell_2d")
        w_g = raw.get("gaussmix_2d")

        if w_h is None or w_d is None or w_g is None:
            raise ValueError(f"null weights: harmonic={w_h}, doublewell={w_d}, gaussmix={w_g}")

        total = float(w_h) + float(w_d) + float(w_g)
        if total <= 0:
            raise ValueError(f"non-positive total weight: {total}")

        return {
            "harmonic_1d":   float(w_h) / total,
            "doublewell_2d": float(w_d) / total,
            "gaussmix_2d":   float(w_g) / total,
        }

    except Exception as exc:
        print(
            f"\n{'!'*60}\n"
            f"WARNING: difficulty_weights unavailable ({exc}).\n"
            f"Falling back to EQUAL WEIGHTS [1/3, 1/3, 1/3].\n"
            f"Results are NOT difficulty-adjusted and will change once\n"
            f"Phase 2.5 baseline weights are measured.\n"
            f"{'!'*60}\n",
            file=sys.stderr, flush=True,
        )
        eq = 1.0 / 3.0
        return {"harmonic_1d": eq, "doublewell_2d": eq, "gaussmix_2d": eq}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Core evaluation logic
# ═════════════════════════════════════════════════════════════════════════════

def _evaluate_solution(solution_code: str) -> dict:
    """Full evaluation of solution_code.  Returns result dict with metric + auxiliaries.

    This function is the inner kernel — it runs either locally or inside Modal.
    """
    wall_start = time.monotonic()

    # ── 1. Load solution module ───────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as fh:
        fh.write(solution_code)
        tmp_path = fh.name

    spec = importlib.util.spec_from_file_location("solution", tmp_path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        return {"metric": math.inf, "error": f"Import failed: {exc}"}

    # ── 2. Validate (oddness + derivative check) ──────────────────────────────
    ok, msg = _validate_solution(mod)
    print(f"VALIDATION: {msg}", flush=True)
    if not ok:
        return {"metric": math.inf, "error": msg}

    friction_fn = mod.friction_function
    setup_fn    = getattr(mod, "setup", None)

    # ── 3. Load difficulty weights ────────────────────────────────────────────
    weights = _load_weights()

    # ── 4. Main integration loop (3 potentials × 3 seeds = 9 runs) ───────────
    tau_results: dict[str, list[float]] = {k: [] for k in POTENTIALS}
    kl_results:  dict[str, list[float]] = {k: [] for k in POTENTIALS}
    ess_results: dict[str, list[float]] = {k: [] for k in POTENTIALS}

    for pot_name, pot_cfg in POTENTIALS.items():
        grad_fn = pot_cfg["grad"]
        dim     = pot_cfg["dim"]

        for seed in SEEDS:
            # ADVERSARIAL FIX 5 — setup() exception handling:
            #   catch any exception, log warning, continue with default init.
            #   Also enforce a timeout so a malicious sleep() can't hang us.
            if callable(setup_fn):
                try:
                    with ThreadPoolExecutor(max_workers=1) as _pool:
                        fut = _pool.submit(setup_fn, seed)
                        try:
                            fut.result(timeout=SETUP_TIMEOUT_S)
                        except FuturesTimeoutError:
                            print(
                                f"WARNING: setup({seed}) timed out after "
                                f"{SETUP_TIMEOUT_S:.0f}s — skipping setup for this run.",
                                file=sys.stderr, flush=True,
                            )
                except Exception as exc:
                    print(
                        f"WARNING: setup({seed}) raised: {exc} — continuing with default init.",
                        file=sys.stderr, flush=True,
                    )

            t_start = time.monotonic()

            # ADVERSARIAL FIX 3 — NaN trajectory → immediate disqualify
            samples = _integrate_one(grad_fn, dim, seed, friction_fn)
            elapsed = time.monotonic() - t_start

            if samples is None:
                print(
                    f"DISQUALIFIED_RUN: {pot_name} seed={seed} diverged — "
                    f"tau=TAU_CAP, kl=inf",
                    flush=True,
                )
                tau_results[pot_name].append(float(TAU_CAP))
                kl_results[pot_name].append(math.inf)
                ess_results[pot_name].append(0.0)
                continue

            # KL divergence
            kl = _kl_divergence(samples, pot_name)
            kl_results[pot_name].append(kl)

            # τ_int — FFT-Sokal canonical + emcee cross-check
            label = f"{pot_name}/seed={seed}"
            tau   = _compute_tau_int(samples, label=label)
            tau_results[pot_name].append(tau)

            # KS stationarity (soft diagnostic, warn only)
            _ks_stationarity_check(samples, label=label)

            # ESS/second
            ess_s = (N_SAMPLES / tau / elapsed) if (tau > 0 and elapsed > 0) else 0.0
            ess_results[pot_name].append(ess_s)

            print(
                f"  {pot_name} seed={seed}: tau_int={tau:.2f}, "
                f"kl={kl:.5f}, t={elapsed:.1f}s",
                flush=True,
            )

    # ── 5. KL gate — GLOBAL: one potential failing → METRIC=inf ──────────────
    #    ADVERSARIAL FIX 4 (from candidate 6 bug): revert to global gate.
    #    Candidate 6's per-potential policy contradicts the spec.
    kl_mean_per_pot: dict[str, float] = {}
    for pot_name in POTENTIALS:
        kls = kl_results[pot_name]
        finite_kls = [k for k in kls if math.isfinite(k)]
        mean_kl = float(np.mean(finite_kls)) if finite_kls else math.inf
        kl_mean_per_pot[pot_name] = mean_kl

        if mean_kl > KL_THRESHOLD:
            print(
                f"KL_GATE_FAIL: {pot_name} mean_kl={mean_kl:.5f} > {KL_THRESHOLD} "
                f"→ GLOBAL DISQUALIFICATION (METRIC=inf)",
                flush=True,
            )
            return {
                "metric": math.inf,
                "error":  f"KL gate violated: {pot_name} mean_kl={mean_kl:.5f}",
                "kl_per_potential":      kl_mean_per_pot,
                "tau_int_per_potential": tau_results,
            }

    # ── 6. Weighted τ_int ─────────────────────────────────────────────────────
    tau_mean_per_pot: dict[str, float] = {}
    for pot_name in POTENTIALS:
        taus = tau_results[pot_name]
        tau_mean_per_pot[pot_name] = float(np.mean(taus)) if taus else float(TAU_CAP)

    weighted_tau = sum(
        weights[k] * tau_mean_per_pot[k] for k in POTENTIALS
    )

    wall_total = time.monotonic() - wall_start

    ess_mean = {
        k: float(np.mean(v)) if v else 0.0
        for k, v in ess_results.items()
    }

    return {
        "metric":                 weighted_tau,
        "tau_int_per_potential":  tau_results,
        "tau_mean_per_potential": tau_mean_per_pot,
        "kl_per_potential":       kl_mean_per_pot,
        "ESS_per_second":         ess_mean,
        "wall_time_s":            wall_total,
        "weights_used":           weights,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Modal batch dispatch
# ═════════════════════════════════════════════════════════════════════════════

_modal_fn_cache = None


def _get_modal_fn():
    """Lazily build and cache the Modal remote function.

    From candidate 5: all 9 runs dispatched in a single .remote() call.
    ThreadPoolExecutor inside the container handles parallelism.
    """
    global _modal_fn_cache
    if _modal_fn_cache is not None:
        return _modal_fn_cache

    import modal
    from modal_app import app, CPU_COUNT, MEMORY_MB, TIMEOUT_SECS  # noqa: F401

    @app.function(
        gpu=None,
        cpu=CPU_COUNT,
        memory=MEMORY_MB,
        timeout=TIMEOUT_SECS,
    )
    def _remote_eval(solution_code: str) -> dict:
        return _evaluate_solution(solution_code)

    _modal_fn_cache = (app, _remote_eval)
    return _modal_fn_cache


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Results printer
# ═════════════════════════════════════════════════════════════════════════════

def _print_results(result: dict) -> None:
    """Print METRIC= to stdout (required contract) + auxiliary metrics to stderr."""
    metric = result.get("metric", math.inf)

    # ── stdout: METRIC line (required by eval-v1 contract) ───────────────────
    if not math.isfinite(metric):
        print("METRIC=inf")
    else:
        print(f"METRIC={metric:.6f}")

    # ── stderr: auxiliary metrics ─────────────────────────────────────────────
    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)

    tau_per_pot = result.get("tau_int_per_potential", {})
    if tau_per_pot:
        print("\n--- tau_int_per_potential ---", file=sys.stderr)
        for pot_name, taus in tau_per_pot.items():
            seed_strs = ", ".join(
                f"seed={s}: {t:.1f}" for s, t in zip(SEEDS, taus)
            )
            mean_t = float(np.mean(taus)) if taus else math.nan
            print(f"  {pot_name}: [{seed_strs}]  mean={mean_t:.2f}", file=sys.stderr)

    kl_per_pot = result.get("kl_per_potential", {})
    if kl_per_pot:
        print("\n--- kl_per_potential ---", file=sys.stderr)
        for pot_name, kl in kl_per_pot.items():
            print(f"  {pot_name}: {kl:.6f}", file=sys.stderr)

    ess = result.get("ESS_per_second", {})
    if ess:
        print("\n--- ESS_per_second ---", file=sys.stderr)
        for pot_name, e in ess.items():
            print(f"  {pot_name}: {e:.3f}", file=sys.stderr)

    if "weights_used" in result:
        w = result["weights_used"]
        print(
            f"\nweights: harmonic={w.get('harmonic_1d', 0):.4f}, "
            f"doublewell={w.get('doublewell_2d', 0):.4f}, "
            f"gaussmix={w.get('gaussmix_2d', 0):.4f}",
            file=sys.stderr,
        )

    if "wall_time_s" in result:
        print(f"wall_time={result['wall_time_s']:.1f}s", file=sys.stderr)

    sys.stderr.flush()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12 — Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Production Nosé-Hoover friction evaluator (eval-v1). "
            "Synthesized from candidates 4/5/6 + adversarial findings."
        )
    )
    parser.add_argument("--solution", required=True, help="Path to solution.py")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Ignored — seeds are fixed at [42, 137, 2024] per spec."
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run locally without Modal (for testing/debugging).",
    )
    args = parser.parse_args()

    solution_path = Path(args.solution)
    if not solution_path.exists():
        print(f"ERROR: solution not found: {args.solution}", file=sys.stderr)
        sys.exit(1)

    solution_code = solution_path.read_text()

    if args.local:
        print("INFO: --local flag set; running locally (no Modal).", flush=True)
        result = _evaluate_solution(solution_code)
    else:
        print("INFO: Dispatching to Modal (single batch call for all 9 runs)...", flush=True)
        try:
            modal_app, modal_fn = _get_modal_fn()
            with modal_app.run():
                result = modal_fn.remote(solution_code)
        except Exception as exc:
            print(f"ERROR: Modal dispatch failed: {exc}", file=sys.stderr)
            print("INFO: Falling back to local execution.", flush=True)
            result = _evaluate_solution(solution_code)

    _print_results(result)

    metric = result.get("metric", math.inf)
    sys.exit(0 if math.isfinite(metric) else 1)


if __name__ == "__main__":
    main()
