#!/usr/bin/env python3
"""
Focused search for gaussmix optimization.
Uses longer integration (500k steps) for more reliable tau estimates.
Tests different (a, b, c, alpha) combinations around the known-good baseline.
"""
import math
import sys
import time
import numpy as np
from optimize import integrate_one, _sokal_tau_int_fft, _kl_divergence, POTENTIALS

# Override mini-eval to use longer runs for gaussmix
N_BURNIN_LONG = 10_000
N_MAIN_STEPS_LONG = 500_000
THIN_LONG = 10
N_SAMPLES_LONG = N_MAIN_STEPS_LONG // THIN_LONG  # 50,000
N_TOTAL_STEPS_LONG = N_BURNIN_LONG + N_MAIN_STEPS_LONG
TAU_CAP_LONG = 25_000

DT = 0.01
M = 1.0
KT = 1.0
Q = 1.0

def integrate_long(pot_name, seed, a, b, c, alpha):
    """Run longer integration for more reliable tau estimate."""
    pot = POTENTIALS[pot_name]
    grad_fn = pot["grad"]
    dim = pot["dim"]
    d_kT = float(dim) * KT

    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim)
    p = rng.standard_normal(dim)
    xi = 0.0

    samples = np.empty(N_SAMPLES_LONG, dtype=np.float64)
    rec_idx = 0
    half_dt = DT / 2.0

    for step in range(N_TOTAL_STEPS_LONG):
        p = p - grad_fn(q) * half_dt
        xi2 = xi * xi
        gxi = xi * (a + b * xi2) / (1.0 + c * xi2)
        if not math.isfinite(gxi):
            return TAU_CAP_LONG, math.inf
        try:
            exp_fac = math.exp(-gxi * half_dt / Q)
        except OverflowError:
            return TAU_CAP_LONG, math.inf
        p = p * exp_fac
        q = q + p / M * DT
        p = p * exp_fac
        grad_q = grad_fn(q)
        p = p - grad_q * half_dt
        pp = float(np.dot(p, p))
        h_val = alpha * pp - (alpha - 1.0) * d_kT
        xi = xi + (h_val - d_kT) / Q * DT
        if not math.isfinite(xi) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            return TAU_CAP_LONG, math.inf
        if step >= N_BURNIN_LONG and (step - N_BURNIN_LONG) % THIN_LONG == 0:
            samples[rec_idx] = q[0]
            rec_idx += 1

    if rec_idx != N_SAMPLES_LONG:
        return TAU_CAP_LONG, math.inf

    tau, _ = _sokal_tau_int_fft(samples)
    kl = _kl_divergence(samples, pot_name)
    return tau, kl


if __name__ == "__main__":
    # ── Search 1: Alpha scan for gaussmix with default Pade ──
    print("="*60)
    print("Gaussmix: Alpha scan (a=0.7, b=3.0, c=0.06)")
    print("="*60)

    alphas = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    for alpha in alphas:
        taus = []
        kls = []
        for seed in [42, 137, 2024]:
            t0 = time.time()
            tau, kl = integrate_long("gaussmix_2d", seed, 0.7, 3.0, 0.06, alpha)
            elapsed = time.time() - t0
            taus.append(tau)
            kls.append(kl)
        mean_tau = np.mean(taus)
        mean_kl = np.mean([k for k in kls if math.isfinite(k)])
        print(f"  alpha={alpha:.1f}: tau={mean_tau:.2f} ({taus[0]:.1f},{taus[1]:.1f},{taus[2]:.1f}), kl={mean_kl:.5f}")

    # ── Search 2: Pade parameter variations with alpha=1.0 ──
    print("\n" + "="*60)
    print("Gaussmix: Pade variations (alpha=1.0)")
    print("="*60)

    # Vary a (linear slope at origin)
    for a_val in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]:
        taus = []
        kls = []
        for seed in [42, 137]:
            tau, kl = integrate_long("gaussmix_2d", seed, a_val, 3.0, 0.06, 1.0)
            taus.append(tau)
            kls.append(kl)
        mean_tau = np.mean(taus)
        mean_kl = np.mean([k for k in kls if math.isfinite(k)])
        print(f"  a={a_val:.1f}: tau={mean_tau:.2f}, kl={mean_kl:.5f}")

    # Vary b (cubic growth)
    print()
    for b_val in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        taus = []
        kls = []
        for seed in [42, 137]:
            tau, kl = integrate_long("gaussmix_2d", seed, 0.7, b_val, 0.06, 1.0)
            taus.append(tau)
            kls.append(kl)
        mean_tau = np.mean(taus)
        mean_kl = np.mean([k for k in kls if math.isfinite(k)])
        print(f"  b={b_val:.1f}: tau={mean_tau:.2f}, kl={mean_kl:.5f}")

    # Vary c (saturation)
    print()
    for c_val in [0.001, 0.01, 0.03, 0.06, 0.10, 0.20, 0.50]:
        taus = []
        kls = []
        for seed in [42, 137]:
            tau, kl = integrate_long("gaussmix_2d", seed, 0.7, 3.0, c_val, 1.0)
            taus.append(tau)
            kls.append(kl)
        mean_tau = np.mean(taus)
        mean_kl = np.mean([k for k in kls if math.isfinite(k)])
        print(f"  c={c_val:.3f}: tau={mean_tau:.2f}, kl={mean_kl:.5f}")

    # ── Search 3: Doublewell alpha confirmation ──
    print("\n" + "="*60)
    print("Doublewell: Alpha scan (a=0.7, b=3.0, c=0.06)")
    print("="*60)

    for alpha in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        taus = []
        kls = []
        for seed in [42, 137, 2024]:
            tau, kl = integrate_long("doublewell_2d", seed, 0.7, 3.0, 0.06, alpha)
            taus.append(tau)
            kls.append(kl)
        mean_tau = np.mean(taus)
        mean_kl = np.mean([k for k in kls if math.isfinite(k)])
        print(f"  alpha={alpha:.1f}: tau={mean_tau:.2f} ({taus[0]:.1f},{taus[1]:.1f},{taus[2]:.1f}), kl={mean_kl:.5f}")

    # ── Search 4: Harmonic alpha confirmation ──
    print("\n" + "="*60)
    print("Harmonic: Alpha scan (a=0.7, b=3.0, c=0.06)")
    print("="*60)

    for alpha in [1.5, 2.0, 2.5, 3.0]:
        taus = []
        kls = []
        for seed in [42, 137, 2024]:
            tau, kl = integrate_long("harmonic_1d", seed, 0.7, 3.0, 0.06, alpha)
            taus.append(tau)
            kls.append(kl)
        mean_tau = np.mean(taus)
        mean_kl = np.mean([k for k in kls if math.isfinite(k)])
        print(f"  alpha={alpha:.1f}: tau={mean_tau:.2f} ({taus[0]:.1f},{taus[1]:.1f},{taus[2]:.1f}), kl={mean_kl:.5f}")

    print("\nDone.")
