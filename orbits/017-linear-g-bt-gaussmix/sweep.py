"""Short proxy sweep: run gaussmix only at reduced steps to scan parameters.

For each config, uses 250k steps instead of 1M, single seed.
Returns (tau_int, kl) for gaussmix so we can filter out divergent configs
before spending full-eval budget.
"""
from __future__ import annotations

import importlib
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.eval.evaluator import (  # noqa: E402
    _grad_gaussmix_2d,
    _sokal_tau_int_fft,
    _kl_divergence,
    M, KT, Q, DT,
)

# Short-run parameters
N_BURNIN_SHORT = 10_000
N_MAIN_SHORT   = 250_000
THIN           = 10
N_SAMPLES_SHORT = N_MAIN_SHORT // THIN


def run_gaussmix(friction_fn, driving_fn, seed: int = 42) -> tuple[float, float, float]:
    """One gaussmix run at reduced steps.  Returns (tau_int, kl, elapsed_s)."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(2)
    p = rng.standard_normal(2)
    xi = 0.0
    half = DT / 2.0
    d_kT = 2.0 * KT

    samples = np.empty(N_SAMPLES_SHORT, dtype=np.float64)
    rec_idx = 0
    t0 = time.monotonic()
    n_total = N_BURNIN_SHORT + N_MAIN_SHORT

    for step in range(n_total):
        p -= _grad_gaussmix_2d(q) * half
        gxi = float(friction_fn(np.array([xi]))[0])
        if not math.isfinite(gxi):
            return math.inf, math.inf, time.monotonic() - t0
        try:
            ef = math.exp(-gxi * half / Q)
        except OverflowError:
            return math.inf, math.inf, time.monotonic() - t0
        p *= ef
        q = q + p / M * DT
        p *= ef
        grad_q = _grad_gaussmix_2d(q)
        p -= grad_q * half
        h = float(driving_fn(q, p, grad_q, xi))
        if not math.isfinite(h):
            return math.inf, math.inf, time.monotonic() - t0
        xi = xi + (h - d_kT) / Q * DT
        if not math.isfinite(xi) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            return math.inf, math.inf, time.monotonic() - t0
        if step >= N_BURNIN_SHORT and (step - N_BURNIN_SHORT) % THIN == 0:
            samples[rec_idx] = q[0]
            rec_idx += 1

    if rec_idx != N_SAMPLES_SHORT:
        return math.inf, math.inf, time.monotonic() - t0

    tau, _ = _sokal_tau_int_fft(samples)
    kl = _kl_divergence(samples, "gaussmix_2d")
    return float(tau), float(kl), time.monotonic() - t0


def load_solution_with(mode: str, alpha: float = 1.0, lam: float = 0.5, k: float = 1.0):
    """Hot-load solution module with overridden gaussmix parameters."""
    mod_path = Path(__file__).parent / "solution.py"
    # remove any cached module
    sys.modules.pop("solution", None)
    sys.modules.pop("orbits.017-linear-g-bt-gaussmix.solution", None)
    spec = importlib.util.spec_from_file_location("solution", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._GAUSSMIX_MODE = mode
    mod._GAUSSMIX_ALPHA = alpha
    mod._GAUSSMIX_LAMBDA = lam
    mod._GAUSSMIX_LINEAR_K = k
    return mod


def _worker(args):
    """Top-level (picklable) worker.  One process per config*seed."""
    mode, alpha, lam, k, seed = args
    mod = load_solution_with(mode, alpha=alpha, lam=lam, k=k)
    mod.setup(seed)
    tau, kl, _ = run_gaussmix(mod.friction_function, mod.driving_function, seed=seed)
    return mode, alpha, lam, k, seed, tau, kl


def main():
    # Configs to sweep
    configs = [
        ("effectiveQ", 0.74, 0.0, 1.0),     # orbit-015 reference
        # Refined hybrid sweep — promising region is K in [1.5,2.5], lam in [0.75,0.9]
        ("bt_hybrid",  1.0,  0.75, 2.0),
        ("bt_hybrid",  1.0,  0.80, 2.0),
        ("bt_hybrid",  1.0,  0.85, 2.0),
        ("bt_hybrid",  1.0,  0.90, 2.0),
        ("bt_hybrid",  1.0,  0.75, 2.5),
        ("bt_hybrid",  1.0,  0.85, 2.5),
        ("bt_hybrid",  1.0,  0.85, 1.5),
        ("bt_hybrid",  1.0,  0.75, 1.5),
    ]

    # Multi-seed (42, 137, 2024) for each config to reduce noise
    seeds = [42, 137, 2024]

    print(f"{'mode':<12s} {'alpha':>5s} {'lam':>5s} {'k':>4s} {'tau_mean':>9s} {'kl_mean':>8s} {'taus':>30s}")
    for mode, alpha, lam, k in configs:
        taus, kls = [], []
        for s in seeds:
            mod = load_solution_with(mode, alpha=alpha, lam=lam, k=k)
            mod.setup(s)
            tau, kl, _ = run_gaussmix(mod.friction_function, mod.driving_function, seed=s)
            taus.append(tau)
            kls.append(kl)
        tau_mean = sum(taus) / len(taus)
        kl_mean  = sum(kls) / len(kls)
        taus_str = ",".join(f"{t:.1f}" for t in taus)
        print(f"{mode:<12s} {alpha:>5.2f} {lam:>5.2f} {k:>4.2f} {tau_mean:>9.1f} {kl_mean:>8.3f} {taus_str:>30s}",
              flush=True)


if __name__ == "__main__":
    main()
