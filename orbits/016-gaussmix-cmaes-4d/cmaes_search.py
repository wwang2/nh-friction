#!/usr/bin/env python3
"""
Fast parameter search for gaussmix friction function parameters.
Phase 1: Focused grid on (b, alpha) with a,c fixed at known-good (100 pts, ~5min)
Phase 2: Refine (a, c) around best (b, alpha) from Phase 1 (50 pts, ~2.5min)
Phase 3: CMA-ES 4D refinement from top-2 candidates (15 gens x 8 pop, ~12min)
Phase 4: Verify best with 1M steps x 3 seeds (~2min)

Total budget: ~22 minutes

Usage:
  uv run python3 orbits/016-gaussmix-cmaes-4d/cmaes_search.py
"""

import math
import sys
import time
import numpy as np
from pathlib import Path

M, KT, Q, DT = 1.0, 1.0, 1.0, 0.01
N_BURNIN = 10_000
THIN = 10
TAU_CAP = 50_000
TAU_INT_FLOOR = 1.0
KL_THRESHOLD = 0.05
KL_BINS = 100
KL_RANGE = (-6.0, 6.0)
C_SOKAL = 6.0

MU = np.array([[3.0 * math.cos(2.0 * math.pi * k / 5),
                3.0 * math.sin(2.0 * math.pi * k / 5)] for k in range(5)])

def _grad_gaussmix_2d(q):
    diff = q[np.newaxis, :] - MU
    log_w = -0.5 * np.sum(diff ** 2, axis=1)
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()
    return np.einsum("k,kd->d", w, diff)

def _marginal_gaussmix_2d(bin_edges):
    MU_X = [3.0 * math.cos(2.0 * math.pi * k / 5) for k in range(5)]
    lo, hi = bin_edges[:-1], bin_edges[1:]
    def _norm_cdf(x):
        return 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in x]))
    probs = np.zeros(len(lo))
    for mu_x in MU_X:
        probs += (_norm_cdf(hi - mu_x) - _norm_cdf(lo - mu_x)) / 5.0
    probs = np.clip(probs, 1e-300, None)
    return probs / probs.sum()

def _kl_divergence(samples_q0):
    if len(samples_q0) == 0 or not np.all(np.isfinite(samples_q0)):
        return math.inf
    bin_edges = np.linspace(KL_RANGE[0], KL_RANGE[1], KL_BINS + 1)
    counts, _ = np.histogram(np.clip(samples_q0, KL_RANGE[0], KL_RANGE[1]), bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return math.inf
    p_emp = counts.astype(np.float64) / total
    p_ref = _marginal_gaussmix_2d(bin_edges)
    kl = 0.0
    for p_i, q_i in zip(p_emp, p_ref):
        if p_i > 0.0:
            if q_i <= 0.0: return math.inf
            kl += p_i * math.log(p_i / q_i)
    return max(0.0, kl)

def _sokal_tau(x, c=C_SOKAL):
    x = np.asarray(x, dtype=np.float64); N = len(x)
    if N < 3 or not np.all(np.isfinite(x)): return float(TAU_CAP)
    xc = x - x.mean(); var = float(np.var(xc))
    if var == 0.0 or not math.isfinite(var): return float(TAU_CAP)
    nfft = 1
    while nfft < 2 * N: nfft <<= 1
    F = np.fft.rfft(xc, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N].real
    acf = acf / (N - np.arange(N))
    C0 = float(acf[0])
    if C0 <= 0.0 or not math.isfinite(C0): return float(TAU_CAP)
    rho = acf / C0
    if not np.all(np.isfinite(rho)): return float(TAU_CAP)
    tau = 1.0
    for t in range(1, N):
        tau += 2.0 * rho[t]
        if not math.isfinite(tau) or tau < TAU_INT_FLOOR: tau = TAU_INT_FLOOR
        if t > c * tau:
            return max(min(float(tau), float(TAU_CAP)), TAU_INT_FLOOR)
    return float(TAU_CAP)

def run_gm(a, b, c, alpha, seed=42, n_main=200_000):
    """Run gaussmix integration. Returns (tau, kl, diverged)."""
    dim = 2; d_kT = 2.0; half_dt = DT / 2.0
    n_total = N_BURNIN + n_main; n_samp = n_main // THIN
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim); p = rng.standard_normal(dim); xi = 0.0
    samples = np.empty(n_samp, dtype=np.float64); ri = 0
    probe = 5000
    for step in range(n_total):
        p -= _grad_gaussmix_2d(q) * half_dt
        xi2 = xi * xi
        gxi = xi * (a + b * xi2) / (1.0 + c * xi2)
        if not math.isfinite(gxi): return float(TAU_CAP), math.inf, True
        try: ef = math.exp(-gxi * half_dt / Q)
        except OverflowError: return float(TAU_CAP), math.inf, True
        p *= ef; q += p / M * DT; p *= ef
        gq = _grad_gaussmix_2d(q); p -= gq * half_dt
        pp = float(np.dot(p, p))
        h = pp if step < probe else alpha * pp - (alpha - 1.0) * d_kT
        if not math.isfinite(h): return float(TAU_CAP), math.inf, True
        xi += (h - d_kT) / Q * DT
        if not math.isfinite(xi) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            return float(TAU_CAP), math.inf, True
        if step >= N_BURNIN and (step - N_BURNIN) % THIN == 0 and ri < n_samp:
            samples[ri] = q[0]; ri += 1
    if ri != n_samp: return float(TAU_CAP), math.inf, True
    kl = _kl_divergence(samples)
    if kl > KL_THRESHOLD: return float(TAU_CAP), kl, False
    return _sokal_tau(samples), kl, False

def obj200k(params):
    a, b, c, alpha = params
    tau, kl, div = run_gm(a, b, c, alpha, seed=42, n_main=200_000)
    return 5000.0 if (div or kl > KL_THRESHOLD) else tau

def cmaes(fn, x0, sig, lo, hi, ngen=15, pop=8):
    n = len(x0); mu_ = pop // 2
    w = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1)); w /= w.sum()
    me = 1.0 / np.sum(w ** 2)
    cs = (me + 2) / (n + me + 5); ds = 1 + 2 * max(0, math.sqrt((me - 1) / (n + 1)) - 1) + cs
    cc = (4 + me / n) / (n + 4 + 2 * me / n)
    c1 = 2 / ((n + 1.3) ** 2 + me); cm = min(1 - c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
    chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    m = np.array(x0, dtype=np.float64); C = np.eye(n)
    ps = np.zeros(n); pc = np.zeros(n); bx, bf = m.copy(), math.inf; hist = []
    for g in range(ngen):
        try:
            ev, U = np.linalg.eigh(C); ev = np.maximum(ev, 1e-20)
            sqC = U @ np.diag(np.sqrt(ev)) @ U.T; isqC = U @ np.diag(1 / np.sqrt(ev)) @ U.T
        except: C = np.eye(n); sqC = np.eye(n); isqC = np.eye(n)
        arx = np.array([np.clip(m + sig * sqC @ np.random.randn(n), lo, hi) for _ in range(pop)])
        fit = np.array([fn(arx[k]) for k in range(pop)])
        idx = np.argsort(fit); arx, fit = arx[idx], fit[idx]
        if fit[0] < bf: bf, bx = fit[0], arx[0].copy()
        hist.append(bf)
        old = m.copy(); m = np.clip(np.sum(w[:, None] * arx[:mu_], axis=0), lo, hi)
        dm = (m - old) / sig
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * me) * isqC @ dm
        hs = 1.0 if np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (g + 1))) < (1.4 + 2 / (n + 1)) * chi else 0.0
        pc = (1 - cc) * pc + hs * math.sqrt(cc * (2 - cc) * me) * dm
        art = (arx[:mu_] - old) / sig
        C = (1 - c1 - cm) * C + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C) + \
            cm * np.sum(w[:, None, None] * np.array([np.outer(a_, a_) for a_ in art]), axis=0)
        C = 0.5 * (C + C.T)
        sig = min(sig * math.exp(cs / ds * (np.linalg.norm(ps) / chi - 1)), 10.0)
        print(f"    G{g:2d} best={bf:.1f} sig={sig:.3f} x=({arx[0,0]:.3f},{arx[0,1]:.3f},{arx[0,2]:.4f},{arx[0,3]:.3f})", flush=True)
        if sig < 1e-4: break
    return bx, bf, hist


def main():
    lo = np.array([0.1, 0.1, 0.01, 0.3])
    hi = np.array([2.0, 8.0, 0.50, 2.0])
    all_evals = []

    # ── Phase 1: (b, alpha) grid with a=0.7, c=0.06 fixed ────────────────────
    print("=" * 60, flush=True)
    print("PHASE 1: (b, alpha) grid, a=0.7, c=0.06", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()
    b_vals = np.arange(0.5, 6.1, 0.5)  # 12 values
    alpha_vals = np.arange(0.3, 2.01, 0.1)  # 18 values
    p1_results = []
    for b in b_vals:
        for alpha in alpha_vals:
            tau, kl, div = run_gm(0.7, b, 0.06, alpha, seed=42, n_main=200_000)
            p1_results.append((tau, kl, div, 0.7, b, 0.06, alpha))
            all_evals.append((0.7, b, 0.06, alpha, tau, kl))

    valid1 = [(tau, kl, a, b, c, al) for tau, kl, div, a, b, c, al in p1_results
              if not div and kl < KL_THRESHOLD and tau < TAU_CAP]
    valid1.sort(key=lambda x: x[0])
    print(f"Phase 1: {len(valid1)}/{len(p1_results)} valid, {time.time()-t0:.0f}s", flush=True)
    print("Top 5:", flush=True)
    for tau, kl, a, b, c, al in valid1[:5]:
        print(f"  tau={tau:.1f} b={b:.2f} alpha={al:.2f}", flush=True)

    # ── Phase 2: Refine (a, c) around top-3 (b, alpha) from Phase 1 ──────────
    print(f"\n{'='*60}", flush=True)
    print("PHASE 2: (a, c) refinement around top (b, alpha)", flush=True)
    print(f"{'='*60}", flush=True)
    t1 = time.time()

    # Get unique top (b, alpha) combos
    seen = set()
    top_ba = []
    for _, _, _, b, _, al in valid1[:5]:
        key = (round(b, 2), round(al, 2))
        if key not in seen:
            seen.add(key)
            top_ba.append(key)
        if len(top_ba) >= 3:
            break

    p2_results = []
    a_vals = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
    c_vals = [0.02, 0.04, 0.06, 0.08, 0.12, 0.20]
    for b, al in top_ba:
        for a in a_vals:
            for c_val in c_vals:
                tau, kl, div = run_gm(a, b, c_val, al, seed=42, n_main=200_000)
                p2_results.append((tau, kl, div, a, b, c_val, al))
                all_evals.append((a, b, c_val, al, tau, kl))

    valid2 = [(tau, kl, a, b, c, al) for tau, kl, div, a, b, c, al in p2_results
              if not div and kl < KL_THRESHOLD and tau < TAU_CAP]
    valid2.sort(key=lambda x: x[0])
    print(f"Phase 2: {len(valid2)}/{len(p2_results)} valid, {time.time()-t1:.0f}s", flush=True)
    print("Top 5:", flush=True)
    for tau, kl, a, b, c, al in valid2[:5]:
        print(f"  tau={tau:.1f} a={a:.2f} b={b:.2f} c={c:.4f} alpha={al:.2f}", flush=True)

    # ── Phase 3: CMA-ES refinement ───────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("PHASE 3: CMA-ES 4D refinement from top 2", flush=True)
    print(f"{'='*60}", flush=True)

    # Merge and deduplicate top candidates from both phases
    all_valid = valid1 + valid2
    all_valid.sort(key=lambda x: x[0])
    cma_starts = []
    seen_starts = set()
    for tau, kl, a, b, c, al in all_valid[:6]:
        key = (round(a, 1), round(b, 1), round(c, 2), round(al, 1))
        if key not in seen_starts:
            seen_starts.add(key)
            cma_starts.append([a, b, c, al])
        if len(cma_starts) >= 2:
            break

    cma_results = []
    for i, x0 in enumerate(cma_starts):
        print(f"\n  CMA-ES #{i} from ({x0[0]:.2f},{x0[1]:.2f},{x0[2]:.4f},{x0[3]:.2f})", flush=True)
        bx, bf, hist = cmaes(obj200k, x0, sig=0.1, lo=lo, hi=hi, ngen=15, pop=8)
        cma_results.append((bf, bx, hist))
        print(f"  -> tau={bf:.1f} at ({bx[0]:.4f},{bx[1]:.4f},{bx[2]:.6f},{bx[3]:.4f})", flush=True)

    # ── Phase 4: Verify with 1M steps x 3 seeds ─────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("PHASE 4: Verification (1M steps, 3 seeds)", flush=True)
    print(f"{'='*60}", flush=True)

    candidates = []
    for bf, bx, _ in sorted(cma_results, key=lambda x: x[0]):
        candidates.append(bx.copy())
    # Add best from grid if distinct
    if len(all_valid) > 0:
        gx = np.array([all_valid[0][2], all_valid[0][3], all_valid[0][4], all_valid[0][5]])
        if not any(np.allclose(gx, c, atol=0.01) for c in candidates):
            candidates.append(gx)
    # Also add the known baseline for comparison
    baseline = np.array([0.70, 3.00, 0.06, 0.74])
    if not any(np.allclose(baseline, c, atol=0.01) for c in candidates):
        candidates.append(baseline)

    best_mean = math.inf
    best_p = None
    best_det = None

    for ci, params in enumerate(candidates):
        a, b, c_val, alpha = params
        print(f"\n  Cand {ci}: a={a:.4f} b={b:.4f} c={c_val:.6f} alpha={alpha:.4f}", flush=True)
        taus, kls = [], []
        for seed in [42, 137, 2024]:
            tau, kl, div = run_gm(a, b, c_val, alpha, seed=seed, n_main=1_000_000)
            taus.append(tau); kls.append(kl)
            print(f"    seed={seed}: tau={tau:.2f} kl={kl:.5f} div={div}", flush=True)
        mt = np.mean(taus); st = np.std(taus)
        print(f"    Mean: tau={mt:.2f}+/-{st:.2f}", flush=True)
        if mt < best_mean and all(k < KL_THRESHOLD for k in kls):
            best_mean, best_p = mt, params.copy()
            best_det = {'taus': list(taus), 'kls': list(kls), 'mean': mt, 'std': st}

    print(f"\n{'='*60}", flush=True)
    print("FINAL RESULT", flush=True)
    print(f"{'='*60}", flush=True)
    if best_p is not None:
        print(f"Best: a={best_p[0]:.4f}, b={best_p[1]:.4f}, c={best_p[2]:.6f}, alpha={best_p[3]:.4f}", flush=True)
        print(f"tau_gm (1M, 3-seed) = {best_mean:.2f}", flush=True)
        print(f"Detail: {best_det}", flush=True)
    else:
        print("No valid candidate!", flush=True)

    # Save
    ae = np.array(all_evals)
    np.savez(str(Path(__file__).parent / 'search_results.npz'),
             all_evals=ae,
             best_params=best_p if best_p is not None else np.zeros(4),
             best_mean_tau=np.array([best_mean]),
             best_detail_taus=np.array(best_det['taus']) if best_det else np.zeros(3),
             best_detail_kls=np.array(best_det['kls']) if best_det else np.zeros(3))
    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
