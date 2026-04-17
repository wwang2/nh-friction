#!/usr/bin/env python3
"""
Fast coordinate descent + local CMA-ES for gaussmix parameters.
Budget: ~100 evaluations = ~5 minutes.

Strategy:
1. Coordinate descent from (0.7, 3.0, 0.06, 0.74) — 4 rounds, 7 values each = ~28 evals
2. Coordinate descent from (0.7, 1.0, 0.06, 1.0) — second basin
3. Tiny CMA-ES from best found — 10 gens x 6 pop = 60 evals
4. Verify top 2 with 1M steps x 3 seeds

Usage:
  PYTHONUNBUFFERED=1 uv run python3 orbits/016-gaussmix-cmaes-4d/fast_search.py
"""

import math, sys, time
import numpy as np
from pathlib import Path

M, KT, Q, DT = 1.0, 1.0, 1.0, 0.01
N_BURNIN = 10_000; THIN = 10
TAU_CAP = 50_000; TAU_INT_FLOOR = 1.0
KL_THRESHOLD = 0.05; KL_BINS = 100; KL_RANGE = (-6.0, 6.0); C_SOKAL = 6.0

MU = np.array([[3.0*math.cos(2*math.pi*k/5), 3.0*math.sin(2*math.pi*k/5)] for k in range(5)])

def _grad(q):
    diff = q[np.newaxis,:] - MU
    lw = -0.5*np.sum(diff**2, axis=1); lw -= lw.max()
    w = np.exp(lw); w /= w.sum()
    return np.einsum("k,kd->d", w, diff)

def _kl(samples):
    if len(samples)==0 or not np.all(np.isfinite(samples)): return math.inf
    be = np.linspace(KL_RANGE[0],KL_RANGE[1],KL_BINS+1)
    cnt,_ = np.histogram(np.clip(samples,KL_RANGE[0],KL_RANGE[1]),bins=be)
    t = cnt.sum()
    if t==0: return math.inf
    pe = cnt.astype(np.float64)/t
    MUX = [3.0*math.cos(2*math.pi*k/5) for k in range(5)]
    lo,hi = be[:-1],be[1:]
    def ncdf(x): return 0.5*(1+np.array([math.erf(v/math.sqrt(2)) for v in x]))
    pr = np.zeros(len(lo))
    for mx in MUX: pr += (ncdf(hi-mx)-ncdf(lo-mx))/5.0
    pr = np.clip(pr,1e-300,None); pr /= pr.sum()
    kl = 0.0
    for p,q in zip(pe,pr):
        if p>0:
            if q<=0: return math.inf
            kl += p*math.log(p/q)
    return max(0.0,kl)

def _tau(x,c=C_SOKAL):
    x = np.asarray(x,dtype=np.float64); N=len(x)
    if N<3 or not np.all(np.isfinite(x)): return float(TAU_CAP)
    xc=x-x.mean(); v=float(np.var(xc))
    if v==0 or not math.isfinite(v): return float(TAU_CAP)
    nfft=1
    while nfft<2*N: nfft<<=1
    F=np.fft.rfft(xc,n=nfft); acf=np.fft.irfft(F*np.conj(F),n=nfft)[:N].real
    acf /= (N-np.arange(N)); C0=float(acf[0])
    if C0<=0 or not math.isfinite(C0): return float(TAU_CAP)
    rho=acf/C0
    if not np.all(np.isfinite(rho)): return float(TAU_CAP)
    t_r=1.0
    for t in range(1,N):
        t_r+=2*rho[t]
        if not math.isfinite(t_r) or t_r<TAU_INT_FLOOR: t_r=TAU_INT_FLOOR
        if t>c*t_r: return max(min(float(t_r),float(TAU_CAP)),TAU_INT_FLOOR)
    return float(TAU_CAP)

def run(a,b,c,alpha,seed=42,n_main=500_000):
    dim=2; dkT=2.0; hdt=DT/2; n_tot=N_BURNIN+n_main; ns=n_main//THIN
    rng=np.random.default_rng(seed); q=rng.standard_normal(dim); p=rng.standard_normal(dim); xi=0.0
    samp=np.empty(ns,dtype=np.float64); ri=0; probe=5000
    for step in range(n_tot):
        p -= _grad(q)*hdt
        xi2=xi*xi; gxi=xi*(a+b*xi2)/(1+c*xi2)
        if not math.isfinite(gxi): return float(TAU_CAP),math.inf,True
        try: ef=math.exp(-gxi*hdt/Q)
        except OverflowError: return float(TAU_CAP),math.inf,True
        p*=ef; q+=p/M*DT; p*=ef
        gq=_grad(q); p-=gq*hdt
        pp=float(np.dot(p,p))
        h = pp if step<probe else alpha*pp-(alpha-1)*dkT
        if not math.isfinite(h): return float(TAU_CAP),math.inf,True
        xi+=(h-dkT)/Q*DT
        if not math.isfinite(xi) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            return float(TAU_CAP),math.inf,True
        if step>=N_BURNIN and (step-N_BURNIN)%THIN==0 and ri<ns:
            samp[ri]=q[0]; ri+=1
    if ri!=ns: return float(TAU_CAP),math.inf,True
    kl=_kl(samp)
    if kl>KL_THRESHOLD: return float(TAU_CAP),kl,False
    return _tau(samp),kl,False

def eval_pt(a,b,c,alpha):
    tau,kl,div = run(a,b,c,alpha,seed=42,n_main=200_000)
    return 5000.0 if (div or kl>KL_THRESHOLD) else tau

def coord_descent(start, param_ranges, n_rounds=2):
    """Coordinate descent: optimize one param at a time."""
    best = list(start)
    best_f = eval_pt(*best)
    print(f"    Start: ({best[0]:.3f},{best[1]:.3f},{best[2]:.4f},{best[3]:.3f}) -> tau={best_f:.1f}", flush=True)

    for rd in range(n_rounds):
        for pi, vals in enumerate(param_ranges):
            pnames = ['a','b','c','alpha']
            best_val = best[pi]
            best_tau = best_f
            for v in vals:
                trial = list(best)
                trial[pi] = v
                tau = eval_pt(*trial)
                if tau < best_tau:
                    best_tau = tau
                    best_val = v
            if best_val != best[pi]:
                best[pi] = best_val
                best_f = best_tau
                print(f"    Round {rd} {pnames[pi]}: {best_val:.4f} -> tau={best_f:.1f}", flush=True)

    return best, best_f

def main():
    t0 = time.time()

    # Parameter ranges for coordinate descent (keep concise for speed)
    a_vals = [0.4, 0.6, 0.7, 0.8, 1.0, 1.3]
    b_vals = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    c_vals = [0.02, 0.04, 0.06, 0.08, 0.12, 0.20]
    alpha_vals = [0.4, 0.6, 0.7, 0.74, 0.8, 0.9, 1.0, 1.3]
    ranges = [a_vals, b_vals, c_vals, alpha_vals]

    # ── Basin A: known optimum ────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("BASIN A: Coordinate descent from (0.7, 3.0, 0.06, 0.74)", flush=True)
    print("=" * 60, flush=True)
    bestA, fA = coord_descent([0.7, 3.0, 0.06, 0.74], ranges, n_rounds=1)
    print(f"  Basin A result: ({bestA[0]:.4f},{bestA[1]:.4f},{bestA[2]:.4f},{bestA[3]:.4f}) tau={fA:.1f}", flush=True)

    # ── Basin B: second known optimum ─────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("BASIN B: Coordinate descent from (0.7, 1.0, 0.06, 1.0)", flush=True)
    print("=" * 60, flush=True)
    bestB, fB = coord_descent([0.7, 1.0, 0.06, 1.0], ranges, n_rounds=1)
    print(f"  Basin B result: ({bestB[0]:.4f},{bestB[1]:.4f},{bestB[2]:.4f},{bestB[3]:.4f}) tau={fB:.1f}", flush=True)

    # ── Basin C: unexplored region ────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("BASIN C: Coordinate descent from (1.0, 2.0, 0.03, 0.9)", flush=True)
    print("=" * 60, flush=True)
    bestC, fC = coord_descent([1.0, 2.0, 0.03, 0.9], ranges, n_rounds=1)
    print(f"  Basin C result: ({bestC[0]:.4f},{bestC[1]:.4f},{bestC[2]:.4f},{bestC[3]:.4f}) tau={fC:.1f}", flush=True)

    # ── Fine-tune best with smaller steps ─────────────────────────────────────
    results = [(fA, bestA, 'A'), (fB, bestB, 'B'), (fC, bestC, 'C')]
    results.sort(key=lambda x: x[0])
    best_f, best_x, best_name = results[0]

    print(f"\n{'='*60}", flush=True)
    print(f"FINE-TUNING: Basin {best_name} ({best_x[0]:.3f},{best_x[1]:.3f},{best_x[2]:.4f},{best_x[3]:.3f}) tau={best_f:.1f}", flush=True)
    print("=" * 60, flush=True)

    # Fine grid around best
    a0, b0, c0, al0 = best_x
    fine_a = [a0-0.1, a0-0.05, a0, a0+0.05, a0+0.1]
    fine_b = [b0-0.5, b0-0.25, b0, b0+0.25, b0+0.5]
    fine_c = [max(0.01, c0-0.02), max(0.01, c0-0.01), c0, c0+0.01, c0+0.02]
    fine_al = [al0-0.1, al0-0.05, al0, al0+0.05, al0+0.1]
    fine_ranges = [fine_a, fine_b, fine_c, fine_al]

    best_x2, best_f2 = coord_descent(best_x, fine_ranges, n_rounds=2)

    if best_f2 < best_f:
        best_f, best_x = best_f2, best_x2

    print(f"\nFine-tuned: ({best_x[0]:.4f},{best_x[1]:.4f},{best_x[2]:.4f},{best_x[3]:.4f}) tau={best_f:.1f}", flush=True)

    # ── Verify with 1M steps x 3 seeds ───────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("VERIFICATION (1M steps, 3 seeds)", flush=True)
    print("=" * 60, flush=True)

    candidates = [best_x]
    # Add second-best basin if significantly different
    if results[1][0] < best_f * 1.1:
        candidates.append(results[1][1])
    # Always include baseline for comparison
    candidates.append([0.70, 3.00, 0.06, 0.74])

    final_best_mean = math.inf
    final_best_params = None
    final_best_detail = None

    for ci, params in enumerate(candidates):
        a,b,c,alpha = params
        print(f"\n  Cand {ci}: a={a:.4f} b={b:.4f} c={c:.4f} alpha={alpha:.4f}", flush=True)
        taus, kls = [], []
        for seed in [42, 137, 2024]:
            tau, kl, div = run(a, b, c, alpha, seed=seed, n_main=1_000_000)
            taus.append(tau); kls.append(kl)
            print(f"    seed={seed}: tau={tau:.2f} kl={kl:.5f} div={div}", flush=True)
        mt = np.mean(taus); st = np.std(taus)
        print(f"    Mean: {mt:.2f} +/- {st:.2f}", flush=True)
        if mt < final_best_mean and all(k < KL_THRESHOLD for k in kls):
            final_best_mean = mt
            final_best_params = list(params)
            final_best_detail = {'taus': taus, 'kls': kls, 'mean': mt, 'std': st}

    print(f"\n{'='*60}", flush=True)
    print("FINAL RESULT", flush=True)
    print("=" * 60, flush=True)
    if final_best_params:
        a,b,c,alpha = final_best_params
        print(f"Best: a={a:.4f}, b={b:.4f}, c={c:.6f}, alpha={alpha:.4f}", flush=True)
        print(f"tau_gm (1M, 3-seed) = {final_best_mean:.2f} +/- {final_best_detail['std']:.2f}", flush=True)
        print(f"Seeds: {final_best_detail['taus']}", flush=True)
        print(f"KLs: {final_best_detail['kls']}", flush=True)
    else:
        print("No valid candidate!", flush=True)

    # Save all results
    all_results = {
        'basin_A': np.array(bestA), 'tau_A': np.array([fA]),
        'basin_B': np.array(bestB), 'tau_B': np.array([fB]),
        'basin_C': np.array(bestC), 'tau_C': np.array([fC]),
        'best_params': np.array(final_best_params) if final_best_params else np.zeros(4),
        'best_mean_tau': np.array([final_best_mean]),
        'best_taus': np.array(final_best_detail['taus']) if final_best_detail else np.zeros(3),
        'best_kls': np.array(final_best_detail['kls']) if final_best_detail else np.zeros(3),
    }
    np.savez(str(Path(__file__).parent / 'search_results.npz'), **all_results)

    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
    print("Results saved to search_results.npz", flush=True)

if __name__ == "__main__":
    main()
