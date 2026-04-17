#!/usr/bin/env python3
"""
Search for per-potential Pade parameter optimization.
Focus: Can we improve doublewell tau by tuning (a,b,c) at alpha=3.0?
Also: Can we improve gaussmix tau by tuning (a,b,c) at alpha=1.0?

Uses 500k step integration with 3 seeds for reliability.
"""
import math
import sys
import time
import numpy as np

# Import from optimize.py
sys.path.insert(0, '/Users/wujiewang/code/bath/.worktrees/012-joint-cmaes-alpha/orbits/012-joint-cmaes-alpha')
from gaussmix_search import integrate_long

def scan_2d(pot_name, alpha, param_name1, values1, param_name2, values2, fixed):
    """2D parameter scan. fixed = {'a': ..., 'b': ..., 'c': ...}"""
    results = []
    for v1 in values1:
        for v2 in values2:
            params = dict(fixed)
            params[param_name1] = v1
            params[param_name2] = v2
            taus = []
            kls = []
            for seed in [42, 137]:
                tau, kl = integrate_long(pot_name, seed, params['a'], params['b'], params['c'], alpha)
                taus.append(tau)
                kls.append(kl)
            mean_tau = np.mean(taus)
            mean_kl = np.mean([k for k in kls if math.isfinite(k)])
            results.append({
                param_name1: v1, param_name2: v2,
                'tau': mean_tau, 'kl': mean_kl
            })
            print(f"  {param_name1}={v1:.3f}, {param_name2}={v2:.3f}: tau={mean_tau:.2f}, kl={mean_kl:.5f}")
    return results

if __name__ == "__main__":
    # ── Doublewell: (a, b) scan at alpha=3.0, c=0.06 ──
    print("="*60)
    print("Doublewell (a,b) scan at alpha=3.0, c=0.06")
    print("="*60)
    dw_results = scan_2d(
        "doublewell_2d", 3.0,
        'a', [0.4, 0.6, 0.7, 0.8, 1.0],
        'b', [1.5, 2.0, 3.0, 4.0, 5.0],
        {'a': 0.7, 'b': 3.0, 'c': 0.06}
    )

    # Find best for doublewell
    valid_dw = [r for r in dw_results if r['kl'] < 0.05]
    if valid_dw:
        best_dw = min(valid_dw, key=lambda r: r['tau'])
        print(f"\n  BEST DW: a={best_dw['a']:.3f}, b={best_dw['b']:.3f}, tau={best_dw['tau']:.2f}, kl={best_dw['kl']:.5f}")

    # ── Doublewell: c scan at best (a,b) ──
    print("\n" + "="*60)
    print("Doublewell c scan at best (a,b), alpha=3.0")
    print("="*60)
    if valid_dw:
        best_a, best_b = best_dw['a'], best_dw['b']
    else:
        best_a, best_b = 0.7, 3.0

    for c_val in [0.001, 0.01, 0.03, 0.06, 0.10, 0.15, 0.20]:
        taus = []
        kls = []
        for seed in [42, 137, 2024]:
            tau, kl = integrate_long("doublewell_2d", seed, best_a, best_b, c_val, 3.0)
            taus.append(tau)
            kls.append(kl)
        mean_tau = np.mean(taus)
        mean_kl = np.mean([k for k in kls if math.isfinite(k)])
        print(f"  c={c_val:.3f}: tau={mean_tau:.2f} ({taus[0]:.1f},{taus[1]:.1f},{taus[2]:.1f}), kl={mean_kl:.5f}")

    # ── Gaussmix: (a, c) scan at alpha=1.0, b=3.0 ──
    print("\n" + "="*60)
    print("Gaussmix (a,c) scan at alpha=1.0, b=3.0")
    print("="*60)
    gm_results = scan_2d(
        "gaussmix_2d", 1.0,
        'a', [0.5, 0.7, 0.9, 1.2],
        'c', [0.03, 0.06, 0.10, 0.20, 0.50],
        {'a': 0.7, 'b': 3.0, 'c': 0.06}
    )

    valid_gm = [r for r in gm_results if r['kl'] < 0.05]
    if valid_gm:
        best_gm = min(valid_gm, key=lambda r: r['tau'])
        print(f"\n  BEST GM: a={best_gm['a']:.3f}, c={best_gm['c']:.3f}, tau={best_gm['tau']:.2f}, kl={best_gm['kl']:.5f}")

    # ── Gaussmix: b scan at best (a,c) ──
    print("\n" + "="*60)
    print("Gaussmix b scan at best (a,c), alpha=1.0")
    print("="*60)
    if valid_gm:
        best_a_gm, best_c_gm = best_gm['a'], best_gm['c']
    else:
        best_a_gm, best_c_gm = 0.7, 0.06

    for b_val in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        taus = []
        kls = []
        for seed in [42, 137, 2024]:
            tau, kl = integrate_long("gaussmix_2d", seed, best_a_gm, b_val, best_c_gm, 1.0)
            taus.append(tau)
            kls.append(kl)
        mean_tau = np.mean(taus)
        mean_kl = np.mean([k for k in kls if math.isfinite(k)])
        print(f"  b={b_val:.1f}: tau={mean_tau:.2f} ({taus[0]:.1f},{taus[1]:.1f},{taus[2]:.1f}), kl={mean_kl:.5f}")

    print("\nDone.")
