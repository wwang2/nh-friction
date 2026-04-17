#!/usr/bin/env python3
"""Fine-tune gaussmix parameters around b=1.0 sweet spot."""
import math
import sys
import time
import numpy as np

sys.path.insert(0, '/Users/wujiewang/code/bath/.worktrees/012-joint-cmaes-alpha/orbits/012-joint-cmaes-alpha')
from gaussmix_search import integrate_long

def test_combo(a, b, c, alpha=1.0, seeds=[42, 137, 2024]):
    taus = []
    kls = []
    for seed in seeds:
        tau, kl = integrate_long("gaussmix_2d", seed, a, b, c, alpha)
        taus.append(tau)
        kls.append(kl)
    mean_tau = np.mean(taus)
    mean_kl = np.mean([k for k in kls if math.isfinite(k)])
    return mean_tau, mean_kl, taus

if __name__ == "__main__":
    print("Fine-tuning gaussmix around b=1.0")
    print("="*60)

    # b fine scan
    print("\nb scan (a=0.7, c=0.06):")
    for b in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
        mt, mk, taus = test_combo(0.7, b, 0.06)
        print(f"  b={b:.1f}: tau={mt:.2f} ({taus[0]:.1f},{taus[1]:.1f},{taus[2]:.1f}), kl={mk:.5f}")

    # a fine scan at b=1.0
    print("\na scan (b=1.0, c=0.06):")
    for a in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mt, mk, taus = test_combo(a, 1.0, 0.06)
        print(f"  a={a:.1f}: tau={mt:.2f} ({taus[0]:.1f},{taus[1]:.1f},{taus[2]:.1f}), kl={mk:.5f}")

    # c fine scan at best a, b=1.0
    print("\nc scan (a=0.7, b=1.0):")
    for c in [0.01, 0.03, 0.06, 0.10, 0.15, 0.20]:
        mt, mk, taus = test_combo(0.7, 1.0, c)
        print(f"  c={c:.2f}: tau={mt:.2f} ({taus[0]:.1f},{taus[1]:.1f},{taus[2]:.1f}), kl={mk:.5f}")

    # Also test doublewell improvements
    print("\n" + "="*60)
    print("Doublewell fine-tuning around a=0.8, b=3.0, c=0.03, alpha=3.0")
    print("="*60)

    print("\nalpha fine scan (a=0.7, b=3.0, c=0.06):")
    for alpha in [2.5, 2.8, 3.0, 3.2, 3.5]:
        taus = []
        kls = []
        for seed in [42, 137, 2024]:
            tau, kl = integrate_long("doublewell_2d", seed, 0.7, 3.0, 0.06, alpha)
            taus.append(tau)
            kls.append(kl)
        mean_tau = np.mean(taus)
        mean_kl = np.mean([k for k in kls if math.isfinite(k)])
        print(f"  alpha={alpha:.1f}: tau={mean_tau:.2f} ({taus[0]:.1f},{taus[1]:.1f},{taus[2]:.1f}), kl={mean_kl:.5f}")

    print("\nDone.")
