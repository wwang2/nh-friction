#!/usr/bin/env python3
"""
Offline parameter sweep for extended Padé friction function.
Runs the actual evaluator for each candidate — no proxy shortcuts.

Strategy: the parent's (a=0.5, b=3.0, c=0.06) has metric=97.9.
The bottleneck is gaussmix_2d (tau~87, weight 0.68) and doublewell_2d (tau~131, weight 0.29).

We sweep c (denominator damping) and b (cubic strength) around the parent's values,
plus the new parameters e (quintic numerator) and d (quartic denominator).
"""

import subprocess
import concurrent.futures
import sys
import itertools

def run_eval(params_label, a, b, c, d, e, f):
    """Write a temp solution, run evaluator, parse METRIC."""
    import tempfile, os

    code = f'''
import numpy as np

_a = {a}
_b = {b}
_c = {c}
_d = {d}
_e = {e}
_f = {f}

def friction_function(xi: np.ndarray) -> np.ndarray:
    xi2 = xi * xi
    xi4 = xi2 * xi2
    num = _a + _b * xi2 + _e * xi4
    den = 1.0 + _c * xi2 + _d * xi4 + _f * xi4 * xi2
    return xi * num / den

def friction_derivative(xi: np.ndarray) -> np.ndarray:
    xi2 = xi * xi
    u = xi2
    u2 = u * u
    N = _a + _b * u + _e * u2
    D = 1.0 + _c * u + _d * u2 + _f * u2 * u
    D2 = D * D
    Np = _b + 2.0 * _e * u
    Dp = _c + 2.0 * _d * u + 3.0 * _f * u2
    return N / D + 2.0 * u * (Np * D - N * Dp) / D2

def setup(seed: int = 42) -> None:
    pass
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                       dir='/tmp') as tf:
        tf.write(code)
        tf.flush()
        tmp_path = tf.name

    try:
        result = subprocess.run(
            ["uv", "run", "python3", "research/eval/evaluator.py",
             "--solution", tmp_path, "--seed", "42"],
            capture_output=True, text=True, timeout=700,
            cwd="/Users/wujiewang/code/bath/.worktrees/002-pade-cmaes-refine"
        )

        metric = float('inf')
        details = ""
        for line in result.stdout.splitlines():
            if line.startswith("METRIC="):
                metric = float(line.split("=")[1])

        # Extract per-potential taus from stderr
        for line in result.stderr.splitlines():
            if "mean=" in line and ("harmonic" in line or "doublewell" in line or "gaussmix" in line):
                details += line.strip() + " | "

        return params_label, metric, details
    except Exception as ex:
        return params_label, float('inf'), str(ex)
    finally:
        import os
        os.unlink(tmp_path)


if __name__ == "__main__":
    # Phase 1: Sweep c and b around parent's optimum
    candidates = []

    # Sweep c (denominator damping) — this controls tail behavior
    for c_val in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]:
        candidates.append((f"c={c_val:.2f}", 0.5, 3.0, c_val, 0.0, 0.0, 0.0))

    # Sweep b (cubic strength) at c=0.06
    for b_val in [2.0, 2.5, 3.5, 4.0, 4.5, 5.0]:
        candidates.append((f"b={b_val:.1f},c=0.06", 0.5, b_val, 0.06, 0.0, 0.0, 0.0))

    # Sweep a (linear term) at b=3, c=0.06
    for a_val in [0.3, 0.4, 0.6, 0.7, 0.8, 1.0]:
        candidates.append((f"a={a_val:.1f}", a_val, 3.0, 0.06, 0.0, 0.0, 0.0))

    # Phase 2: Try adding quartic denominator d
    for d_val in [0.001, 0.005, 0.01, 0.02]:
        candidates.append((f"d={d_val:.3f}", 0.5, 3.0, 0.06, d_val, 0.0, 0.0))

    # Phase 3: Try adding quintic numerator e
    for e_val in [0.1, 0.5, 1.0, 2.0]:
        candidates.append((f"e={e_val:.1f}", 0.5, 3.0, 0.06, 0.0, e_val, 0.0))

    print(f"Running {len(candidates)} candidates...")
    sys.stdout.flush()

    results = []
    # Run 4 at a time to not overwhelm the machine
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(run_eval, *c): c[0] for c in candidates}
        for fut in concurrent.futures.as_completed(futures):
            label, metric, details = fut.result()
            results.append((metric, label, details))
            print(f"  {label}: METRIC={metric:.2f}  {details[:120]}")
            sys.stdout.flush()

    print("\n=== SORTED RESULTS ===")
    for metric, label, details in sorted(results):
        print(f"  {metric:8.2f}  {label}")
