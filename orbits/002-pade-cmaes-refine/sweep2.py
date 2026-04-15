#!/usr/bin/env python3
"""
Refined sweep around a=0.7, b=3.0, c=0.06 — the best from sweep 1.
Also try combinations: a+e, a+d, a+c variations.
"""

import subprocess
import concurrent.futures
import sys
import tempfile
import os

def run_eval(params_label, a, b, c, d, e, f):
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

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as tf:
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
        for line in result.stdout.splitlines():
            if line.startswith("METRIC="):
                metric = float(line.split("=")[1])

        details = ""
        for line in result.stderr.splitlines():
            if "mean=" in line and ("harmonic" in line or "doublewell" in line or "gaussmix" in line):
                details += line.strip() + " | "

        return params_label, metric, details
    except Exception as ex:
        return params_label, float('inf'), str(ex)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    candidates = []

    # Fine-tune a around 0.7
    for a_val in [0.65, 0.68, 0.70, 0.72, 0.75]:
        candidates.append((f"a={a_val:.2f}", a_val, 3.0, 0.06, 0.0, 0.0, 0.0))

    # a=0.7 + vary c
    for c_val in [0.03, 0.04, 0.05, 0.07, 0.08]:
        candidates.append((f"a=0.7,c={c_val:.2f}", 0.7, 3.0, c_val, 0.0, 0.0, 0.0))

    # a=0.7 + add e (quintic numerator)
    for e_val in [0.05, 0.1, 0.2, 0.5]:
        candidates.append((f"a=0.7,e={e_val}", 0.7, 3.0, 0.06, 0.0, e_val, 0.0))

    # a=0.7 + add d (quartic denominator)
    for d_val in [0.002, 0.005, 0.01]:
        candidates.append((f"a=0.7,d={d_val}", 0.7, 3.0, 0.06, d_val, 0.0, 0.0))

    # a=0.7 + vary b
    for b_val in [2.5, 2.8, 3.2, 3.5]:
        candidates.append((f"a=0.7,b={b_val}", 0.7, b_val, 0.06, 0.0, 0.0, 0.0))

    # Best combos from sweep 1
    candidates.append(("a=0.7,e=0.1,d=0.005", 0.7, 3.0, 0.06, 0.005, 0.1, 0.0))
    candidates.append(("a=0.7,c=0.03", 0.7, 3.0, 0.03, 0.0, 0.0, 0.0))
    candidates.append(("a=0.7,c=0.04,e=0.1", 0.7, 3.0, 0.04, 0.0, 0.1, 0.0))
    candidates.append(("a=0.75,e=0.1", 0.75, 3.0, 0.06, 0.0, 0.1, 0.0))

    print(f"Running {len(candidates)} candidates...")
    sys.stdout.flush()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(run_eval, *c): c[0] for c in candidates}
        for fut in concurrent.futures.as_completed(futures):
            label, metric, details = fut.result()
            results.append((metric, label, details))
            print(f"  {label}: METRIC={metric:.2f}  {details[:150]}")
            sys.stdout.flush()

    print("\n=== SORTED RESULTS ===")
    for metric, label, details in sorted(results):
        print(f"  {metric:8.2f}  {label}")
