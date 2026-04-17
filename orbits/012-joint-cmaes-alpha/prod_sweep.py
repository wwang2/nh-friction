#!/usr/bin/env python3
"""
Direct production-evaluator sweep for final parameter tuning.
Tests specific (a,b,c,alpha) combos by rewriting solution.py and running evaluator.
"""
import subprocess
import sys
import re
import os

SOLUTION_TEMPLATE = '''"""Auto-generated solution for parameter sweep."""
import numpy as np

_PARAMS = {{
    'harmonic':   {{'a': 0.70, 'b': 3.00, 'c': 0.06, 'alpha': 2.0}},
    'doublewell': {{'a': {dw_a:.2f}, 'b': {dw_b:.2f}, 'c': {dw_c:.3f}, 'alpha': {dw_alpha:.1f}}},
    'gaussmix':   {{'a': {gm_a:.2f}, 'b': {gm_b:.2f}, 'c': {gm_c:.3f}, 'alpha': {gm_alpha:.1f}}},
}}

_a = 0.7
_b = 3.0
_c = 0.06
_alpha = 1.0
_detect_done = False
_probe_n = 0
_probe_q_norm_accum = 0.0

def friction_function(xi):
    xi = np.asarray(xi, dtype=float)
    xi2 = xi * xi
    return xi * (_a + _b * xi2) / (1.0 + _c * xi2)

def friction_derivative(xi):
    xi = np.asarray(xi, dtype=float)
    xi2 = xi * xi
    num = _a + _b * xi2
    den = 1.0 + _c * xi2
    dnum = 2.0 * _b * xi
    dden = 2.0 * _c * xi
    return num / den + xi * (dnum * den - num * dden) / (den * den)

def setup(seed):
    global _a, _b, _c, _alpha, _detect_done, _probe_n, _probe_q_norm_accum
    _a = 0.7
    _b = 3.0
    _c = 0.06
    _alpha = 1.0
    _detect_done = False
    _probe_n = 0
    _probe_q_norm_accum = 0.0

def driving_function(q, p, grad_V):
    global _a, _b, _c, _alpha, _detect_done, _probe_n, _probe_q_norm_accum
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    d = len(q)
    pp = float(np.dot(p, p))
    d_kT = float(d) * 1.0
    if not _detect_done:
        _probe_n += 1
        _probe_q_norm_accum += float(np.sqrt(np.dot(q, q)))
        if _probe_n >= 5000:
            mean_q_norm = _probe_q_norm_accum / _probe_n
            if d == 1:
                params = _PARAMS['harmonic']
            elif mean_q_norm > 2.0:
                params = _PARAMS['gaussmix']
            else:
                params = _PARAMS['doublewell']
            _a = params['a']
            _b = params['b']
            _c = params['c']
            _alpha = params['alpha']
            _detect_done = True
        return pp
    return _alpha * pp - (_alpha - 1.0) * d_kT
'''

WORKTREE = "/Users/wujiewang/code/bath/.worktrees/012-joint-cmaes-alpha"
SOLUTION_PATH = os.path.join(WORKTREE, "orbits/012-joint-cmaes-alpha/_sweep_solution.py")
EVALUATOR = os.path.join(WORKTREE, "research/eval/evaluator.py")


def run_eval(dw_a, dw_b, dw_c, dw_alpha, gm_a, gm_b, gm_c, gm_alpha):
    code = SOLUTION_TEMPLATE.format(
        dw_a=dw_a, dw_b=dw_b, dw_c=dw_c, dw_alpha=dw_alpha,
        gm_a=gm_a, gm_b=gm_b, gm_c=gm_c, gm_alpha=gm_alpha,
    )
    with open(SOLUTION_PATH, 'w') as f:
        f.write(code)

    result = subprocess.run(
        ["uv", "run", "python3", EVALUATOR, "--solution", SOLUTION_PATH, "--local"],
        capture_output=True, text=True, cwd=WORKTREE, timeout=600,
    )

    metric = None
    tau_info = {}
    for line in result.stdout.splitlines():
        m = re.match(r'METRIC=(.+)', line)
        if m:
            metric = float(m.group(1))
    for line in result.stderr.splitlines():
        m = re.match(r'\s+(\w+): \[.+\]\s+mean=(.+)', line)
        if m:
            tau_info[m.group(1)] = float(m.group(2))

    return metric, tau_info


if __name__ == "__main__":
    # Current best: dw(1.0, 4.0, 0.06, 3.0) + gm(0.7, 1.0, 0.06, 1.0) = 49.54
    # Try variations on doublewell
    combos = [
        # label, dw_params, gm_params
        ("baseline", (1.0, 4.0, 0.06, 3.0), (0.7, 1.0, 0.06, 1.0)),
        ("dw:a=1.2,b=4.0", (1.2, 4.0, 0.06, 3.0), (0.7, 1.0, 0.06, 1.0)),
        ("dw:a=1.0,b=5.0", (1.0, 5.0, 0.06, 3.0), (0.7, 1.0, 0.06, 1.0)),
        ("dw:a=1.0,b=4.0,c=0.03", (1.0, 4.0, 0.03, 3.0), (0.7, 1.0, 0.06, 1.0)),
        ("dw:a=0.8,b=5.0", (0.8, 5.0, 0.06, 3.0), (0.7, 1.0, 0.06, 1.0)),
        # Try gaussmix variations
        ("gm:a=0.6,b=1.0", (1.0, 4.0, 0.06, 3.0), (0.6, 1.0, 0.06, 1.0)),
        ("gm:a=0.9,b=1.0", (1.0, 4.0, 0.06, 3.0), (0.9, 1.0, 0.06, 1.0)),
        ("gm:b=0.8", (1.0, 4.0, 0.06, 3.0), (0.7, 0.8, 0.06, 1.0)),
        ("gm:b=1.2", (1.0, 4.0, 0.06, 3.0), (0.7, 1.2, 0.06, 1.0)),
    ]

    print(f"{'Label':<30} {'METRIC':>8} {'tau_h':>7} {'tau_dw':>7} {'tau_gm':>7}")
    print("-" * 65)

    for label, dw, gm in combos:
        metric, taus = run_eval(*dw, *gm)
        tau_h = taus.get('harmonic_1d', -1)
        tau_dw = taus.get('doublewell_2d', -1)
        tau_gm = taus.get('gaussmix_2d', -1)
        print(f"{label:<30} {metric:>8.2f} {tau_h:>7.2f} {tau_dw:>7.2f} {tau_gm:>7.2f}")

    # Cleanup
    os.remove(SOLUTION_PATH)
    print("\nDone.")
