"""Quick parameter sweep for Pade friction with pure kinetic driving.

Tests focused variations around the known optimum (a=0.7, b=3.0, c=0.06).
"""
import subprocess
import sys
import tempfile
import os
import re

TEMPLATE = '''"""Auto-generated solution: a={a}, b={b}, c={c}"""
import numpy as np

_a = {a}
_b = {b}
_c = {c}

def friction_function(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=float)
    xi2 = xi * xi
    return xi * (_a + _b * xi2) / (1.0 + _c * xi2)

def friction_derivative(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=float)
    xi2 = xi * xi
    num = _a + _b * xi2
    den = 1.0 + _c * xi2
    dnum = 2.0 * _b * xi
    dden = 2.0 * _c * xi
    return num / den + xi * (dnum * den - num * dden) / (den * den)
'''

# Focused parameter grid
params = [
    # stronger linear + cubic
    (1.0, 5.0, 0.04),
    (1.2, 5.0, 0.03),
    (1.5, 5.0, 0.02),
    (1.0, 6.0, 0.03),
    (1.0, 8.0, 0.02),
    (1.5, 8.0, 0.01),
]

results = []
for a, b, c in params:
    code = TEMPLATE.format(a=a, b=b, c=c)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, dir="/tmp") as f:
        f.write(code)
        tmp = f.name

    try:
        proc = subprocess.run(
            ["uv", "run", "python3", "research/eval/evaluator.py",
             "--solution", tmp, "--local"],
            capture_output=True, text=True, timeout=600,
            cwd="/Users/wujiewang/code/bath"
        )
        output = proc.stdout + proc.stderr
        metric = float("inf")
        for line in output.split("\n"):
            if line.startswith("METRIC="):
                val = line.split("=")[1]
                metric = float(val) if val != "inf" else float("inf")

        # Extract per-potential tau means
        tau_info = ""
        for line in output.split("\n"):
            if "mean=" in line and ("harmonic" in line or "doublewell" in line or "gaussmix" in line):
                tau_info += line.strip() + " | "

        results.append((a, b, c, metric))
        print(f"a={a:.1f} b={b:.1f} c={c:.2f} -> METRIC={metric:.2f}  {tau_info}")
        sys.stdout.flush()
    except subprocess.TimeoutExpired:
        print(f"a={a:.1f} b={b:.1f} c={c:.2f} -> TIMEOUT")
        results.append((a, b, c, float("inf")))
    finally:
        os.unlink(tmp)

print("\n=== SORTED RESULTS ===")
results.sort(key=lambda x: x[3])
for a, b, c, metric in results:
    print(f"a={a:.2f} b={b:.2f} c={c:.3f} -> METRIC={metric:.4f}")
