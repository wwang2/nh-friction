"""Sweep lambda for hybrid kinetic-configurational driving.

h = lambda * |p|^2/m + (1-lambda) * |grad_V|^2 / (d * E_ref)

This technically breaks canonical invariance but for small (1-lambda),
the KL divergence may stay below 0.05.
"""
import subprocess
import sys
import tempfile
import os

TEMPLATE = '''"""Auto-generated: lambda={lam}, a={a}, b={b}, c={c}"""
import numpy as np

_a = {a}
_b = {b}
_c = {c}

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

_lam = {lam}
_E_ref = 1.0
_n = 0
_accum = 0.0
_warmup_done = False

def setup(seed):
    global _E_ref, _n, _accum, _warmup_done
    _E_ref = 1.0
    _n = 0
    _accum = 0.0
    _warmup_done = False

def driving_function(q, p, grad_V):
    global _E_ref, _n, _accum, _warmup_done
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    grad_V = np.asarray(grad_V, dtype=float)
    d = len(q)
    pp = float(np.dot(p, p))
    gv2 = float(np.dot(grad_V, grad_V))

    if not _warmup_done:
        _n += 1
        _accum += gv2 / d
        if _n >= 5000:
            _E_ref = max(_accum / _n, 1e-6)
            _warmup_done = True
        return pp

    alpha = 0.001
    _E_ref = max((1-alpha)*_E_ref + alpha*(gv2/d), 1e-6)
    return _lam * pp + (1-_lam) * d * 1.0 * (gv2/d) / _E_ref
'''

# Lambda values to test
lambdas = [1.0, 0.99, 0.97, 0.95, 0.90]

a, b, c = 0.7, 3.0, 0.06

results = []
for lam in lambdas:
    code = TEMPLATE.format(lam=lam, a=a, b=b, c=c)
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

        # Extract KL and tau info
        info = ""
        for line in output.split("\n"):
            if "mean=" in line and ("harmonic" in line or "doublewell" in line or "gaussmix" in line):
                info += line.strip() + "\n"
            if "kl_per_potential" in line or ("harmonic" in line and "0." in line and "kl" not in line.lower()):
                pass
        for line in output.split("\n"):
            if line.strip().startswith("harmonic_1d:") or line.strip().startswith("doublewell_2d:") or line.strip().startswith("gaussmix_2d:"):
                info += "  " + line.strip() + "\n"

        results.append((lam, metric))
        print(f"\nlambda={lam:.2f} -> METRIC={metric:.4f}")
        print(info)
        sys.stdout.flush()
    except subprocess.TimeoutExpired:
        print(f"lambda={lam:.2f} -> TIMEOUT")
        results.append((lam, float("inf")))
    finally:
        os.unlink(tmp)

print("\n=== SORTED ===")
results.sort(key=lambda x: x[1])
for lam, metric in results:
    print(f"lambda={lam:.2f} -> METRIC={metric:.4f}")
