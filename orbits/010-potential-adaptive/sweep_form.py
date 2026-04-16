"""Sweep alternative g(xi) functional forms.

Test 1: Pure polynomial g = a*xi + b*xi^3 (no denominator damping)
Test 2: Boosted Pade g = xi*(a+b*xi^2)/(1+c*xi^2) + d*xi^3/(1+e*xi^4)
Test 3: Double-linear Pade g = xi*(a+b*xi^2+f*xi^4)/(1+c*xi^2+g*xi^4)
Test 4: Modified power g = a*xi*(1+b*xi^2)^alpha for fractional alpha
"""
import subprocess
import sys
import tempfile
import os

def run_solution(code, label):
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
        info = ""
        for line in output.split("\n"):
            if "mean=" in line and ("harmonic" in line or "doublewell" in line or "gaussmix" in line):
                info += line.strip() + " | "
        print(f"{label} -> METRIC={metric:.2f}  {info}")
        sys.stdout.flush()
        return metric
    except subprocess.TimeoutExpired:
        print(f"{label} -> TIMEOUT")
        return float("inf")
    finally:
        os.unlink(tmp)

# Test 1: Pade with different c values (less damping at large xi)
code_pade = '''import numpy as np
_a, _b, _c = {a}, {b}, {c}
def friction_function(xi):
    xi = np.asarray(xi, dtype=float)
    xi2 = xi*xi
    return xi*(_a + _b*xi2)/(1.0 + _c*xi2)
def friction_derivative(xi):
    xi = np.asarray(xi, dtype=float)
    xi2 = xi*xi
    num = _a + _b*xi2; den = 1.0 + _c*xi2
    return num/den + xi*(2*_b*xi*den - num*2*_c*xi)/(den*den)
'''

# Test 2: Boosted Pade with extra mid-range term
code_boosted = '''import numpy as np
_a, _b, _c, _d, _e = {a}, {b}, {c}, {d}, {e}
def friction_function(xi):
    xi = np.asarray(xi, dtype=float)
    xi2 = xi*xi
    base = xi*(_a + _b*xi2)/(1.0 + _c*xi2)
    boost = _d*xi*xi2/(1.0 + _e*xi2*xi2)
    return base + boost
def friction_derivative(xi):
    xi = np.asarray(xi, dtype=float)
    xi2 = xi*xi
    # base derivative
    num = _a + _b*xi2; den = 1.0 + _c*xi2
    dbase = num/den + xi*(2*_b*xi*den - num*2*_c*xi)/(den*den)
    # boost derivative: d/dxi [d*xi^3/(1+e*xi^4)]
    n2 = _d*xi2*xi; d2 = 1.0 + _e*xi2*xi2
    dboost = (3*_d*xi2*d2 - n2*4*_e*xi*xi2)/(d2*d2)
    return dbase + dboost
'''

# Test 3: Sum of two Pade terms
code_twopade = '''import numpy as np
_a1, _b1, _c1 = {a1}, {b1}, {c1}
_a2, _b2, _c2 = {a2}, {b2}, {c2}
def friction_function(xi):
    xi = np.asarray(xi, dtype=float)
    xi2 = xi*xi
    return xi*(_a1 + _b1*xi2)/(1.0 + _c1*xi2) + xi*(_a2 + _b2*xi2)/(1.0 + _c2*xi2)
def friction_derivative(xi):
    xi = np.asarray(xi, dtype=float)
    xi2 = xi*xi
    def dpade(a,b,c):
        num = a + b*xi2; den = 1.0 + c*xi2
        return num/den + xi*(2*b*xi*den - num*2*c*xi)/(den*den)
    return dpade(_a1,_b1,_c1) + dpade(_a2,_b2,_c2)
'''

# Run the tests
print("=== Pade parameter refinement (near baseline) ===")
# Slightly lower c = less damping at large xi = stronger friction
for a, b, c in [(0.7, 3.0, 0.04), (0.7, 3.0, 0.03), (0.7, 3.0, 0.02),
                 (0.7, 4.0, 0.06), (0.7, 4.0, 0.04),
                 (0.8, 3.5, 0.05), (0.6, 3.0, 0.06)]:
    run_solution(code_pade.format(a=a, b=b, c=c), f"Pade a={a} b={b} c={c}")

print("\n=== Boosted Pade ===")
for a, b, c, d, e in [(0.7, 3.0, 0.06, 1.0, 0.1), (0.7, 3.0, 0.06, 2.0, 0.05),
                       (0.7, 3.0, 0.06, 0.5, 0.2)]:
    run_solution(code_boosted.format(a=a, b=b, c=c, d=d, e=e), f"Boosted d={d} e={e}")

print("\n=== Two-Pade sum ===")
for a1,b1,c1,a2,b2,c2 in [(0.7, 3.0, 0.06, 0.3, 0.0, 1.0),
                            (0.5, 3.0, 0.06, 0.3, 1.0, 0.5)]:
    run_solution(code_twopade.format(a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2),
                 f"TwoPade ({a1},{b1},{c1})+({a2},{b2},{c2})")
