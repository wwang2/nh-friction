#!/usr/bin/env python3
"""
Compute the finite-difference Hessian of the weighted tau_int metric
w.r.t. Pade parameters (a, b, c) at the known optimum (0.7, 3.0, 0.06).

Strategy:
  - Central finite differences for diagonal: H_ii = (f(+h) - 2*f(0) + f(-h)) / h^2
  - Central finite differences for off-diagonal: H_ij = (f(+hi,+hj) - f(+hi,-hj) - f(-hi,+hj) + f(-hi,-hj)) / (4*hi*hj)
  - This requires: 1 center + 6 axis-aligned + 6 corner = 13 evaluations (but center is reused)
  - Actually: 6 axis (+-h for each of 3 params) + 6 off-diag corners + 1 center = 13 total

We parallelize all 13 evaluations using concurrent.futures since each dispatches to Modal independently.
"""

import subprocess
import sys
import tempfile
import concurrent.futures
import numpy as np
import os
import json
import time

WORKTREE = "/Users/wujiewang/code/bath/.worktrees/005-hessian-landscape"
ORBIT_DIR = os.path.join(WORKTREE, "orbits/005-hessian-landscape")
EVALUATOR = os.path.join(WORKTREE, "research/eval/evaluator.py")

# Known optimum
THETA_0 = np.array([0.7, 3.0, 0.06])
PARAM_NAMES = ["a", "b", "c"]

# Step sizes for finite differences
# a=0.7 -> h=0.02 (~3%), b=3.0 -> h=0.1 (~3%), c=0.06 -> h=0.005 (~8%)
STEP_SIZES = np.array([0.02, 0.1, 0.005])


def make_solution_code(a, b, c):
    """Generate solution.py code for given parameters."""
    return f'''"""Auto-generated Pade friction for Hessian sweep. a={a}, b={b}, c={c}"""
import numpy as np

_a = {a}
_b = {b}
_c = {c}

def friction_function(xi: np.ndarray) -> np.ndarray:
    xi2 = xi * xi
    num = _a + _b * xi2
    den = 1.0 + _c * xi2
    return xi * num / den

def friction_derivative(xi: np.ndarray) -> np.ndarray:
    xi2 = xi * xi
    u = xi2
    N = _a + _b * u
    D = 1.0 + _c * u
    D2 = D * D
    return N / D + 2.0 * u * (_b * D - N * _c) / D2

def setup(seed: int = 42) -> None:
    pass
'''


def evaluate_params(a, b, c, label=""):
    """Write a temp solution.py, run evaluator, parse METRIC."""
    code = make_solution_code(a, b, c)

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', dir=ORBIT_DIR, delete=False, prefix='tmp_sol_'
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        t0 = time.time()
        result = subprocess.run(
            ["uv", "run", "python3", EVALUATOR, "--solution", tmp_path, "--seed", "42", "--local"],
            capture_output=True, text=True, timeout=600,
            cwd=WORKTREE
        )
        elapsed = time.time() - t0

        metric = float('inf')
        for line in result.stdout.splitlines():
            if line.startswith("METRIC="):
                val = line.split("=")[1].strip()
                metric = float(val)
                break

        print(f"  [{label}] a={a:.4f} b={b:.4f} c={c:.5f} -> metric={metric:.4f} ({elapsed:.1f}s)", flush=True)
        return metric
    except subprocess.TimeoutExpired:
        print(f"  [{label}] TIMEOUT a={a:.4f} b={b:.4f} c={c:.5f}", flush=True)
        return float('inf')
    except Exception as e:
        print(f"  [{label}] ERROR a={a:.4f} b={b:.4f} c={c:.5f}: {e}", flush=True)
        return float('inf')
    finally:
        os.unlink(tmp_path)


def build_evaluation_points():
    """Build all (a,b,c,label) tuples needed for the Hessian."""
    points = []
    a0, b0, c0 = THETA_0
    ha, hb, hc = STEP_SIZES

    # Center point
    points.append((a0, b0, c0, "center"))

    # Axis-aligned perturbations (6 points)
    for i, (name, h) in enumerate(zip(PARAM_NAMES, STEP_SIZES)):
        theta_p = THETA_0.copy(); theta_p[i] += h
        theta_m = THETA_0.copy(); theta_m[i] -= h
        points.append((*theta_p, f"+{name}"))
        points.append((*theta_m, f"-{name}"))

    # Off-diagonal corner points (12 points, but we only need 4 per pair = 12 total for 3 pairs)
    # H_ij uses: f(+i,+j), f(+i,-j), f(-i,+j), f(-i,-j)
    # But f(+i,+j) might overlap with axis points only if j=0, which it doesn't.
    for i in range(3):
        for j in range(i+1, 3):
            hi, hj = STEP_SIZES[i], STEP_SIZES[j]
            ni, nj = PARAM_NAMES[i], PARAM_NAMES[j]

            for si in [+1, -1]:
                for sj in [+1, -1]:
                    theta = THETA_0.copy()
                    theta[i] += si * hi
                    theta[j] += sj * hj
                    sign_i = "+" if si > 0 else "-"
                    sign_j = "+" if sj > 0 else "-"
                    points.append((*theta, f"{sign_i}{ni},{sign_j}{nj}"))

    return points


def compute_hessian(results):
    """Compute Hessian from evaluation results dict {label: metric}."""
    f0 = results["center"]
    H = np.zeros((3, 3))

    # Diagonal elements: H_ii = (f(+h) - 2*f(0) + f(-h)) / h^2
    for i, name in enumerate(PARAM_NAMES):
        fp = results[f"+{name}"]
        fm = results[f"-{name}"]
        h = STEP_SIZES[i]
        H[i, i] = (fp - 2*f0 + fm) / (h * h)

    # Off-diagonal: H_ij = (f(+i,+j) - f(+i,-j) - f(-i,+j) + f(-i,-j)) / (4*hi*hj)
    for i in range(3):
        for j in range(i+1, 3):
            ni, nj = PARAM_NAMES[i], PARAM_NAMES[j]
            hi, hj = STEP_SIZES[i], STEP_SIZES[j]

            fpp = results[f"+{ni},+{nj}"]
            fpm = results[f"+{ni},-{nj}"]
            fmp = results[f"-{ni},+{nj}"]
            fmm = results[f"-{ni},-{nj}"]

            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * hi * hj)
            H[j, i] = H[i, j]  # symmetric

    return H


def main():
    print("=" * 60)
    print("Hessian Landscape Analysis")
    print(f"Center: a={THETA_0[0]}, b={THETA_0[1]}, c={THETA_0[2]}")
    print(f"Steps:  ha={STEP_SIZES[0]}, hb={STEP_SIZES[1]}, hc={STEP_SIZES[2]}")
    print("=" * 60)

    points = build_evaluation_points()
    print(f"\nTotal evaluation points: {len(points)}")

    # Run all evaluations in parallel (each dispatches to Modal independently)
    results = {}
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for a, b, c, label in points:
            future = executor.submit(evaluate_params, a, b, c, label)
            futures[future] = label

        for future in concurrent.futures.as_completed(futures):
            label = futures[future]
            try:
                metric = future.result()
                results[label] = metric
            except Exception as e:
                print(f"  [{label}] FAILED: {e}")
                results[label] = float('inf')

    t_total = time.time() - t_start
    print(f"\nAll {len(points)} evaluations completed in {t_total:.1f}s")

    # Check for any inf results
    inf_labels = [k for k, v in results.items() if not np.isfinite(v)]
    if inf_labels:
        print(f"\nWARNING: {len(inf_labels)} evaluations returned inf: {inf_labels}")
        print("Hessian computation may be unreliable.")

    # Compute Hessian
    H = compute_hessian(results)

    print("\n" + "=" * 60)
    print("HESSIAN MATRIX:")
    print("=" * 60)
    print(f"Parameters: {PARAM_NAMES}")
    for i in range(3):
        row = "  ".join(f"{H[i,j]:10.2f}" for j in range(3))
        print(f"  [{row}]")

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    print("\nEIGENVALUES:")
    for i, ev in enumerate(eigenvalues):
        print(f"  lambda_{i} = {ev:.4f}")

    print("\nEIGENVECTORS (columns):")
    for i in range(3):
        v = eigenvectors[:, i]
        components = ", ".join(f"{PARAM_NAMES[j]}={v[j]:.4f}" for j in range(3))
        print(f"  v_{i} (lambda={eigenvalues[i]:.4f}): [{components}]")

    # Gradient estimation (from axis-aligned evaluations)
    grad = np.zeros(3)
    for i, name in enumerate(PARAM_NAMES):
        fp = results[f"+{name}"]
        fm = results[f"-{name}"]
        h = STEP_SIZES[i]
        grad[i] = (fp - fm) / (2 * h)

    print(f"\nGRADIENT at optimum:")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  d(metric)/d{name} = {grad[i]:.4f}")

    # Save results for plotting
    output = {
        "center_metric": results["center"],
        "all_results": {k: v for k, v in results.items()},
        "hessian": H.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "eigenvectors": eigenvectors.tolist(),
        "gradient": grad.tolist(),
        "theta_0": THETA_0.tolist(),
        "step_sizes": STEP_SIZES.tolist(),
        "param_names": PARAM_NAMES,
    }

    outpath = os.path.join(ORBIT_DIR, "hessian_results.json")
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")

    # Identify soft directions
    print("\n" + "=" * 60)
    print("LANDSCAPE ANALYSIS:")
    print("=" * 60)

    soft_threshold = 50.0  # eigenvalue threshold for "soft"
    for i, ev in enumerate(eigenvalues):
        v = eigenvectors[:, i]
        direction_str = ", ".join(f"d{PARAM_NAMES[j]}={v[j]:+.4f}" for j in range(3))
        if ev < soft_threshold:
            print(f"  SOFT direction (lambda={ev:.2f}): [{direction_str}]")
        else:
            print(f"  STIFF direction (lambda={ev:.2f}): [{direction_str}]")

    condition_number = max(abs(eigenvalues)) / max(abs(min(eigenvalues)), 1e-10)
    print(f"\nCondition number: {condition_number:.1f}")
    print(f"Center metric: {results['center']:.4f}")


if __name__ == "__main__":
    main()
