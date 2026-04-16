#!/usr/bin/env python3
"""
Compute diagonal Hessian + gradient of the metric w.r.t. Pade parameters.

Only 7 evaluations needed: center + 2 per parameter (+-h).
This gives:
  - Gradient: g_i = (f(+h_i) - f(-h_i)) / (2*h_i)
  - Diagonal Hessian: H_ii = (f(+h_i) - 2*f(0) + f(-h_i)) / h_i^2

Runs sequentially to avoid CPU contention on local execution.
"""

import subprocess
import sys
import tempfile
import numpy as np
import os
import json
import time

WORKTREE = "/Users/wujiewang/code/bath/.worktrees/005-hessian-landscape"
ORBIT_DIR = os.path.join(WORKTREE, "orbits/005-hessian-landscape")
EVALUATOR = os.path.join(WORKTREE, "research/eval/evaluator.py")

THETA_0 = np.array([0.7, 3.0, 0.06])
PARAM_NAMES = ["a", "b", "c"]
STEP_SIZES = np.array([0.02, 0.1, 0.005])


def make_solution_code(a, b, c):
    return f'''import numpy as np

_a = {a}
_b = {b}
_c = {c}

def friction_function(xi: np.ndarray) -> np.ndarray:
    xi2 = xi * xi
    return xi * (_a + _b * xi2) / (1.0 + _c * xi2)

def friction_derivative(xi: np.ndarray) -> np.ndarray:
    xi2 = xi * xi
    N = _a + _b * xi2
    D = 1.0 + _c * xi2
    return N / D + 2.0 * xi2 * (_b * D - N * _c) / (D * D)

def setup(seed: int = 42) -> None:
    pass
'''


def evaluate_params(a, b, c, label=""):
    code = make_solution_code(a, b, c)
    # Use /tmp to avoid git issues
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir='/tmp', delete=False, prefix='hess_') as f:
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
                metric = float(line.split("=")[1].strip())
                break

        # Also parse per-potential taus from stderr
        per_pot = {}
        for line in result.stderr.splitlines():
            for pot in ["harmonic_1d", "doublewell_2d", "gaussmix_2d"]:
                if pot in line and "mean=" in line:
                    mean_val = float(line.split("mean=")[1])
                    per_pot[pot] = mean_val

        print(f"  [{label:>8s}] a={a:.4f} b={b:.4f} c={c:.5f} -> metric={metric:.4f} ({elapsed:.0f}s)")
        sys.stdout.flush()
        return metric, per_pot, elapsed
    except Exception as e:
        print(f"  [{label:>8s}] ERROR: {e}")
        sys.stdout.flush()
        return float('inf'), {}, 0
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


def main():
    print("=" * 70)
    print("Diagonal Hessian + Gradient Analysis")
    print(f"Center: a={THETA_0[0]}, b={THETA_0[1]}, c={THETA_0[2]}")
    print(f"Steps:  ha={STEP_SIZES[0]}, hb={STEP_SIZES[1]}, hc={STEP_SIZES[2]}")
    print("=" * 70)

    # We already know center=84.1421 and +a=100.3394 from partial sweep
    # But let's re-run center for consistency and run all 7 points
    results = {}
    per_pot_results = {}
    timings = {}

    # Evaluation order: center first, then +/- for each param
    points = [
        ("center", THETA_0),
    ]
    for i, name in enumerate(PARAM_NAMES):
        theta_p = THETA_0.copy(); theta_p[i] += STEP_SIZES[i]
        theta_m = THETA_0.copy(); theta_m[i] -= STEP_SIZES[i]
        points.append((f"+{name}", theta_p))
        points.append((f"-{name}", theta_m))

    print(f"\nRunning {len(points)} evaluations sequentially...")
    t_start = time.time()

    for label, theta in points:
        metric, per_pot, elapsed = evaluate_params(*theta, label=label)
        results[label] = metric
        per_pot_results[label] = per_pot
        timings[label] = elapsed

    t_total = time.time() - t_start
    print(f"\nAll {len(points)} evaluations completed in {t_total:.0f}s")

    # Compute gradient
    f0 = results["center"]
    grad = np.zeros(3)
    for i, name in enumerate(PARAM_NAMES):
        fp = results[f"+{name}"]
        fm = results[f"-{name}"]
        h = STEP_SIZES[i]
        grad[i] = (fp - fm) / (2 * h)

    # Compute diagonal Hessian
    H_diag = np.zeros(3)
    for i, name in enumerate(PARAM_NAMES):
        fp = results[f"+{name}"]
        fm = results[f"-{name}"]
        h = STEP_SIZES[i]
        H_diag[i] = (fp - 2*f0 + fm) / (h * h)

    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Center metric: {f0:.4f}")
    print()

    print(f"{'Param':>8s}  {'f(+h)':>10s}  {'f(-h)':>10s}  {'Gradient':>10s}  {'H_diag':>10s}  {'Verdict':>12s}")
    print("-" * 70)
    for i, name in enumerate(PARAM_NAMES):
        fp = results[f"+{name}"]
        fm = results[f"-{name}"]
        verdict = "STIFF" if H_diag[i] > 100 else "MODERATE" if H_diag[i] > 10 else "SOFT"
        print(f"{name:>8s}  {fp:10.4f}  {fm:10.4f}  {grad[i]:+10.4f}  {H_diag[i]:10.2f}  {verdict:>12s}")

    print()
    print(f"Gradient magnitude: |g| = {np.linalg.norm(grad):.4f}")

    # Approximate eigenvalues (diagonal Hessian = eigenvalues if off-diagonal small)
    idx_sorted = np.argsort(H_diag)
    print(f"\nSorted curvatures (ascending):")
    for i in idx_sorted:
        print(f"  {PARAM_NAMES[i]}: H_ii = {H_diag[i]:.2f}, gradient = {grad[i]:+.4f}")

    # Check if any direction shows a descent opportunity
    print(f"\n{'='*70}")
    print("DESCENT OPPORTUNITY ANALYSIS:")
    print(f"{'='*70}")
    for i in range(3):
        # Newton step along this direction: delta_i = -g_i / H_ii
        if H_diag[i] > 0:
            newton_step = -grad[i] / H_diag[i]
            predicted_improvement = -0.5 * grad[i]**2 / H_diag[i]
            theta_newton = THETA_0.copy()
            theta_newton[i] += newton_step
            print(f"  {PARAM_NAMES[i]}: Newton step = {newton_step:+.6f}, predicted delta(metric) = {predicted_improvement:.4f}")
            print(f"    -> theta_new = ({theta_newton[0]:.4f}, {theta_newton[1]:.4f}, {theta_newton[2]:.5f})")
        else:
            print(f"  {PARAM_NAMES[i]}: Negative curvature H_ii = {H_diag[i]:.2f} -- saddle point direction!")

    # Per-potential breakdown
    print(f"\n{'='*70}")
    print("PER-POTENTIAL TAU BREAKDOWN:")
    print(f"{'='*70}")
    for label in results:
        pp = per_pot_results.get(label, {})
        if pp:
            pp_str = ", ".join(f"{k}={v:.1f}" for k, v in pp.items())
            print(f"  {label:>15s}: metric={results[label]:.2f}  [{pp_str}]")

    # Save results
    # Construct a full 3x3 Hessian (diagonal only, off-diagonal = 0)
    H = np.diag(H_diag)

    output = {
        "center_metric": f0,
        "all_results": results,
        "per_pot_results": per_pot_results,
        "hessian": H.tolist(),
        "hessian_diagonal": H_diag.tolist(),
        "eigenvalues": sorted(H_diag.tolist()),  # diagonal = eigenvalues
        "eigenvectors": np.eye(3).tolist(),  # axis-aligned for diagonal Hessian
        "gradient": grad.tolist(),
        "theta_0": THETA_0.tolist(),
        "step_sizes": STEP_SIZES.tolist(),
        "param_names": PARAM_NAMES,
        "timings": timings,
    }

    outpath = os.path.join(ORBIT_DIR, "hessian_results.json")
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
