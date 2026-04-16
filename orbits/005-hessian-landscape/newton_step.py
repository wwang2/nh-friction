#!/usr/bin/env python3
"""
Try the Newton-step-adjusted parameters from the Hessian analysis.
Also try a combined Newton step using all three gradients.
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
    if a <= 0 or b <= 0 or c < 0:
        print(f"  [{label:>15s}] SKIP: invalid params a={a:.4f} b={b:.4f} c={c:.5f}")
        sys.stdout.flush()
        return float('inf')

    code = make_solution_code(a, b, c)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir='/tmp', delete=False, prefix='newton_') as f:
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

        per_pot = {}
        for line in result.stderr.splitlines():
            for pot in ["harmonic_1d", "doublewell_2d", "gaussmix_2d"]:
                if pot in line and "mean=" in line:
                    per_pot[pot] = float(line.split("mean=")[1])

        print(f"  [{label:>15s}] a={a:.4f} b={b:.4f} c={c:.5f} -> metric={metric:.4f} ({elapsed:.0f}s)")
        if per_pot:
            print(f"                   h1d={per_pot.get('harmonic_1d', 0):.1f} dw={per_pot.get('doublewell_2d', 0):.1f} gm={per_pot.get('gaussmix_2d', 0):.1f}")
        sys.stdout.flush()
        return metric
    except Exception as e:
        print(f"  [{label:>15s}] ERROR: {e}")
        sys.stdout.flush()
        return float('inf')
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


def main():
    # Load Hessian results
    with open(os.path.join(ORBIT_DIR, "hessian_results.json")) as f:
        data = json.load(f)

    theta_0 = np.array(data["theta_0"])
    grad = np.array(data["gradient"])
    H_diag = np.array(data["hessian_diagonal"])
    center_metric = data["center_metric"]

    print("=" * 70)
    print("Newton Step Exploration")
    print(f"Center: a={theta_0[0]}, b={theta_0[1]}, c={theta_0[2]}, metric={center_metric:.4f}")
    print(f"Gradient: {grad}")
    print(f"H_diag: {H_diag}")
    print("=" * 70)

    # Newton steps along individual axes
    newton_steps = -grad / H_diag
    print(f"\nIndividual Newton steps: da={newton_steps[0]:.6f}, db={newton_steps[1]:.6f}, dc={newton_steps[2]:.6f}")

    candidates = [
        ("center", theta_0),
    ]

    # Individual Newton steps
    for i, name in enumerate(["a", "b", "c"]):
        theta_new = theta_0.copy()
        theta_new[i] += newton_steps[i]
        candidates.append((f"newton_{name}", theta_new))

    # Combined Newton step (all axes)
    theta_combined = theta_0 + newton_steps
    candidates.append(("newton_all", theta_combined))

    # Half Newton steps (more conservative)
    theta_half = theta_0 + 0.5 * newton_steps
    candidates.append(("newton_half", theta_half))

    # Try scaling the b parameter more aggressively toward -gradient
    # Since b has the smallest curvature, it's the softest direction
    for scale in [0.5, 1.0, 2.0, 3.0, 5.0]:
        theta_b = theta_0.copy()
        theta_b[1] += -grad[1] / H_diag[1] * scale
        candidates.append((f"b_scale_{scale}", theta_b))

    # A few fine-tuned candidates near the gradient-informed optimum
    fine_tuned_candidates = [
        ("fine_combo1", np.array([0.702, 2.98, 0.06])),
        ("fine_combo2", np.array([0.701, 2.99, 0.0599])),
    ]
    candidates.extend(fine_tuned_candidates)

    best_metric = center_metric
    best_params = theta_0.copy()

    print(f"\nRunning {len(candidates)} evaluations sequentially...")
    for label, theta in candidates:
        metric = evaluate_params(*theta, label=label)
        if np.isfinite(metric) and metric < best_metric:
            best_metric = metric
            best_params = np.array(theta).copy()
            print(f"  *** NEW BEST: {best_metric:.4f} ***")

    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  Center: metric={center_metric:.4f}")
    print(f"  Best:   metric={best_metric:.4f}")
    print(f"  Best params: a={best_params[0]:.6f} b={best_params[1]:.6f} c={best_params[2]:.6f}")
    print(f"  Improvement: {center_metric - best_metric:.4f} ({(center_metric - best_metric)/center_metric*100:.2f}%)")
    print(f"{'='*70}")

    # Save results
    output = {
        "center_metric": center_metric,
        "best_metric": best_metric,
        "best_params": best_params.tolist(),
    }
    with open(os.path.join(ORBIT_DIR, "newton_results.json"), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {os.path.join(ORBIT_DIR, 'newton_results.json')}")


if __name__ == "__main__":
    main()
