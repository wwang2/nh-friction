#!/usr/bin/env python3
"""
Hill-climb along soft eigenvector directions identified by the Hessian analysis.

After computing the Hessian at the known optimum (a=0.7, b=3.0, c=0.06),
this script:
1. Loads the eigendecomposition
2. Identifies soft directions (eigenvalue < threshold)
3. Does a line search along each soft direction
4. Reports the best parameters found
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


def make_solution_code(a, b, c):
    """Generate solution.py code for given parameters."""
    return f'''"""Auto-generated Pade friction for hill-climb. a={a}, b={b}, c={c}"""
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
    # Validate params are physically reasonable
    if a <= 0 or b <= 0 or c < 0:
        print(f"  [{label}] SKIP: invalid params a={a:.4f} b={b:.4f} c={c:.5f}", flush=True)
        return float('inf')

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
        print(f"  [{label}] TIMEOUT", flush=True)
        return float('inf')
    except Exception as e:
        print(f"  [{label}] ERROR: {e}", flush=True)
        return float('inf')
    finally:
        os.unlink(tmp_path)


def main():
    # Load Hessian results
    results_path = os.path.join(ORBIT_DIR, "hessian_results.json")
    if not os.path.exists(results_path):
        print("ERROR: hessian_results.json not found. Run hessian_sweep.py first.")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    theta_0 = np.array(data["theta_0"])
    eigenvalues = np.array(data["eigenvalues"])
    eigenvectors = np.array(data["eigenvectors"])
    gradient = np.array(data["gradient"])
    center_metric = data["center_metric"]
    param_names = data["param_names"]

    print("=" * 60)
    print("Hill-Climb Along Soft Directions")
    print(f"Center: a={theta_0[0]}, b={theta_0[1]}, c={theta_0[2]}")
    print(f"Center metric: {center_metric:.4f}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Gradient: {gradient}")
    print("=" * 60)

    best_metric = center_metric
    best_params = theta_0.copy()
    climb_log = []

    # For each eigenvector, try stepping in both +/- directions with various step sizes
    for ev_idx in range(3):
        ev = eigenvalues[ev_idx]
        v = eigenvectors[:, ev_idx]

        print(f"\n--- Eigenvector {ev_idx}: lambda={ev:.2f} ---")
        print(f"    Direction: {', '.join(f'd{param_names[k]}={v[k]:+.4f}' for k in range(3))}")

        # Adaptive step sizes: smaller for stiff, larger for soft
        if ev < 10:
            scales = [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]
        elif ev < 100:
            scales = [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]
        else:
            scales = [-0.5, -0.3, 0.3, 0.5]

        # Also try gradient-informed direction: step along -gradient projected onto this eigenvector
        grad_proj = np.dot(gradient, v)
        if abs(grad_proj) > 0.1:
            # Add a gradient-descent step scaled by 1/eigenvalue
            gd_scale = -grad_proj / max(ev, 1.0)
            if abs(gd_scale) > 0.1 and abs(gd_scale) < 5.0:
                scales.append(gd_scale)
                print(f"    Added gradient-informed scale: {gd_scale:.3f}")

        # Use step_sizes norm as the base step magnitude
        base_step = np.linalg.norm(data["step_sizes"])

        for scale in scales:
            theta_new = theta_0 + scale * base_step * v

            # Check parameter validity
            if theta_new[0] <= 0 or theta_new[1] <= 0 or theta_new[2] < 0:
                print(f"    scale={scale:+.1f}: SKIP (invalid params: a={theta_new[0]:.3f} b={theta_new[1]:.3f} c={theta_new[2]:.4f})")
                continue

            label = f"ev{ev_idx}_s{scale:+.1f}"
            metric = evaluate_params(*theta_new, label=label)

            entry = {
                "eigenvector_idx": ev_idx,
                "eigenvalue": ev,
                "scale": scale,
                "params": theta_new.tolist(),
                "metric": metric,
            }
            climb_log.append(entry)

            if np.isfinite(metric) and metric < best_metric:
                best_metric = metric
                best_params = theta_new.copy()
                print(f"    *** NEW BEST: metric={metric:.4f} at a={theta_new[0]:.4f} b={theta_new[1]:.4f} c={theta_new[2]:.5f}")

    # Also try along the negative gradient direction directly
    print(f"\n--- Direct gradient descent ---")
    grad_norm = np.linalg.norm(gradient)
    if grad_norm > 0.01:
        grad_dir = -gradient / grad_norm
        for scale in [0.5, 1.0, 2.0, 3.0, 5.0]:
            base_step = np.linalg.norm(data["step_sizes"])
            theta_new = theta_0 + scale * base_step * grad_dir

            if theta_new[0] <= 0 or theta_new[1] <= 0 or theta_new[2] < 0:
                continue

            label = f"grad_s{scale:.1f}"
            metric = evaluate_params(*theta_new, label=label)

            entry = {
                "direction": "gradient_descent",
                "scale": scale,
                "params": theta_new.tolist(),
                "metric": metric,
            }
            climb_log.append(entry)

            if np.isfinite(metric) and metric < best_metric:
                best_metric = metric
                best_params = theta_new.copy()
                print(f"    *** NEW BEST: metric={metric:.4f} at a={theta_new[0]:.4f} b={theta_new[1]:.4f} c={theta_new[2]:.5f}")

    print(f"\n{'='*60}")
    print(f"HILL-CLIMB RESULTS:")
    print(f"  Center metric: {center_metric:.4f}")
    print(f"  Best metric:   {best_metric:.4f}")
    print(f"  Best params:   a={best_params[0]:.4f} b={best_params[1]:.4f} c={best_params[2]:.5f}")
    print(f"  Improvement:   {center_metric - best_metric:.4f} ({(center_metric - best_metric)/center_metric*100:.2f}%)")
    print(f"{'='*60}")

    # Save results
    output = {
        "center_metric": center_metric,
        "best_metric": best_metric,
        "best_params": best_params.tolist(),
        "climb_log": climb_log,
    }
    outpath = os.path.join(ORBIT_DIR, "hillclimb_results.json")
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
