#!/usr/bin/env python3
"""
CMA-ES joint optimization of (a, b, c, alpha) per potential.

Mini-evaluator: runs a shorter integration (200k steps, 1 seed) for fast inner loop.
Uses the same integrator as the production evaluator but with reduced budget.
"""
import math
import sys
import time
import numpy as np

# ── Physical constants ──
M = 1.0
KT = 1.0
Q = 1.0
DT = 0.01

# ── Mini-eval parameters (faster than production) ──
N_BURNIN = 5_000
N_MAIN_STEPS = 200_000
THIN = 10
N_SAMPLES = N_MAIN_STEPS // THIN  # 20,000
N_TOTAL_STEPS = N_BURNIN + N_MAIN_STEPS
TAU_CAP = 10_000
TAU_INT_FLOOR = 1.0
KL_THRESHOLD = 0.05
C_SOKAL = 6.0

# ── Potential definitions ──
def _grad_harmonic_1d(q):
    return q.copy()

def _grad_doublewell_2d(q):
    x = q[0]
    return np.array([4.0 * x * (x * x - 1.0), q[1]])

def _grad_gaussmix_2d(q):
    MU = np.array([
        [3.0 * math.cos(2.0 * math.pi * k / 5),
         3.0 * math.sin(2.0 * math.pi * k / 5)]
        for k in range(5)
    ])
    diff = q[np.newaxis, :] - MU
    log_w = -0.5 * np.sum(diff ** 2, axis=1)
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    return np.einsum("k,kd->d", w, diff)

POTENTIALS = {
    "harmonic_1d":   {"dim": 1, "grad": _grad_harmonic_1d},
    "doublewell_2d": {"dim": 2, "grad": _grad_doublewell_2d},
    "gaussmix_2d":   {"dim": 2, "grad": _grad_gaussmix_2d},
}

# ── KL marginals ──
def _marginal_harmonic_1d(bin_edges):
    lo, hi = bin_edges[:-1], bin_edges[1:]
    def _norm_cdf(x):
        return 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in x]))
    probs = _norm_cdf(hi) - _norm_cdf(lo)
    probs = np.clip(probs, 1e-300, None)
    return probs / probs.sum()

def _marginal_doublewell_2d(bin_edges):
    mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    log_p = -(mids ** 2 - 1.0) ** 2
    log_p -= log_p.max()
    p = np.exp(log_p)
    bin_w = bin_edges[1] - bin_edges[0]
    probs = p * bin_w
    probs = np.clip(probs, 1e-300, None)
    return probs / probs.sum()

def _marginal_gaussmix_2d(bin_edges):
    MU_X = [3.0 * math.cos(2.0 * math.pi * k / 5) for k in range(5)]
    lo, hi = bin_edges[:-1], bin_edges[1:]
    def _norm_cdf(x):
        return 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in x]))
    probs = np.zeros(len(lo))
    for mu_x in MU_X:
        probs += (_norm_cdf(hi - mu_x) - _norm_cdf(lo - mu_x)) / 5.0
    probs = np.clip(probs, 1e-300, None)
    return probs / probs.sum()

_MARGINALS = {
    "harmonic_1d": _marginal_harmonic_1d,
    "doublewell_2d": _marginal_doublewell_2d,
    "gaussmix_2d": _marginal_gaussmix_2d,
}

def _kl_divergence(samples_q0, potential_name):
    if len(samples_q0) == 0 or not np.all(np.isfinite(samples_q0)):
        return math.inf
    bin_edges = np.linspace(-6.0, 6.0, 101)
    counts, _ = np.histogram(np.clip(samples_q0, -6.0, 6.0), bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return math.inf
    p_emp = counts.astype(np.float64) / total
    p_ref = _MARGINALS[potential_name](bin_edges)
    kl = 0.0
    for p_i, q_i in zip(p_emp, p_ref):
        if p_i > 0.0:
            if q_i <= 0.0:
                return math.inf
            kl += p_i * math.log(p_i / q_i)
    return max(0.0, kl)

def _sokal_tau_int_fft(x, c=C_SOKAL):
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < 3:
        return float(TAU_CAP), False
    if not np.all(np.isfinite(x)):
        return float(TAU_CAP), False
    x_centered = x - x.mean()
    var = float(np.var(x_centered))
    if var == 0.0 or not math.isfinite(var):
        return float(TAU_CAP), False
    nfft = 1
    while nfft < 2 * N:
        nfft <<= 1
    F = np.fft.rfft(x_centered, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N].real
    acf = acf / (N - np.arange(N))
    C0 = float(acf[0])
    if C0 <= 0.0 or not math.isfinite(C0):
        return float(TAU_CAP), False
    rho = acf / C0
    if not np.all(np.isfinite(rho)):
        return float(TAU_CAP), False
    tau_running = 1.0
    converged = False
    for t in range(1, N):
        tau_running += 2.0 * rho[t]
        if not math.isfinite(tau_running) or tau_running < TAU_INT_FLOOR:
            tau_running = TAU_INT_FLOOR
        if t > c * tau_running:
            converged = True
            break
    if not converged:
        return float(TAU_CAP), False
    tau = float(tau_running)
    if not math.isfinite(tau) or tau > TAU_CAP:
        return float(TAU_CAP), False
    tau = max(tau, TAU_INT_FLOOR)
    return min(tau, float(TAU_CAP)), True

def integrate_one(pot_name, seed, a, b, c, alpha):
    """Run one short integration with given (a, b, c, alpha). Returns (tau, kl)."""
    pot = POTENTIALS[pot_name]
    grad_fn = pot["grad"]
    dim = pot["dim"]
    d_kT = float(dim) * KT

    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim)
    p = rng.standard_normal(dim)
    xi = 0.0

    samples = np.empty(N_SAMPLES, dtype=np.float64)
    rec_idx = 0
    half_dt = DT / 2.0

    for step in range(N_TOTAL_STEPS):
        # half kick
        p = p - grad_fn(q) * half_dt

        # friction
        xi2 = xi * xi
        gxi = xi * (a + b * xi2) / (1.0 + c * xi2)

        if not math.isfinite(gxi):
            return TAU_CAP, math.inf

        try:
            exp_fac = math.exp(-gxi * half_dt / Q)
        except OverflowError:
            return TAU_CAP, math.inf

        p = p * exp_fac

        # drift
        q = q + p / M * DT

        # friction again
        p = p * exp_fac

        # second half kick
        grad_q = grad_fn(q)
        p = p - grad_q * half_dt

        # driving: h = alpha*|p|^2 - (alpha-1)*d*kT
        pp = float(np.dot(p, p))
        h_val = alpha * pp - (alpha - 1.0) * d_kT
        xi = xi + (h_val - d_kT) / Q * DT

        if not math.isfinite(xi) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            return TAU_CAP, math.inf

        if step >= N_BURNIN and (step - N_BURNIN) % THIN == 0:
            samples[rec_idx] = q[0]
            rec_idx += 1

    if rec_idx != N_SAMPLES:
        return TAU_CAP, math.inf

    tau, _ = _sokal_tau_int_fft(samples)
    kl = _kl_divergence(samples, pot_name)
    return tau, kl


def evaluate_candidate(pot_name, a, b, c, alpha, seeds=[42, 137]):
    """Evaluate a candidate on given potential with multiple seeds.
    Returns (mean_tau, mean_kl, all_taus, all_kls)."""
    taus = []
    kls = []
    for seed in seeds:
        tau, kl = integrate_one(pot_name, seed, a, b, c, alpha)
        taus.append(tau)
        kls.append(kl)
    mean_tau = np.mean(taus)
    mean_kl = np.mean([k for k in kls if math.isfinite(k)]) if any(math.isfinite(k) for k in kls) else math.inf
    return float(mean_tau), float(mean_kl), taus, kls


def cmaes_optimize(pot_name, x0, sigma0=0.3, max_iter=60, popsize=12):
    """CMA-ES optimization over (a, b, c, alpha) for a single potential.

    x0: initial [a, b, c, alpha]
    Returns: best_params, best_tau, history
    """
    try:
        import cma
    except ImportError:
        print("Installing cma...", file=sys.stderr)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cma"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import cma

    # Bounds: a in (0.01, 3], b in (0.01, 10], c in (0.001, 1], alpha in [1.0, 5.0]
    bounds = [[0.01, 0.01, 0.001, 1.0], [3.0, 10.0, 1.0, 5.0]]

    opts = cma.CMAOptions()
    opts['bounds'] = bounds
    opts['maxiter'] = max_iter
    opts['popsize'] = popsize
    opts['verbose'] = -1  # suppress output
    opts['seed'] = 42
    opts['tolfun'] = 0.1

    history = []
    best_tau = float('inf')
    best_params = x0.copy()

    def objective(x):
        a, b, c, alpha = x
        mean_tau, mean_kl, _, _ = evaluate_candidate(pot_name, a, b, c, alpha, seeds=[42])
        # Penalize KL violations heavily
        if mean_kl > KL_THRESHOLD:
            return mean_tau + 10000.0 * mean_kl
        return mean_tau

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    gen = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)

        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < best_tau:
            best_tau = fitnesses[best_idx]
            best_params = solutions[best_idx].copy()

        history.append({
            'gen': gen,
            'best_tau': best_tau,
            'best_params': best_params.tolist(),
            'mean_fitness': float(np.mean(fitnesses)),
        })

        gen += 1
        if gen % 5 == 0:
            print(f"  Gen {gen}: best_tau={best_tau:.2f}, params=[{best_params[0]:.3f}, {best_params[1]:.3f}, {best_params[2]:.3f}, {best_params[3]:.3f}]",
                  file=sys.stderr, flush=True)

    return best_params, best_tau, history


def grid_search_alpha(pot_name, a=0.7, b=3.0, c=0.06, alphas=None):
    """Quick grid search over alpha values with fixed (a,b,c)."""
    if alphas is None:
        alphas = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

    results = []
    for alpha in alphas:
        mean_tau, mean_kl, taus, kls = evaluate_candidate(pot_name, a, b, c, alpha, seeds=[42, 137])
        results.append({
            'alpha': alpha,
            'mean_tau': mean_tau,
            'mean_kl': mean_kl,
            'taus': taus,
            'kls': kls,
        })
        print(f"  alpha={alpha:.1f}: tau={mean_tau:.2f}, kl={mean_kl:.5f}", flush=True)

    return results


if __name__ == "__main__":
    import json

    all_results = {}

    # ── Step 1: Grid search alpha for each potential ──
    for pot_name in ["harmonic_1d", "doublewell_2d", "gaussmix_2d"]:
        print(f"\n{'='*60}")
        print(f"Grid search alpha for {pot_name}")
        print(f"{'='*60}")
        results = grid_search_alpha(pot_name)
        all_results[f"{pot_name}_grid"] = results

        # Find best alpha
        valid = [r for r in results if r['mean_kl'] < KL_THRESHOLD]
        if valid:
            best = min(valid, key=lambda r: r['mean_tau'])
            print(f"  Best: alpha={best['alpha']:.1f}, tau={best['mean_tau']:.2f}, kl={best['mean_kl']:.5f}")
        else:
            print(f"  No valid alpha found (all KL > {KL_THRESHOLD})")

    # ── Step 2: CMA-ES joint optimization for each potential ──
    # Use best alpha from grid search as warm start
    for pot_name in ["harmonic_1d", "doublewell_2d", "gaussmix_2d"]:
        print(f"\n{'='*60}")
        print(f"CMA-ES joint optimization for {pot_name}")
        print(f"{'='*60}")

        # Get best alpha from grid
        grid = all_results[f"{pot_name}_grid"]
        valid = [r for r in grid if r['mean_kl'] < KL_THRESHOLD]
        if valid:
            best_alpha = min(valid, key=lambda r: r['mean_tau'])['alpha']
        else:
            best_alpha = 1.0

        x0 = np.array([0.7, 3.0, 0.06, best_alpha])
        best_params, best_tau, history = cmaes_optimize(pot_name, x0, sigma0=0.3, max_iter=50, popsize=10)

        # Verify with 2 seeds
        a, b, c, alpha = best_params
        mean_tau, mean_kl, taus, kls = evaluate_candidate(pot_name, a, b, c, alpha, seeds=[42, 137])

        all_results[f"{pot_name}_cmaes"] = {
            'best_params': best_params.tolist(),
            'best_tau': float(best_tau),
            'verified_tau': float(mean_tau),
            'verified_kl': float(mean_kl),
            'history': history,
        }

        print(f"  CMA-ES best: a={a:.4f}, b={b:.4f}, c={c:.4f}, alpha={alpha:.4f}")
        print(f"  Verified: tau={mean_tau:.2f}, kl={mean_kl:.5f}")

    # Save results
    output_path = "/Users/wujiewang/code/bath/.worktrees/012-joint-cmaes-alpha/orbits/012-joint-cmaes-alpha/optimization_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY — Optimal parameters per potential")
    print(f"{'='*60}")
    for pot_name in ["harmonic_1d", "doublewell_2d", "gaussmix_2d"]:
        key = f"{pot_name}_cmaes"
        if key in all_results:
            r = all_results[key]
            p = r['best_params']
            print(f"  {pot_name}: a={p[0]:.4f}, b={p[1]:.4f}, c={p[2]:.4f}, alpha={p[3]:.4f} → tau={r['verified_tau']:.2f} (kl={r['verified_kl']:.5f})")
