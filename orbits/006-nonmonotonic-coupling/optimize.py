"""
CMA-ES optimizer for non-monotonic Pade friction parameters.
Runs a proxy evaluation (50k steps, 1 seed) for speed, then
validates the best with full eval.
"""

import numpy as np
import math
import time
import sys
import os

# Physical constants matching evaluator
M = 1.0
KT = 1.0
Q = 1.0
DT = 0.01

# Proxy integration: shorter for speed
PROXY_STEPS = 200_000
PROXY_BURNIN = 5_000
PROXY_THIN = 10
PROXY_SAMPLES = (PROXY_STEPS - PROXY_BURNIN) // PROXY_THIN

# Full integration params
FULL_STEPS = 1_000_000
FULL_BURNIN = 10_000
FULL_THIN = 10
FULL_SAMPLES = (FULL_STEPS - FULL_BURNIN) // FULL_THIN

# Difficulty weights
W = {"harmonic_1d": 0.0240, "doublewell_2d": 0.2944, "gaussmix_2d": 0.6816}

# KL params
KL_BINS = 100
KL_RANGE = (-6.0, 6.0)
KL_THRESHOLD = 0.05
TAU_CAP = 50000
TAU_FLOOR = 1.0


# ── Potential gradients ───────────────────────────────────────────────────

def grad_harmonic(q):
    return q.copy()

def grad_doublewell(q):
    x = q[0]
    return np.array([4.0 * x * (x * x - 1.0), q[1]])

_MU_GM = np.array([
    [3.0 * math.cos(2.0 * math.pi * k / 5),
     3.0 * math.sin(2.0 * math.pi * k / 5)]
    for k in range(5)
])

def grad_gaussmix(q):
    diff = q[np.newaxis, :] - _MU_GM
    log_w = -0.5 * np.sum(diff ** 2, axis=1)
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    return np.einsum("k,kd->d", w, diff)


POTENTIALS = {
    "harmonic_1d": {"dim": 1, "grad": grad_harmonic},
    "doublewell_2d": {"dim": 2, "grad": grad_doublewell},
    "gaussmix_2d": {"dim": 2, "grad": grad_gaussmix},
}


# ── KL divergence ────────────────────────────────────────────────────────

def _norm_cdf(x):
    return 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in x]))

def marginal_harmonic(edges):
    lo, hi = edges[:-1], edges[1:]
    p = _norm_cdf(hi) - _norm_cdf(lo)
    p = np.clip(p, 1e-300, None)
    return p / p.sum()

def marginal_doublewell(edges):
    mids = 0.5 * (edges[:-1] + edges[1:])
    log_p = -(mids**2 - 1.0)**2
    log_p -= log_p.max()
    p = np.exp(log_p) * (edges[1] - edges[0])
    p = np.clip(p, 1e-300, None)
    return p / p.sum()

def marginal_gaussmix(edges):
    MU_X = [3.0 * math.cos(2.0 * math.pi * k / 5) for k in range(5)]
    lo, hi = edges[:-1], edges[1:]
    p = np.zeros(len(lo))
    for mu_x in MU_X:
        p += (_norm_cdf(hi - mu_x) - _norm_cdf(lo - mu_x)) / 5.0
    p = np.clip(p, 1e-300, None)
    return p / p.sum()

MARGINALS = {
    "harmonic_1d": marginal_harmonic,
    "doublewell_2d": marginal_doublewell,
    "gaussmix_2d": marginal_gaussmix,
}

def kl_divergence(samples, pot_name):
    if len(samples) == 0 or not np.all(np.isfinite(samples)):
        return math.inf
    edges = np.linspace(KL_RANGE[0], KL_RANGE[1], KL_BINS + 1)
    counts, _ = np.histogram(np.clip(samples, KL_RANGE[0], KL_RANGE[1]), bins=edges)
    total = counts.sum()
    if total == 0:
        return math.inf
    p_emp = counts.astype(np.float64) / total
    p_ref = MARGINALS[pot_name](edges)
    kl = 0.0
    for pi, qi in zip(p_emp, p_ref):
        if pi > 0.0:
            if qi <= 0.0:
                return math.inf
            kl += pi * math.log(pi / qi)
    return max(0.0, kl)


# ── Sokal tau_int ─────────────────────────────────────────────────────────

def sokal_tau(x, c=6.0):
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < 3:
        return float(TAU_CAP)
    xc = x - x.mean()
    var = float(np.var(xc))
    if var == 0.0 or not math.isfinite(var):
        return float(TAU_CAP)
    nfft = 1
    while nfft < 2 * N:
        nfft <<= 1
    F = np.fft.rfft(xc, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N].real
    acf = acf / (N - np.arange(N))
    C0 = float(acf[0])
    if C0 <= 0 or not math.isfinite(C0):
        return float(TAU_CAP)
    rho = acf / C0
    if not np.all(np.isfinite(rho)):
        return float(TAU_CAP)
    tau_run = 1.0
    for t in range(1, N):
        tau_run += 2.0 * rho[t]
        if not math.isfinite(tau_run) or tau_run < TAU_FLOOR:
            tau_run = TAU_FLOOR
        if t > c * tau_run:
            return max(min(tau_run, TAU_CAP), TAU_FLOOR)
    return float(TAU_CAP)


# ── Integrator ────────────────────────────────────────────────────────────

def integrate(grad_fn, dim, seed, friction_fn, n_steps, n_burnin, thin):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim)
    p = rng.standard_normal(dim)
    xi = 0.0
    n_samples = (n_steps - n_burnin) // thin
    samples = np.empty(n_samples, dtype=np.float64)
    rec_idx = 0
    half_dt = DT / 2.0
    d_kT = float(dim) * KT

    for step in range(n_steps):
        p = p - grad_fn(q) * half_dt
        try:
            gxi = float(friction_fn(np.array([xi]))[0])
        except:
            return None
        if not math.isfinite(gxi):
            return None
        try:
            ef = math.exp(-gxi * half_dt / Q)
        except OverflowError:
            return None
        p = p * ef
        q = q + p / M * DT
        p = p * ef
        p = p - grad_fn(q) * half_dt
        xi = xi + (float(np.dot(p, p)) / M - d_kT) / Q * DT
        if not math.isfinite(xi) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            return None
        if step >= n_burnin and (step - n_burnin) % thin == 0:
            samples[rec_idx] = q[0]
            rec_idx += 1

    if rec_idx != n_samples:
        return None
    return samples


# ── Proxy evaluation ──────────────────────────────────────────────────────

def proxy_eval(a, b, c, d, seed=42, n_steps=PROXY_STEPS, n_burnin=PROXY_BURNIN, thin=PROXY_THIN):
    """Evaluate parameters with a shorter integration for speed."""
    def friction_fn(xi):
        xi2 = xi * xi
        xi4 = xi2 * xi2
        num = a + b * xi2
        den = 1.0 + c * xi2 + d * xi4
        return xi * num / den

    weighted_tau = 0.0
    for pot_name, pot_cfg in POTENTIALS.items():
        samples = integrate(
            pot_cfg["grad"], pot_cfg["dim"], seed, friction_fn,
            n_steps, n_burnin, thin
        )
        if samples is None:
            return math.inf
        kl = kl_divergence(samples, pot_name)
        if kl > KL_THRESHOLD:
            return math.inf  # KL gate
        tau = sokal_tau(samples)
        weighted_tau += W[pot_name] * tau

    return weighted_tau


def full_eval(a, b, c, d, seeds=[42, 137, 2024]):
    """Full evaluation matching evaluator.py specs."""
    def friction_fn(xi):
        xi2 = xi * xi
        xi4 = xi2 * xi2
        num = a + b * xi2
        den = 1.0 + c * xi2 + d * xi4
        return xi * num / den

    results = {}
    for pot_name, pot_cfg in POTENTIALS.items():
        taus = []
        kls = []
        for seed in seeds:
            samples = integrate(
                pot_cfg["grad"], pot_cfg["dim"], seed, friction_fn,
                FULL_STEPS, FULL_BURNIN, FULL_THIN
            )
            if samples is None:
                return math.inf, {}
            kl = kl_divergence(samples, pot_name)
            if kl > KL_THRESHOLD:
                return math.inf, {}
            tau = sokal_tau(samples)
            taus.append(tau)
            kls.append(kl)
        results[pot_name] = {"taus": taus, "kls": kls, "mean_tau": np.mean(taus)}

    weighted = sum(W[k] * results[k]["mean_tau"] for k in POTENTIALS)
    return weighted, results


# ── CMA-ES ────────────────────────────────────────────────────────────────

def cmaes_optimize(init_params, sigma0=0.3, max_iter=80, popsize=16):
    """Simple CMA-ES in log-space for (a, b, c, d) with d > 0."""
    # Work in transformed space: x = [log(a), log(b), log(c), log(d)]
    dim = len(init_params)
    mean = np.log(np.array(init_params))
    sigma = sigma0
    C = np.eye(dim)
    pc = np.zeros(dim)
    ps = np.zeros(dim)

    # CMA-ES constants
    lam = popsize
    mu = lam // 2
    weights_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights_cma = weights_raw / weights_raw.sum()
    mueff = 1.0 / np.sum(weights_cma**2)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3)**2 + mueff)
    cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))

    best_metric = math.inf
    best_params = init_params[:]
    stagnation = 0

    print(f"CMA-ES: dim={dim}, lambda={lam}, mu={mu}, sigma0={sigma0}")
    print(f"Init params: a={init_params[0]:.3f}, b={init_params[1]:.3f}, "
          f"c={init_params[2]:.4f}, d={init_params[3]:.4f}")

    for gen in range(max_iter):
        t0 = time.time()

        # Sample population
        try:
            sqrtC = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C = np.eye(dim)
            sqrtC = np.eye(dim)

        z_all = np.random.randn(lam, dim)
        x_all = mean[np.newaxis, :] + sigma * (z_all @ sqrtC.T)

        # Evaluate
        fitness = np.full(lam, math.inf)
        for i in range(lam):
            params = np.exp(x_all[i])
            a_i, b_i, c_i, d_i = params

            # Sanity bounds
            if a_i < 0.1 or a_i > 5.0:
                continue
            if b_i < 0.1 or b_i > 50.0:
                continue
            if c_i < 0.001 or c_i > 5.0:
                continue
            if d_i < 1e-4 or d_i > 10.0:
                continue

            try:
                fitness[i] = proxy_eval(a_i, b_i, c_i, d_i, seed=42)
            except:
                fitness[i] = math.inf

        # Sort by fitness
        idx = np.argsort(fitness)
        if fitness[idx[0]] < best_metric:
            best_metric = fitness[idx[0]]
            best_params = list(np.exp(x_all[idx[0]]))
            stagnation = 0
        else:
            stagnation += 1

        elapsed = time.time() - t0
        print(f"Gen {gen:3d}: best={best_metric:.2f}, "
              f"gen_best={fitness[idx[0]]:.2f}, sigma={sigma:.4f}, "
              f"params=[{best_params[0]:.3f},{best_params[1]:.3f},"
              f"{best_params[2]:.4f},{best_params[3]:.4f}], "
              f"t={elapsed:.1f}s")

        if stagnation >= 15:
            print(f"Stagnation after {stagnation} generations, stopping.")
            break

        # CMA-ES update
        selected = x_all[idx[:mu]]
        z_sel = z_all[idx[:mu]]

        old_mean = mean.copy()
        mean = np.sum(weights_cma[:, np.newaxis] * selected, axis=0)

        # Evolution paths
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (mean - old_mean) / sigma
        hsig = (np.linalg.norm(ps) /
                math.sqrt(1 - (1 - cs)**(2 * (gen + 1))) /
                chiN) < 1.4 + 2.0 / (dim + 1)

        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma

        # Covariance update
        artmp = (selected - old_mean[np.newaxis, :]) / sigma
        C = ((1 - c1 - cmu_val) * C +
             c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) +
             cmu_val * np.sum(weights_cma[:, np.newaxis, np.newaxis] *
                              artmp[:, :, np.newaxis] * artmp[:, np.newaxis, :], axis=0))

        # Symmetrize
        C = 0.5 * (C + C.T)

        # Step-size update
        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = max(0.01, min(sigma, 2.0))

    return best_params, best_metric


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sweep", "cmaes", "full"], default="cmaes")
    parser.add_argument("--a", type=float, default=0.7)
    parser.add_argument("--b", type=float, default=3.0)
    parser.add_argument("--c", type=float, default=0.06)
    parser.add_argument("--d", type=float, default=0.5)
    args = parser.parse_args()

    if args.mode == "sweep":
        # Quick d-sweep to see the landscape
        print("d-sweep with parent (a=0.7, b=3.0, c=0.06):")
        for d_val in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
            m = proxy_eval(0.7, 3.0, 0.06, d_val)
            print(f"  d={d_val:.3f}: proxy_metric={m:.2f}")

    elif args.mode == "cmaes":
        best_params, best_metric = cmaes_optimize(
            [args.a, args.b, args.c, args.d],
            sigma0=0.3, max_iter=80, popsize=16
        )
        print(f"\nBest found: a={best_params[0]:.4f}, b={best_params[1]:.4f}, "
              f"c={best_params[2]:.5f}, d={best_params[3]:.5f}")
        print(f"Proxy metric: {best_metric:.2f}")

        # Full eval on best
        print("\nRunning full eval on best params...")
        metric, results = full_eval(*best_params)
        print(f"Full metric: {metric:.2f}")
        if results:
            for pot_name, res in results.items():
                print(f"  {pot_name}: taus={[f'{t:.1f}' for t in res['taus']]}, "
                      f"kls={[f'{k:.4f}' for k in res['kls']]}")

    elif args.mode == "full":
        print(f"Full eval with a={args.a}, b={args.b}, c={args.c}, d={args.d}")
        metric, results = full_eval(args.a, args.b, args.c, args.d)
        print(f"Full metric: {metric:.2f}")
        if results:
            for pot_name, res in results.items():
                print(f"  {pot_name}: taus={[f'{t:.1f}' for t in res['taus']]}, "
                      f"kls={[f'{k:.4f}' for k in res['kls']]}")
