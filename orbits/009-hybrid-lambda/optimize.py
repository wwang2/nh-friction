#!/usr/bin/env python3
"""
CMA-ES optimization of hybrid driving parameters (a, b, c, lambda).

Hybrid driving: h = lambda*|p|^2 + (1-lambda)*|grad_V|^2_norm
where |grad_V|^2_norm = (gv2/d) / E_ref, with E_ref tracking E[gv2/d].

Optimizes over all 3 benchmark potentials using a proxy evaluation
(fewer steps) for speed, then writes best parameters.
"""

import math
import sys
import json
import time
import numpy as np

# ── Physical constants (match evaluator) ─────────────────────────────────
M = 1.0
KT = 1.0
Q = 1.0
DT = 0.01

# ── Proxy eval: fewer steps for speed ───────────────────────────────────
N_BURNIN = 5_000
N_MAIN_STEPS = 200_000
THIN = 10
N_SAMPLES = N_MAIN_STEPS // THIN
N_TOTAL_STEPS = N_BURNIN + N_MAIN_STEPS
SEEDS = [42, 137]  # 2 seeds for proxy (faster)

TAU_CAP = 25_000
TAU_INT_FLOOR = 1.0
KL_THRESHOLD = 0.05
KL_BINS = 100
KL_RANGE = (-6.0, 6.0)
C_SOKAL = 6.0

# Difficulty weights from config
WEIGHTS = {
    "harmonic_1d": 0.0240,
    "doublewell_2d": 0.2944,
    "gaussmix_2d": 0.6816,
}


# ── Potentials ───────────────────────────────────────────────────────────

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


# ── KL divergence ───────────────────────────────────────────────────────

def _norm_cdf(x):
    return 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in x]))

def _marginal_harmonic_1d(bin_edges):
    lo, hi = bin_edges[:-1], bin_edges[1:]
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
    bin_edges = np.linspace(KL_RANGE[0], KL_RANGE[1], KL_BINS + 1)
    counts, _ = np.histogram(np.clip(samples_q0, KL_RANGE[0], KL_RANGE[1]), bins=bin_edges)
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


# ── Tau int (Sokal FFT) ─────────────────────────────────────────────────

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


# ── Integrator ───────────────────────────────────────────────────────────

def _integrate_one(grad_fn, dim, seed, friction_fn, driving_fn):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim)
    p = rng.standard_normal(dim)
    xi = 0.0

    samples = np.empty(N_SAMPLES, dtype=np.float64)
    rec_idx = 0
    half_dt = DT / 2.0
    d_kT = float(dim) * KT

    for step in range(N_TOTAL_STEPS):
        p = p - grad_fn(q) * half_dt

        gxi = friction_fn(xi)
        try:
            exp_fac = math.exp(-gxi * half_dt / Q)
        except OverflowError:
            return None
        p = p * exp_fac

        q = q + p / M * DT
        p = p * exp_fac

        grad_q = grad_fn(q)
        p = p - grad_q * half_dt

        h_val = driving_fn(q, p, grad_q)
        if not math.isfinite(h_val):
            return None
        xi = xi + (h_val - d_kT) / Q * DT

        if not math.isfinite(xi) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            return None

        if step >= N_BURNIN and (step - N_BURNIN) % THIN == 0:
            samples[rec_idx] = q[0]
            rec_idx += 1

    if rec_idx != N_SAMPLES:
        return None
    return samples


# ── Objective function ───────────────────────────────────────────────────

def evaluate_params(a, b, c, lam):
    """Evaluate (a, b, c, lambda) -> weighted tau_int (proxy)."""

    def friction_fn(xi_val):
        xi2 = xi_val * xi_val
        num = a + b * xi2
        den = 1.0 + c * xi2
        return xi_val * num / den

    # State for driving function - reset per run via closure
    tau_results = {}

    for pot_name, pot_cfg in POTENTIALS.items():
        grad_fn = pot_cfg["grad"]
        dim = pot_cfg["dim"]
        taus = []

        for seed in SEEDS:
            # Fresh driving function state per run
            state = {"grad_sq_ema": 1.0, "call_count": 0}

            def driving_fn(q, p, grad_V, _state=state, _lam=lam):
                d = len(q)
                gv2 = float(np.dot(grad_V, grad_V))
                pp = float(np.dot(p, p))

                _state["call_count"] += 1
                alpha = max(0.001, 1.0 / _state["call_count"])
                _state["grad_sq_ema"] = (1.0 - alpha) * _state["grad_sq_ema"] + alpha * (gv2 / d)

                E_ref = max(_state["grad_sq_ema"], 1e-10)
                return _lam * pp + (1.0 - _lam) * (gv2 / d) / E_ref

            samples = _integrate_one(grad_fn, dim, seed, friction_fn, driving_fn)
            if samples is None:
                return 1e6  # diverged

            # KL check
            kl = _kl_divergence(samples, pot_name)
            if kl > KL_THRESHOLD:
                return 1e6  # KL gate fail

            tau, _ = _sokal_tau_int_fft(samples)
            taus.append(tau)

        tau_results[pot_name] = float(np.mean(taus))

    # Weighted metric
    metric = sum(WEIGHTS[k] * tau_results[k] for k in POTENTIALS)
    return metric


# ── CMA-ES ───────────────────────────────────────────────────────────────

class SimpleCMAES:
    """Minimal CMA-ES implementation (Hansen 2016)."""

    def __init__(self, x0, sigma0, pop_size=None, bounds=None):
        self.dim = len(x0)
        self.mean = np.array(x0, dtype=np.float64)
        self.sigma = sigma0
        self.pop_size = pop_size or (4 + int(3 * math.log(self.dim)))
        self.bounds = bounds  # list of (lo, hi) per dim

        # Selection
        mu = self.pop_size // 2
        weights = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)])
        weights /= weights.sum()
        self.weights = weights
        self.mu = mu
        self.mu_eff = 1.0 / np.sum(weights ** 2)

        # Adaptation parameters
        n = self.dim
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0, math.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0) + self.c_sigma
        self.c_c = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)
        self.c_1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1.0 - self.c_1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((n + 2.0) ** 2 + self.mu_eff))

        # State
        self.p_sigma = np.zeros(n)
        self.p_c = np.zeros(n)
        self.C = np.eye(n)
        self.chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
        self.gen = 0

    def ask(self):
        """Sample pop_size candidates."""
        n = self.dim
        try:
            A = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            self.C = np.eye(n)
            A = np.eye(n)

        zs = np.random.randn(self.pop_size, n)
        xs = self.mean[np.newaxis, :] + self.sigma * (zs @ A.T)

        # Clip to bounds
        if self.bounds is not None:
            for i, (lo, hi) in enumerate(self.bounds):
                xs[:, i] = np.clip(xs[:, i], lo, hi)

        return xs

    def tell(self, xs, fitnesses):
        """Update distribution from evaluated candidates."""
        n = self.dim
        idx = np.argsort(fitnesses)
        xs_sorted = xs[idx[:self.mu]]

        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, np.newaxis] * xs_sorted, axis=0)

        # Clip mean to bounds
        if self.bounds is not None:
            for i, (lo, hi) in enumerate(self.bounds):
                self.mean[i] = np.clip(self.mean[i], lo, hi)

        diff = (self.mean - old_mean) / self.sigma

        try:
            C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(self.C)).T
        except np.linalg.LinAlgError:
            self.C = np.eye(n)
            C_inv_sqrt = np.eye(n)

        # Evolution path updates
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + math.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * (C_inv_sqrt @ diff)

        h_sigma = 1.0 if np.linalg.norm(self.p_sigma) / math.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.gen + 1))) < (1.4 + 2.0 / (n + 1.0)) * self.chi_n else 0.0

        self.p_c = (1.0 - self.c_c) * self.p_c + h_sigma * math.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * diff

        # Covariance update
        diffs = (xs_sorted - old_mean[np.newaxis, :]) / self.sigma
        rank_mu = sum(self.weights[i] * np.outer(diffs[i], diffs[i]) for i in range(self.mu))

        self.C = ((1.0 - self.c_1 - self.c_mu) * self.C
                   + self.c_1 * (np.outer(self.p_c, self.p_c) + (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c) * self.C)
                   + self.c_mu * rank_mu)

        # Step size update
        self.sigma *= math.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self.chi_n - 1.0))
        self.sigma = min(self.sigma, 10.0)  # cap

        self.gen += 1
        return fitnesses[idx[0]], xs[idx[0]]


def main():
    print("=" * 60)
    print("CMA-ES optimization: (a, b, c, lambda)")
    print("=" * 60)

    x0 = [0.7, 3.0, 0.06, 0.5]
    bounds = [(0.1, 3.0), (0.5, 10.0), (0.001, 1.0), (0.0, 1.0)]

    cma = SimpleCMAES(x0, sigma0=0.3, pop_size=10, bounds=bounds)

    best_metric = float("inf")
    best_params = x0[:]
    history = []

    n_generations = 12  # ~120 evaluations total

    for gen in range(n_generations):
        xs = cma.ask()
        fitnesses = np.full(len(xs), 1e6)

        for i, x in enumerate(xs):
            a, b, c, lam = x
            t0 = time.time()
            try:
                metric = evaluate_params(a, b, c, lam)
            except Exception as e:
                print(f"  ERROR: {e}")
                metric = 1e6
            elapsed = time.time() - t0
            fitnesses[i] = metric
            print(f"  gen={gen} i={i} a={a:.3f} b={b:.3f} c={c:.4f} lam={lam:.3f} metric={metric:.2f} ({elapsed:.1f}s)")

        best_fit, best_x = cma.tell(xs, fitnesses)

        if best_fit < best_metric:
            best_metric = best_fit
            best_params = best_x.tolist()

        history.append({
            "gen": gen,
            "best_gen": float(best_fit),
            "best_overall": float(best_metric),
            "best_params": best_params[:],
            "mean": cma.mean.tolist(),
            "sigma": float(cma.sigma),
        })

        print(f"\nGen {gen}: best_gen={best_fit:.2f}, best_overall={best_metric:.2f}")
        print(f"  params: a={best_params[0]:.4f} b={best_params[1]:.4f} c={best_params[2]:.5f} lam={best_params[3]:.4f}")
        print(f"  CMA mean: {cma.mean}")
        print(f"  CMA sigma: {cma.sigma:.4f}")
        print()

    # Save results
    result = {
        "best_metric": best_metric,
        "best_params": {"a": best_params[0], "b": best_params[1], "c": best_params[2], "lambda": best_params[3]},
        "history": history,
    }

    out_path = "/Users/wujiewang/code/bath/.worktrees/009-hybrid-lambda/orbits/009-hybrid-lambda/cma_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nFinal best: metric={best_metric:.4f}")
    print(f"  a={best_params[0]:.4f}, b={best_params[1]:.4f}, c={best_params[2]:.5f}, lambda={best_params[3]:.4f}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
