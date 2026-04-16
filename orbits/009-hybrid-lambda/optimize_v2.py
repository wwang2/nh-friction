#!/usr/bin/env python3
"""
CMA-ES optimization of Pade friction parameters (a, b, c) with pure kinetic driving.

Hybrid driving was found to break canonical distribution (configurational
normalization creates feedback loops). Instead, optimize friction parameters
more aggressively.

Also explores extended Pade forms:
  g(xi) = xi * (a + b*xi^2) / (1 + c*xi^2 + d*xi^4)
"""

import math
import sys
import json
import time
import numpy as np

M, KT, Q, DT = 1.0, 1.0, 1.0, 0.01
N_BURNIN = 5000
N_MAIN = 200000
THIN = 10
N_SAMPLES = N_MAIN // THIN
N_TOTAL = N_BURNIN + N_MAIN
TAU_CAP = 25000
TAU_INT_FLOOR = 1.0
C_SOKAL = 6.0
WEIGHTS = {"harmonic_1d": 0.0240, "doublewell_2d": 0.2944, "gaussmix_2d": 0.6816}


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

POTS = {
    "harmonic_1d":   {"dim": 1, "grad": _grad_harmonic_1d},
    "doublewell_2d": {"dim": 2, "grad": _grad_doublewell_2d},
    "gaussmix_2d":   {"dim": 2, "grad": _grad_gaussmix_2d},
}

def _norm_cdf(x):
    return 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in x]))

def _kl(samples, pot_name):
    be = np.linspace(-6, 6, 101)
    counts, _ = np.histogram(np.clip(samples, -6, 6), bins=be)
    if counts.sum() == 0:
        return float("inf")
    pe = counts / counts.sum()
    lo, hi = be[:-1], be[1:]

    if pot_name == "harmonic_1d":
        pr = _norm_cdf(hi) - _norm_cdf(lo)
    elif pot_name == "doublewell_2d":
        m = 0.5 * (lo + hi)
        lp = -(m ** 2 - 1) ** 2
        lp -= lp.max()
        pr = np.exp(lp) * (hi[0] - lo[0])
    elif pot_name == "gaussmix_2d":
        MU_X = [3.0 * math.cos(2 * math.pi * k / 5) for k in range(5)]
        pr = np.zeros(100)
        for mu in MU_X:
            pr += (_norm_cdf(hi - mu) - _norm_cdf(lo - mu)) / 5.0
    pr = np.clip(pr, 1e-300, None)
    pr /= pr.sum()
    return sum(p * np.log(p / q) for p, q in zip(pe, pr) if p > 0)


def _sokal_tau(x):
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < 3:
        return float(TAU_CAP)
    xc = x - x.mean()
    var = float(np.var(xc))
    if var == 0:
        return float(TAU_CAP)
    nfft = 1
    while nfft < 2 * N:
        nfft <<= 1
    F = np.fft.rfft(xc, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N].real
    acf = acf / (N - np.arange(N))
    C0 = float(acf[0])
    if C0 <= 0:
        return float(TAU_CAP)
    rho = acf / C0
    tau_r = 1.0
    for t in range(1, N):
        tau_r += 2.0 * rho[t]
        if tau_r < TAU_INT_FLOOR:
            tau_r = TAU_INT_FLOOR
        if t > C_SOKAL * tau_r:
            return max(tau_r, TAU_INT_FLOOR)
    return float(TAU_CAP)


def _run_one(friction_fn_scalar, grad_fn, dim, seed):
    """Integrate with pure kinetic driving h=|p|^2."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim)
    p = rng.standard_normal(dim)
    xi = 0.0
    half_dt = DT / 2.0
    d_kT = float(dim) * KT
    samples = np.empty(N_SAMPLES)
    rec = 0

    for step in range(N_TOTAL):
        p = p - grad_fn(q) * half_dt
        gxi = friction_fn_scalar(xi)
        try:
            ef = math.exp(-gxi * half_dt / Q)
        except OverflowError:
            return None
        p = p * ef
        q = q + p / M * DT
        p = p * ef
        gq = grad_fn(q)
        p = p - gq * half_dt
        pp = float(np.dot(p, p))
        xi = xi + (pp - d_kT) / Q * DT
        if not (math.isfinite(xi) and np.all(np.isfinite(q)) and np.all(np.isfinite(p))):
            return None
        if step >= N_BURNIN and (step - N_BURNIN) % THIN == 0:
            samples[rec] = q[0]
            rec += 1

    return samples if rec == N_SAMPLES else None


def evaluate_3param(params):
    """Evaluate standard Pade: g(xi) = xi*(a + b*xi^2)/(1 + c*xi^2)."""
    a, b, c = params
    if a <= 0 or b <= 0 or c < 0:
        return 1e6

    def ff(xi):
        xi2 = xi * xi
        return xi * (a + b * xi2) / (1.0 + c * xi2)

    tau_per = {}
    for pn, pc in POTS.items():
        taus = []
        for seed in [42, 137]:
            s = _run_one(ff, pc["grad"], pc["dim"], seed)
            if s is None:
                return 1e6
            kl = _kl(s, pn)
            if kl > 0.05:
                return 1e6
            taus.append(_sokal_tau(s))
        tau_per[pn] = np.mean(taus)

    return sum(WEIGHTS[k] * tau_per[k] for k in POTS)


def evaluate_4param(params):
    """Extended Pade: g(xi) = xi*(a + b*xi^2 + e*xi^4)/(1 + c*xi^2 + d*xi^4)."""
    a, b, c, d = params[:4]
    e = params[4] if len(params) > 4 else 0.0
    if a <= 0 or b <= 0 or c < 0 or d < 0 or e < 0:
        return 1e6

    def ff(xi):
        xi2 = xi * xi
        xi4 = xi2 * xi2
        return xi * (a + b * xi2 + e * xi4) / (1.0 + c * xi2 + d * xi4)

    tau_per = {}
    for pn, pc in POTS.items():
        taus = []
        for seed in [42, 137]:
            s = _run_one(ff, pc["grad"], pc["dim"], seed)
            if s is None:
                return 1e6
            kl = _kl(s, pn)
            if kl > 0.05:
                return 1e6
            taus.append(_sokal_tau(s))
        tau_per[pn] = np.mean(taus)

    return sum(WEIGHTS[k] * tau_per[k] for k in POTS)


class CMA:
    def __init__(self, x0, sigma0, pop_size=None, bounds=None):
        self.dim = len(x0)
        self.mean = np.array(x0, dtype=np.float64)
        self.sigma = sigma0
        self.pop_size = pop_size or max(8, 4 + int(3 * math.log(self.dim)))
        self.bounds = bounds

        mu = self.pop_size // 2
        w = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)])
        w /= w.sum()
        self.weights = w
        self.mu = mu
        self.mu_eff = 1.0 / np.sum(w ** 2)

        n = self.dim
        self.c_sigma = (self.mu_eff + 2) / (n + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, math.sqrt((self.mu_eff - 1) / (n + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / n) / (n + 4 + 2 * self.mu_eff / n)
        self.c_1 = 2 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((n + 2) ** 2 + self.mu_eff))

        self.p_sigma = np.zeros(n)
        self.p_c = np.zeros(n)
        self.C = np.eye(n)
        self.chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.gen = 0

    def ask(self):
        try:
            A = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            self.C = np.eye(self.dim)
            A = np.eye(self.dim)
        zs = np.random.randn(self.pop_size, self.dim)
        xs = self.mean[None, :] + self.sigma * (zs @ A.T)
        if self.bounds:
            for i, (lo, hi) in enumerate(self.bounds):
                xs[:, i] = np.clip(xs[:, i], lo, hi)
        return xs

    def tell(self, xs, fits):
        n = self.dim
        idx = np.argsort(fits)
        xs_s = xs[idx[:self.mu]]
        old = self.mean.copy()
        self.mean = np.sum(self.weights[:, None] * xs_s, axis=0)
        if self.bounds:
            for i, (lo, hi) in enumerate(self.bounds):
                self.mean[i] = np.clip(self.mean[i], lo, hi)
        d = (self.mean - old) / self.sigma
        try:
            Ci = np.linalg.inv(np.linalg.cholesky(self.C)).T
        except np.linalg.LinAlgError:
            self.C = np.eye(n)
            Ci = np.eye(n)
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + math.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * (Ci @ d)
        hs = 1.0 if np.linalg.norm(self.p_sigma) / math.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.gen + 1))) < (1.4 + 2 / (n + 1)) * self.chi_n else 0.0
        self.p_c = (1 - self.c_c) * self.p_c + hs * math.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * d
        diffs = (xs_s - old[None, :]) / self.sigma
        rm = sum(self.weights[i] * np.outer(diffs[i], diffs[i]) for i in range(self.mu))
        self.C = ((1 - self.c_1 - self.c_mu) * self.C
                   + self.c_1 * (np.outer(self.p_c, self.p_c) + (1 - hs) * self.c_c * (2 - self.c_c) * self.C)
                   + self.c_mu * rm)
        self.sigma *= math.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self.chi_n - 1))
        self.sigma = min(self.sigma, 5.0)
        self.gen += 1
        return fits[idx[0]], xs[idx[0]]


def run_cma(eval_fn, x0, bounds, sigma0, n_gen, pop_size, label):
    cma = CMA(x0, sigma0, pop_size=pop_size, bounds=bounds)
    best_metric = float("inf")
    best_params = x0[:]
    history = []

    for gen in range(n_gen):
        xs = cma.ask()
        fits = np.full(len(xs), 1e6)
        for i, x in enumerate(xs):
            t0 = time.time()
            try:
                fits[i] = eval_fn(x)
            except Exception as e:
                fits[i] = 1e6
            el = time.time() - t0
            status = f"{fits[i]:.2f}" if fits[i] < 1e5 else "FAIL"
            print(f"  [{label}] gen={gen} i={i} params={np.round(x,4)} metric={status} ({el:.1f}s)")

        bf, bx = cma.tell(xs, fits)
        if bf < best_metric:
            best_metric = bf
            best_params = bx.tolist()

        history.append({
            "gen": gen, "best_gen": float(bf),
            "best_overall": float(best_metric),
            "best_params": best_params[:],
        })
        print(f"  [{label}] Gen {gen}: best_gen={bf:.2f}, best_overall={best_metric:.2f}, params={np.round(best_params, 5)}\n")

    return best_metric, best_params, history


def main():
    all_results = {}

    # ── Phase 1: Optimize standard 3-param Pade ─────────────────────────
    print("=" * 60)
    print("Phase 1: CMA-ES on standard Pade (a, b, c)")
    print("=" * 60)

    m3, p3, h3 = run_cma(
        evaluate_3param,
        x0=[0.7, 3.0, 0.06],
        bounds=[(0.1, 2.0), (0.5, 10.0), (0.001, 0.5)],
        sigma0=0.1,
        n_gen=15,
        pop_size=12,
        label="3p",
    )
    all_results["3param"] = {"metric": m3, "params": p3, "history": h3}
    print(f"\nPhase 1 best: metric={m3:.4f}, a={p3[0]:.4f}, b={p3[1]:.4f}, c={p3[2]:.5f}\n")

    # ── Phase 2: Optimize extended 4-param Pade ──────────────────────────
    print("=" * 60)
    print("Phase 2: CMA-ES on extended Pade (a, b, c, d)")
    print("=" * 60)

    m4, p4, h4 = run_cma(
        evaluate_4param,
        x0=[p3[0], p3[1], p3[2], 0.001],
        bounds=[(0.1, 2.0), (0.5, 10.0), (0.001, 0.5), (0.0, 0.1)],
        sigma0=0.05,
        n_gen=10,
        pop_size=10,
        label="4p",
    )
    all_results["4param"] = {"metric": m4, "params": p4, "history": h4}
    print(f"\nPhase 2 best: metric={m4:.4f}, params={p4}\n")

    # ── Phase 3: Extended 5-param Pade (numerator quartic) ───────────────
    print("=" * 60)
    print("Phase 3: CMA-ES on 5-param Pade (a, b, c, d, e)")
    print("=" * 60)

    def eval5(params):
        return evaluate_4param(params)

    best_start = p4 if m4 < m3 else list(p3) + [0.001]
    if len(best_start) < 5:
        best_start.append(0.0)

    m5, p5, h5 = run_cma(
        eval5,
        x0=best_start,
        bounds=[(0.1, 2.0), (0.5, 10.0), (0.001, 0.5), (0.0, 0.1), (0.0, 5.0)],
        sigma0=0.05,
        n_gen=10,
        pop_size=10,
        label="5p",
    )
    all_results["5param"] = {"metric": m5, "params": p5, "history": h5}
    print(f"\nPhase 3 best: metric={m5:.4f}, params={p5}\n")

    # Save all results
    out_path = "/Users/wujiewang/code/bath/.worktrees/009-hybrid-lambda/orbits/009-hybrid-lambda/cma_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Final summary
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for name, res in all_results.items():
        m = res["metric"]
        p = res["params"]
        status = f"{m:.4f}" if m < 1e5 else "FAIL"
        print(f"  {name}: metric={status}, params={[round(x, 5) for x in p]}")


if __name__ == "__main__":
    main()
