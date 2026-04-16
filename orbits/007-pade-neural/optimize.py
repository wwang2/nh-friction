"""
Offline optimizer for neural-enhanced Pade friction function.
Searches for better g(xi) shapes using a fast proxy evaluator
(short integration) before committing to full eval.

Strategy:
1. Start from Pade baseline (a=0.7, b=3.0, c=0.06)
2. Try parametric variations with extra degrees of freedom
3. Use short proxy runs (50k steps) to screen candidates
4. Full eval on the best candidates

The NN parameterization: g(xi) = xi * [a + NN(xi^2)] / (1 + c*xi^2)
We'll optimize the NN weights to modify the transition profile.
"""

import numpy as np
import time
import sys
import math
from pathlib import Path

# Add research dir for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "eval"))

# ── Physical constants ───────────────────────────────────────────────────
M = 1.0
KT = 1.0
Q = 1.0
DT = 0.01

# ── Proxy integration parameters (shorter for speed) ────────────────────
N_BURNIN = 5000
N_MAIN_STEPS = 200000
THIN = 10
N_SAMPLES = N_MAIN_STEPS // THIN
N_TOTAL_STEPS = N_BURNIN + N_MAIN_STEPS

SEEDS = [42, 137, 2024]

TAU_CAP = N_SAMPLES // 2
TAU_INT_FLOOR = 1.0
KL_THRESHOLD = 0.05
C_SOKAL = 6.0


# ── Potential gradients ──────────────────────────────────────────────────
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

WEIGHTS = {"harmonic_1d": 0.0240, "doublewell_2d": 0.2944, "gaussmix_2d": 0.6816}


# ── KL divergence ───────────────────────────────────────────────────────
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


# ── tau_int computation ──────────────────────────────────────────────────
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
def _integrate_one(grad_fn, dim, seed, friction_fn):
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
        p = p - grad_fn(q) * half_dt
        xi = xi + (float(np.dot(p, p)) / M - d_kT) / Q * DT
        if not math.isfinite(xi) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            return None
        if step >= N_BURNIN and (step - N_BURNIN) % THIN == 0:
            samples[rec_idx] = q[0]
            rec_idx += 1

    return samples[:rec_idx] if rec_idx > 0 else None


# ── Proxy evaluator ─────────────────────────────────────────────────────
def proxy_eval(friction_fn_scalar, verbose=False):
    """Evaluate a friction function using short proxy runs.
    friction_fn_scalar takes a float, returns a float.
    Returns weighted tau_int (lower is better), or inf if KL fails.
    """
    tau_per_pot = {}
    kl_per_pot = {}

    for pot_name, pot_cfg in POTENTIALS.items():
        grad_fn = pot_cfg["grad"]
        dim = pot_cfg["dim"]
        taus = []
        kls = []

        for seed in SEEDS:
            samples = _integrate_one(grad_fn, dim, seed, friction_fn_scalar)
            if samples is None:
                taus.append(TAU_CAP)
                kls.append(math.inf)
                continue
            kl = _kl_divergence(samples, pot_name)
            kls.append(kl)
            tau, _ = _sokal_tau_int_fft(samples)
            taus.append(tau)

        mean_kl = np.mean([k for k in kls if math.isfinite(k)]) if any(math.isfinite(k) for k in kls) else math.inf
        mean_tau = np.mean(taus)

        if mean_kl > KL_THRESHOLD:
            if verbose:
                print(f"  KL FAIL: {pot_name} mean_kl={mean_kl:.4f}")
            return math.inf

        tau_per_pot[pot_name] = mean_tau
        kl_per_pot[pot_name] = mean_kl

    weighted = sum(WEIGHTS[k] * tau_per_pot[k] for k in POTENTIALS)
    if verbose:
        for k in POTENTIALS:
            print(f"  {k}: tau={tau_per_pot[k]:.1f}, kl={kl_per_pot[k]:.4f}")
        print(f"  weighted_tau = {weighted:.2f}")
    return weighted


# ── Parametric families to try ───────────────────────────────────────────

def make_pade(a, b, c):
    """Standard Pade: g(xi) = xi*(a + b*xi^2)/(1 + c*xi^2)"""
    def ff(xi):
        xi2 = xi * xi
        return xi * (a + b * xi2) / (1.0 + c * xi2)
    return ff

def make_pade_plus_bump(a, b, c, amp, center, width):
    """Pade + Gaussian bump: g(xi) = xi*[(a + b*xi^2)/(1+c*xi^2) + amp*exp(-(xi^2-center)^2/width^2)]
    The bump adds localized enhancement at a specific |xi| range.
    """
    def ff(xi):
        xi2 = xi * xi
        pade = (a + b * xi2) / (1.0 + c * xi2)
        bump = amp * np.exp(-(xi2 - center)**2 / (width**2 + 1e-10))
        return xi * (pade + bump)
    return ff

def make_double_pade(a, b1, c1, b2, c2):
    """Sum of two Pade terms: g(xi) = xi*[a + b1*xi^2/(1+c1*xi^2) + b2*xi^2/(1+c2*xi^2)]"""
    def ff(xi):
        xi2 = xi * xi
        term1 = b1 * xi2 / (1.0 + c1 * xi2)
        term2 = b2 * xi2 / (1.0 + c2 * xi2)
        return xi * (a + term1 + term2)
    return ff

def make_pade_with_quartic(a, b, c, d):
    """Extended Pade with quartic: g(xi) = xi*(a + b*xi^2 + d*xi^4)/(1 + c*xi^2)"""
    def ff(xi):
        xi2 = xi * xi
        xi4 = xi2 * xi2
        return xi * (a + b * xi2 + d * xi4) / (1.0 + c * xi2)
    return ff

def make_pade_extended_den(a, b, c, e):
    """Pade with extended denominator: g(xi) = xi*(a + b*xi^2)/(1 + c*xi^2 + e*xi^4)"""
    def ff(xi):
        xi2 = xi * xi
        xi4 = xi2 * xi2
        return xi * (a + b * xi2) / (1.0 + c * xi2 + e * xi4)
    return ff

def make_softplus_transition(a, b, k, x0):
    """Softplus transition: g(xi) = xi * [a + (b-a)*softplus(k*(xi^2 - x0))/softplus(k*(-x0+100))]
    Allows sharper transitions than rational functions.
    """
    def _sp(x):
        return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -50.0, 20.0))))
    norm = float(_sp(np.array([k * (100.0 - x0)]))[0]) if k > 0 else 1.0
    def ff(xi):
        xi2 = xi * xi
        t = _sp(k * (xi2 - x0)) / (norm + 1e-10)
        return xi * (a + (b - a) * np.clip(t, 0.0, 1.0))
    return ff


if __name__ == "__main__":
    print("=" * 60)
    print("Proxy optimization for neural-enhanced Pade")
    print("=" * 60)

    # 1. Baseline Pade
    print("\n--- Baseline Pade (a=0.7, b=3.0, c=0.06) ---")
    baseline_ff = make_pade(0.7, 3.0, 0.06)
    t0 = time.time()
    baseline_metric = proxy_eval(baseline_ff, verbose=True)
    t_baseline = time.time() - t0
    print(f"  Time: {t_baseline:.1f}s")

    results = [("Pade(0.7,3.0,0.06)", baseline_metric)]

    # 2. Grid search over Pade parameters (fine-grained around optimum)
    print("\n--- Fine Pade grid search ---")
    best_metric = baseline_metric
    best_params = None
    for a in [0.60, 0.65, 0.70, 0.75, 0.80]:
        for b in [2.0, 2.5, 3.0, 3.5, 4.0]:
            for c in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
                if a == 0.7 and b == 3.0 and c == 0.06:
                    continue
                ff = make_pade(a, b, c)
                m = proxy_eval(ff, verbose=False)
                if m < best_metric:
                    best_metric = m
                    best_params = (a, b, c)
                    print(f"  NEW BEST: Pade({a},{b},{c}) -> {m:.2f}")

    if best_params:
        print(f"\n  Best Pade: {best_params} -> {best_metric:.2f}")
        results.append((f"Pade{best_params}", best_metric))
    else:
        print("  No improvement found in Pade grid.")

    # 3. Double Pade search
    print("\n--- Double Pade search ---")
    best_dp_metric = baseline_metric
    best_dp_params = None
    for a in [0.5, 0.6, 0.7, 0.8]:
        for b1 in [1.0, 2.0, 3.0]:
            for c1 in [0.03, 0.06, 0.10]:
                for b2 in [0.5, 1.0, 2.0]:
                    for c2 in [0.01, 0.03, 0.10, 0.30]:
                        ff = make_double_pade(a, b1, c1, b2, c2)
                        m = proxy_eval(ff, verbose=False)
                        if m < best_dp_metric:
                            best_dp_metric = m
                            best_dp_params = (a, b1, c1, b2, c2)
                            print(f"  NEW BEST: DoublePade({a},{b1},{c1},{b2},{c2}) -> {m:.2f}")

    if best_dp_params:
        results.append((f"DoublePade{best_dp_params}", best_dp_metric))

    # 4. Pade + bump search
    print("\n--- Pade + Gaussian bump search ---")
    best_bump_metric = baseline_metric
    best_bump_params = None
    for amp in [0.1, 0.3, 0.5, 1.0, 2.0]:
        for center in [0.5, 1.0, 2.0, 5.0, 10.0]:
            for width in [0.5, 1.0, 2.0, 5.0]:
                ff = make_pade_plus_bump(0.7, 3.0, 0.06, amp, center, width)
                m = proxy_eval(ff, verbose=False)
                if m < best_bump_metric:
                    best_bump_metric = m
                    best_bump_params = (amp, center, width)
                    print(f"  NEW BEST: Bump(amp={amp},ctr={center},w={width}) -> {m:.2f}")

    if best_bump_params:
        results.append((f"PadeBump{best_bump_params}", best_bump_metric))

    # 5. Extended denominator
    print("\n--- Extended denominator search ---")
    best_ed_metric = baseline_metric
    best_ed_params = None
    for e in [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01]:
        for a in [0.6, 0.7, 0.8]:
            for b in [2.5, 3.0, 3.5]:
                for c in [0.04, 0.06, 0.08]:
                    ff = make_pade_extended_den(a, b, c, e)
                    m = proxy_eval(ff, verbose=False)
                    if m < best_ed_metric:
                        best_ed_metric = m
                        best_ed_params = (a, b, c, e)
                        print(f"  NEW BEST: ExtDen({a},{b},{c},{e}) -> {m:.2f}")

    if best_ed_params:
        results.append((f"ExtDen{best_ed_params}", best_ed_metric))

    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results.sort(key=lambda x: x[1])
    for name, metric in results:
        marker = " <-- BEST" if metric == results[0][1] else ""
        print(f"  {name}: {metric:.2f}{marker}")
