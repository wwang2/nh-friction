"""
Offline CMA-ES optimizer for composite friction parameters.
Runs proxy integrations on all 3 potentials to find optimal (a, b, e, f, c).
Results are printed for hardcoding into solution.py.
"""

import numpy as np
import math
import time


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


def proxy_integrate(grad_fn, dim, seed, friction_fn, n_steps=50000, dt=0.01,
                    thin=10, burn_in=5000):
    """Short proxy integration."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim)
    p = rng.standard_normal(dim)
    xi = 0.0

    n_samples = (n_steps - burn_in) // thin
    samples = np.empty(n_samples, dtype=np.float64)
    rec_idx = 0
    half_dt = dt / 2.0
    d_kT = float(dim)

    for step in range(n_steps):
        p = p - grad_fn(q) * half_dt

        g_val = float(friction_fn(np.array([xi]))[0])
        if not math.isfinite(g_val):
            return None

        try:
            exp_fac = math.exp(-g_val * half_dt)
        except OverflowError:
            return None

        p = p * exp_fac
        q = q + p * dt
        p = p * exp_fac
        p = p - grad_fn(q) * half_dt
        xi = xi + (float(np.dot(p, p)) - float(dim)) * dt

        if not math.isfinite(xi) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            return None

        if step >= burn_in and (step - burn_in) % thin == 0:
            if rec_idx < n_samples:
                samples[rec_idx] = q[0]
                rec_idx += 1

    if rec_idx < n_samples:
        return None
    return samples


def sokal_tau_fft(x, c=6.0):
    """Quick Sokal tau_int."""
    N = len(x)
    if N < 10:
        return 50000.0

    x_c = x - x.mean()
    var = float(np.var(x_c))
    if var == 0.0 or not math.isfinite(var):
        return 50000.0

    nfft = 1
    while nfft < 2 * N:
        nfft <<= 1

    F = np.fft.rfft(x_c, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N].real
    acf = acf / (N - np.arange(N))

    C0 = float(acf[0])
    if C0 <= 0.0 or not math.isfinite(C0):
        return 50000.0

    rho = acf / C0
    tau = 1.0

    for t in range(1, N):
        tau += 2.0 * rho[t]
        if not math.isfinite(tau) or tau < 1.0:
            tau = 1.0
        if t > c * tau:
            return max(1.0, min(tau, 50000.0))

    return 50000.0


def kl_proxy(samples, potential_name):
    """Quick KL divergence."""
    if samples is None or len(samples) == 0:
        return float('inf')

    bin_edges = np.linspace(-6, 6, 101)
    counts, _ = np.histogram(np.clip(samples, -6, 6), bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return float('inf')

    p_emp = counts.astype(np.float64) / total
    mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bw = bin_edges[1] - bin_edges[0]

    if potential_name == 'harmonic_1d':
        log_p = -0.5 * mids**2
    elif potential_name == 'doublewell_2d':
        log_p = -(mids**2 - 1.0)**2
    elif potential_name == 'gaussmix_2d':
        MU_X = [3.0 * math.cos(2.0 * math.pi * k / 5) for k in range(5)]
        p_ref = np.zeros(len(mids))
        for mu_x in MU_X:
            p_ref += np.exp(-0.5 * (mids - mu_x)**2) / 5.0
        log_p = np.log(np.clip(p_ref, 1e-300, None))
    else:
        return float('inf')

    if potential_name != 'gaussmix_2d':
        log_p -= log_p.max()
        p_ref_vals = np.exp(log_p) * bw
    else:
        p_ref_vals = np.exp(log_p) * bw

    p_ref_vals = np.clip(p_ref_vals, 1e-300, None)
    p_ref_vals = p_ref_vals / p_ref_vals.sum()

    kl = 0.0
    for pi, qi in zip(p_emp, p_ref_vals):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            kl += pi * math.log(pi / qi)
    return max(0.0, kl)


def evaluate_params(params_vec, proxy_seeds=(42, 137)):
    """Evaluate composite friction parameters on proxy integrations.
    Uses 2 seeds for speed during optimization."""
    a, b, e, f, c = params_vec

    # Sanity bounds
    if a < 0.01 or b < 0.01 or e < -0.01 or f < -0.01 or c < 0.01:
        return 1e6
    if a > 10.0 or b > 50.0 or e > 5.0 or f > 10.0 or c > 20.0:
        return 1e6

    def ff(xi):
        xi2 = xi * xi
        return xi * (a + b * xi2) * np.exp(-e * xi2) + f * np.tanh(c * xi)

    potentials = [
        ('harmonic_1d', _grad_harmonic_1d, 1),
        ('doublewell_2d', _grad_doublewell_2d, 2),
        ('gaussmix_2d', _grad_gaussmix_2d, 2),
    ]
    weights = {'harmonic_1d': 0.024, 'doublewell_2d': 0.294, 'gaussmix_2d': 0.682}

    total = 0.0
    for pot_name, grad_fn, dim in potentials:
        tau_sum = 0.0
        for seed in proxy_seeds:
            samples = proxy_integrate(grad_fn, dim, seed, ff, n_steps=50000)
            if samples is None:
                return 1e6
            kl = kl_proxy(samples, pot_name)
            if kl > 0.04:
                return 1e6
            tau = sokal_tau_fft(samples)
            tau_sum += tau
        total += weights[pot_name] * tau_sum / len(proxy_seeds)

    return total


def cmaes_optimize(x0, sigma0=0.3, popsize=14, max_iter=80):
    """CMA-ES with diagonal covariance."""
    n = len(x0)
    xmean = np.array(x0, dtype=np.float64)
    sigma = sigma0

    lam = popsize
    mu = lam // 2

    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / weights.sum()
    mueff = 1.0 / np.sum(weights**2)

    cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    c1 = 2.0 / ((n + 1.3)**2 + mueff)
    cmu_val = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0)**2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chiN = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))

    pc = np.zeros(n)
    ps = np.zeros(n)
    C_diag = np.ones(n)

    best_x = xmean.copy()
    best_f = float('inf')

    for gen in range(max_iter):
        t0 = time.time()

        arz = np.random.randn(lam, n)
        arx = xmean[np.newaxis, :] + sigma * np.sqrt(C_diag)[np.newaxis, :] * arz

        fitvals = np.array([evaluate_params(arx[k]) for k in range(lam)])

        idx = np.argsort(fitvals)
        arx = arx[idx]
        arz = arz[idx]
        fitvals = fitvals[idx]

        if fitvals[0] < best_f:
            best_f = fitvals[0]
            best_x = arx[0].copy()

        xold = xmean.copy()
        xmean = np.dot(weights, arx[:mu])
        zmean = np.dot(weights, arz[:mu])

        ps = (1.0 - cs) * ps + math.sqrt(cs * (2.0 - cs) * mueff) * zmean
        hsig = (np.linalg.norm(ps) /
                math.sqrt(1.0 - (1.0 - cs)**(2.0 * (gen + 1))) / chiN
                < 1.4 + 2.0 / (n + 1.0))

        pc = (1.0 - cc) * pc + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * (xmean - xold) / sigma
        artmp = (arx[:mu] - xold[np.newaxis, :]) / sigma
        C_diag = ((1.0 - c1 - cmu_val) * C_diag +
                  c1 * (pc**2 + (1.0 - hsig) * cc * (2.0 - cc) * C_diag) +
                  cmu_val * np.dot(weights, artmp**2))
        C_diag = np.clip(C_diag, 1e-20, 1e20)

        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1.0))
        sigma = min(sigma, 5.0)

        elapsed = time.time() - t0
        print(f"Gen {gen:3d}: best_f={best_f:.4f} cur_best={fitvals[0]:.4f} "
              f"sigma={sigma:.4f} params={best_x} ({elapsed:.1f}s)")

        if sigma < 1e-8:
            break

    return best_x, best_f


if __name__ == '__main__':
    np.random.seed(2024)

    print("=== CMA-ES Optimization for Composite Friction ===")
    print("g(xi) = xi*(a + b*xi^2)*exp(-e*xi^2) + f*tanh(c*xi)")
    print()

    # Start from parent Padé approximation
    x0 = [0.7, 3.0, 0.03, 0.5, 1.0]  # [a, b, e, f, c]

    best_x, best_f = cmaes_optimize(x0, sigma0=0.4, popsize=14, max_iter=60)

    print()
    print("=" * 60)
    print(f"Best parameters: a={best_x[0]:.6f}, b={best_x[1]:.6f}, "
          f"e={best_x[2]:.6f}, f={best_x[3]:.6f}, c={best_x[4]:.6f}")
    print(f"Best proxy metric: {best_f:.4f}")
    print()

    # Verify with longer runs and more seeds
    print("Verifying with longer proxy (100k steps, 3 seeds)...")
    a, b, e, f, c = best_x
    def ff(xi):
        xi2 = xi * xi
        return xi * (a + b * xi2) * np.exp(-e * xi2) + f * np.tanh(c * xi)

    potentials = [
        ('harmonic_1d', _grad_harmonic_1d, 1),
        ('doublewell_2d', _grad_doublewell_2d, 2),
        ('gaussmix_2d', _grad_gaussmix_2d, 2),
    ]
    wts = {'harmonic_1d': 0.024, 'doublewell_2d': 0.294, 'gaussmix_2d': 0.682}

    total = 0.0
    for pot_name, grad_fn, dim in potentials:
        taus = []
        kls = []
        for seed in [42, 137, 2024]:
            samples = proxy_integrate(grad_fn, dim, seed, ff, n_steps=100000)
            if samples is None:
                print(f"  {pot_name} seed={seed}: DIVERGED")
                taus.append(50000)
                kls.append(float('inf'))
            else:
                tau = sokal_tau_fft(samples)
                kl = kl_proxy(samples, pot_name)
                taus.append(tau)
                kls.append(kl)
                print(f"  {pot_name} seed={seed}: tau={tau:.2f} kl={kl:.5f}")
        mean_tau = np.mean(taus)
        mean_kl = np.mean([k for k in kls if math.isfinite(k)])
        total += wts[pot_name] * mean_tau
        print(f"  {pot_name} mean: tau={mean_tau:.2f} kl={mean_kl:.5f}")

    print(f"\nProxy weighted tau: {total:.4f}")
    print(f"\nHardcode these in solution.py:")
    print(f"'a': {best_x[0]:.10f},")
    print(f"'b': {best_x[1]:.10f},")
    print(f"'e': {best_x[2]:.10f},")
    print(f"'f': {best_x[3]:.10f},")
    print(f"'c': {best_x[4]:.10f},")
