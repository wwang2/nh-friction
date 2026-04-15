"""
Quick parameter scan for composite friction.
Uses short proxy runs (50k steps) to find parameters that pass KL gate.
"""
import numpy as np
import math
import time
import itertools


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


def proxy_integrate(grad_fn, dim, seed, a, b, e, f, c, n_steps=50000):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim)
    p = rng.standard_normal(dim)
    xi = 0.0
    dt = 0.01
    half_dt = dt / 2.0
    d_kT = float(dim)
    burn_in = 5000
    thin = 10
    n_samples = (n_steps - burn_in) // thin
    samples = np.empty(n_samples, dtype=np.float64)
    rec_idx = 0

    for step in range(n_steps):
        p = p - grad_fn(q) * half_dt
        xi2_val = xi * xi
        g_val = xi * (a + b * xi2_val) * math.exp(-e * xi2_val) + f * math.tanh(c * xi)
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
        xi = xi + (float(np.dot(p, p)) - d_kT) * dt
        if not math.isfinite(xi) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            return None
        if step >= burn_in and (step - burn_in) % thin == 0:
            if rec_idx < n_samples:
                samples[rec_idx] = q[0]
                rec_idx += 1

    return samples if rec_idx == n_samples else None


def sokal_tau(x, c_sokal=6.0):
    N = len(x)
    if N < 10:
        return 50000.0
    x_c = x - x.mean()
    var = float(np.var(x_c))
    if var == 0 or not math.isfinite(var):
        return 50000.0
    nfft = 1
    while nfft < 2 * N:
        nfft <<= 1
    F = np.fft.rfft(x_c, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N].real
    acf = acf / (N - np.arange(N))
    C0 = float(acf[0])
    if C0 <= 0 or not math.isfinite(C0):
        return 50000.0
    rho = acf / C0
    tau = 1.0
    for t in range(1, N):
        tau += 2.0 * rho[t]
        if not math.isfinite(tau) or tau < 1.0:
            tau = 1.0
        if t > c_sokal * tau:
            return max(1.0, min(tau, 50000.0))
    return 50000.0


def kl_1d_ho(samples):
    """KL for 1D harmonic oscillator (reference: N(0,1))."""
    if samples is None:
        return float('inf')
    bin_edges = np.linspace(-6, 6, 101)
    counts, _ = np.histogram(np.clip(samples, -6, 6), bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return float('inf')
    p_emp = counts / total
    lo, hi = bin_edges[:-1], bin_edges[1:]
    from math import erf as merf
    def ncdf(x):
        return np.array([0.5 * (1.0 + merf(v / math.sqrt(2.0))) for v in x])
    p_ref = ncdf(hi) - ncdf(lo)
    p_ref = np.clip(p_ref, 1e-300, None)
    p_ref = p_ref / p_ref.sum()
    kl = 0.0
    for pi, qi in zip(p_emp, p_ref):
        if pi > 0:
            kl += pi * math.log(pi / qi)
    return max(0.0, kl)


def evaluate_candidate(a, b, e, f, c):
    """Quick evaluation on 1D HO only (KL gate check) + all 3 potentials for tau."""
    seed = 42

    # First check KL on 1D HO
    samples_ho = proxy_integrate(_grad_harmonic_1d, 1, seed, a, b, e, f, c, n_steps=80000)
    if samples_ho is None:
        return None
    kl_ho = kl_1d_ho(samples_ho)
    if kl_ho > 0.03:  # strict gate
        return {'kl_ho': kl_ho, 'passed': False}

    tau_ho = sokal_tau(samples_ho)

    # Double well
    samples_dw = proxy_integrate(_grad_doublewell_2d, 2, seed, a, b, e, f, c, n_steps=80000)
    if samples_dw is None:
        return None
    tau_dw = sokal_tau(samples_dw)

    # Gaussmix
    samples_gm = proxy_integrate(_grad_gaussmix_2d, 2, seed, a, b, e, f, c, n_steps=80000)
    if samples_gm is None:
        return None
    tau_gm = sokal_tau(samples_gm)

    weighted = 0.024 * tau_ho + 0.294 * tau_dw + 0.682 * tau_gm
    return {
        'kl_ho': kl_ho, 'tau_ho': tau_ho, 'tau_dw': tau_dw, 'tau_gm': tau_gm,
        'weighted': weighted, 'passed': True
    }


if __name__ == '__main__':
    print("Parameter scan for composite friction")
    print("g(xi) = xi*(a + b*xi^2)*exp(-e*xi^2) + f*tanh(c*xi)")
    print()

    # Parameter grid - focused around promising regions
    # Key: need stronger damping e or stronger tanh f to pass KL gate on 1D HO
    candidates = [
        # (a, b, e, f, c) - description
        (0.7, 3.0, 0.05, 0.5, 1.0),   # more damping
        (0.7, 3.0, 0.1, 0.5, 1.0),    # even more damping
        (0.7, 3.0, 0.15, 0.5, 1.0),   # strong damping
        (0.5, 3.0, 0.1, 0.7, 1.0),    # lower a, higher f
        (0.5, 2.0, 0.1, 0.8, 1.5),    # lower b, stronger tanh
        (0.7, 2.5, 0.08, 0.6, 1.2),   # balanced
        (0.8, 3.5, 0.1, 0.3, 0.8),    # strong core, weak tanh
        (0.6, 4.0, 0.15, 0.4, 1.0),   # very strong cubic, heavy damping
        (0.5, 5.0, 0.2, 0.5, 1.0),    # extreme cubic, heavy damping
        (0.7, 3.0, 0.08, 0.5, 1.5),   # moderate damping, steeper tanh
        (0.3, 3.0, 0.1, 0.8, 0.8),    # low a, high f
        (0.4, 4.0, 0.12, 0.6, 1.0),   # moderate
        (1.0, 2.0, 0.05, 0.3, 1.0),   # strong linear, weak cubic
        (0.6, 3.0, 0.1, 0.6, 1.0),    # slightly lower a
        (0.7, 3.0, 0.07, 0.5, 2.0),   # steep tanh
        (0.5, 3.5, 0.1, 0.5, 1.0),    # lower a, higher b
    ]

    results = []
    for i, (a, b, e, f, c) in enumerate(candidates):
        t0 = time.time()
        res = evaluate_candidate(a, b, e, f, c)
        elapsed = time.time() - t0
        if res is None:
            print(f"[{i:2d}] a={a:.1f} b={b:.1f} e={e:.2f} f={f:.1f} c={c:.1f} -> DIVERGED ({elapsed:.1f}s)")
        elif not res['passed']:
            print(f"[{i:2d}] a={a:.1f} b={b:.1f} e={e:.2f} f={f:.1f} c={c:.1f} -> KL_FAIL kl={res['kl_ho']:.4f} ({elapsed:.1f}s)")
        else:
            print(f"[{i:2d}] a={a:.1f} b={b:.1f} e={e:.2f} f={f:.1f} c={c:.1f} -> "
                  f"weighted={res['weighted']:.2f} tau_ho={res['tau_ho']:.1f} "
                  f"tau_dw={res['tau_dw']:.1f} tau_gm={res['tau_gm']:.1f} "
                  f"kl={res['kl_ho']:.4f} ({elapsed:.1f}s)")
            results.append((res['weighted'], a, b, e, f, c, res))

    if results:
        results.sort()
        print("\n=== Top 5 ===")
        for rank, (w, a, b, e, f, c, res) in enumerate(results[:5]):
            print(f"#{rank+1}: weighted={w:.2f} | a={a:.4f} b={b:.4f} e={e:.4f} f={f:.4f} c={c:.4f}")
            print(f"     tau_ho={res['tau_ho']:.1f} tau_dw={res['tau_dw']:.1f} tau_gm={res['tau_gm']:.1f} kl_ho={res['kl_ho']:.5f}")
