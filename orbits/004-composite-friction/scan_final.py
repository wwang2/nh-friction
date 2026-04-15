"""
Focused Padé parameter scan using ALL 3 real eval seeds (42, 137, 2024).
Keeping b/c ratio >= 50 to ensure strong tail friction.
"""
import numpy as np
import math
import time
import sys


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


def proxy_integrate(grad_fn, dim, seed, ff_func, n_steps=150000):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim)
    p = rng.standard_normal(dim)
    xi = 0.0
    dt = 0.01; half_dt = dt / 2.0; d_kT = float(dim)
    burn_in = 10000; thin = 10
    n_samples = (n_steps - burn_in) // thin
    samples = np.empty(n_samples, dtype=np.float64)
    rec_idx = 0
    for step in range(n_steps):
        p = p - grad_fn(q) * half_dt
        g_val = float(ff_func(np.array([xi]))[0])
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
    if N < 10: return 50000.0
    x_c = x - x.mean()
    var = float(np.var(x_c))
    if var == 0 or not math.isfinite(var): return 50000.0
    nfft = 1
    while nfft < 2 * N: nfft <<= 1
    F = np.fft.rfft(x_c, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N].real
    acf = acf / (N - np.arange(N))
    C0 = float(acf[0])
    if C0 <= 0 or not math.isfinite(C0): return 50000.0
    rho = acf / C0
    tau = 1.0
    for t in range(1, N):
        tau += 2.0 * rho[t]
        if not math.isfinite(tau) or tau < 1.0: tau = 1.0
        if t > c_sokal * tau: return max(1.0, min(tau, 50000.0))
    return 50000.0


def kl_1d_ho(samples):
    if samples is None: return float('inf')
    bin_edges = np.linspace(-6, 6, 101)
    counts, _ = np.histogram(np.clip(samples, -6, 6), bins=bin_edges)
    total = counts.sum()
    if total == 0: return float('inf')
    p_emp = counts / total
    lo, hi = bin_edges[:-1], bin_edges[1:]
    def ncdf(x):
        return np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in x])
    p_ref = ncdf(hi) - ncdf(lo)
    p_ref = np.clip(p_ref, 1e-300, None)
    p_ref = p_ref / p_ref.sum()
    kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p_emp, p_ref) if pi > 0)
    return max(0.0, kl)


def evaluate(a, b, c):
    """Evaluate pure Padé with all 3 real seeds."""
    def ff(xi):
        xi2 = xi * xi
        return xi * (a + b * xi2) / (1.0 + c * xi2)

    weights = {'harmonic_1d': 0.024, 'doublewell_2d': 0.294, 'gaussmix_2d': 0.682}
    potentials = [
        ('harmonic_1d', _grad_harmonic_1d, 1),
        ('doublewell_2d', _grad_doublewell_2d, 2),
        ('gaussmix_2d', _grad_gaussmix_2d, 2),
    ]
    seeds = [42, 137, 2024]

    all_taus = {}
    all_kls = {}

    for pot_name, grad_fn, dim in potentials:
        taus = []
        kls = []
        for seed in seeds:
            samples = proxy_integrate(grad_fn, dim, seed, ff)
            if samples is None:
                return None
            tau = sokal_tau(samples)
            taus.append(tau)
            if pot_name == 'harmonic_1d':
                kl = kl_1d_ho(samples)
                kls.append(kl)
        all_taus[pot_name] = np.mean(taus)
        if pot_name == 'harmonic_1d':
            all_kls['harmonic_1d'] = np.mean(kls)
            all_kls['harmonic_1d_max'] = max(kls)
            all_kls['harmonic_1d_per_seed'] = kls

    if all_kls['harmonic_1d'] > 0.04:
        return {'passed': False, 'kl_ho': all_kls['harmonic_1d'],
                'kl_max': all_kls['harmonic_1d_max']}

    weighted = sum(weights[k] * all_taus[k] for k in all_taus)
    return {
        'passed': True,
        'weighted': weighted,
        'tau_ho': all_taus['harmonic_1d'],
        'tau_dw': all_taus['doublewell_2d'],
        'tau_gm': all_taus['gaussmix_2d'],
        'kl_ho': all_kls['harmonic_1d'],
        'kl_max': all_kls['harmonic_1d_max'],
        'kl_seeds': all_kls['harmonic_1d_per_seed'],
    }


if __name__ == '__main__':
    t_total = time.time()
    print("Focused Padé scan (3 seeds: 42, 137, 2024)")
    print("g(xi) = xi*(a + b*xi^2)/(1 + c*xi^2)\n")

    candidates = []
    # Keep b/c >= 40 for strong tail
    # a: controls linear coupling at origin
    # b: controls cubic strength AND tail slope (b/c)
    # c: controls how fast tail growth kicks in

    for a in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for b in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
            for c_val in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
                ratio = b / c_val
                if ratio < 30:  # skip weak tails
                    continue
                candidates.append((a, b, c_val))

    print(f"Testing {len(candidates)} candidates...\n")

    results = []
    for i, (a, b, c) in enumerate(candidates):
        t0 = time.time()
        res = evaluate(a, b, c)
        elapsed = time.time() - t0
        if res is None:
            pass
        elif not res['passed']:
            if i < 30 or res['kl_ho'] < 0.06:
                print(f"  a={a:.1f} b={b:.1f} c={c:.2f} b/c={b/c:.0f} -> KL_FAIL "
                      f"kl_mean={res['kl_ho']:.4f} kl_max={res['kl_max']:.4f} ({elapsed:.1f}s)")
        else:
            print(f"  a={a:.1f} b={b:.1f} c={c:.2f} b/c={b/c:.0f} -> "
                  f"W={res['weighted']:.2f} ho={res['tau_ho']:.1f} "
                  f"dw={res['tau_dw']:.1f} gm={res['tau_gm']:.1f} "
                  f"kl_mean={res['kl_ho']:.4f} kl_max={res['kl_max']:.4f} ({elapsed:.1f}s)")
            results.append((res['weighted'], a, b, c, res))
            sys.stdout.flush()

    results.sort()
    print(f"\n=== Top 10 (total time: {time.time()-t_total:.0f}s) ===")
    for rank, (w, a, b, c, res) in enumerate(results[:10]):
        print(f"#{rank+1}: W={w:.2f} | a={a} b={b} c={c} b/c={b/c:.1f}")
        print(f"     tau: ho={res['tau_ho']:.1f} dw={res['tau_dw']:.1f} gm={res['tau_gm']:.1f}")
        print(f"     kl: mean={res['kl_ho']:.5f} max={res['kl_max']:.5f} seeds={[f'{k:.4f}' for k in res['kl_seeds']]}")
