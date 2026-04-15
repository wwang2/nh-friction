"""
Sweep d and e for the Padé + cubic boost form.
g(xi) = xi*(0.7 + 3*xi^2)/(1 + 0.06*xi^2) + d*xi^3*exp(-e*xi^2)

Also sweep the Padé params a, b, c to look for improvements.
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


def proxy_integrate(grad_fn, dim, seed, ff_func, n_steps=120000):
    """Longer proxy for more reliable tau estimates."""
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


def evaluate(a, b, c, d, e, seeds=(42, 137)):
    """Full proxy evaluation with all 3 potentials, 3 seeds."""
    def ff(xi):
        xi2 = xi * xi
        pade = xi * (a + b * xi2) / (1.0 + c * xi2)
        boost = d * xi * xi2 * np.exp(-e * xi2)
        return pade + boost

    weights = {'harmonic_1d': 0.024, 'doublewell_2d': 0.294, 'gaussmix_2d': 0.682}
    potentials = [
        ('harmonic_1d', _grad_harmonic_1d, 1),
        ('doublewell_2d', _grad_doublewell_2d, 2),
        ('gaussmix_2d', _grad_gaussmix_2d, 2),
    ]

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

    if all_kls['harmonic_1d'] > 0.035:
        return {'passed': False, 'kl_ho': all_kls['harmonic_1d']}

    weighted = sum(weights[k] * all_taus[k] for k in all_taus)
    return {
        'passed': True,
        'weighted': weighted,
        'tau_ho': all_taus['harmonic_1d'],
        'tau_dw': all_taus['doublewell_2d'],
        'tau_gm': all_taus['gaussmix_2d'],
        'kl_ho': all_kls['harmonic_1d'],
    }


if __name__ == '__main__':
    t_total = time.time()

    # Phase 1: Sweep boost params d, e with parent Padé backbone
    print("Phase 1: Padé backbone (a=0.7, b=3.0, c=0.06) + boost sweep\n")

    a0, b0, c0 = 0.7, 3.0, 0.06
    candidates = []

    for d_val in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]:
        for e_val in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
            if d_val == 0.0 and e_val != 0.1:
                continue  # only need one run for d=0
            candidates.append((a0, b0, c0, d_val, e_val))

    results = []
    for i, (a, b, c, d, e) in enumerate(candidates):
        t0 = time.time()
        res = evaluate(a, b, c, d, e)
        elapsed = time.time() - t0
        if res is None:
            print(f"  a={a} b={b} c={c} d={d:.1f} e={e:.1f} -> DIVERGED ({elapsed:.1f}s)")
        elif not res['passed']:
            print(f"  a={a} b={b} c={c} d={d:.1f} e={e:.1f} -> KL_FAIL kl={res['kl_ho']:.4f} ({elapsed:.1f}s)")
        else:
            print(f"  a={a} b={b} c={c} d={d:.1f} e={e:.1f} -> "
                  f"W={res['weighted']:.2f} ho={res['tau_ho']:.1f} "
                  f"dw={res['tau_dw']:.1f} gm={res['tau_gm']:.1f} "
                  f"kl={res['kl_ho']:.4f} ({elapsed:.1f}s)")
            results.append((res['weighted'], a, b, c, d, e, res))
            sys.stdout.flush()

    # Phase 2: Sweep Padé params with best boost
    print("\nPhase 2: Sweep Padé params\n")

    for a in [0.5, 0.7, 0.9]:
        for b in [2.0, 3.0, 4.0, 5.0]:
            for c_val in [0.03, 0.06, 0.10]:
                t0 = time.time()
                res = evaluate(a, b, c_val, 0.0, 0.3)  # no boost, just Padé sweep
                elapsed = time.time() - t0
                if res is None:
                    continue
                elif not res['passed']:
                    continue
                else:
                    if res['weighted'] < 84.0:  # only print improvements
                        print(f"  a={a} b={b} c={c_val} -> "
                              f"W={res['weighted']:.2f} ho={res['tau_ho']:.1f} "
                              f"dw={res['tau_dw']:.1f} gm={res['tau_gm']:.1f} "
                              f"kl={res['kl_ho']:.4f} ({elapsed:.1f}s)")
                    results.append((res['weighted'], a, b, c_val, 0.0, 0.3, res))
                    sys.stdout.flush()

    results.sort()
    print(f"\n=== Top 10 (total time: {time.time()-t_total:.0f}s) ===")
    for rank, (w, a, b, c, d, e, res) in enumerate(results[:10]):
        print(f"#{rank+1}: W={w:.2f} | a={a} b={b} c={c} d={d} e={e}")
        print(f"     tau: ho={res['tau_ho']:.1f} dw={res['tau_dw']:.1f} gm={res['tau_gm']:.1f} kl_ho={res['kl_ho']:.5f}")
