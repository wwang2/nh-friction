"""Generate figures for orbit 003-cmaes-3pot."""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10.0,
    "axes.labelpad": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handletextpad": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
})

COLORS = {
    'baseline': '#888888',
    'linear': '#C44E52',
    'pade': '#4C72B0',
    'pade_light': '#8BACD4',
}


def _integrate_trajectory(grad_fn, dim, seed, friction_fn, n_steps=200_000):
    """Integrate and return full trajectory for visualization."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim)
    p = rng.standard_normal(dim)
    xi = 0.0
    dt = 0.01
    half_dt = dt / 2.0
    d_kT = float(dim)

    burnin = 5000
    thin = 10
    n_main = n_steps - burnin
    n_samples = n_main // thin
    qs = np.empty(n_samples, dtype=np.float64)
    xis = np.empty(n_samples, dtype=np.float64)
    rec = 0

    for step in range(n_steps):
        p = p - grad_fn(q) * half_dt
        gxi = friction_fn(xi)
        try:
            ef = math.exp(-gxi * half_dt)
        except OverflowError:
            break
        p = p * ef
        q = q + p * dt
        p = p * ef
        p = p - grad_fn(q) * half_dt
        xi = xi + (float(np.dot(p, p)) - d_kT) * dt
        if not math.isfinite(xi) or not np.all(np.isfinite(q)):
            break
        if step >= burnin and (step - burnin) % thin == 0 and rec < n_samples:
            qs[rec] = q[0]
            xis[rec] = xi
            rec += 1

    return qs[:rec], xis[:rec]


def linear_friction(xi_val):
    return xi_val

def pade_friction(xi_val):
    xi2 = xi_val * xi_val
    return xi_val * (0.7 + 3.0 * xi2) / (1.0 + 0.06 * xi2)


def make_narrative():
    """Narrative figure: friction function shape + phase-space trajectories + density comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel (a): Friction function shape
    ax = axes[0, 0]
    xi = np.linspace(-4, 4, 500)
    g_linear = xi
    xi2 = xi * xi
    g_pade = xi * (0.7 + 3.0 * xi2) / (1.0 + 0.06 * xi2)

    ax.plot(xi, g_linear, '--', color=COLORS['linear'], lw=2, label='Linear: g(xi)=xi')
    ax.plot(xi, g_pade, '-', color=COLORS['pade'], lw=2.5, label='Pade: a=0.7, b=3.0, c=0.06')
    ax.set_xlabel('xi')
    ax.set_ylabel('g(xi)')
    ax.set_title('Friction function shape')
    ax.legend(loc='upper left')
    ax.set_ylim(-80, 80)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # Panel (b): Phase-space trajectory — 1D harmonic, linear (KAM tori)
    ax = axes[0, 1]

    def grad_h1d(q):
        return q.copy()

    qs_lin, xis_lin = _integrate_trajectory(grad_h1d, 1, 2024, linear_friction, 100_000)
    ax.plot(qs_lin[:2000], xis_lin[:2000], '.', color=COLORS['linear'], alpha=0.1, markersize=1, rasterized=True)
    ax.set_xlabel('q')
    ax.set_ylabel('xi')
    ax.set_title('1D Harmonic: linear g(xi)=xi')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.text(0.5, 0.95, 'KAM tori trapping', transform=ax.transAxes,
            ha='center', va='top', fontsize=10, color=COLORS['linear'], fontweight='medium')

    # Panel (c): Phase-space trajectory — 1D harmonic, Pade (ergodic)
    ax = axes[0, 2]
    qs_pade, xis_pade = _integrate_trajectory(grad_h1d, 1, 2024, pade_friction, 100_000)
    ax.plot(qs_pade[:2000], xis_pade[:2000], '.', color=COLORS['pade'], alpha=0.1, markersize=1, rasterized=True)
    ax.set_xlabel('q')
    ax.set_ylabel('xi')
    ax.set_title('1D Harmonic: Pade friction')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.text(0.5, 0.95, 'Ergodic mixing', transform=ax.transAxes,
            ha='center', va='top', fontsize=10, color=COLORS['pade'], fontweight='medium')

    # Panel (d): 1D harmonic density comparison
    ax = axes[1, 0]
    x_ref = np.linspace(-4, 4, 200)
    p_ref = np.exp(-0.5 * x_ref**2) / np.sqrt(2 * np.pi)
    ax.plot(x_ref, p_ref, 'k-', lw=2, label='Boltzmann (exact)')
    if len(qs_lin) > 100:
        ax.hist(qs_lin, bins=60, density=True, alpha=0.4, color=COLORS['linear'], label='Linear NH')
    if len(qs_pade) > 100:
        ax.hist(qs_pade, bins=60, density=True, alpha=0.4, color=COLORS['pade'], label='Pade NH')
    ax.set_xlabel('q')
    ax.set_ylabel('P(q)')
    ax.set_title('Harmonic: marginal density')
    ax.legend(loc='upper right')
    ax.set_xlim(-4, 4)
    ax.text(-0.12, 1.05, '(d)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # Panel (e): 2D gaussmix trajectory
    ax = axes[1, 1]
    GM_MU = np.array([
        [3.0 * math.cos(2.0 * math.pi * k / 5),
         3.0 * math.sin(2.0 * math.pi * k / 5)]
        for k in range(5)
    ])

    def grad_gm(q):
        diff = q[np.newaxis, :] - GM_MU
        log_w = -0.5 * np.sum(diff ** 2, axis=1)
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()
        return np.einsum("k,kd->d", w, diff)

    # Run 2D integration
    rng = np.random.default_rng(42)
    q = rng.standard_normal(2)
    p = rng.standard_normal(2)
    xi = 0.0
    dt = 0.01
    half_dt = dt / 2.0
    qs_2d = []
    for step in range(200_000):
        p = p - grad_gm(q) * half_dt
        gxi = pade_friction(xi)
        try:
            ef = math.exp(-gxi * half_dt)
        except OverflowError:
            break
        p = p * ef
        q = q + p * dt
        p = p * ef
        p = p - grad_gm(q) * half_dt
        xi = xi + (float(np.dot(p, p)) - 2.0) * dt
        if not math.isfinite(xi) or not np.all(np.isfinite(q)):
            break
        if step >= 5000 and step % 10 == 0:
            qs_2d.append(q.copy())

    qs_2d = np.array(qs_2d)
    if len(qs_2d) > 100:
        ax.hexbin(qs_2d[:, 0], qs_2d[:, 1], gridsize=35, cmap='GnBu', mincnt=1)
        for k, mu in enumerate(GM_MU):
            ax.plot(mu[0], mu[1], 'r*', markersize=10, zorder=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Gaussmix 2D: Pade sampling')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.text(-0.12, 1.05, '(e)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # Panel (f): Parameter landscape (grid search results)
    ax = axes[1, 2]
    # Show the search space with parent optimum highlighted
    # Param sensitivity: metric vs perturbation
    perturbations = {
        'parent\n(0.7,3.0,0.06)': 84.14,
        'a=0.75': 101.7,
        'a=0.8': 92.6,
        'b=6,c=0\nd=0.02': 124.7,
        'd=0.02': 100.2,
    }
    names = list(perturbations.keys())
    vals = list(perturbations.values())
    colors_bar = [COLORS['pade'] if v == 84.14 else COLORS['baseline'] for v in vals]

    bars = ax.bar(range(len(names)), vals, color=colors_bar, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Metric (weighted tau)')
    ax.set_title('Parameter sensitivity')
    ax.axhline(84.14, color=COLORS['pade'], ls='--', lw=1, alpha=0.5)
    ax.text(-0.12, 1.05, '(f)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    fig.savefig('orbits/003-cmaes-3pot/figures/narrative.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Saved narrative.png")


def make_results():
    """Results figure: metric breakdown, seed comparison, search landscape."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a): Per-potential tau breakdown
    ax = axes[0]
    potentials = ['harmonic_1d', 'doublewell_2d', 'gaussmix_2d']
    weights = [0.024, 0.294, 0.682]
    taus_parent = [10.36, 114.08, 73.81]
    weighted_parent = [w * t for w, t in zip(weights, taus_parent)]

    # NHC baseline (from config)
    taus_nhc = [5.7, 70.5, 163.2]  # from config
    weighted_nhc = [w * t for w, t in zip(weights, taus_nhc)]

    x = np.arange(len(potentials))
    width = 0.35
    ax.bar(x - width/2, weighted_nhc, width, color=COLORS['baseline'], alpha=0.7, label='NHC (M=3) baseline')
    ax.bar(x + width/2, weighted_parent, width, color=COLORS['pade'], alpha=0.8, label='Pade (this orbit)')

    ax.set_xticks(x)
    ax.set_xticklabels(['1D Harm.\n(2.4%)', '2D DW\n(29.4%)', '2D GM\n(68.2%)'], fontsize=9)
    ax.set_ylabel('Weighted tau contribution')
    ax.set_title('Metric breakdown by potential')
    ax.legend()

    # Add value labels
    for i, (nhc, pade) in enumerate(zip(weighted_nhc, weighted_parent)):
        ax.text(i - width/2, nhc + 0.5, f'{nhc:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, pade + 0.5, f'{pade:.1f}', ha='center', va='bottom', fontsize=8)

    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # Panel (b): Per-seed tau values
    ax = axes[1]
    seeds = [42, 137, 2024]

    h1d_taus = [9.13, 9.98, 11.98]
    dw_taus = [122.59, 94.21, 125.45]
    gm_taus = [72.08, 61.94, 87.40]

    x = np.arange(3)
    width = 0.25
    ax.bar(x - width, h1d_taus, width, color='#55A868', alpha=0.8, label='1D Harmonic')
    ax.bar(x, dw_taus, width, color='#DD8452', alpha=0.8, label='2D Double-well')
    ax.bar(x + width, gm_taus, width, color='#4C72B0', alpha=0.8, label='2D Gaussmix')

    ax.set_xticks(x)
    ax.set_xticklabels([f'seed={s}' for s in seeds])
    ax.set_ylabel('tau_int')
    ax.set_title('Per-seed autocorrelation times')
    ax.legend(loc='upper left', fontsize=8)
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # Panel (c): Grid search overview — scatter of metric vs candidate
    ax = axes[2]

    # Tested candidates (from real eval)
    labels = ['parent\na=0.7,b=3', 'a=0.75', 'a=0.8', 'b=6,c=0,d=.02', 'c=.06,d=.02']
    metrics = [84.14, 101.74, 92.64, 124.74, 100.23]
    colors_scatter = [COLORS['pade']] + [COLORS['baseline']] * 4

    ax.scatter(range(len(labels)), metrics, c=colors_scatter, s=100, zorder=5, edgecolors='black', linewidths=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha='right')
    ax.set_ylabel('Metric (weighted tau)')
    ax.set_title('Candidate comparison (real eval)')
    ax.axhline(84.14, color=COLORS['pade'], ls='--', lw=1, alpha=0.5)
    ax.axhline(150, color='red', ls=':', lw=1, alpha=0.3)
    ax.text(len(labels)-0.5, 152, 'NHC target', fontsize=8, color='red', alpha=0.5)

    for i, m in enumerate(metrics):
        ax.text(i, m + 2, f'{m:.1f}', ha='center', va='bottom', fontsize=9)

    ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    fig.savefig('orbits/003-cmaes-3pot/figures/results.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Saved results.png")


if __name__ == "__main__":
    make_narrative()
    make_results()
    print("Done.")
