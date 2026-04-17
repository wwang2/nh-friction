"""Generate narrative.png and results.png for orbit-015."""
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
    'orbit012': '#4C72B0',
    'orbit014': '#55A868',
    'combined': '#C44E52',
    'best': '#DD8452',
}

FIGDIR = "/Users/wujiewang/code/bath/.worktrees/015-combined-b1-alpha074/orbits/015-combined-b1-alpha074/figures"

# ============================================================================
# FIGURE 1: narrative.png — Why the combination fails
# ============================================================================

def make_narrative():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel (a): Friction function g(xi) for different b values
    ax = axes[0]
    xi = np.linspace(-5, 5, 500)
    a, c = 0.70, 0.06

    for b_val, label, color, ls in [
        (3.0, 'b=3.0 (orbit-014)', COLORS['orbit014'], '-'),
        (1.0, 'b=1.0 (orbit-012)', COLORS['orbit012'], '--'),
    ]:
        g = xi * (a + b_val * xi**2) / (1.0 + c * xi**2)
        ax.plot(xi, g, color=color, linestyle=ls, linewidth=2, label=label)

    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$g(\xi)$')
    ax.set_title('Friction function shape')
    ax.legend(loc='upper left')
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.set_ylim(-150, 150)

    # Panel (b): Effective thermostat strength diagram
    ax = axes[1]
    # Show how b and alpha both reduce effective thermostat coupling
    alphas = [1.0, 0.74]
    bs = [3.0, 1.0]
    labels = ['Standard\n(b=3, alpha=1)', 'orbit-014\n(b=3, alpha=0.74)',
              'orbit-012\n(b=1, alpha=1)', 'Combined\n(b=1, alpha=0.74)']
    # Effective coupling ~ b * alpha (rough proxy)
    couplings = [3.0*1.0, 3.0*0.74, 1.0*1.0, 1.0*0.74]
    tau_gms = [73.8, 57.7, 59.2, 169.5]
    colors_bar = [COLORS['baseline'], COLORS['orbit014'], COLORS['orbit012'], COLORS['combined']]

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, tau_gms, color=colors_bar, width=0.6, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, tau_gms):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='medium')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r'$\tau_{\mathrm{gm}}$ (gaussmix)')
    ax.set_title('Gaussmix autocorrelation time')
    ax.axhline(y=73.8, color=COLORS['baseline'], linestyle=':', alpha=0.5, linewidth=1)
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # Panel (c): b sweep with alpha=0.74 showing the interaction
    ax = axes[2]
    b_vals =   [1.0,  1.5,   2.0,  2.5,  3.0,  3.5,  4.0]
    tau_sweep = [169.5, 124.4, 92.5, 67.4, 57.7, 60.4, 73.9]

    ax.plot(b_vals, tau_sweep, 'o-', color=COLORS['orbit014'], linewidth=2, markersize=8)
    ax.axhline(y=57.7, color=COLORS['orbit014'], linestyle=':', alpha=0.4, linewidth=1)
    ax.axhline(y=59.2, color=COLORS['orbit012'], linestyle=':', alpha=0.4, linewidth=1)
    ax.text(3.8, 58.5, 'b=3.0 optimum', fontsize=9, color=COLORS['orbit014'], alpha=0.7)
    ax.text(3.8, 61.0, 'b=1.0, alpha=1.0', fontsize=9, color=COLORS['orbit012'], alpha=0.7)

    # Mark the b=1.0 point as the failed combination
    ax.annotate('Combined\n(REJECTED)', xy=(1.0, 169.5), fontsize=9,
                color=COLORS['combined'],
                arrowprops=dict(arrowstyle='->', color=COLORS['combined'], lw=1.2),
                xytext=(1.5, 155))

    ax.set_xlabel('b (friction cubic coefficient)')
    ax.set_ylabel(r'$\tau_{\mathrm{gm}}$ (gaussmix)')
    ax.set_title('b sweep with alpha=0.74')
    ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    fig.suptitle('Orbit-015: Why b=1.0 + alpha=0.74 fails for gaussmix', fontsize=14, fontweight='medium', y=1.02)
    fig.savefig(f'{FIGDIR}/narrative.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("narrative.png saved")


# ============================================================================
# FIGURE 2: results.png — Quantitative comparison
# ============================================================================

def make_results():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel (a): METRIC comparison across configurations
    ax = axes[0]
    configs = [
        ('Standard\n(no alpha)', 60.34, COLORS['baseline']),
        ('orbit-012\n(b=1, al=1)', 49.54, COLORS['orbit012']),
        ('orbit-014\n(b=3, al=0.74)', 48.45, COLORS['orbit014']),
        ('Combined\n(b=1, al=0.74)', 124.69, COLORS['combined']),
    ]
    labels = [c[0] for c in configs]
    metrics = [c[1] for c in configs]
    colors_bar = [c[2] for c in configs]

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, metrics, color=colors_bar, width=0.6, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, metrics):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='medium')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('METRIC (weighted tau_int)')
    ax.set_title('Weighted metric comparison')
    ax.axhline(y=65, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(3.3, 66, 'target < 65', fontsize=9, color='gray', alpha=0.6)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # Panel (b): Per-potential tau breakdown for best config
    ax = axes[1]
    pots = ['harmonic\n(2.4%)', 'doublewell\n(29.4%)', 'gaussmix\n(68.2%)']
    seeds_data = {
        42:   [6.72, 29.17, 64.53],
        137:  [7.01, 28.73, 60.59],
        2024: [8.13, 33.55, 47.86],
    }
    means = [7.29, 30.48, 57.66]

    x_pos = np.arange(len(pots))
    width = 0.2
    seed_colors = ['#4C72B0', '#DD8452', '#55A868']
    for i, (seed, vals) in enumerate(seeds_data.items()):
        ax.bar(x_pos + (i-1)*width, vals, width=width, label=f'Seed {seed}',
               color=seed_colors[i], alpha=0.7, edgecolor='white')

    # Mean markers
    for j, m in enumerate(means):
        ax.plot([j-0.3, j+0.3], [m, m], 'k-', linewidth=2, alpha=0.8)
        ax.text(j+0.35, m, f'{m:.1f}', fontsize=9, va='center')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(pots, fontsize=9)
    ax.set_ylabel(r'$\tau_{\mathrm{int}}$')
    ax.set_title('Per-potential breakdown (best config)')
    ax.legend(loc='upper left')
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # Panel (c): Doublewell parameter sensitivity
    ax = axes[2]
    # Show that all params are at local optima
    # Alpha sweep for doublewell
    dw_alphas = [2.0, 3.0, 4.0]
    dw_taus_alpha = [51.4, 30.5, 49.1]
    ax.plot(dw_alphas, dw_taus_alpha, 'o-', color=COLORS['orbit012'], linewidth=2,
            markersize=8, label='DW alpha sweep (b=4)')

    # b sweep for doublewell
    ax2 = ax.twiny()
    dw_bs = [3.0, 4.0, 5.0, 6.0]
    dw_taus_b = [35.7, 30.5, 36.2, 40.2]
    ax2.plot(dw_bs, dw_taus_b, 's--', color=COLORS['best'], linewidth=2,
             markersize=8, label='DW b sweep (alpha=3)')
    ax2.set_xlabel('b (doublewell)', color=COLORS['best'])
    ax2.tick_params(axis='x', colors=COLORS['best'])

    ax.set_xlabel('alpha (doublewell)', color=COLORS['orbit012'])
    ax.tick_params(axis='x', colors=COLORS['orbit012'])
    ax.set_ylabel(r'$\tau_{\mathrm{dw}}$ (doublewell)')
    ax.set_title('Doublewell params at local optimum')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    fig.suptitle('Orbit-015: Quantitative Results', fontsize=14, fontweight='medium', y=1.02)
    fig.savefig(f'{FIGDIR}/results.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("results.png saved")


if __name__ == '__main__':
    make_narrative()
    make_results()
    print("All figures generated.")
