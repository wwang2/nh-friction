"""
Generate narrative and results figures for orbit 004-composite-friction.
Shows why composite friction forms fail and the parent Padé remains optimal.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

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
    'parent_pade': '#4C72B0',
    'composite_exp_tanh': '#DD8452',
    'composite_exp_linear': '#55A868',
    'composite_pade_boost': '#C44E52',
    'linear_nh': '#888888',
}

# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Narrative — friction profiles + why composite fails
# ═══════════════════════════════════════════════════════════════════════

def make_narrative():
    xi = np.linspace(-8, 8, 500)

    # Friction functions
    def pade(xi, a=0.7, b=3.0, c=0.06):
        xi2 = xi * xi
        return xi * (a + b * xi2) / (1.0 + c * xi2)

    def composite_exp_tanh(xi, a=0.7, b=3.0, e=0.30, f=2.0, c=0.3):
        xi2 = xi * xi
        return xi * (a + b * xi2) * np.exp(-e * xi2) + f * np.tanh(c * xi)

    def composite_exp_linear(xi, a=0.7, b=3.0, e=0.30, f=0.3):
        xi2 = xi * xi
        return xi * (a + b * xi2) * np.exp(-e * xi2) + f * xi

    def composite_pade_boost(xi, a=0.7, b=3.0, c=0.06, d=1.0, e=0.3):
        xi2 = xi * xi
        return xi * (a + b * xi2) / (1.0 + c * xi2) + d * xi * xi2 * np.exp(-e * xi2)

    def linear_nh(xi):
        return xi

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (a): Friction function profiles
    ax = axes[0, 0]
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.plot(xi, pade(xi), color=COLORS['parent_pade'], lw=2.5, label='Padé (parent)')
    ax.plot(xi, composite_exp_tanh(xi), color=COLORS['composite_exp_tanh'], lw=1.5,
            ls='--', label='Exp-cubic + tanh')
    ax.plot(xi, composite_exp_linear(xi), color=COLORS['composite_exp_linear'], lw=1.5,
            ls='-.', label='Exp-cubic + linear')
    ax.plot(xi, composite_pade_boost(xi), color=COLORS['composite_pade_boost'], lw=1.5,
            ls=':', label='Padé + boost')
    ax.plot(xi, linear_nh(xi), color=COLORS['linear_nh'], lw=1, ls='--', label='Standard NH (g=ξ)')
    ax.set_xlabel('ξ')
    ax.set_ylabel('g(ξ)')
    ax.set_title('Friction function profiles')
    ax.set_ylim(-150, 150)
    ax.legend(loc='upper left', fontsize=9)

    # Panel (b): Zoomed view near origin (KAM tori region)
    ax = axes[0, 1]
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')
    xi_zoom = np.linspace(-3, 3, 300)
    ax.plot(xi_zoom, pade(xi_zoom), color=COLORS['parent_pade'], lw=2.5, label='Padé')
    ax.plot(xi_zoom, composite_exp_tanh(xi_zoom), color=COLORS['composite_exp_tanh'], lw=1.5,
            ls='--', label='Exp-cubic + tanh')
    ax.plot(xi_zoom, composite_exp_linear(xi_zoom), color=COLORS['composite_exp_linear'], lw=1.5,
            ls='-.', label='Exp-cubic + linear')
    ax.plot(xi_zoom, composite_pade_boost(xi_zoom), color=COLORS['composite_pade_boost'], lw=1.5,
            ls=':', label='Padé + boost')
    ax.plot(xi_zoom, linear_nh(xi_zoom), color=COLORS['linear_nh'], lw=1, ls='--', label='NH')
    ax.set_xlabel('ξ')
    ax.set_ylabel('g(ξ)')
    ax.set_title('Core region (KAM tori breaking)')
    ax.legend(loc='upper left', fontsize=9)

    # Panel (c): Tail behavior — why bounded forms fail
    ax = axes[1, 0]
    ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')
    xi_tail = np.linspace(0, 15, 300)
    ax.plot(xi_tail, pade(xi_tail), color=COLORS['parent_pade'], lw=2.5, label='Padé (~50ξ)')
    ax.plot(xi_tail, composite_exp_tanh(xi_tail), color=COLORS['composite_exp_tanh'], lw=1.5,
            ls='--', label='Exp+tanh (bounded)')
    ax.plot(xi_tail, composite_exp_linear(xi_tail), color=COLORS['composite_exp_linear'], lw=1.5,
            ls='-.', label='Exp+linear (~0.3ξ)')

    # Annotate the saturation problem
    ax.axhline(y=2.0, color=COLORS['composite_exp_tanh'], ls=':', alpha=0.5, lw=1)
    ax.annotate('tanh saturates at f=2.0',
                xy=(12, 2.0), fontsize=9, color=COLORS['composite_exp_tanh'],
                arrowprops=dict(arrowstyle='->', color=COLORS['composite_exp_tanh'], lw=0.8),
                xytext=(8, 80))

    ax.set_xlabel('ξ')
    ax.set_ylabel('g(ξ)')
    ax.set_title('Tail behavior: bounded vs unbounded')
    ax.legend(loc='upper left', fontsize=9)

    # Panel (d): KL divergence across variants
    ax = axes[1, 1]
    ax.text(-0.12, 1.05, '(d)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # Data from actual eval runs
    variants = [
        'Padé\n(parent)',
        'Exp+tanh\nv4',
        'Exp+linear\nv5',
        'Padé+boost\nv9',
        'Padé b=4\nv12',
        'Padé c=0.03\nv15',
        'Padé c=0.05\nv16',
    ]
    kl_vals = [0.018, 0.099, 0.042, 0.072, 0.086, 0.014, 0.012]
    colors_bar = [
        COLORS['parent_pade'],
        COLORS['composite_exp_tanh'],
        COLORS['composite_exp_linear'],
        COLORS['composite_pade_boost'],
        '#8172B3',
        '#937860',
        '#4C72B0',
    ]
    passed = [True, False, True, False, False, True, True]

    bars = ax.bar(range(len(variants)), kl_vals, color=colors_bar, alpha=0.8, edgecolor='white')
    # Mark failing variants
    for i, (p, bar) in enumerate(zip(passed, bars)):
        if not p:
            bar.set_hatch('///')
            bar.set_alpha(0.4)

    ax.axhline(y=0.05, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.annotate('KL threshold = 0.05', xy=(5.5, 0.052), fontsize=9, color='red')
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, fontsize=8, rotation=0)
    ax.set_ylabel('KL(empirical || analytical)')
    ax.set_title('1D HO canonical measure preservation')
    ax.set_ylim(0, 0.12)

    fig.suptitle('Composite Friction Hypothesis: Padé Backbone Proved Irreplaceable', y=1.02,
                 fontsize=14, fontweight='medium')

    fig.savefig('orbits/004-composite-friction/figures/narrative.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Saved narrative.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Results — quantitative comparison
# ═══════════════════════════════════════════════════════════════════════

def make_results():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Data from all eval runs
    variants = [
        'Parent Padé\na=0.7 b=3 c=0.06',
        'Exp+tanh\nv4 (KL fail)',
        'Exp+linear\nv5',
        'Padé+boost\nv9 (KL fail)',
        'Padé b=6\nc=0.06 (v13)',
        'Padé b=3\nc=0.02 (v14)',
        'Padé b=3\nc=0.05 (v16)',
    ]
    short_labels = ['Parent', 'Exp+tanh', 'Exp+lin', 'Padé+boost', 'b=6', 'c=0.02', 'c=0.05']

    # tau_int per potential (mean across 3 seeds)
    tau_ho = [10.36, 4.66, 8.96, 6.81, 9.76, 9.80, 11.06]
    tau_dw = [114.08, 139.90, 163.01, 206.36, 96.84, 127.83, 127.92]
    tau_gm = [73.81, 76.33, 69.00, 128.40, 125.47, 84.36, 82.96]
    metrics = [84.14, float('inf'), 95.23, float('inf'), 114.26, 95.37, 94.47]
    kl_ho = [0.018, 0.099, 0.042, 0.072, 0.008, 0.013, 0.012]

    valid_mask = [math.isfinite(m) for m in metrics]

    bar_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#4C72B0']

    # Panel (a): tau_int per potential
    ax = axes[0]
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')
    x = np.arange(len(short_labels))
    w = 0.25
    bars1 = ax.bar(x - w, tau_dw, w, label='doublewell_2d', color='#4C72B0', alpha=0.8)
    bars2 = ax.bar(x, tau_gm, w, label='gaussmix_2d', color='#DD8452', alpha=0.8)
    bars3 = ax.bar(x + w, tau_ho, w, label='harmonic_1d (x10)', color='#55A868', alpha=0.8)

    # Hatch KL-failing variants
    for i in range(len(short_labels)):
        if not valid_mask[i]:
            bars1[i].set_hatch('///')
            bars1[i].set_alpha(0.3)
            bars2[i].set_hatch('///')
            bars2[i].set_alpha(0.3)
            bars3[i].set_hatch('///')
            bars3[i].set_alpha(0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel('tau_int (lower is better)')
    ax.set_title('Per-potential autocorrelation time')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 250)

    # Panel (b): Weighted metric
    ax = axes[1]
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')
    finite_metrics = [m if math.isfinite(m) else 0 for m in metrics]
    bars = ax.bar(x, finite_metrics, color=bar_colors, alpha=0.8, edgecolor='white')
    for i in range(len(short_labels)):
        if not valid_mask[i]:
            bars[i].set_hatch('///')
            bars[i].set_alpha(0.3)
            ax.annotate('KL\nfail', xy=(i, 5), fontsize=8, ha='center', color='red',
                       fontweight='bold')

    ax.axhline(y=84.14, color='#4C72B0', ls='--', lw=1.5, alpha=0.7)
    ax.annotate('Parent = 84.14', xy=(5.5, 85), fontsize=9, color='#4C72B0')

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel('Weighted tau_int (lower is better)')
    ax.set_title('Overall metric comparison')
    ax.set_ylim(0, 130)

    # Panel (c): KL divergence
    ax = axes[2]
    ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')
    bars = ax.bar(x, kl_ho, color=bar_colors, alpha=0.8, edgecolor='white')
    for i in range(len(short_labels)):
        if not valid_mask[i]:
            bars[i].set_hatch('///')
            bars[i].set_alpha(0.4)

    ax.axhline(y=0.05, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.annotate('KL threshold', xy=(5, 0.052), fontsize=9, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel('KL divergence (1D HO)')
    ax.set_title('Canonical measure preservation')
    ax.set_ylim(0, 0.12)

    fig.suptitle('Composite Friction: Quantitative Results Across 7 Variants',
                 y=1.02, fontsize=14, fontweight='medium')

    fig.savefig('orbits/004-composite-friction/figures/results.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Saved results.png")


if __name__ == '__main__':
    make_narrative()
    make_results()
