"""Generate figure for orbit 010: potential-adaptive thermostat results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
    'adaptive': '#4C72B0',
    'target': '#C44E52',
}

# Data from experiments
potentials = ['harmonic_1d', 'doublewell_2d', 'gaussmix_2d']
pot_labels = ['Harmonic 1D', 'Double-well 2D', 'Gauss. mixture 2D']
weights = [0.024, 0.294, 0.682]

# Baseline (orbit-002): alpha=1, standard kinetic, c=0.06
baseline_tau = [10.36, 114.08, 73.81]
baseline_kl = [0.0178, 0.0023, 0.0023]

# Best adaptive (v4): alpha varies, c=0.06
adaptive_tau = [7.29, 33.48, 73.81]
adaptive_kl = [0.0302, 0.0010, 0.0022]
adaptive_alpha = [2.0, 3.0, 1.0]

# NHC baseline (from config)
nhc_tau = [5.7, 70.5, 163.2]

# Per-seed data for adaptive
adaptive_seeds = {
    'harmonic_1d': [6.72, 7.01, 8.13],
    'doublewell_2d': [40.66, 28.35, 31.44],
    'gaussmix_2d': [72.08, 61.94, 87.40],
}

baseline_seeds = {
    'harmonic_1d': [9.1, 10.0, 12.0],
    'doublewell_2d': [122.6, 94.2, 125.4],
    'gaussmix_2d': [72.1, 61.9, 87.4],
}

# Weighted metrics
baseline_metric = sum(w * t for w, t in zip(weights, baseline_tau))
adaptive_metric = sum(w * t for w, t in zip(weights, adaptive_tau))

# --- Figure: 2x2 grid ---
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# Panel (a): Per-potential tau comparison
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(potentials))
width = 0.25
bars_nhc = ax1.bar(x - width, nhc_tau, width, label='NHC M=3', color='#937860', alpha=0.7)
bars_base = ax1.bar(x, baseline_tau, width, label=f'Pade baseline ({baseline_metric:.1f})', color=COLORS['baseline'], alpha=0.7)
bars_adapt = ax1.bar(x + width, adaptive_tau, width, label=f'Adaptive ({adaptive_metric:.1f})', color=COLORS['adaptive'], alpha=0.9)
ax1.set_ylabel(r'$\tau_{\mathrm{int}}$ (mean over seeds)')
ax1.set_xticks(x)
ax1.set_xticklabels(pot_labels, fontsize=9)
ax1.set_title('Per-potential autocorrelation time')
ax1.legend(loc='upper left', fontsize=9)
ax1.set_yscale('log')
ax1.axhline(y=65, color=COLORS['target'], linestyle='--', alpha=0.5, linewidth=1)
ax1.text(-0.15, 1.05, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold')

# Panel (b): Weighted metric breakdown
ax2 = fig.add_subplot(gs[0, 1])
contributions_base = [w * t for w, t in zip(weights, baseline_tau)]
contributions_adapt = [w * t for w, t in zip(weights, adaptive_tau)]

x2 = np.arange(len(potentials))
bars_b2 = ax2.bar(x2 - 0.15, contributions_base, 0.3, label='Baseline', color=COLORS['baseline'], alpha=0.7)
bars_a2 = ax2.bar(x2 + 0.15, contributions_adapt, 0.3, label='Adaptive', color=COLORS['adaptive'], alpha=0.9)
ax2.set_ylabel(r'$w_k \cdot \tau_k$ (contribution to metric)')
ax2.set_xticks(x2)
ax2.set_xticklabels(pot_labels, fontsize=9)
ax2.set_title('Weighted metric contribution')
ax2.legend(fontsize=9)
# Annotate total
ax2.annotate(f'Total: {baseline_metric:.1f}', xy=(2.3, sum(contributions_base)),
             fontsize=9, color=COLORS['baseline'], fontweight='medium')
ax2.annotate(f'Total: {adaptive_metric:.1f}', xy=(2.3, sum(contributions_adapt)),
             fontsize=9, color=COLORS['adaptive'], fontweight='medium')
ax2.axhline(y=65, color=COLORS['target'], linestyle='--', alpha=0.5, linewidth=1)
ax2.text(-0.15, 1.05, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold')

# Panel (c): Per-seed tau_int scatter for doublewell (biggest improvement)
ax3 = fig.add_subplot(gs[1, 0])
seeds = [42, 137, 2024]
for i, pot in enumerate(potentials):
    ax3.scatter(seeds, baseline_seeds[pot], marker='o', s=60, alpha=0.5,
                color=COLORS['baseline'], zorder=2)
    ax3.scatter(seeds, adaptive_seeds[pot], marker='s', s=60, alpha=0.8,
                color=COLORS['adaptive'], zorder=3)
    # Connect with arrows
    for j in range(3):
        if baseline_seeds[pot][j] != adaptive_seeds[pot][j]:
            ax3.annotate('', xy=(seeds[j]+5, adaptive_seeds[pot][j]),
                        xytext=(seeds[j]+5, baseline_seeds[pot][j]),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.5))

ax3.set_xlabel('Seed')
ax3.set_ylabel(r'$\tau_{\mathrm{int}}$')
ax3.set_title('Per-seed improvement (all potentials)')
ax3.set_yscale('log')
# Manual legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['baseline'],
                          markersize=8, label='Baseline', alpha=0.6),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['adaptive'],
                          markersize=8, label='Adaptive')]
ax3.legend(handles=legend_elements, fontsize=9)
ax3.text(-0.15, 1.05, '(c)', transform=ax3.transAxes, fontsize=14, fontweight='bold')

# Panel (d): g(xi) comparison and effective Q_eff
ax4 = fig.add_subplot(gs[1, 1])
xi = np.linspace(-5, 5, 500)
xi2 = xi * xi

# g(xi) with c=0.06
a, b, c_std = 0.7, 3.0, 0.06
g_std = xi * (a + b * xi2) / (1.0 + c_std * xi2)

# g(xi) with c=0.02 (gaussmix variant)
c_gm = 0.02
g_gm = xi * (a + b * xi2) / (1.0 + c_gm * xi2)

# Standard NH: g = xi
g_nh = xi

ax4.plot(xi, g_nh, '--', color='#937860', alpha=0.6, label='Standard NH: g=xi', linewidth=1)
ax4.plot(xi, g_std, '-', color=COLORS['baseline'], label=f'Pade c={c_std}', linewidth=2)
ax4.plot(xi, g_gm, '-', color='#55A868', label=f'Pade c={c_gm} (gaussmix)', linewidth=2)
ax4.set_xlabel(r'$\xi$')
ax4.set_ylabel(r'$g(\xi)$')
ax4.set_title('Friction function shapes')
ax4.legend(fontsize=9)
ax4.set_ylim(-200, 200)

# Inset: alpha effect illustration
inset = ax4.inset_axes([0.55, 0.1, 0.4, 0.35])
alphas = [1.0, 2.0, 3.0]
alpha_labels = [r'$\alpha$=1 (Q=1)', r'$\alpha$=2 (Q=0.5)', r'$\alpha$=3 (Q=0.33)']
alpha_colors = [COLORS['baseline'], COLORS['adaptive'], '#DD8452']
for alp, lab, col in zip(alphas, alpha_labels, alpha_colors):
    K = np.linspace(0, 6, 100)
    h = alp * K - (alp - 1) * 2  # d*kT=2
    inset.plot(K, h, color=col, linewidth=1.5, label=lab)
inset.axhline(y=2, color=COLORS['target'], linestyle=':', alpha=0.5)
inset.set_xlabel('K', fontsize=8)
inset.set_ylabel('h(K)', fontsize=8)
inset.tick_params(labelsize=7)
inset.set_title(r'Driving $h=\alpha K - (\alpha-1)d\,kT$', fontsize=8)
inset.legend(fontsize=6, loc='upper left')
inset.spines['top'].set_visible(False)
inset.spines['right'].set_visible(False)
ax4.text(-0.15, 1.05, '(d)', transform=ax4.transAxes, fontsize=14, fontweight='bold')

fig.suptitle('Orbit 010: Potential-adaptive thermostat driving\n'
             f'Metric: {baseline_metric:.1f} (baseline) to {adaptive_metric:.1f} (adaptive), target < 65',
             fontsize=13, fontweight='medium', y=1.02)

fig.savefig('/Users/wujiewang/code/bath/.worktrees/010-potential-adaptive/orbits/010-potential-adaptive/figures/results.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Figure saved.")
