"""Generate narrative and results figures for orbit-013."""
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
    'orbit010': '#4C72B0',
    'failed': '#C44E52',
    'gaussmix': '#DD8452',
    'doublewell': '#55A868',
    'harmonic': '#8172B3',
}

# ── Figure 1: Narrative ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel (a): Alpha sweep for doublewell
ax = axes[0, 0]
alphas_dw = [2.5, 3.0, 3.5, 4.0]
taus_dw = [44.81, 33.48, 37.93, 56.49]
ax.plot(alphas_dw, taus_dw, 'o-', color=COLORS['doublewell'], linewidth=2, markersize=8)
ax.axhline(y=33.48, color=COLORS['baseline'], linestyle='--', alpha=0.7, label='Best (alpha=3.0)')
ax.set_xlabel('alpha (effective-Q scaling)')
ax.set_ylabel('tau_int (doublewell)')
ax.set_title('Doublewell: alpha sweep')
ax.legend()
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# Panel (b): Approaches tried for gaussmix
ax = axes[0, 1]
approaches = ['Standard\n(alpha=1)', 'alpha=1.5', 'Stronger\nfriction', 'Weaker\nfriction',
              'Grad-gated\nalpha', 'Cross-term\np.gradV', 'Quadratic\nK^2']
gm_taus = [73.81, 133.99, 97.07, 82.11, 75.82, np.inf, 245.01]
# Replace inf with a visible cap
gm_taus_plot = [min(t, 300) for t in gm_taus]
colors_bar = [COLORS['orbit010']] + [COLORS['failed']] * 6
bars = ax.bar(range(len(approaches)), gm_taus_plot, color=colors_bar, alpha=0.8)
ax.set_xticks(range(len(approaches)))
ax.set_xticklabels(approaches, fontsize=8)
ax.axhline(y=73.81, color=COLORS['baseline'], linestyle='--', alpha=0.7, label='Baseline (alpha=1)')
ax.set_ylabel('tau_int (gaussmix)')
ax.set_title('Gaussmix: all approaches tried')
ax.legend()
# Mark the KL-failed one
ax.annotate('KL fail', xy=(5, 300), fontsize=9, ha='center', color=COLORS['failed'],
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# Panel (c): Friction function g(xi) — the Pade form
ax = axes[1, 0]
xi = np.linspace(-5, 5, 500)
xi2 = xi * xi

# Standard Pade
g_pade = xi * (0.7 + 3.0 * xi2) / (1.0 + 0.06 * xi2)
# Stronger friction (tried for gaussmix)
g_strong = xi * (1.0 + 4.0 * xi2) / (1.0 + 0.04 * xi2)
# Weaker friction (tried for gaussmix)
g_weak = xi * (0.5 + 2.0 * xi2) / (1.0 + 0.06 * xi2)
# Linear (standard NH)
g_linear = xi

ax.plot(xi, g_pade, color=COLORS['orbit010'], linewidth=2, label='Pade (0.7, 3.0, 0.06)')
ax.plot(xi, g_strong, color=COLORS['failed'], linewidth=1.5, linestyle='--', label='Stronger (1.0, 4.0, 0.04)')
ax.plot(xi, g_weak, color=COLORS['gaussmix'], linewidth=1.5, linestyle=':', label='Weaker (0.5, 2.0, 0.06)')
ax.plot(xi, g_linear, color=COLORS['baseline'], linewidth=1, linestyle='-.', label='Linear g(xi)=xi')
ax.set_xlabel('xi')
ax.set_ylabel('g(xi)')
ax.set_title('Friction functions tested')
ax.legend(fontsize=9)
ax.set_ylim(-60, 60)
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# Panel (d): Effective-Q driving — how alpha scales the kinetic driving
ax = axes[1, 1]
K = np.linspace(0, 8, 200)
d = 2
d_kT = 2.0

for alpha, label, color in [
    (1.0, 'alpha=1.0 (standard)', COLORS['baseline']),
    (2.0, 'alpha=2.0 (harmonic)', COLORS['harmonic']),
    (3.0, 'alpha=3.0 (doublewell)', COLORS['doublewell']),
]:
    h = alpha * K - (alpha - 1) * d_kT
    ax.plot(K, h, color=color, linewidth=2, label=label)

ax.axhline(y=d_kT, color='gray', linestyle=':', alpha=0.5, label='d*kT = 2')
ax.axvline(x=d_kT, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('K = |p|^2/m')
ax.set_ylabel('h(q,p) driving value')
ax.set_title('Effective-Q driving: h = alpha*K - (alpha-1)*d*kT')
ax.legend(fontsize=9)
ax.text(-0.12, 1.05, '(d)', transform=ax.transAxes, fontsize=14, fontweight='bold')

fig.suptitle('Orbit 013: NHC Chain Attempt via Eval-v2 Driving Function', y=1.02, fontsize=14)
fig.savefig('orbits/013-nhc-chain-v3/figures/narrative.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("narrative.png saved")


# ── Figure 2: Results ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Per-potential tau_int comparison
ax = axes[0]
potentials = ['harmonic_1d', 'doublewell_2d', 'gaussmix_2d']
pot_labels = ['Harmonic 1D', 'Double-well 2D', 'Gaussmix 2D']
weights = [0.024, 0.294, 0.682]

# Final results (orbit-010 = orbit-013)
taus_final = {
    'harmonic_1d': [6.72, 7.01, 8.13],
    'doublewell_2d': [40.66, 28.35, 31.44],
    'gaussmix_2d': [72.08, 61.94, 87.40],
}

seeds = [42, 137, 2024]
x = np.arange(3)
width = 0.25

for i, seed in enumerate(seeds):
    vals = [taus_final[p][i] for p in potentials]
    ax.bar(x + i * width, vals, width, label=f'Seed {seed}', alpha=0.8,
           color=[COLORS['harmonic'], COLORS['doublewell'], COLORS['gaussmix']][0])

# Mean bars
means = [np.mean(taus_final[p]) for p in potentials]
ax.bar(x + 3 * width, means, width, label='Mean', color='#333333', alpha=0.9)

ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(pot_labels, fontsize=10)
ax.set_ylabel('tau_int')
ax.set_title('Per-potential tau_int (3 seeds)')
ax.legend(fontsize=9)
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# Panel (b): Metric contribution breakdown
ax = axes[1]
contributions = [w * m for w, m in zip(weights, means)]
colors_pie = [COLORS['harmonic'], COLORS['doublewell'], COLORS['gaussmix']]
wedges, texts, autotexts = ax.pie(
    contributions, labels=pot_labels, autopct='%1.1f%%',
    colors=colors_pie, startangle=90
)
ax.set_title(f'Metric breakdown (total={sum(contributions):.1f})')
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# Panel (c): Summary table as text
ax = axes[2]
ax.axis('off')
table_data = [
    ['Approach', 'Gaussmix tau', 'Metric', 'Status'],
    ['Orbit-010 baseline\n(alpha=1.0 for GM)', '73.81', '60.34', 'BEST'],
    ['alpha=1.5 for GM', '133.99', '101.36', 'Worse'],
    ['Stronger friction\n(a=1.0, b=4.0)', '97.07', '76.20', 'Worse'],
    ['Weaker friction\n(a=0.5, b=2.0)', '82.11', '66.00', 'Worse'],
    ['Grad-gated alpha\n(delta=0.3)', '75.82', '61.71', 'Marginal'],
    ['Cross-term p.gradV\n(gamma=0.15)', 'inf', 'inf', 'KL fail'],
    ['Quadratic K^2\n(qa=0.05)', '245.01', '177.03', 'Much worse'],
]

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Color the status column
for i in range(len(table_data) - 1):
    cell = table[i + 1, 3]
    status = table_data[i + 1][3]
    if status == 'BEST':
        cell.set_facecolor('#d4edda')
    elif status == 'KL fail':
        cell.set_facecolor('#f8d7da')
    elif 'Worse' in status:
        cell.set_facecolor('#fff3cd')

ax.set_title('Approaches tested for gaussmix improvement')
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

fig.suptitle('Orbit 013: Results Summary — metric=60.34 (no improvement over orbit-010)', fontsize=13)
fig.savefig('orbits/013-nhc-chain-v3/figures/results.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("results.png saved")
