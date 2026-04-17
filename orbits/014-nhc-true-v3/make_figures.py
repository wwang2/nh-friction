"""Generate narrative.png and results.png for orbit-014-nhc-true-v3."""
import matplotlib
matplotlib.use("Agg")
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
    'ours': '#4C72B0',
    'highlight': '#DD8452',
    'green': '#55A868',
    'red': '#C44E52',
}

OUT = "/Users/wujiewang/code/bath/.worktrees/014-nhc-true-v3/orbits/014-nhc-true-v3/figures"

# ═══════════════════════════════════════════════════════════════════════
# Figure 1: narrative.png — Alpha sweep landscape + physical mechanism
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Alpha sweep for gaussmix — the optimization landscape
ax = axes[0]
alphas = [0.4, 0.5, 0.6, 0.7, 0.71, 0.72, 0.73, 0.735, 0.74, 0.745, 0.75, 0.76, 0.77, 0.78, 0.8, 0.85, 0.9, 1.0]
taus =   [369.65, 190.05, 121.36, 69.51, 71.22, 83.78, 62.59, 59.86, 57.66, 59.79, 63.05, 63.47, 64.09, 69.02, 64.59, 71.27, 73.39, 73.81]

ax.plot(alphas, taus, 'o-', color=COLORS['ours'], markersize=5, linewidth=1.5)
ax.axhline(73.81, color=COLORS['baseline'], linestyle='--', linewidth=1, label='Baseline (alpha=1.0)')
ax.axvline(0.74, color=COLORS['highlight'], linestyle=':', linewidth=1, alpha=0.7)
ax.annotate('alpha=0.74\ntau=57.7', xy=(0.74, 57.66), xytext=(0.55, 90),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
ax.set_xlabel('alpha (effective-Q scaling)')
ax.set_ylabel('gaussmix tau_int')
ax.set_xlim(0.35, 1.05)
ax.set_ylim(0, 400)
ax.legend(loc='upper left')
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')
ax.set_title('Alpha sweep: gaussmix tau_int')

# Panel (b): Physical mechanism — effective Q vs momentum persistence
ax = axes[1]
# Show how Q_eff = Q/alpha changes with alpha
alpha_range = np.linspace(0.5, 1.5, 100)
q_eff = 1.0 / alpha_range  # Q_eff = Q/alpha with Q=1
ax.plot(alpha_range, q_eff, color=COLORS['ours'], linewidth=2)
ax.axvline(0.74, color=COLORS['highlight'], linestyle=':', linewidth=1.5)
ax.axvline(1.0, color=COLORS['baseline'], linestyle='--', linewidth=1)
ax.fill_between(alpha_range[alpha_range < 1.0], q_eff[alpha_range < 1.0], 1.0,
                alpha=0.1, color=COLORS['green'])
ax.fill_between(alpha_range[alpha_range > 1.0], q_eff[alpha_range > 1.0], 1.0,
                alpha=0.1, color=COLORS['red'])
ax.annotate('Weaker thermostat\n(more inertia)', xy=(0.65, 1.4), fontsize=9, color=COLORS['green'])
ax.annotate('Stronger thermostat\n(less inertia)', xy=(1.1, 0.75), fontsize=9, color=COLORS['red'])
ax.set_xlabel('alpha')
ax.set_ylabel('Q_eff = Q / alpha')
ax.set_title('Effective thermostat mass')
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# Panel (c): Per-potential tau comparison (baseline vs ours)
ax = axes[2]
pots = ['harmonic\n(2.4%)', 'doublewell\n(29.4%)', 'gaussmix\n(68.2%)']
baseline_tau = [7.29, 33.48, 73.81]
ours_tau = [7.29, 33.48, 57.66]

x = np.arange(len(pots))
width = 0.35
bars1 = ax.bar(x - width/2, baseline_tau, width, label='Baseline (alpha=1.0)', color=COLORS['baseline'], alpha=0.7)
bars2 = ax.bar(x + width/2, ours_tau, width, label='Ours (alpha=0.74)', color=COLORS['ours'], alpha=0.9)

# Add value labels
for bar, val in zip(bars1, baseline_tau):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9, color=COLORS['baseline'])
for bar, val in zip(bars2, ours_tau):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9, color=COLORS['ours'])

ax.set_xticks(x)
ax.set_xticklabels(pots)
ax.set_ylabel('tau_int (lower is better)')
ax.set_title('Per-potential tau comparison')
ax.legend(loc='upper left')
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

fig.savefig(f"{OUT}/narrative.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved narrative.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: results.png — Quantitative results table + seed breakdown
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): Seed breakdown for gaussmix across alpha values
ax = axes[0]
alpha_sel = [0.7, 0.73, 0.74, 0.75, 0.8, 1.0]
seed_data = {
    0.7:  [72.1, 66.9, 69.5],
    0.73: [60.4, 64.6, 62.8],
    0.74: [64.5, 60.6, 47.9],
    0.75: [54.8, 69.0, 65.4],
    0.8:  [62.6, 73.6, 57.6],
    1.0:  [72.1, 61.9, 87.4],
}
seed_labels = ['Seed 42', 'Seed 137', 'Seed 2024']
seed_colors = ['#4C72B0', '#DD8452', '#55A868']

x = np.arange(len(alpha_sel))
width = 0.25
for i, (label, color) in enumerate(zip(seed_labels, seed_colors)):
    vals = [seed_data[a][i] for a in alpha_sel]
    ax.bar(x + (i - 1) * width, vals, width, label=label, color=color, alpha=0.8)

# Mean line
means = [np.mean(seed_data[a]) for a in alpha_sel]
ax.plot(x, means, 'k-o', markersize=6, linewidth=1.5, label='Mean', zorder=5)

ax.set_xticks(x)
ax.set_xticklabels([f'{a:.2f}' for a in alpha_sel])
ax.set_xlabel('alpha (gaussmix)')
ax.set_ylabel('tau_int')
ax.set_title('Gaussmix tau_int: seed breakdown')
ax.legend(loc='upper right')
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# Panel (b): Summary results table
ax = axes[1]
ax.axis('off')

table_data = [
    ['', 'Baseline\n(orbit-010)', 'This orbit\n(014)', 'Change'],
    ['harmonic tau', '7.29', '7.29', '0%'],
    ['harmonic KL', '0.030', '0.030', '0%'],
    ['doublewell tau', '33.48', '33.48', '0%'],
    ['doublewell KL', '0.001', '0.001', '0%'],
    ['gaussmix tau', '73.81', '57.66', '-21.9%'],
    ['gaussmix KL', '0.002', '0.002', '0%'],
    ['', '', '', ''],
    ['METRIC', '60.34', '49.33', '-18.2%'],
    ['', '', '', ''],
    ['alpha (h)', '2.0', '2.0', ''],
    ['alpha (dw)', '3.0', '3.0', ''],
    ['alpha (gm)', '1.0', '0.74', 'NEW'],
]

table = ax.table(
    cellText=table_data,
    cellLoc='center',
    loc='center',
    bbox=[0.0, 0.0, 1.0, 1.0],
)
table.auto_set_font_size(False)
table.set_fontsize(10)

# Style header row
for j in range(4):
    cell = table[0, j]
    cell.set_facecolor('#E8E8E8')
    cell.set_text_props(fontweight='bold')

# Highlight METRIC row
for j in range(4):
    cell = table[8, j]
    cell.set_facecolor('#D4E6F1')
    cell.set_text_props(fontweight='bold')

# Highlight gaussmix tau row
for j in range(4):
    cell = table[5, j]
    cell.set_facecolor('#D5F5E3')

# Highlight alpha_gm row
for j in range(4):
    cell = table[12, j]
    cell.set_facecolor('#FDEBD0')

ax.set_title('Summary: orbit-010 vs orbit-014', fontsize=13, fontweight='medium', pad=15)
ax.text(-0.02, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

fig.savefig(f"{OUT}/results.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved results.png")
