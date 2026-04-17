#!/usr/bin/env python3
"""Generate narrative.png and results.png for orbit-012."""
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
    'orbit012': '#DD8452',
    'harmonic': '#55A868',
    'doublewell': '#C44E52',
    'gaussmix': '#8172B3',
}

FIGDIR = "/Users/wujiewang/code/bath/.worktrees/012-joint-cmaes-alpha/orbits/012-joint-cmaes-alpha/figures"

# ════════════════════════════════════════════════════════════════
# FIGURE 1: narrative.png --friction function comparison
# ════════════════════════════════════════════════════════════════

def pade_g(xi, a, b, c):
    xi2 = xi * xi
    return xi * (a + b * xi2) / (1.0 + c * xi2)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

xi = np.linspace(-5, 5, 500)

# Panel (a): Friction functions for baseline vs per-potential
ax = axes[0]
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')
ax.set_title("Friction function g(xi)")

# Baseline (orbit-010: same for all potentials)
g_base = pade_g(xi, 0.7, 3.0, 0.06)
ax.plot(xi, g_base, color=COLORS['baseline'], linestyle='--', linewidth=2, label='Orbit-010 (all potentials)')

# Per-potential (orbit-012)
g_harm = pade_g(xi, 0.7, 3.0, 0.06)  # same as baseline
g_dw = pade_g(xi, 1.0, 4.0, 0.06)
g_gm = pade_g(xi, 0.7, 1.0, 0.06)

ax.plot(xi, g_dw, color=COLORS['doublewell'], linewidth=1.8, label='Orbit-012: doublewell')
ax.plot(xi, g_gm, color=COLORS['gaussmix'], linewidth=1.8, label='Orbit-012: gaussmix')
ax.set_xlabel(r'$\xi$')
ax.set_ylabel(r'$g(\xi)$')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(-5, 5)

# Panel (b): Effective driving h vs p^2 for different alpha
ax = axes[1]
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')
ax.set_title("Driving function h(p)")

pp = np.linspace(0, 8, 200)
d = 2  # 2D case
d_kT = 2.0

# Standard NH
h_std = pp
ax.plot(pp, h_std, color=COLORS['baseline'], linestyle='--', linewidth=2, label=r'Standard NH ($\alpha$=1)')

# Alpha=2 harmonic
alpha = 2.0
h_a2 = alpha * pp - (alpha - 1) * 1.0  # d=1 for harmonic
ax.plot(pp, h_a2, color=COLORS['harmonic'], linewidth=1.8, label=r'Harmonic ($\alpha$=2, d=1)')

# Alpha=3 doublewell
alpha = 3.0
h_a3 = alpha * pp - (alpha - 1) * d_kT
ax.plot(pp, h_a3, color=COLORS['doublewell'], linewidth=1.8, label=r'Doublewell ($\alpha$=3, d=2)')

ax.axhline(y=d_kT, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.annotate(r'$d \cdot kT = 2$', xy=(0.5, d_kT), fontsize=9, color='gray')
ax.set_xlabel(r'$|p|^2$')
ax.set_ylabel(r'$h(q,p)$')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(0, 8)
ax.set_ylim(-6, 20)

# Panel (c): b parameter effect on friction profile
ax = axes[2]
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')
ax.set_title("Effect of b on friction (a=0.7, c=0.06)")

xi_pos = np.linspace(0, 5, 200)
for b_val, ls, lw, alpha_c in [(0.5, ':', 1.2, 0.6), (1.0, '-', 2.0, 1.0), (1.5, '-', 1.5, 0.7), (3.0, '--', 2.0, 1.0), (5.0, ':', 1.2, 0.6)]:
    g = pade_g(xi_pos, 0.7, b_val, 0.06)
    color = COLORS['gaussmix'] if b_val == 1.0 else COLORS['baseline'] if b_val == 3.0 else '#aaaaaa'
    label_str = f'b={b_val}'
    if b_val == 1.0:
        label_str += ' (gaussmix opt.)'
    elif b_val == 3.0:
        label_str += ' (orbit-010)'
    ax.plot(xi_pos, g, color=color, linestyle=ls, linewidth=lw, alpha=alpha_c, label=label_str)

ax.set_xlabel(r'$\xi$')
ax.set_ylabel(r'$g(\xi)$')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(0, 5)

fig.suptitle("Per-potential Pade friction optimization: orbit-012 vs orbit-010 baseline", y=1.02, fontsize=13, fontweight='medium')

fig.savefig(f"{FIGDIR}/narrative.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved narrative.png")


# ════════════════════════════════════════════════════════════════
# FIGURE 2: results.png --quantitative comparison
# ════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

# Panel (a): Per-potential tau comparison
ax = axes[0]
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')
ax.set_title(r'$\tau_{int}$ per potential')

potentials = ['harmonic', 'doublewell', 'gaussmix']
pot_labels = ['Harmonic\n(2.4%)', 'Doublewell\n(29.4%)', 'Gaussmix\n(68.2%)']

# Orbit-010 results
tau_010 = [7.29, 33.48, 73.81]
# Orbit-012 results
tau_012 = [7.29, 30.48, 59.25]

x = np.arange(len(potentials))
width = 0.35

bars1 = ax.bar(x - width/2, tau_010, width, label='Orbit-010', color=COLORS['orbit010'], alpha=0.8)
bars2 = ax.bar(x + width/2, tau_012, width, label='Orbit-012', color=COLORS['orbit012'], alpha=0.8)

# Value labels
for bar, val in zip(bars1, tau_010):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9, color=COLORS['orbit010'])
for bar, val in zip(bars2, tau_012):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9, color=COLORS['orbit012'])

ax.set_xticks(x)
ax.set_xticklabels(pot_labels)
ax.set_ylabel(r'$\tau_{int}$')
ax.legend(loc='upper left')
ax.set_ylim(0, 85)

# Panel (b): Per-seed breakdown for gaussmix (the bottleneck)
ax = axes[1]
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')
ax.set_title('Gaussmix per-seed breakdown')

seeds = ['42', '137', '2024', 'Mean']
tau_gm_010 = [72.08, 61.94, 87.40, 73.81]
tau_gm_012 = [68.85, 57.91, 51.00, 59.25]

x = np.arange(len(seeds))
bars1 = ax.bar(x - width/2, tau_gm_010, width, label='Orbit-010 (b=3.0)', color=COLORS['orbit010'], alpha=0.8)
bars2 = ax.bar(x + width/2, tau_gm_012, width, label='Orbit-012 (b=1.0)', color=COLORS['orbit012'], alpha=0.8)

for bar, val in zip(bars1, tau_gm_010):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9, color=COLORS['orbit010'])
for bar, val in zip(bars2, tau_gm_012):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9, color=COLORS['orbit012'])

ax.set_xticks(x)
ax.set_xticklabels(seeds)
ax.set_xlabel('Seed')
ax.set_ylabel(r'$\tau_{int}$')
ax.legend(loc='upper right')
ax.set_ylim(0, 100)

# Panel (c): Weighted metric comparison
ax = axes[2]
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')
ax.set_title('Weighted metric (lower is better)')

methods = ['NHC\nM=3', 'Orbit-003\n(eval-v1)', 'Orbit-010\nbaseline', 'Orbit-012\n(this work)']
metrics = [132.1, 84.14, 60.34, 49.54]
colors = [COLORS['baseline'], COLORS['baseline'], COLORS['orbit010'], COLORS['orbit012']]

bars = ax.bar(range(len(methods)), metrics, color=colors, alpha=0.8, width=0.6)
for bar, val in zip(bars, metrics):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='medium')

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('Weighted ' + r'$\tau_{int}$')
ax.set_ylim(0, 155)

# Target line
ax.axhline(y=65, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
ax.annotate('Target: 65', xy=(3.3, 65), fontsize=9, color='green')

fig.suptitle("Orbit-012: joint Pade + effective-Q optimization -- metric 60.34 to 49.54", y=1.02, fontsize=13, fontweight='medium')

fig.savefig(f"{FIGDIR}/results.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved results.png")
