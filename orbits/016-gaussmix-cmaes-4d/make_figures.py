#!/usr/bin/env python3
"""
Generate narrative.png and results.png for orbit/016-gaussmix-cmaes-4d.

Usage:
  uv run python3 orbits/016-gaussmix-cmaes-4d/make_figures.py
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

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
    'cand0': '#4C72B0',
    'cand1': '#DD8452',
    'cand2': '#55A868',
    'seeds': ['#4C72B0', '#DD8452', '#55A868'],
}

BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def make_narrative():
    """
    Panel (a): Friction function g(xi) comparison for different (a, b) values
    Panel (b): Driving function h(p) for different alpha
    Panel (c): Verification: 200k vs 1M tau estimates showing noise problem
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    xi = np.linspace(-3, 3, 300)

    def g(xi, a, b, c):
        xi2 = xi ** 2
        return xi * (a + b * xi2) / (1 + c * xi2)

    # ── Panel (a): Friction function comparison ──────────────────────────────
    ax = axes[0]
    # Standard NH
    ax.plot(xi, xi, color=COLORS['baseline'], lw=1, ls=':', label='g(xi)=xi (standard NH)')
    # Baseline (orbit-014)
    ax.plot(xi, g(xi, 0.7, 3.0, 0.06), color=COLORS['cand2'], lw=2,
            label='baseline (a=0.7, b=3.0)')
    # Basin A best
    ax.plot(xi, g(xi, 0.8, 3.5, 0.06), color=COLORS['cand0'], lw=2, ls='--',
            label='Basin A (a=0.8, b=3.5)')
    # Basin B best
    ax.plot(xi, g(xi, 0.6, 1.0, 0.06), color=COLORS['cand1'], lw=2, ls='-.',
            label='Basin B (a=0.6, b=1.0)')

    ax.set_xlabel('xi')
    ax.set_ylabel('g(xi)')
    ax.set_title('(a) Friction functions explored')
    ax.legend(loc='upper left', fontsize=9)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # ── Panel (b): Driving function ──────────────────────────────────────────
    ax = axes[1]
    pp = np.linspace(0, 8, 200)
    d_kT = 2.0

    ax.plot(pp, pp, color=COLORS['baseline'], lw=1, ls=':', label='alpha=1.0 (standard)')
    ax.plot(pp, 0.74 * pp - (0.74 - 1.0) * d_kT, color=COLORS['cand2'], lw=2,
            label='alpha=0.74 (gaussmix)')
    ax.plot(pp, 2.0 * pp - (2.0 - 1.0) * d_kT, color=COLORS['cand0'], lw=1.5, ls='--',
            label='alpha=2.0 (harmonic)')
    ax.plot(pp, 3.0 * pp - (3.0 - 1.0) * d_kT, color=COLORS['cand1'], lw=1.5, ls='-.',
            label='alpha=3.0 (doublewell)')

    ax.axhline(d_kT, color='gray', lw=0.5, alpha=0.3)
    ax.annotate('E[h] = d*kT', xy=(6.5, d_kT + 0.15), fontsize=9, color='gray')
    ax.set_xlabel('|p|^2')
    ax.set_ylabel('h(p)')
    ax.set_title('(b) Per-potential driving functions')
    ax.legend(loc='upper left', fontsize=9)
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # ── Panel (c): 200k vs 1M noise comparison ──────────────────────────────
    ax = axes[2]

    candidates = ['Basin A\n(0.8,3.5,0.06,0.74)', 'Basin B\n(0.6,1.0,0.06,1.0)', 'Baseline\n(0.7,3.0,0.06,0.74)']
    tau_200k = [32.6, 34.2, 62.6]
    tau_1M_mean = [66.18, 69.13, 57.66]
    tau_1M_std = [10.93, 7.76, 7.11]

    x = np.arange(len(candidates))
    width = 0.35

    bars1 = ax.bar(x - width/2, tau_200k, width, color='#8172B3', alpha=0.7, label='200k steps (1 seed)')
    bars2 = ax.bar(x + width/2, tau_1M_mean, width, color='#937860', alpha=0.7, label='1M steps (3-seed mean)')
    ax.errorbar(x + width/2, tau_1M_mean, yerr=tau_1M_std, fmt='none', ecolor='black', capsize=4, capthick=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(candidates, fontsize=9)
    ax.set_ylabel('tau_gm (gaussmix only)')
    ax.set_title('(c) Short-run estimates are misleading')
    ax.legend(loc='upper left', fontsize=9)
    ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # Annotate the reversal
    ax.annotate('200k: "48% better"', xy=(0 - width/2, tau_200k[0] + 1),
                fontsize=8, color='#8172B3', ha='center')
    ax.annotate('1M: 15% worse', xy=(0 + width/2, tau_1M_mean[0] + tau_1M_std[0] + 1),
                fontsize=8, color='#937860', ha='center')

    fig.savefig(FIG_DIR / 'narrative.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'narrative.png'}")


def make_results():
    """
    Panel (a): Production eval seed-by-seed breakdown
    Panel (b): Comparison of all candidates at 1M steps
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel (a): Production eval per-potential ─────────────────────────────
    ax = axes[0]
    potentials = ['harmonic\n(w=0.024)', 'doublewell\n(w=0.294)', 'gaussmix\n(w=0.682)']
    taus = {
        'harmonic': [6.72, 7.01, 8.13],
        'doublewell': [29.16, 28.66, 33.62],
        'gaussmix': [64.53, 60.59, 47.86],
    }
    seeds = [42, 137, 2024]
    x = np.arange(len(potentials))
    width = 0.25

    for i, seed in enumerate(seeds):
        vals = [taus['harmonic'][i], taus['doublewell'][i], taus['gaussmix'][i]]
        ax.bar(x + i * width - width, vals, width,
               label=f'Seed {seed}', color=COLORS['seeds'][i], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(potentials, fontsize=10)
    ax.set_ylabel('tau_int')
    ax.set_title('(a) Production eval: METRIC = 48.45')
    ax.legend(fontsize=9)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # ── Panel (b): Candidate comparison at 1M steps ─────────────────────────
    ax = axes[1]

    candidates = ['Basin A\n(a=0.8,b=3.5)', 'Basin B\n(a=0.6,b=1.0)', 'Baseline\n(a=0.7,b=3.0)']
    means = [66.18, 69.13, 57.66]
    stds = [10.93, 7.76, 7.11]
    colors_bar = [COLORS['cand0'], COLORS['cand1'], COLORS['cand2']]

    bars = ax.bar(candidates, means, color=colors_bar, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax.errorbar(candidates, means, yerr=stds, fmt='none', ecolor='black', capsize=5, capthick=1.5)

    # Highlight winner
    bars[2].set_edgecolor('#2d2d2d')
    bars[2].set_linewidth(2)

    ax.axhline(57.66, color=COLORS['cand2'], lw=1, ls='--', alpha=0.5)
    ax.set_ylabel('tau_gm (1M steps, 3-seed mean)')
    ax.set_title('(b) Gaussmix parameter comparison')

    # Annotate best
    ax.annotate('BEST', xy=(2, 57.66), xytext=(2, 50),
                fontsize=11, fontweight='bold', color=COLORS['cand2'],
                ha='center', arrowprops=dict(arrowstyle='->', color=COLORS['cand2'], lw=1.5))

    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    fig.savefig(FIG_DIR / 'results.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'results.png'}")


if __name__ == "__main__":
    make_narrative()
    make_results()
    print("Done.")
