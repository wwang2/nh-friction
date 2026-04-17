#!/usr/bin/env python3
"""
Generate narrative.png and results.png for orbit/016-gaussmix-cmaes-4d.
Reads search_results.npz (from cmaes_search.py) + production eval output.

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
    'orbit014': '#4C72B0',
    'cmaes_best': '#DD8452',
    'grid_valid': '#55A868',
    'grid_invalid': '#C44E52',
}

BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def make_narrative(data):
    """
    Panel (a): (b, alpha) heatmap of tau_gm from grid search
    Panel (b): Friction function g(xi) comparison — baseline vs CMA-ES best
    Panel (c): Driving function effective alpha visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel (a): (b, alpha) heatmap ─────────────────────────────────────────
    ax = axes[0]
    grid_params = data['grid_params']  # shape (N, 4): a, b, c, alpha
    grid_taus = data['grid_taus']

    # Filter to a=0.7, c=0.06 (Phase 1 grid)
    mask = (np.abs(grid_params[:, 0] - 0.7) < 0.01) & (np.abs(grid_params[:, 2] - 0.06) < 0.01)
    b_vals = np.sort(np.unique(grid_params[mask, 1]))
    alpha_vals = np.sort(np.unique(grid_params[mask, 3]))

    tau_grid = np.full((len(b_vals), len(alpha_vals)), np.nan)
    for i, b in enumerate(b_vals):
        for j, al in enumerate(alpha_vals):
            idx = mask & (np.abs(grid_params[:, 1] - b) < 0.01) & (np.abs(grid_params[:, 3] - al) < 0.01)
            if np.any(idx):
                tau_val = grid_taus[idx][0]
                if tau_val < TAU_CAP:
                    tau_grid[i, j] = tau_val

    TAU_CAP = 50_000
    im = ax.imshow(tau_grid.T, origin='lower', aspect='auto',
                   extent=[b_vals[0]-0.25, b_vals[-1]+0.25,
                           alpha_vals[0]-0.05, alpha_vals[-1]+0.05],
                   cmap='viridis_r', vmin=40, vmax=200)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='tau_gm (200k)')
    ax.set_xlabel('b')
    ax.set_ylabel('alpha')
    ax.set_title('(a) Gaussmix tau_int landscape')

    # Mark known optima
    best_p = data['best_params']
    ax.plot(3.0, 0.74, 'D', color=COLORS['orbit014'], markersize=8, label='orbit-014')
    if best_p is not None and np.any(best_p != 0):
        ax.plot(best_p[1], best_p[3], '*', color=COLORS['cmaes_best'], markersize=12, label='CMA-ES best')
    ax.legend(loc='upper right')
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # ── Panel (b): Friction function comparison ──────────────────────────────
    ax = axes[1]
    xi = np.linspace(-3, 3, 200)

    def g(xi, a, b, c):
        xi2 = xi ** 2
        return xi * (a + b * xi2) / (1 + c * xi2)

    # Baseline (orbit-014)
    ax.plot(xi, g(xi, 0.7, 3.0, 0.06), color=COLORS['orbit014'], lw=2, label='orbit-014 (a=0.7,b=3.0,c=0.06)')
    # CMA-ES best
    if best_p is not None and np.any(best_p != 0):
        ax.plot(xi, g(xi, best_p[0], best_p[1], best_p[2]),
                color=COLORS['cmaes_best'], lw=2, ls='--',
                label=f'CMA-ES (a={best_p[0]:.2f},b={best_p[1]:.2f},c={best_p[2]:.3f})')
    # Standard NH
    ax.plot(xi, xi, color=COLORS['baseline'], lw=1, ls=':', label='g(xi)=xi (standard NH)')

    ax.set_xlabel('xi')
    ax.set_ylabel('g(xi)')
    ax.set_title('(b) Friction function comparison')
    ax.legend(loc='upper left', fontsize=9)
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # ── Panel (c): Driving function behavior ─────────────────────────────────
    ax = axes[2]
    pp = np.linspace(0, 8, 200)  # |p|^2 range
    d_kT = 2.0  # dim=2, kT=1

    # Standard: h = pp (alpha=1)
    ax.plot(pp, pp, color=COLORS['baseline'], lw=1, ls=':', label='alpha=1.0 (standard)')
    # Orbit-014: alpha=0.74
    ax.plot(pp, 0.74 * pp - (0.74 - 1.0) * d_kT, color=COLORS['orbit014'], lw=2, label='alpha=0.74 (orbit-014)')
    # CMA-ES best
    if best_p is not None and np.any(best_p != 0):
        al = best_p[3]
        ax.plot(pp, al * pp - (al - 1.0) * d_kT,
                color=COLORS['cmaes_best'], lw=2, ls='--',
                label=f'alpha={al:.3f} (CMA-ES)')
    ax.axhline(d_kT, color='gray', lw=0.5, ls='-', alpha=0.3)
    ax.annotate('E[h] = d*kT', xy=(6.5, d_kT + 0.1), fontsize=9, color='gray')

    ax.set_xlabel('|p|^2')
    ax.set_ylabel('h(p)')
    ax.set_title('(c) Driving function h(p)')
    ax.legend(loc='upper left', fontsize=9)
    ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    fig.savefig(FIG_DIR / 'narrative.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'narrative.png'}")


def make_results(data, prod_metric=None, prod_taus=None):
    """
    Panel (a): Grid search valid tau distribution
    Panel (b): Seed-by-seed breakdown (production eval)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel (a): Distribution of valid tau values from grid ─────────────────
    ax = axes[0]
    grid_taus = data['grid_taus']
    TAU_CAP = 50_000
    valid_taus = grid_taus[grid_taus < TAU_CAP]

    if len(valid_taus) > 0:
        ax.hist(valid_taus, bins='fd', color=COLORS['grid_valid'], alpha=0.7, edgecolor='white')
        ax.axvline(data['best_mean_tau'][0], color=COLORS['cmaes_best'], lw=2, ls='--',
                   label=f"CMA-ES best: {data['best_mean_tau'][0]:.1f}")
        # Mark baseline
        ax.axvline(57.66, color=COLORS['orbit014'], lw=2, ls='-.',
                   label='orbit-014 baseline: 57.66')
    ax.set_xlabel('tau_gm (gaussmix only, 200k steps)')
    ax.set_ylabel('Count')
    ax.set_title('(a) Grid search tau distribution')
    ax.legend(fontsize=9)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # ── Panel (b): Production eval seed breakdown ─────────────────────────────
    ax = axes[1]

    if prod_taus is not None:
        potentials = list(prod_taus.keys())
        x = np.arange(len(potentials))
        width = 0.25
        seeds = [42, 137, 2024]
        seed_colors = ['#4C72B0', '#DD8452', '#55A868']

        for i, seed in enumerate(seeds):
            vals = [prod_taus[pot][i] for pot in potentials]
            ax.bar(x + i * width, vals, width, label=f'Seed {seed}', color=seed_colors[i], alpha=0.8)

        ax.set_xticks(x + width)
        ax.set_xticklabels([p.replace('_', '\n') for p in potentials])
        ax.set_ylabel('tau_int')
        ax.set_title(f'(b) Production eval (METRIC={prod_metric:.2f})')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Production eval\nnot yet run', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        ax.set_title('(b) Production eval (pending)')

    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    fig.savefig(FIG_DIR / 'results.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'results.png'}")


def main():
    data_path = BASE_DIR / 'search_results.npz'
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run cmaes_search.py first.")
        return

    data = dict(np.load(data_path, allow_pickle=True))

    # Check if production eval results exist
    prod_path = BASE_DIR / 'prod_results.npz'
    prod_metric = None
    prod_taus = None
    if prod_path.exists():
        prod = dict(np.load(prod_path, allow_pickle=True))
        prod_metric = float(prod.get('metric', [0])[0])
        prod_taus = prod.get('tau_per_potential', None)
        if prod_taus is not None:
            prod_taus = prod_taus.item()  # dict from npz

    make_narrative(data)
    make_results(data, prod_metric, prod_taus)


if __name__ == "__main__":
    main()
