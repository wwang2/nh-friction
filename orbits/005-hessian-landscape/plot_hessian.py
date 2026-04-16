#!/usr/bin/env python3
"""
Generate figures for the Hessian landscape analysis.

Produces two figures:
  1. figures/narrative.png - Qualitative panel showing the Hessian landscape:
     (a) Friction function g(xi) comparison (baseline vs best found)
     (b) 2D metric surface slice along the two softest eigenvector directions
     (c) Eigenvalue spectrum of the Hessian
  2. figures/results.png - Quantitative results panel
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Style from research/style.md
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
    'optimum': '#4C72B0',
    'best': '#DD8452',
    'soft': '#55A868',
    'stiff': '#C44E52',
}

ORBIT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ORBIT_DIR, "figures")


def load_results():
    with open(os.path.join(ORBIT_DIR, "hessian_results.json")) as f:
        data = json.load(f)
    return data


def pade_friction(xi, a, b, c):
    xi2 = xi * xi
    return xi * (a + b * xi2) / (1.0 + c * xi2)


def plot_narrative(data, best_params=None):
    """Generate narrative.png: qualitative 3-panel figure."""

    theta_0 = np.array(data["theta_0"])
    eigenvalues = np.array(data["eigenvalues"])
    eigenvectors = np.array(data["eigenvectors"])
    hessian = np.array(data["hessian"])
    gradient = np.array(data["gradient"])
    center_metric = data["center_metric"]
    param_names = data["param_names"]

    if best_params is None:
        best_params = theta_0

    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.2, 1])

    # (a) Friction function comparison
    ax1 = fig.add_subplot(gs[0])
    xi = np.linspace(-5, 5, 500)

    # Linear baseline g(xi) = xi
    g_linear = xi
    g_baseline = pade_friction(xi, *theta_0)
    g_best = pade_friction(xi, *best_params)

    ax1.plot(xi, g_linear, '--', color=COLORS['baseline'], lw=1.5, label='Linear g(xi)=xi')
    ax1.plot(xi, g_baseline, '-', color=COLORS['optimum'], lw=2, label=f'Pade optimum')
    if not np.allclose(best_params, theta_0):
        ax1.plot(xi, g_best, '-', color=COLORS['best'], lw=2, label=f'Hill-climb best')

    ax1.set_xlabel('xi')
    ax1.set_ylabel('g(xi)')
    ax1.set_title('Friction function comparison')
    ax1.legend(loc='upper left')
    ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold')

    # (b) 2D metric surface slice along two eigenvectors
    ax2 = fig.add_subplot(gs[1])

    # Get the two softest directions
    idx_sorted = np.argsort(eigenvalues)
    v0 = eigenvectors[:, idx_sorted[0]]  # softest
    v1 = eigenvectors[:, idx_sorted[1]]  # second softest
    lam0 = eigenvalues[idx_sorted[0]]
    lam1 = eigenvalues[idx_sorted[1]]

    # Create a quadratic approximation of the metric surface
    # f(theta) ~ f0 + grad^T delta + 0.5 delta^T H delta
    t_range = np.linspace(-3, 3, 100)
    T0, T1 = np.meshgrid(t_range, t_range)

    # Scale: t is in units of step size
    scale0 = np.linalg.norm(data["step_sizes"])
    metric_surface = np.zeros_like(T0)
    for i in range(len(t_range)):
        for j in range(len(t_range)):
            delta = T0[i, j] * v0 * scale0 + T1[i, j] * v1 * scale0
            metric_surface[i, j] = center_metric + gradient @ delta + 0.5 * delta @ hessian @ delta

    im = ax2.contourf(T0, T1, metric_surface, levels=20, cmap='viridis')
    ax2.contour(T0, T1, metric_surface, levels=10, colors='white', linewidths=0.3, alpha=0.5)
    ax2.plot(0, 0, 'w*', markersize=12, markeredgecolor='black', markeredgewidth=1, label='Current optimum')

    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Approx. tau_int')

    # Label eigenvector directions
    ev0_label = ", ".join(f"{param_names[k]}={v0[k]:+.2f}" for k in range(3))
    ev1_label = ", ".join(f"{param_names[k]}={v1[k]:+.2f}" for k in range(3))
    ax2.set_xlabel(f'Soft dir (lam={lam0:.1f})\n[{ev0_label}]', fontsize=9)
    ax2.set_ylabel(f'2nd dir (lam={lam1:.1f})\n[{ev1_label}]', fontsize=9)
    ax2.set_title('Quadratic metric surface')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold')

    # (c) Eigenvalue spectrum
    ax3 = fig.add_subplot(gs[2])
    bar_colors = [COLORS['soft'] if ev < 50 else COLORS['stiff'] for ev in eigenvalues]
    bars = ax3.bar(range(3), eigenvalues, color=bar_colors, edgecolor='black', linewidth=0.5)

    # Annotate eigenvector compositions
    for i, ev in enumerate(eigenvalues):
        v = eigenvectors[:, i]
        dominant = param_names[np.argmax(np.abs(v))]
        ax3.text(i, ev + max(eigenvalues)*0.03, f'{dominant}-dominant\nlam={ev:.1f}',
                ha='center', va='bottom', fontsize=9)

    ax3.set_xticks(range(3))
    ax3.set_xticklabels([f'v_{i}' for i in range(3)])
    ax3.set_ylabel('Eigenvalue')
    ax3.set_title('Hessian eigenvalue spectrum')
    ax3.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax3.text(2.4, 50, 'soft threshold', va='bottom', ha='right', fontsize=9, color='gray')
    ax3.text(-0.12, 1.05, '(c)', transform=ax3.transAxes, fontsize=14, fontweight='bold')

    fig.savefig(os.path.join(FIG_DIR, "narrative.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {os.path.join(FIG_DIR, 'narrative.png')}")


def plot_results(data, best_params=None, best_metric=None, hill_climb_log=None):
    """Generate results.png: quantitative panel."""

    theta_0 = np.array(data["theta_0"])
    eigenvalues = np.array(data["eigenvalues"])
    eigenvectors = np.array(data["eigenvectors"])
    hessian = np.array(data["hessian"])
    gradient = np.array(data["gradient"])
    center_metric = data["center_metric"]
    param_names = data["param_names"]
    all_results = data.get("all_results", {})

    if best_params is None:
        best_params = theta_0
    if best_metric is None:
        best_metric = center_metric

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig)

    # (a) Hessian matrix heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(np.array(data["hessian"]), cmap='RdBu_r', aspect='auto')
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(param_names)
    ax1.set_yticklabels(param_names)
    for i in range(3):
        for j in range(3):
            val = data["hessian"][i][j]
            ax1.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=10,
                    color='white' if abs(val) > np.max(np.abs(data["hessian"]))*0.6 else 'black')
    ax1.set_title('Hessian matrix H_ij')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold')

    # (b) All evaluation results as scatter
    ax2 = fig.add_subplot(gs[0, 1])
    labels_sorted = sorted(all_results.keys(), key=lambda k: all_results[k])
    metrics = [all_results[k] for k in labels_sorted]
    y_pos = range(len(labels_sorted))

    colors = ['#DD8452' if m < center_metric else '#C44E52' if m > center_metric*1.2 else '#4C72B0' for m in metrics]
    ax2.barh(y_pos, metrics, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_sorted, fontsize=8)
    ax2.axvline(x=center_metric, color='gray', linestyle='--', linewidth=1, label=f'Center={center_metric:.2f}')
    ax2.set_xlabel('Weighted tau_int')
    ax2.set_title('All Hessian evaluation points')
    ax2.legend(fontsize=9)
    ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold')

    # (c) Gradient vector
    ax3 = fig.add_subplot(gs[1, 0])
    grad = np.array(data["gradient"])
    bar_colors_grad = ['#55A868' if g < 0 else '#C44E52' for g in grad]
    ax3.bar(param_names, grad, color=bar_colors_grad, edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('d(metric)/d(param)')
    ax3.set_title('Gradient at optimum')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.text(-0.12, 1.05, '(c)', transform=ax3.transAxes, fontsize=14, fontweight='bold')

    # (d) Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    table_data = [
        ['Parameter', 'Optimum', 'Best Found', 'Gradient'],
    ]
    for i, name in enumerate(param_names):
        table_data.append([
            name,
            f'{theta_0[i]:.4f}',
            f'{best_params[i]:.4f}',
            f'{grad[i]:+.2f}',
        ])
    table_data.append(['', '', '', ''])
    table_data.append(['Eigenvalues', f'{eigenvalues[0]:.1f}', f'{eigenvalues[1]:.1f}', f'{eigenvalues[2]:.1f}'])
    table_data.append(['', '', '', ''])
    table_data.append(['Center metric', f'{center_metric:.2f}', '', ''])
    table_data.append(['Best metric', f'{best_metric:.2f}', '', ''])

    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Style header row
    for j in range(4):
        table[0, j].set_facecolor('#4C72B0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax4.set_title('Summary')
    ax4.text(-0.02, 1.05, '(d)', transform=ax4.transAxes, fontsize=14, fontweight='bold')

    fig.savefig(os.path.join(FIG_DIR, "results.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {os.path.join(FIG_DIR, 'results.png')}")


if __name__ == "__main__":
    data = load_results()
    plot_narrative(data)
    plot_results(data)
