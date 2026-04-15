#!/usr/bin/env python3
"""Generate figures for orbit/002-pade-cmaes-refine."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
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
    "parent": "#888888",
    "refined": "#4C72B0",
    "standard_nh": "#C44E52",
}

OUTDIR = "/Users/wujiewang/code/bath/.worktrees/002-pade-cmaes-refine/orbits/002-pade-cmaes-refine/figures"

# ── Panel figure: g(xi) comparison + bar chart of per-potential tau ──────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel (a): g(xi) vs xi
ax = axes[0]
xi = np.linspace(-5, 5, 500)

# Standard NH: g(xi) = xi
g_nh = xi

# Parent (orbit 001): a=0.5, b=3, c=0.06
def g_parent(xi):
    xi2 = xi * xi
    return xi * (0.5 + 3.0 * xi2) / (1.0 + 0.06 * xi2)

# Refined (this orbit): a=0.7, b=3, c=0.06
def g_refined(xi):
    xi2 = xi * xi
    return xi * (0.7 + 3.0 * xi2) / (1.0 + 0.06 * xi2)

ax.plot(xi, g_nh, color=COLORS["standard_nh"], linestyle="--", linewidth=1.5,
        label="Standard NH: g = $\\xi$")
ax.plot(xi, g_parent(xi), color=COLORS["parent"], linestyle="-.", linewidth=2.0,
        label="Parent (a=0.5): metric=97.9")
ax.plot(xi, g_refined(xi), color=COLORS["refined"], linewidth=2.5,
        label="Refined (a=0.7): metric=84.1")

ax.set_xlabel("$\\xi$ (thermostat variable)")
ax.set_ylabel("$g(\\xi)$")
ax.set_title("Friction functions")
ax.legend(loc="upper left")
ax.set_xlim(-5, 5)
ax.set_ylim(-200, 200)
ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# Panel (b): Per-potential tau comparison
ax = axes[1]
potentials = ["harmonic\n1D", "doublewell\n2D", "gaussmix\n2D", "weighted\nmean"]

# Parent orbit 001 values (from log)
parent_taus = [12.23, 130.87, 86.68, 97.90]
# This orbit values
refined_taus = [10.36, 114.08, 73.81, 84.14]

x = np.arange(len(potentials))
width = 0.35

bars1 = ax.bar(x - width/2, parent_taus, width, color=COLORS["parent"],
               label="Parent (a=0.5)", alpha=0.8)
bars2 = ax.bar(x + width/2, refined_taus, width, color=COLORS["refined"],
               label="Refined (a=0.7)", alpha=0.8)

# Add value labels
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}",
            ha="center", va="bottom", fontsize=9, color=COLORS["parent"])
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}",
            ha="center", va="bottom", fontsize=9, color=COLORS["refined"])

# Target line
ax.axhline(y=90, color="#DD8452", linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(3.5, 91, "target = 90", fontsize=9, color="#DD8452", va="bottom")

ax.set_xticks(x)
ax.set_xticklabels(potentials)
ax.set_ylabel("$\\tau_{\\mathrm{int}}$")
ax.set_title("Autocorrelation time by potential")
ax.legend(loc="upper right")
ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

fig.savefig(f"{OUTDIR}/results.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ── Second figure: parameter sweep heatmap ──────────────────────────────

# Sweep data from sweep 1 (a vs metric, c vs metric)
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Panel (a): c sweep (a=0.5, b=3)
ax = axes2[0]
c_vals = [0.03, 0.04, 0.05, 0.06, 0.12, 0.15]
c_metrics = [94.58, 117.53, 106.44, 97.90, 131.76, 270.23]
# Mark inf values
c_vals_all = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]
c_metrics_all = [94.58, 117.53, 106.44, 97.90, None, None, None, 131.76, 270.23]

c_finite = [(c, m) for c, m in zip(c_vals_all, c_metrics_all) if m is not None]
c_inf = [c for c, m in zip(c_vals_all, c_metrics_all) if m is None]

ax.plot([c for c, _ in c_finite], [m for _, m in c_finite],
        "o-", color=COLORS["refined"], linewidth=2, markersize=7)
if c_inf:
    ax.scatter(c_inf, [300]*len(c_inf), marker="x", s=80, color=COLORS["standard_nh"],
               zorder=5, linewidths=2)
    ax.annotate("KL fail", xy=(c_inf[0], 300), fontsize=9, color=COLORS["standard_nh"],
                xytext=(c_inf[0]+0.01, 310))

ax.axhline(y=90, color="#DD8452", linestyle=":", linewidth=1.5, alpha=0.7)
ax.set_xlabel("$c$ (denominator damping)")
ax.set_ylabel("weighted $\\tau_{\\mathrm{int}}$")
ax.set_title("$c$ sweep (a=0.5, b=3)")
ax.set_ylim(70, 320)
ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# Panel (b): a sweep (b=3, c=0.06)
ax = axes2[1]
a_vals_all = [0.3, 0.4, 0.5, 0.6, 0.65, 0.68, 0.7, 0.72, 0.75, 0.8, 1.0]
a_metrics_all = [None, None, 97.90, 105.75, 108.11, 105.60, 84.14, 100.34, 101.74, 92.64, None]

a_finite = [(a, m) for a, m in zip(a_vals_all, a_metrics_all) if m is not None]
a_inf = [a for a, m in zip(a_vals_all, a_metrics_all) if m is None]

ax.plot([a for a, _ in a_finite], [m for _, m in a_finite],
        "o-", color=COLORS["refined"], linewidth=2, markersize=7)
if a_inf:
    ax.scatter(a_inf, [130]*len(a_inf), marker="x", s=80, color=COLORS["standard_nh"],
               zorder=5, linewidths=2)
    ax.annotate("KL fail", xy=(a_inf[-1], 130), fontsize=9, color=COLORS["standard_nh"],
                xytext=(a_inf[-1]-0.15, 135))

# Highlight optimum
ax.scatter([0.7], [84.14], s=120, color=COLORS["refined"], zorder=10, edgecolors="black", linewidths=1.5)
ax.annotate("a=0.7, metric=84.1", xy=(0.7, 84.14), fontsize=9,
            xytext=(0.75, 78), arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

ax.axhline(y=90, color="#DD8452", linestyle=":", linewidth=1.5, alpha=0.7)
ax.axhline(y=97.9, color=COLORS["parent"], linestyle="--", linewidth=1, alpha=0.5)
ax.text(0.95, 98.5, "parent=97.9", fontsize=9, color=COLORS["parent"], ha="right")

ax.set_xlabel("$a$ (linear coupling)")
ax.set_ylabel("weighted $\\tau_{\\mathrm{int}}$")
ax.set_title("$a$ sweep (b=3, c=0.06)")
ax.set_ylim(70, 140)
ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

fig2.savefig(f"{OUTDIR}/sweep_analysis.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig2)

print("Figures saved to", OUTDIR)
