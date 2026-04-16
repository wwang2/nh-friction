#!/usr/bin/env python3
"""Generate figures for orbit 009-hybrid-lambda."""
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
    "kinetic": "#4C72B0",
    "hybrid_ema": "#DD8452",
    "hybrid_frozen": "#C44E52",
    "hybrid_mom": "#55A868",
    "baseline": "#888888",
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# ── Panel (a): Lambda scan for adaptive EMA hybrid ──────────────────────
ax = axes[0]
ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# Data from experiments (proxy eval, seed=42)
lambdas_ema = [0.0, 0.5, 0.7, 0.9, 1.0]
kl_harmonic_ema = [5.50, 0.216, 0.036, 0.029, 0.023]
kl_doublewell_ema = [5.50, 0.010, 0.022, 6.53, 0.006]
kl_gaussmix_ema = [3.84, 0.017, 0.019, 0.009, 0.011]

ax.semilogy(lambdas_ema, kl_harmonic_ema, "o-", color="#4C72B0", label="harmonic 1d", markersize=5)
ax.semilogy(lambdas_ema, kl_doublewell_ema, "s-", color="#DD8452", label="doublewell 2d", markersize=5)
ax.semilogy(lambdas_ema, kl_gaussmix_ema, "^-", color="#55A868", label="gaussmix 2d", markersize=5)
ax.axhline(y=0.05, color="#888888", linestyle="--", linewidth=1, label="KL threshold")
ax.set_xlabel(r"$\lambda$ (kinetic fraction)")
ax.set_ylabel("KL divergence")
ax.set_title("Adaptive EMA hybrid driving")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(1e-3, 20)
ax.legend(loc="upper left", fontsize=9)

# ── Panel (b): Lambda scan for frozen E_ref hybrid ──────────────────────
ax = axes[1]
ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

lambdas_frozen = [0.5, 0.7, 0.9, 1.0]
kl_h_frozen = [4.75, 5.88, 0.030, 0.027]
kl_dw_frozen = [3.55, 6.84, 6.53, 0.006]
kl_gm_frozen = [3.84, 0.019, 0.009, 0.011]

ax.semilogy(lambdas_frozen, kl_h_frozen, "o-", color="#4C72B0", label="harmonic 1d", markersize=5)
ax.semilogy(lambdas_frozen, kl_dw_frozen, "s-", color="#DD8452", label="doublewell 2d", markersize=5)
ax.semilogy(lambdas_frozen, kl_gm_frozen, "^-", color="#55A868", label="gaussmix 2d", markersize=5)
ax.axhline(y=0.05, color="#888888", linestyle="--", linewidth=1, label="KL threshold")
ax.set_xlabel(r"$\lambda$ (kinetic fraction)")
ax.set_ylabel("KL divergence")
ax.set_title("Frozen E_ref hybrid driving")
ax.set_xlim(0.45, 1.05)
ax.set_ylim(1e-3, 20)
ax.legend(loc="upper left", fontsize=9)

# ── Panel (c): Momentum hybrid tau comparison ───────────────────────────
ax = axes[2]
ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")

alphas_mom = [0.0, 0.08, 0.3, 0.7, 1.0]
tau_h_mom = [8.6, 11.8, 10.1, 7.2, 7.1]
tau_dw_mom = [101.7, 84.0, 277.8, 104.8, 64.0]
tau_gm_mom = [59.2, 84.2, 123.1, 524.7, 300.2]

# Weighted tau (approx)
weighted_mom = [0.024*h + 0.2944*d + 0.6816*g for h,d,g in zip(tau_h_mom, tau_dw_mom, tau_gm_mom)]

x_pos = np.arange(len(alphas_mom))
width = 0.2
ax.bar(x_pos - width, [0.024*t for t in tau_h_mom], width, color="#4C72B0", label="harmonic (w=0.024)")
ax.bar(x_pos, [0.2944*t for t in tau_dw_mom], width, color="#DD8452", label="doublewell (w=0.294)")
ax.bar(x_pos + width, [0.6816*t for t in tau_gm_mom], width, color="#55A868", label="gaussmix (w=0.682)")
ax.plot(x_pos, weighted_mom, "ko-", markersize=6, linewidth=1.5, label="weighted total", zorder=5)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{a:.2f}" for a in alphas_mom])
ax.set_xlabel(r"$\alpha$ (momentum mixing)")
ax.set_ylabel(r"Weighted $\tau_{\mathrm{int}}$")
ax.set_title(r"Momentum hybrid $h=(1-\alpha)|p|^2 + \alpha|p|^4/((d+2)kT)$")
ax.legend(loc="upper right", fontsize=8)

fig.suptitle("Orbit 009: Hybrid driving exploration — all formulations fail to beat pure kinetic", y=1.02, fontsize=13)

outpath = "/Users/wujiewang/code/bath/.worktrees/009-hybrid-lambda/orbits/009-hybrid-lambda/figures/hybrid_driving_analysis.png"
fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {outpath}")
