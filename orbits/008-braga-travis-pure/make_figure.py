"""Generate figures for orbit 008-braga-travis-pure."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

# ── Panel 1: Driving function h vs |∇V|² for each potential ──────────────
ax = axes[0]
gv2_range = np.linspace(0, 20, 500)
# Estimated E_ref values (from warmup): harmonic≈1.0, doublewell≈7.0, gaussmix≈1.26
configs = [
    ("1D harmonic\n(E_ref≈1.0, d=1)", 1.0, 1, "#2196F3"),
    ("2D double-well\n(E_ref≈7.0, d=2)", 7.0, 2, "#FF5722"),
    ("2D Gaussian mix\n(E_ref≈1.26, d=2)", 1.26, 2, "#4CAF50"),
]
for label, e_ref, d, color in configs:
    h_vals = gv2_range * d / e_ref
    target = d  # d·kT = d for kT=1
    ax.plot(gv2_range, h_vals, color=color, lw=1.8, label=label)
    ax.axhline(target, color=color, lw=0.8, ls="--", alpha=0.6)
ax.set_xlabel("|∇V|²")
ax.set_ylabel("h(grad_V) = |∇V|² · d / E_ref")
ax.set_title("Braga-Travis driving function", fontsize=13, fontweight="medium")
ax.legend(fontsize=9)
ax.set_xlim(0, 15)
ax.set_ylim(0, 30)

# ── Panel 2: Coupling strength comparison: BT vs kinetic ──────────────────
ax = axes[1]
potentials = ["1D harmonic", "2D double-well", "2D Gaussian mix"]
# eval-v1 (kinetic) baseline per-potential τ_int
tau_v1 = [10.36, 114.08, 73.81]
# Expected BT improvement (rough estimates based on theory)
# BT provides stronger coupling where forces are large
# For now show the structure, actual values will be filled from eval
x = np.arange(len(potentials))
w = 0.35
bars = ax.bar(x, tau_v1, width=w, color=["#2196F3", "#FF5722", "#4CAF50"], alpha=0.8, label="eval-v1 (kinetic)")
ax.set_xticks(x)
ax.set_xticklabels(potentials, fontsize=9, rotation=15, ha="right")
ax.set_ylabel("τ_int (mean over 3 seeds)")
ax.set_title("Baseline per-potential τ_int\n(eval-v1 reference)", fontsize=13, fontweight="medium")
ax.set_ylim(0, 140)
# Add weights as annotations
weights = [0.024, 0.294, 0.682]
for i, (bar, w_val) in enumerate(zip(bars, weights)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"w={w_val:.1%}", ha="center", va="bottom", fontsize=8)

# ── Panel 3: BT coupling principle diagram ───────────────────────────────
ax = axes[2]
# Show |∇V|² along x for double-well V=(x²-1)²
x_vals = np.linspace(-2, 2, 500)
V = (x_vals**2 - 1)**2
dV = 4 * x_vals * (x_vals**2 - 1)
gV2 = dV**2

ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
ax.spines["top"].set_visible(False)

l1, = ax.plot(x_vals, V, color="#FF5722", lw=2, label="V(x)")
l2, = ax2.plot(x_vals, gV2, color="#9C27B0", lw=2, ls="--", label="|∇V|²")
ax.set_xlabel("x")
ax.set_ylabel("V(x) = (x²−1)²", color="#FF5722")
ax2.set_ylabel("|∇V|² = [4x(x²−1)]²", color="#9C27B0")
ax.set_title("BT coupling: strong at barrier\nflanks (|∇V|² peaks)", fontsize=13, fontweight="medium")
ax.tick_params(axis="y", labelcolor="#FF5722")
ax2.tick_params(axis="y", labelcolor="#9C27B0")
# Shade barrier region
ax.axvspan(-0.4, 0.4, alpha=0.08, color="purple")
ax.annotate("barrier flanks\n(strong BT coupling)", xy=(0.58, 0.5), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8), xytext=(1.1, 1.5))

lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc="upper center", fontsize=9)
ax.set_xlim(-2, 2)

fig.suptitle("Orbit 008: Pure Braga-Travis configurational driving (eval-v2 baseline)",
             fontsize=13, fontweight="medium", y=1.02)

plt.savefig("orbits/008-braga-travis-pure/figures/bt_driving_analysis.png", dpi=150, bbox_inches="tight")
print("Saved bt_driving_analysis.png")
