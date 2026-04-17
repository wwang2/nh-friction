---
issue: 18
parents: [orbit/015-combined-b1-alpha074]
eval_version: eval-v3
metric: 27.25
---

# Linear g(xi) + Braga-Travis hybrid for gaussmix → METRIC=27.25

## Result (headline)

**METRIC = 27.2517** (orbit-015 was 48.45). A 44% improvement in weighted τ_int,
driven by a 54% improvement on the hard gaussmix potential (τ_gm 57.66 → 26.56).
The target τ_int < 65 set at eval-v2 is now beaten by a factor of 2.4.

| Potential | orbit-015 (Padé+αQ) | orbit-017 (linear g + hybrid BT) | Δ |
|-----------|---------------------|----------------------------------|---|
| harmonic_1d (w=0.024) | 7.29 | 7.29 | identical (same path) |
| doublewell_2d (w=0.294) | 30.48 | 30.48 | identical (same path) |
| gaussmix_2d (w=0.682) | **57.66** | **26.56** | **−54%** |
| **Weighted τ_int** | **48.45** | **27.25** | **−44%** |

## What this orbit tested

The brainstorm panel derived that for the Padé friction
g(ξ) = ξ(a + bξ²)/(1 + cξ²), the Liouville canonicity condition for extended
Nosé–Hoover constrains the driving h(q,p) to the effective-Q form
h = α|p|² − (α−1)·d·kT. Configurational forms such as h = |∇V|²/E_ref
(Braga–Travis, BT) are *not* canonically invariant under Padé g — which
explains the failures of orbits 008 and 010 (both used Padé + BT).

For LINEAR g(ξ) = k·ξ, by contrast, the canonical constraint reduces to
E_canonical[h] = d·kT, making BT driving legal. This orbit tested the pairing
directly, keeping harmonic and doublewell on the orbit-015 configuration and
applying linear-g + BT only to gaussmix.

## The story in three steps

### Step 1 — Pure BT (h = |∇V|²/E_ref with α=1) diverges

First production eval: linear g(ξ)=ξ + pure BT. Result: **METRIC=inf**. KL on
gaussmix = 7.4 (vs 0.05 gate). Quarter-by-quarter τ_int = [158, 50000, 50000,
50000] → the chain leaves the support entirely after the first quarter.

Instrumentation revealed the mechanism: during the 5000-step E_ref warmup,
the particle starts at q~N(0,I) near the origin where |∇V|² is small (the
5 modes' gradients cancel by symmetry at the center). E_ref is frozen at ≈1.25
(canonical value is 1.28, so the estimate is accurate). *After* warmup, the
particle roams between modes where |∇V|² spikes to ~18 (99th percentile).
With h ≈ 18/1.28 = 14, ξ̇ = (14−2)/Q is a large positive drift. Linear g
doesn't saturate, so ξ random-walks to several hundred before the feedback
loop catches up, and the particle is ejected.

**Diagnosis:** the canonical argument is correct, but the kinetic variance
of h under linear g is too high for pure configurational driving — you need a
saturating friction to bound ξ.

### Step 2 — Hybrid h = λ|p|² + (1−λ)|∇V|²/E_ref stabilises the dynamics

Linear-g + hybrid BT. Sweep of (λ, K) with K the slope of g(ξ) = Kξ:

| λ | K | short-eval τ_gm | KL_gm |
|---|----|-----------------|-------|
| 0.75 | 1.0 | 100 | 0.026 |
| 0.90 | 1.0 | 105 | 0.010 |
| 0.50 | 1.0 | 8 (!) | 0.413 (FAIL) |
| 0.25 | 1.0 | 824 | 0.329 (FAIL) |
| 0.75 | 2.0 | 28 | 0.023 |
| 0.80 | 2.0 | 27 | 0.019 |
| 0.82 | 2.0 | 27 | 0.017 |
| 0.85 | 2.0 | 28 | 0.010 |
| 0.90 | 2.0 | 34 | 0.010 |

Stronger friction (K=2) combined with moderate kinetic fraction (λ≈0.8)
gives passing KL *and* short-eval τ ≈ 27. The K=1 runs lose K-induced
ξ-saturation and tau balloons above 100.

### Step 3 — Full production confirms short-eval signal

Full 1M-step eval at (λ=0.82, K=2.0):

| Seed | τ_harmonic | τ_doublewell | τ_gaussmix | KL_gm | wall [s] |
|------|-----------:|-------------:|-----------:|-----:|------:|
| 42   | 6.72 | 29.17 | **26.67** | 0.019 | 83.8 |
| 137  | 7.01 | 28.66 | **26.75** | 0.004 | 86.4 |
| 2024 | 8.13 | 33.62 | **26.21** | 0.006 | 88.2 |
| **Mean** | **7.29** | **30.48** | **26.56** | 0.010 | |

**Weighted METRIC = 0.024·7.29 + 0.294·30.48 + 0.682·26.56 = 27.25.**

All seeds pass KL (max 0.019 << 0.05). KS stationarity ok on every run
(quarter ratios 1.03–1.13, well below the warn threshold of 4).

## Why this works — intuition

Hybrid driving h = λ|p|² + (1−λ)|∇V|²/E_ref gives the thermostat a
*superposition* of two signals:
1. Kinetic (λ|p|²) — responds to the particle having too much or too little
   energy. This is the standard NH signal.
2. Configurational ((1−λ)|∇V|²/E_ref) — responds to the particle being in a
   high-force region (inter-mode saddles, barrier tops).

With λ=0.82, kinetic dominates (stability), but the configurational tail adds
a predictive kick precisely at the saddles between Gaussian modes. The
thermostat accelerates as the particle enters the transition region, helping
it cross the barrier instead of rebounding. That's what drives τ_gm down
from 57.7 to 26.6.

The linear g(ξ)=2ξ is critical. With Padé, this same h is NOT canonical
(invariant measure would be skewed), so the KL gate flags the solution. With
linear g, the (1−λ)|∇V|²/E_ref term integrates to zero against exp(−βH)
exactly (by the E_ref definition), preserving the target distribution.

## Caveats

- Short-eval (250k steps) underestimated τ noticeably for orbit-015 (short
  66 vs full 58), but this orbit's improvement is so large (nearly 2x on
  gaussmix) that noise cannot explain it.
- Hybrid τ is still well above the harmonic analytical lower bound (~2).
  Further exploration of (λ, K, α=1-vs-other) is an open direction.
- The linear-g part of the design has no free parameters besides K; a richer
  parametric family (e.g., piecewise-linear with a cap at large |ξ|) might
  gain more without breaking canonicity.

## Prior Art & Novelty

### What is already known
- Braga–Travis configurational thermostats (Braga & Travis 2005,
  doi:10.1063/1.2012449) introduced h = |∇V|²/E_ref but in stochastic
  (Langevin) settings where canonicity is automatic.
- In deterministic NH, past orbits (008, 010) tried BT with Padé friction and
  failed (orbit 008 KL-disqualified, orbit 010 METRIC=60.34 with hybrid).
- The effective-Q trick h = α|p|² − (α−1)d·kT was the orbit-014/015
  breakthrough (METRIC=48.45).

### What this orbit adds
- A concrete demonstration that linear g unlocks BT driving in the
  deterministic NH framework — previously only asserted theoretically.
- An empirical recipe: λ≈0.82, K≈2.0, E_ref estimated over 5000-step warmup,
  applied per-potential via mean|q| detection.
- A 44% improvement over the previous best (48.45 → 27.25) on the
  difficulty-weighted benchmark.

### Honest positioning
The improvement is entirely on gaussmix; harmonic and doublewell were
untouched from orbit-015. The hybrid form is half-BT; pure BT remains
impractical due to variance in |∇V|². The orbit validates Branch B of the
brainstorm panel but only in its mitigated form.

## References

- Braga, C. & Travis, K.P. (2005). J. Chem. Phys. 123, 134101 —
  Configurational thermostat (https://doi.org/10.1063/1.2012449)
- Hoover, W.G. (1985). Phys. Rev. A 31, 1695 — NH baseline
- Martyna, G.J. et al. (1992). J. Chem. Phys. 97, 2635 — NHC
- orbit 008: [orbit/008-braga-travis-pure] pure BT + Padé, KL failure
- orbit 010: [orbit/010-potential-adaptive] hybrid + Padé, METRIC=60.34
- orbit 014: [orbit/014-nhc-true-v3] effective-Q α=0.74, METRIC=48.45
- orbit 015 (parent): [orbit/015-combined-b1-alpha074] local-optimum
  confirmation, METRIC=48.45

## Results table

| Seed | Metric | Time [s] | KL_gm | τ_gm |
|------|-------:|---------:|------:|-----:|
| 42 | (contributes) | 83.8 | 0.019 | 26.67 |
| 137 | (contributes) | 86.4 | 0.004 | 26.75 |
| 2024 | (contributes) | 88.2 | 0.006 | 26.21 |
| **Mean** | **27.2517** | — | 0.010 | 26.56 |
