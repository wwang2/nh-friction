---
issue: 17
parents: [orbit/015-combined-b1-alpha074]
eval_version: eval-v3
metric: 48.45
---

# CMA-ES Parameter Search for Gaussmix Friction

## Motivation

The gaussmix potential carries 68.2% of the total metric weight, making it the dominant optimization target. Prior orbits found two local optima for gaussmix parameters:

- (a=0.70, b=3.00, c=0.06, alpha=0.74): tau_gm ~ 57.66 [orbit-014]
- (a=0.70, b=1.00, c=0.06, alpha=1.00): tau_gm ~ 59.25 [orbit-012]

Crucially, combining the best features of both (b=1.0 + alpha=0.74) was destructive (tau_gm=169.5), indicating these parameters interact nonlinearly. This orbit attempted to find better parameters via systematic search.

## Approach

Coordinate descent from three starting points in the 4D parameter space (a, b, c, alpha), followed by fine-tuning and rigorous multi-seed verification.

### Starting Points
- Basin A: (a=0.70, b=3.00, c=0.06, alpha=0.74) -- known local optimum from orbit-014
- Basin B: (a=0.70, b=1.00, c=0.06, alpha=1.00) -- second basin from orbit-012
- Basin C: (a=1.00, b=2.00, c=0.03, alpha=0.90) -- unexplored region

### Search Strategy
1. Coordinate descent: sweep each parameter independently, keep best
2. Fast evaluation: 200k integration steps, single seed (seed=42)
3. Fine-tuning: smaller step sizes around best candidate
4. Verification: 1M steps, 3 seeds (42, 137, 2024)

## Key Finding: Short-Run Estimates Are Unreliable

The 200k-step evaluator showed dramatic "improvements" that did not survive 1M-step verification. This is the central finding of this orbit: **short-trajectory estimates of tau_int are systematically noisy and can mislead optimization**.

### Coordinate Descent Results (200k steps)

| Basin | Start tau | Final tau | Best params found |
|-------|-----------|-----------|-------------------|
| A | 62.6 | 32.6 | (0.80, 3.50, 0.06, 0.74) |
| B | 146.6 | 34.2 | (0.60, 1.00, 0.06, 1.00) |
| C | 56.7 | 46.1 | (0.60, 4.00, 0.12, 0.74) |

Basin A appeared to show a 48% improvement from a=0.7->0.8 and b=3.0->3.5.

### Verification (1M steps, 3 seeds) -- The Reality Check

| Candidate | params | seed=42 | seed=137 | seed=2024 | Mean +/- Std |
|-----------|--------|---------|----------|-----------|-------------|
| Cand 0 (Basin A best) | (0.80, 3.50, 0.06, 0.74) | 62.51 | 55.01 | 81.01 | 66.18 +/- 10.93 |
| Cand 1 (Basin B best) | (0.60, 1.00, 0.06, 1.00) | 73.94 | 58.18 | 75.27 | 69.13 +/- 7.76 |
| Cand 2 (baseline) | (0.70, 3.00, 0.06, 0.74) | 64.53 | 60.59 | 47.86 | **57.66 +/- 7.11** |

**The baseline wins.** The 200k "improvements" were noise artifacts. The parameter changes that looked like 48% improvements at 200k steps turned into 15% degradation at 1M steps.

## Production Eval Results

| Potential | Seed 42 | Seed 137 | Seed 2024 | Mean | KL |
|-----------|---------|----------|-----------|------|-----|
| harmonic_1d | 6.72 | 7.01 | 8.13 | 7.29 | 0.030 |
| doublewell_2d | 29.16 | 28.66 | 33.62 | 30.48 | 0.001 |
| gaussmix_2d | 64.53 | 60.59 | 47.86 | 57.66 | 0.002 |
| **Weighted METRIC** | | | | **48.45** | |

## Why the Baseline is Hard to Beat

The friction function g(xi) = xi*(0.7 + 3.0*xi^2)/(1 + 0.06*xi^2) with alpha=0.74 driving appears to sit near a robust basin minimum. The parameter `a=0.7` controls the linear response at small xi (determines initial thermostat coupling), while `b=3.0` controls the nonlinear saturation at large xi. The ratio a/b and the interplay with alpha create a balance:

- Increasing `a` to 0.8 strengthens small-xi friction but destabilizes inter-mode transitions (higher variance across seeds)
- Increasing `b` to 3.5 pushes the saturation regime closer, which helps some seeds but hurts others
- The combination (a=0.8, b=3.5) has notably higher seed-to-seed variance (sigma=10.93 vs 7.11 for baseline)

## Prior Art and Novelty

### What is already known
- Pade friction form g(xi) = xi*(a + b*xi^2)/(1 + c*xi^2) identified as optimal family [orbit-003/004]
- Per-potential parameter tuning explored [orbit-014, orbit-015]
- Alpha driving h = alpha*|p|^2 - (alpha-1)*d*kT for Liouville preservation [orbit-010]

### What this orbit adds
- Systematic multi-basin coordinate descent confirms (0.70, 3.00, 0.06, 0.74) is robust
- Documents that 200k-step tau_int estimates are unreliable for optimization (noise > signal at fine parameter resolution)
- Establishes the variance-aware criterion: the baseline has both lower mean AND lower variance than alternatives

### Honest positioning
This is a null result confirming the existing optimum. The orbit did not find better gaussmix parameters but provides evidence that the current parameters are near-optimal in this function family.

## References

- Sokal (1997) Cargese lecture notes, section 3 -- tau_int definition and windowing
- Martyna et al. (1992) J. Chem. Phys. 97, 2635 -- NHC equations
- Hansen & Ostermeier (2001) Evolutionary Computation 9(2) -- CMA-ES algorithm
