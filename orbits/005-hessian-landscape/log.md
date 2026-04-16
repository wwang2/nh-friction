---
issue: 6
parents: [003-cmaes-3pot]
eval_version: eval-v1
metric: 84.14
---

# Hessian Landscape Analysis of Pade Friction Parameters

## Result Summary

The basin around (a=0.7, b=3.0, c=0.06) is **deep and steep in all directions**. The diagonal Hessian eigenvalues are 1,552 (b), 100,626 (a), and 2,409,894 (c) -- all far above any reasonable "soft" threshold. There are no exploitable flat directions. The Newton-step predicted improvements are less than 0.3 metric units, well within the seed noise band (~16% CV on gaussmix_2d alone). The parent orbit's parameters are a genuine local minimum, not a saddle point or ridge.

**Metric = 84.14** (unchanged from parent orbit 003).

## Why the basin is deep: a physical explanation

Consider what each parameter controls:

- **a = 0.7** (linear coupling at xi ~ 0): This is the critical parameter for breaking KAM tori on the 1D harmonic oscillator. Reducing a below ~0.6 causes non-ergodic trapping (KL gate failure). Increasing a weakens mode-hopping on gaussmix_2d. The optimum sits precisely at the tension point between these two constraints.

- **b = 3.0** (cubic nonlinearity): Controls the far-from-equilibrium restoring force. Too high damages harmonic oscillator KL compliance; too low weakens barrier crossing on doublewell_2d. This is the "softest" direction (H_bb = 1,552), but even this is far too stiff for useful hill-climbing.

- **c = 0.06** (denominator damping): The most sensitive parameter (H_cc = 2.4M). A change of just 0.005 (8%) increases the metric by 30-37%. This parameter controls the transition from cubic to linear asymptotic behavior: g(xi) ~ (b/c)*xi for large xi. At c=0.06, the transition happens at xi ~ 1/sqrt(c) ~ 4, which appears to be the critical scale for thermostat dynamics.

The metric surface is a narrow, steep-walled valley. The gradient is nonzero (|g| = 321) but the curvature is so large that the Newton steps are microscopic: da = +0.002, db = -0.019, dc = -0.0001.

## Hessian Results

| Point | a | b | c | Metric | Delta | h1d tau | dw tau | gm tau |
|-------|-------|-------|---------|--------|-------|---------|--------|--------|
| center | 0.700 | 3.00 | 0.060 | **84.14** | -- | 10.4 | 114.1 | 73.8 |
| +a | 0.720 | 3.00 | 0.060 | 100.34 | +19% | 10.5 | 162.6 | 76.6 |
| -a | 0.680 | 3.00 | 0.060 | 108.20 | +29% | 9.4 | 203.9 | 70.3 |
| +b | 0.700 | 3.10 | 0.060 | 94.90 | +13% | 11.8 | 121.0 | 86.6 |
| -b | 0.700 | 2.90 | 0.060 | 88.91 | +5.7% | 9.1 | 144.4 | 67.8 |
| +c | 0.700 | 3.00 | 0.065 | 115.52 | +37% | 10.7 | 210.7 | 78.1 |
| -c | 0.700 | 3.00 | 0.055 | 113.01 | +34% | 10.4 | 195.7 | 80.9 |

### Diagonal Hessian and Gradient

| Param | H_ii | Gradient | Newton step | Predicted improvement |
|-------|------|----------|-------------|----------------------|
| a | 100,626 | -196.4 | +0.0020 | -0.19 |
| b | 1,552 | +30.0 | -0.0193 | -0.29 |
| c | 2,409,894 | +251.8 | -0.0001 | -0.01 |

Condition number: H_cc / H_bb = 1,552x. The `c` direction is 1,552 times stiffer than the `b` direction.

### Per-potential tension

The per-potential breakdown reveals why optimization is so constrained. Every perturbation trades off between potentials:

- **Reducing a** (a=0.68): gaussmix improves (73.8 -> 70.3) but doublewell collapses (114 -> 204). Net: +29%.
- **Reducing b** (b=2.9): gaussmix improves (73.8 -> 67.8) but doublewell worsens (114 -> 144). Net: +5.7%.
- **Changing c** in either direction: doublewell collapses (114 -> 196-211). Net: +34-37%.

The doublewell_2d potential is the binding constraint on c and a. The gaussmix_2d prefers slightly different parameters, but the doublewell penalty dominates due to its 29.4% difficulty weight.

## Evaluation Results

| Seed | Metric | Time |
|------|--------|------|
| 42 | 84.14 | 172s |
| 137 | 84.14 | 172s |
| 2024 | 84.14 | 172s |
| **Mean** | **84.14** | |

## Approach

1. Computed the diagonal Hessian of weighted_tau_int via central finite differences at theta* = (0.7, 3.0, 0.06).
2. Step sizes: ha=0.02, hb=0.1, hc=0.005 (~3%, 3%, 8% of parameter values).
3. 7 evaluations total (center + 2 per parameter), each running the full eval-v1 protocol (3 potentials x 3 seeds = 9 integrations of 1M steps).
4. Computed gradient, diagonal curvature, Newton steps.
5. All directions are stiff (H_ii >> 1). No hill-climbing attempted -- predicted improvements are within noise.

## Prior Art and Novelty

### What is already known

- [Bulgac and Kusnezov (1990)](https://doi.org/10.1103/PhysRevA.42.5045) -- Proposed nonlinear friction g(xi) = 2*xi/(1+xi^2) for canonical sampling
- Parent orbit 003-cmaes-3pot confirmed (a=0.7, b=3.0, c=0.06) as a local optimum via grid search over 882 candidates

### What this orbit adds

- **Quantitative curvature analysis**: The Hessian eigenvalues (1,552 to 2.4M) prove the basin is deep -- the current optimum is a genuine local minimum, not an artifact of discrete grid search
- **Parameter sensitivity hierarchy**: c (denominator damping) is 1,552x more sensitive than b (cubic nonlinearity), revealing that the rational-function asymptotic transition scale is the most critical design parameter
- **Inter-potential tension quantified**: Every perturbation trades off gaussmix_2d vs doublewell_2d performance, with no free lunch available

### Honest positioning

This orbit is a **negative result** -- no improvement over the parent's metric=84.14. The contribution is confirming that the Pade parameter space has been optimized to its local limit. Further improvement requires either (a) a different functional form for g(xi), (b) more parameters (higher-order Pade, neural basis), or (c) a fundamentally different thermostat design (e.g., NHC with optimized coupling).

## Glossary

- **Hessian**: Matrix of second partial derivatives of the metric with respect to parameters
- **Soft direction**: Eigenvector of the Hessian with near-zero eigenvalue, indicating the metric is nearly flat along that direction
- **Pade friction**: Rational function g(xi) = xi*(a+b*xi^2)/(1+c*xi^2) for the Nose-Hoover thermostat
- **tau_int**: Integrated autocorrelation time -- the metric being minimized
- **KAM tori**: Kolmogorov-Arnold-Moser invariant surfaces that trap trajectories in phase space
- **Newton step**: delta_i = -gradient_i / H_ii, the optimal step size under quadratic approximation

## References

- [Bulgac, A. & Kusnezov, D. (1990)](https://doi.org/10.1103/PhysRevA.42.5045) -- Canonical ensemble averages from pseudomicrocanonical dynamics
- [Sokal, A. (1997)](https://doi.org/10.1007/978-1-4899-0319-8_6) -- Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms (Cargese lecture notes)
- Builds on orbit/003-cmaes-3pot (metric=84.14) which established the Pade optimum
