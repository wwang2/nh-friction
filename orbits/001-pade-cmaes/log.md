---
issue: 2
parents: []
eval_version: eval-v1
metric: 120.44
---

# Rational Pade Friction with CMA-ES Tuning

## Approach

The standard Nose-Hoover thermostat uses g(xi) = xi, which is non-ergodic on the 1D harmonic oscillator due to KAM tori (Legoll et al. 2007). The Bulgac-Kusnezov form g(xi) = 2*xi/(1+xi^2) is bounded and also fails KL on 1D HO when used without NHC chains.

We propose a rational Pade family that subsumes both:

g(xi) = xi * (a + b*xi^2) / (1 + c*xi^2 + d*xi^4)

This is automatically odd (g(-xi) = -g(xi)) since it factors as xi times an even function of xi. The four parameters (a, b, c, d) control:
- a: linear coupling strength near xi=0 (a=1 recovers standard NH behavior locally)
- b: cubic enhancement at moderate xi (promotes chaos, breaks KAM tori per Hoover & Sprott 2016)
- c: quadratic damping in denominator (bounded tails when c > 0)
- d: quartic damping (stronger tail control)

Special cases:
- a=1, b=c=d=0: standard NH (non-ergodic)
- a=2, c=1, b=d=0: BK thermostat
- a>0, b>0, c,d small: Ju-Bulgac-like cubic enhancement

## Strategy

Use CMA-ES (pycma) in setup() to optimize (a, b, c, d) on a cheap proxy: short integration (50k steps) on the 1D harmonic oscillator. The proxy scores parameter candidates by:
1. KL divergence (must be < 0.05) — hard constraint via penalty
2. Estimated tau_int (lower is better)

The full evaluator then runs the optimized function across all 3 potentials.

## Prior Art and Novelty

### What is already known
- Bulgac & Kusnezov (1990): nonlinear friction g(xi) = 2xi/(1+xi^2)
- Hoover, Sprott, Hoover (2016): cubic terms in g(xi) promote chaos and break KAM tori
- Sergi & Giaquinta (2010): general framework for non-quadratic chain thermostat construction
- Tapias et al. (2017): logistic thermostat g(xi) = tanh(xi) is ergodic but has energy issues

### What this orbit adds (if anything)
- Systematic CMA-ES optimization over a 4-parameter rational Pade family
- The Pade form provides a smooth interpolation between known special cases

### Honest positioning
This orbit applies known functional forms (rational functions of xi) with automated parameter tuning. The novelty, if any, is in the specific parameter values found by optimization rather than in the functional form itself.

## References
- Legoll, Luskin, Moeckel (2007) doi:10.1007/s00205-006-0029-1 — NH non-ergodicity proof
- Bulgac, Kusnezov (1990) doi:10.1103/PhysRevA.42.5045 — BK thermostat
- Hoover, Sprott, Hoover (2016) doi:10.1016/j.cnsns.2015.08.020 — cubic friction for ergodicity
- Tapias, Sanders, Bravetti (2017) doi:10.12921/cmst.2016.0000061 — logistic thermostat
- Sokal (1997) — tau_int definition and automatic windowing

## Research Notes

### Iteration 1: Polynomial g(xi) = 0.5*xi + 3.0*xi^3

Parameters: a=0.5, b=3.0, c=0.0, d=0.0 (pure polynomial, no denominator damping)

The key insight: strong cubic nonlinearity (b=3.0) is essential for breaking KAM tori on the 1D harmonic oscillator. Bounded forms (c, d > 0) that looked good on short proxy integrations failed the KL gate on the full 1M-step eval because the proxy was too short to detect non-ergodicity.

The polynomial form g(xi) = 0.5*xi + 3.0*xi^3 is robustly ergodic across all seeds:
- 1D HO mean KL = 0.008 (all seeds < 0.012)
- doublewell mean KL = 0.004
- gaussmix mean KL = 0.003

| Potential | Seed 42 | Seed 137 | Seed 2024 | Mean |
|-----------|---------|----------|-----------|------|
| harmonic_1d tau | 13.0 | 11.6 | 11.8 | 12.1 |
| harmonic_1d KL | 0.012 | 0.007 | 0.004 | 0.008 |
| doublewell_2d tau | 220.3 | 134.9 | 198.2 | 184.5 |
| doublewell_2d KL | 0.002 | 0.006 | 0.004 | 0.004 |
| gaussmix_2d tau | 99.6 | 95.3 | 95.0 | 96.6 |
| gaussmix_2d KL | 0.003 | 0.003 | 0.001 | 0.003 |
| **weighted metric** | | | | **120.44** |

Wall time: 183s. Beats NHC M=3 baseline (132.1) by 8.8%.

What failed earlier:
- Bounded Pade (a=0.8, b=4.0, c=0.5, d=0.15): gaussmix tau was excellent (52-62) but doublewell seed 2024 got trapped (kl=0.28, tau=7269)
- Near-linear (a=1.5, b=1.0, c=0.1, d=0.0): gaussmix tau excellent (73-94) but 1D HO seed 2024 non-ergodic (kl=0.36)
- CMA-ES on short proxy: unreliable because 50k-step proxy cannot detect long-time non-ergodicity
