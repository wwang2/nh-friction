---
issue: 4
parents: [002-pade-cmaes-refine]
eval_version: eval-v1
metric: null
---

# CMA-ES on All 3 Benchmark Potentials

## Hypothesis

Parent orbit 002 found g(ξ) = ξ·(0.7 + 3.0·ξ²) / (1 + 0.06·ξ²) → metric=84.14, but the parameters were hand-tuned. Orbit 001's CMA-ES only optimized on the 1D harmonic oscillator proxy, which carries only 2.4% of the metric weight. The dominant contributors — gaussmix_2d (68%) and doublewell_2d (29%) — were never used as optimization targets.

This orbit runs CMA-ES in setup() using all 3 potentials as proxy targets with difficulty weights matching the eval. Starting from the parent's best (a=0.7, b=3.0, c=0.06, d=0.0) as the CMA-ES initial guess.

## Research Notes
