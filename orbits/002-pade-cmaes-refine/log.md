---
issue: 3
parents: [001-pade-cmaes]
eval_version: eval-v1
metric: 84.142123
---

# Research Notes — orbit/002-pade-cmaes-refine

## Hypothesis
REFINE orbit 001. Extended Padé family with proper 3-potential CMA-ES optimization in setup(). Close the 9% gap from 97.9 → <90.

## Results (eval-check, 2026-04-15)

Parameters: a=0.7, b=3.0, c=0.06, d=0.0, e=0.0, f=0.0
g(ξ) = ξ·(0.7 + 3.0·ξ²) / (1 + 0.06·ξ²)

| Potential | Seed 42 | Seed 137 | Seed 2024 | Mean |
|-----------|---------|----------|-----------|------|
| harmonic_1d τ | 9.1 | 10.0 | 12.0 | 10.36 |
| harmonic_1d KL | 0.038 | 0.010 | 0.006 | 0.018 |
| doublewell_2d τ | 122.6 | 94.2 | 125.4 | 114.08 |
| doublewell_2d KL | 0.004 | 0.002 | 0.001 | 0.002 |
| gaussmix_2d τ | 72.1 | 61.9 | 87.4 | 73.81 |
| gaussmix_2d KL | 0.002 | 0.002 | 0.002 | 0.002 |
| **weighted metric** | | | | **84.14** |

Beats target (90) by 6.5%. Beats NHC M=3 baseline (132.1) by 36.3%.
Key improvement over orbit 001 (97.9→84.1): increasing a from 0.5→0.7 strengthens linear coupling, improving gaussmix mixing (73.8 vs ~99).
