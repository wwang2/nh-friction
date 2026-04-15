---
issue: 5
parents: [002-pade-cmaes-refine]
eval_version: eval-v1
metric: null
---

# Composite Friction: Gaussian-Damped Cubic + Tanh Residual

## Hypothesis

The Padé rational form g(ξ) = ξ·(a + b·ξ²)/(1 + c·ξ²) couples core and tail behavior through shared parameters. A composite form decouples them:

g(ξ) = ξ·(a + b·ξ²)·exp(-e·ξ²) + f·tanh(c·ξ)

- First term: strong cubic mixing near ξ=0 (breaks KAM tori), Gaussian-damped for large |ξ|
- Second term: bounded tanh(c·ξ) provides smooth saturation at large |ξ|

This gives independent control over:
1. Core nonlinearity (a, b) — KAM tori breaking strength
2. Damping envelope (e) — how quickly cubic term fades
3. Tail behavior (f, c) — asymptotic friction strength

Parent orbit 002 achieved metric=84.14 with rational Padé. The gaussmix_2d (τ=73.8, 60% of metric) and doublewell_2d (τ=114.1, 40%) are the targets for improvement.

## Research Notes
