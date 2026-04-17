---
issue: 17
parents: [orbit/015-combined-b1-alpha074]
eval_version: eval-v3
metric: null
---

# CMA-ES Parameter Search for Gaussmix Friction

## Motivation

The gaussmix potential carries 68.2% of the total metric weight, making it the dominant optimization target. Prior orbits found two local optima for gaussmix parameters:

- (a=0.70, b=3.00, c=0.06, alpha=0.74): tau_gm ~ 57.66 [orbit-014]
- (a=0.70, b=1.00, c=0.06, alpha=1.00): tau_gm ~ 59.25 [orbit-012]

Crucially, combining the best features of both (b=1.0 + alpha=0.74) was destructive (tau_gm=169.5), indicating these parameters interact nonlinearly. This suggests the loss landscape has multiple basins separated by ridges.

## Approach

Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to search the 4D parameter space (a, b, c, alpha) from multiple starting points. CMA-ES is well-suited for:
- Low-dimensional continuous optimization (4D here)
- Non-convex landscapes with multiple local optima
- No gradient information needed

### Starting Points
- Start A: (a=0.70, b=3.00, c=0.06, alpha=0.74) — known local optimum from orbit-014
- Start B: (a=0.70, b=1.00, c=0.06, alpha=1.50) — compensated: lower b, higher alpha
- Start C: (a=1.00, b=2.00, c=0.03, alpha=0.90) — different shape, unexplored region

### Parameter Bounds
- a in [0.1, 2.0]
- b in [0.1, 8.0]
- c in [0.01, 0.5]
- alpha in [0.3, 2.0]

## Results

(To be filled after CMA-ES runs)
