---
issue: 2
parents: []
eval_version: eval-v1
metric: null
---

# Research Notes — orbit/001-pade-cmaes

## Hypothesis
Rational Padé family g(ξ) = ξ·(a + b·ξ²) / (1 + c·ξ² + d·ξ⁴) with CMA-ES parameter optimization in setup().

## Approach
- Implement the Padé parametric family
- In setup(seed): run a CHEAP proxy evaluation (shorter integrator, e.g., 10^4 steps on 1D HO + one 2D potential) to identify good (a, b, c, d)
- Use CMA-ES (pycma) over the 4D parameter space
- Return best parameters; friction_function uses them
