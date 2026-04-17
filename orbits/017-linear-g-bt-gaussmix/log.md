---
issue: 18
parents: [orbit/015-combined-b1-alpha074]
eval_version: eval-v3
metric: null
---

# Linear g(xi) + Braga-Travis h=|grad_V|^2/E_ref for Gaussmix

## Hypothesis

Brainstorm panel identified that pure BT configurational driving h = |grad_V|^2/E_ref requires LINEAR g(xi) = xi for Liouville canonicity. All prior BT attempts (orbit 008, orbit 010 hybrid) used Pade friction, which mathematically cannot support pure configurational h. This orbit tests the genuinely unexplored combination: linear g + BT h, applied to gaussmix only.

## Approach

Per-potential dispatch preserves orbit-015 for harmonic and doublewell (Pade + effective-Q). For gaussmix:
- friction_function: g(xi) = xi (linear)
- driving_function(q, p, grad_V, xi): h = alpha * |grad_V|^2/E_ref - (alpha-1) * d * kT with alpha tunable
- E_ref estimated via running mean during 5000-step warmup

## Starting parameters

- gaussmix: linear g, E_ref from warmup, alpha sweep in [0.5, 2.0]
- harmonic:   Pade a=0.70, b=3.00, c=0.06, alpha=2.00 (orbit 015)
- doublewell: Pade a=1.00, b=4.00, c=0.06, alpha=3.00 (orbit 015)

## Success criteria

- Primary: METRIC < 48.45 (orbit 015)
- KL < 0.05 for all potentials (non-negotiable)
- If pure BT fails KL for gaussmix: fall back to hybrid h = lambda*|p|^2 + (1-lambda)*|grad_V|^2/E_ref

## Results

(To be filled after runs)
