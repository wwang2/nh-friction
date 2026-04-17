---
issue: 13
parents: [010-potential-adaptive]
eval_version: eval-v2
metric: null
---

# Joint CMA-ES Optimization of Pade Friction Shape and Effective-Q Scaling

## Motivation

Orbit-010 achieved metric=60.34 using potential-adaptive effective-Q driving with fixed Pade friction parameters (a=0.7, b=3.0, c=0.06). The per-potential alpha values were: harmonic alpha=2.0 (tau=7.29), doublewell alpha=3.0 (tau=33.48), gaussmix alpha=1.0 (tau=73.81).

Two observations motivate this orbit:

1. **The Pade parameters were optimized at alpha=1 (standard Nose-Hoover).** When alpha != 1, the effective thermostat mass becomes Q_eff = Q/alpha, changing the characteristic timescale of xi oscillations. The optimal friction shape g(xi) should depend on the alpha value -- higher alpha means xi explores a wider range faster, so the nonlinear saturation controlled by (b, c) may need adjustment.

2. **Gaussmix uses alpha=1.0 and dominates the metric (68.2% weight).** The inter-mode barrier crossing in the 5-mode Gaussian mixture is the bottleneck. There may exist an (a, b, c, alpha) tuple for gaussmix that reduces tau below 73.81 while keeping KL < 0.05.

## Approach

Joint 4-parameter CMA-ES optimization over (a, b, c, alpha) per potential:
- a in (0.01, 3], b in (0.01, 10], c in (0.001, 1], alpha in [1.0, 5.0]
- Pade friction: g(xi) = xi * (a + b*xi^2) / (1 + c*xi^2)
- Driving: h = alpha * |p|^2 - (alpha-1) * d*kT
- Inner loop: 200k steps, 1-2 seeds (fast proxy for production 1M steps)

Strategy:
1. Grid search alpha with fixed (a,b,c) = (0.7, 3.0, 0.06) to find promising alpha ranges
2. CMA-ES joint optimization starting from best grid point
3. Assemble per-potential best parameters into full solution.py
4. Validate with production evaluator (1M steps, 3 seeds)

## Theoretical Background

The effective-Q driving h = alpha*K - (alpha-1)*d*kT preserves the canonical (q,p) marginal because:
- E_canonical[h] = alpha*d*kT - (alpha-1)*d*kT = d*kT (zero-mean condition satisfied)
- dxi/dt = (h - d*kT)/Q = alpha*(K - d*kT)/Q, equivalent to Q_eff = Q/alpha
- The xi-marginal changes shape but the physical (q,p) distribution remains exp(-H/kT)

The friction function g(xi) controls how xi oscillations translate into momentum damping. For larger alpha (more aggressive thermostatting), xi may reach larger values more frequently, making the nonlinear behavior of g at large xi more important.

## Glossary

- **Pade friction**: g(xi) = xi*(a + b*xi^2)/(1 + c*xi^2), a rational function form that is odd by construction
- **Effective-Q**: Q_eff = Q/alpha, the effective thermostat mass when using alpha-scaled driving
- **CMA-ES**: Covariance Matrix Adaptation Evolution Strategy, a derivative-free optimizer
- **KL gate**: KL(empirical || analytical) < 0.05, the canonical measure preservation check
- **tau_int**: Integrated autocorrelation time, lower is better

## Results

(Pending optimization run...)
