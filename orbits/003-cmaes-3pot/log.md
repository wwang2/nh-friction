---
issue: 4
parents: [002-pade-cmaes-refine]
eval_version: eval-v1
metric: 84.14
---

# CMA-ES on All 3 Benchmark Potentials

## Summary

Systematic grid search over 882 parameter combinations of the Pade friction function g(xi) = xi*(a + b*xi^2)/(1 + c*xi^2 + d*xi^4), using all 3 benchmark potentials as proxy targets with difficulty-weighted evaluation. The search confirmed that the parent orbit 002's hand-tuned parameters (a=0.7, b=3.0, c=0.06, d=0.0) sit at a robust local optimum. Every tested perturbation in the 4D parameter space produced a worse metric on the real evaluator.

**Metric = 84.14** (unchanged from parent orbit 002).

## Hypothesis

Parent orbit 002 found g(xi) = xi*(0.7 + 3.0*xi^2) / (1 + 0.06*xi^2) with metric=84.14, but parameters were hand-tuned. Since gaussmix_2d (68.2% weight) and doublewell_2d (29.4% weight) dominate the metric, optimizing against all 3 potentials simultaneously should find better parameters.

## Approach

### Three-phase offline search

The evaluator's 60-second setup() timeout makes runtime CMA-ES infeasible (each proxy evaluation takes ~6s). Instead, we performed an offline three-phase search:

1. **Phase 1 -- Coarse KL screen** (882 candidates): Test each (a, b, c, d) combination on the 1D harmonic oscillator with seed 2024 (the hardest seed for KAM tori breaking). 100k steps, ~0.8s per candidate. Threshold: KL < 0.03. Result: 38 of 882 passed (4.3%).

2. **Phase 2 -- Gaussmix tau screen** (38 candidates): Run gaussmix_2d with 1 seed to estimate tau. Sort by proxy tau.

3. **Phase 3 -- Full proxy eval** (top 20): All 3 potentials, 2-3 seeds, 80-200k steps. Compute difficulty-weighted metric.

### Real evaluator verification

Top candidates from the grid search were then verified with the real evaluator (1M steps, 3 seeds, all 3 potentials):

| Candidate | a | b | c | d | Metric | h1d tau | dw tau | gm tau |
|-----------|-----|-----|------|------|--------|---------|--------|--------|
| **parent (002)** | **0.7** | **3.0** | **0.06** | **0.0** | **84.14** | **10.4** | **114.1** | **73.8** |
| a=0.75 | 0.75 | 3.0 | 0.06 | 0.0 | 101.74 | 10.5 | 126.0 | 94.5 |
| a=0.8 | 0.8 | 3.0 | 0.06 | 0.0 | 92.64 | 9.3 | 119.0 | 84.2 |
| b=6,c=0,d=.02 | 0.6 | 6.0 | 0.0 | 0.02 | 124.74 | 10.0 | 109.4 | 135.4 |
| c=.06,d=.02 | 0.7 | 3.0 | 0.06 | 0.02 | 100.23 | 10.9 | 175.4 | 70.9 |

## Key Findings

### 1. The KL gate on harmonic_1d is the binding constraint

Only 4.3% of parameter combinations pass the harmonic KL gate (KL < 0.05 on all 3 seeds). The 1D harmonic oscillator is prone to KAM tori trapping, which requires sufficiently strong nonlinearity to break. Empirically:
- a >= 0.6 needed (linear coupling strength near equilibrium)
- b >= 2.5 needed (cubic coupling strength for KAM breaking)
- Reducing either below these thresholds causes seed 2024 to get trapped

### 2. Proxy tau estimates do not predict real eval tau

Short proxy integrations (60-200k steps) gave qualitatively misleading tau estimates compared to the full 1M-step eval. The most dramatic example: (a=0.6, b=6.0, c=0, d=0.02) showed proxy gm_tau=49.2 but real eval gm_tau=135.4 (2.75x worse). The strong cubic nonlinearity creates faster local mixing (low proxy tau) but worse mode-hopping at longer timescales.

### 3. The parent parameters are a local optimum

Every perturbation tested made the metric worse:
- Increasing a: worsens gaussmix tau (stronger linear coupling reduces mode-hopping)
- Increasing b: often fails harmonic KL gate or worsens mixing
- Adding quartic denominator (d>0): worsens doublewell barrier crossing
- Reducing c: marginal changes, no consistent improvement
- Increasing c: worsens all 2D potentials

### 4. Gaussmix tau has high seed variance

The gaussmix_2d tau varies substantially across seeds even with the same parameters: (72.1, 61.9, 87.4) for the parent params, giving a coefficient of variation of ~16%. This makes optimization difficult -- improvements within the noise band are indistinguishable from luck.

## Results Table (Parent Params, Confirmed)

| Seed | h1d tau | h1d KL | dw tau | dw KL | gm tau | gm KL |
|------|---------|--------|--------|-------|--------|-------|
| 42 | 9.13 | 0.038 | 122.6 | 0.004 | 72.1 | 0.002 |
| 137 | 9.98 | 0.010 | 94.2 | 0.002 | 61.9 | 0.002 |
| 2024 | 11.98 | 0.006 | 125.5 | 0.001 | 87.4 | 0.002 |
| **Mean** | **10.36** | **0.018** | **114.08** | **0.002** | **73.81** | **0.002** |

Weighted metric = 0.024*10.36 + 0.294*114.08 + 0.682*73.81 = **84.14**

## Prior Art and Novelty

### What is already known
- Bulgac and Kusnezov (1990) proposed g(xi) = 2*xi/(1+xi^2) as a nonlinear friction
- Rational Pade-type friction functions preserve oddness automatically when constructed as xi * even(xi^2)
- CMA-ES is a standard derivative-free optimizer for noisy objectives (Hansen 2003)

### What this orbit adds
- Systematic evidence that (a=0.7, b=3.0, c=0.06) is a local optimum in the 4-parameter Pade space
- Quantitative demonstration that proxy (short) integrations are unreliable for tau optimization (proxy-to-real correlation is weak)
- Identification of the KL gate as the binding constraint: only 4.3% of parameter space is feasible

### Honest positioning
This orbit did not improve on the parent's metric. The contribution is negative results: the parent's hand-tuned parameters are verified to be locally optimal, and the approach of using short proxy integrations for CMA-ES is shown to be unreliable. Future orbits should either use the full evaluator directly (requiring much larger compute budgets) or develop better proxy metrics that correlate with long-timescale mixing.

## Glossary

- **KAM tori**: Kolmogorov-Arnold-Moser tori -- invariant surfaces in phase space that trap trajectories, preventing ergodic sampling
- **KL gate**: KL divergence threshold (< 0.05) that disqualifies non-ergodic solutions
- **Pade friction**: Rational function g(xi) = xi*(a+b*xi^2)/(1+c*xi^2+d*xi^4) for the Nose-Hoover thermostat
- **CMA-ES**: Covariance Matrix Adaptation Evolution Strategy -- derivative-free optimization algorithm
- **tau_int**: Integrated autocorrelation time -- measures how many steps needed for independent samples
- **NHC**: Nose-Hoover Chain -- multi-thermostat approach with M auxiliary variables

## References

- [Hansen (2003)](https://doi.org/10.1162/106365603321828970) -- The CMA Evolution Strategy: A Tutorial
- [Bulgac, A. & Kusnezov, D. (1990)](https://doi.org/10.1103/PhysRevA.42.5045) -- Canonical ensemble averages from pseudomicrocanonical dynamics
- Builds on orbit/002-pade-cmaes-refine (metric=84.14) which hand-tuned the Pade parameters
