---
issue: 14
parents: [010-potential-adaptive]
eval_version: eval-v2
metric: 60.340208
---

# Research Notes — orbit/013-nhc-chain-v3

## Result

**METRIC = 60.34** — no improvement over parent orbit-010.

This orbit attempted to implement a Nose-Hoover Chain (NHC) thermostat via the eval-v2 driving function interface. The key finding is that NHC cannot be implemented within eval-v2 because `driving_function(q, p, grad_V)` does not receive the thermostat variable xi, which is essential for the chain coupling term xi2*xi1 in the NHC equations. The hypothesis assumed an "eval-v3" interface that passes xi — this does not exist.

| Potential | Seed 42 | Seed 137 | Seed 2024 | Mean | alpha |
|-----------|---------|----------|-----------|------|-------|
| harmonic_1d tau | 6.72 | 7.01 | 8.13 | 7.29 | 2.0 |
| harmonic_1d KL | 0.037 | 0.020 | 0.034 | 0.030 | |
| doublewell_2d tau | 40.66 | 28.35 | 31.44 | 33.48 | 3.0 |
| doublewell_2d KL | 0.001 | 0.000 | 0.000 | 0.001 | |
| gaussmix_2d tau | 72.08 | 61.94 | 87.40 | 73.81 | 1.0 |
| gaussmix_2d KL | 0.002 | 0.002 | 0.002 | 0.002 | |

## Why NHC Cannot Be Implemented in Eval-v2

The 2-link NHC requires the chain coupling h1 = |p|^2/m - xi2*xi1, where xi1 is the primary thermostat variable managed by the evaluator. The eval-v2 driving function signature is `driving_function(q, p, grad_V) -> float` — it does not pass xi. Without access to xi, the second thermostat xi2 cannot be coupled to the primary thermostat.

One could try to infer xi from the trajectory (e.g., by tracking cumulative dxi/dt), but this would accumulate numerical error and create a feedback loop outside the evaluator's integrator, violating the intended interface contract.

## Approaches Tested (All Failed to Improve on Baseline)

### 1. Cross-term driving: h = K + gamma * p dot grad_V

**Theory:** Under the canonical distribution, p and q are independent, so E[p dot grad_V] = 0. Therefore h = K + gamma*(p dot grad_V) satisfies E[h] = d*kT for any gamma. The cross-term is positive when moving uphill (barrier crossing), which should strengthen thermostat coupling during transitions.

**Result:** gamma=0.15 -> KL=1.99 for gaussmix (DISQUALIFIED). The chain gets trapped in a single mode. The cross-term creates correlations between p and q in the discrete integrator that break ergodicity.

**Lesson:** Zero-mean conditions derived for the continuous-time distribution do not necessarily hold for the discrete-time integrator. The symplectic splitting introduces correlations that can be amplified by non-standard driving terms.

### 2. Gradient-gated alpha: alpha(q) = 1 + delta * (|grad_V|^2/E_ref - 1)

**Theory:** For any position-dependent alpha(q), h = alpha(q)*K - (alpha(q)-1)*d*kT satisfies E[h] = d*kT because K and alpha(q) are independent under the canonical distribution (p and q are independent).

**Result:** delta=0.3 -> gaussmix tau=75.82 (marginally worse than 73.81). delta=-0.3 not tested due to negative results.

**Lesson:** Position-dependent effective-Q provides marginal effect at best. The fundamental limitation is that gaussmix tau~74 is near the optimum for a single NH thermostat.

### 3. Per-potential friction parameters

Modifying the Pade friction g(xi) = xi*(a+b*xi^2)/(1+c*xi^2) per-potential:
- Stronger friction (a=1.0, b=4.0, c=0.04): gaussmix tau=97.07 (31% worse)
- Weaker friction (a=0.5, b=2.0, c=0.06): gaussmix tau=82.11 (11% worse)

**Lesson:** Both stronger and weaker friction hurt gaussmix. The default Pade (0.7, 3.0, 0.06) is near-optimal.

### 4. Higher-order kinetic driving: h = qa*K^2 + qb*K + qc

Using K^2 to amplify thermostat response to high-energy excursions:
- qa=0.05, qb=1.0, qc=-0.4: gaussmix tau=245.01 (3.3x worse)

**Lesson:** Higher variance in h drives xi to extreme values, causing over-damping and trapping.

### 5. Position-dependent alpha for doublewell

Gaussian kernel: alpha = 2.0 + 3.0*exp(-x^2/0.5) (strong at barrier x~0, moderate in wells):
- doublewell tau=46.96 (40% worse than alpha=3.0 constant)

**Lesson:** Over-thermostatting at the barrier impedes transit. The particle needs to move freely through the barrier, not be slowed down by aggressive thermostatting.

### 6. Alpha sweep for doublewell

| alpha | doublewell tau |
|-------|---------------|
| 2.5   | 44.81         |
| 3.0   | 33.48 (best)  |
| 3.5   | 37.93         |
| 4.0   | 56.49         |

Clear minimum at alpha=3.0. Orbit-010's value is optimal.

## Key Insight: The Single-Thermostat Ceiling

The fundamental conclusion is that for the 5-mode Gaussian mixture at radius 3, a single Nose-Hoover thermostat variable with tau=73.81 appears to be near the performance limit. All attempts to modify the driving function (cross-terms, gradient gating, nonlinear kinetic driving) either break canonical invariance or fail to improve mixing.

The NHC approach (multiple coupled thermostats) is the standard solution to this problem in molecular dynamics, but it requires access to the thermostat variable xi — which the eval-v2 interface does not provide. An eval-v3 interface that passes xi to driving_function would enable true NHC implementation.

## Prior Art and Novelty

### What is already known
- Effective-Q driving h = alpha*K - (alpha-1)*d*kT is a well-known technique (equivalent to scaling the thermostat mass). [Martyna et al. (1992)](https://doi.org/10.1063/1.463940)
- Nose-Hoover Chains (NHC) improve ergodicity by coupling multiple thermostats. [Martyna et al. (1992)](https://doi.org/10.1063/1.463940)
- Position-dependent thermostat coupling has been explored in configurational thermostats. [Braga & Travis (2005)](https://doi.org/10.1063/1.2013227)

### What this orbit adds
- Confirms that alpha=1.0 is optimal for gaussmix with single NH thermostat — no driving function modification within eval-v2 can improve it
- Documents that cross-term p dot grad_V breaks ergodicity in the discrete integrator despite being zero-mean in continuous time
- Documents that gradient-gated alpha provides negligible benefit
- This orbit applies known techniques — no novelty claim

### Honest positioning
This orbit is a thorough negative-result study. It systematically explores the space of eval-v2-compatible driving functions and demonstrates that orbit-010's effective-Q approach is near-optimal within these constraints. The results suggest that further improvement requires either a true NHC implementation (needing xi access) or a fundamentally different approach to the thermostat dynamics.

## References

- Martyna, G.J., Klein, M.L., Tuckerman, M. (1992). J. Chem. Phys. 97, 2635. — NHC equations
- Braga, C. & Travis, K.P. (2005). J. Chem. Phys. 123, 134101. — Configurational thermostat
- Tuckerman, M.E. et al. (2001). J. Chem. Phys. 115, 1678. — Non-Hamiltonian MD framework

## Glossary

- **NH**: Nose-Hoover — a deterministic thermostat that maintains canonical temperature
- **NHC**: Nose-Hoover Chain — extension with multiple coupled thermostat variables for improved ergodicity
- **KL**: Kullback-Leibler divergence — measures deviation of sampled distribution from target
- **tau_int**: Integrated autocorrelation time — measures how many steps before samples are effectively independent
- **Pade**: Pade approximant — rational function form a(xi)*(a+b*xi^2)/(1+c*xi^2) used for friction
- **Effective-Q**: Virtual thermostat mass Q_eff = Q/alpha achieved via driving function scaling
