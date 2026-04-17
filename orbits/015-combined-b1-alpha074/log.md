---
issue: 16
parents: [orbit/012-joint-cmaes-alpha, orbit/014-nhc-true-v3]
eval_version: eval-v2
metric: 48.45
---

# Research Notes — orbit/015-combined-b1-alpha074

## Result

**METRIC = 48.45** (identical to orbit-014 baseline; combination hypothesis rejected).

The hypothesis that two independently discovered gaussmix improvements (b=1.0 from orbit-012, alpha=0.74 from orbit-014) would combine additively is **firmly rejected**. They interact destructively: together they produce tau_gm=169.5, far worse than either alone (57.7 or 59.2). An exhaustive parameter sweep confirms orbit-014's configuration as a local optimum.

## Why the Combination Fails

The two improvements both weaken the thermostat, but through different mechanisms:

**b=1.0 (orbit-012):** Reduces the cubic coefficient in g(xi) = xi*(a + b*xi^2)/(1 + c*xi^2). With b=3.0, friction grows as ~3*xi^3 for moderate xi, providing strong nonlinear damping. With b=1.0, friction grows only as ~xi^3, so the thermostat variable xi can wander further before friction pulls it back. This slower saturation was beneficial when alpha=1.0 (standard driving).

**alpha=0.74 (orbit-014):** The driving term h = alpha*|p|^2 - (alpha-1)*d*kT with alpha < 1 makes the effective thermostat mass Q_eff = Q/alpha = 1.35, meaning the thermostat responds more sluggishly to kinetic energy fluctuations. This was beneficial with b=3.0 (strong friction) because it allowed the particle to maintain momentum through inter-mode barriers.

**Combined failure mode:** With both weaker friction (b=1.0) AND weaker driving (alpha=0.74), the thermostat becomes doubly sluggish. The particle lacks the friction restoring force to regulate xi AND the driving force to push xi in response to energy fluctuations. The result is poor ergodicity: the chain gets trapped in single modes for long periods, producing tau_gm=169.5.

This is a textbook example of non-additivity in parameter optimization. Each change compensates for a different aspect of the default thermostat's behavior, and applying both compensations simultaneously overshoots the optimum.

## Parameter Sweep Summary

### Gaussmix (b, alpha) sweep (a=0.70, c=0.06 fixed)

| b | alpha | tau_gm | METRIC | Note |
|---|-------|--------|--------|------|
| 3.0 | 0.74 | 57.7 | 48.45 | **Best (orbit-014)** |
| 3.0 | 0.73 | 62.6 | 51.81 | |
| 3.0 | 0.75 | 63.0 | 52.12 | |
| 3.5 | 0.74 | 60.4 | 50.32 | |
| 2.5 | 0.74 | 67.4 | 55.09 | |
| 2.0 | 0.74 | 92.5 | 72.18 | |
| 1.5 | 0.74 | 124.4 | 93.94 | |
| 1.0 | 0.74 | 169.5 | 124.69 | **Combined (REJECTED)** |
| 1.0 | 1.00 | 59.2 | 49.54 | orbit-012 alone |
| 1.0 | 1.50 | 108.8 | 83.31 | |
| 4.0 | 0.74 | 73.9 | 59.51 | |
| 3.0 | 1.00 | 73.8 | 60.34 | Standard (no alpha) |

### Gaussmix (a, c) sweep (b=3.0, alpha=0.74 fixed)

| a | c | tau_gm | METRIC |
|---|---|--------|--------|
| 0.70 | 0.06 | 57.7 | 48.45 | **Best** |
| 0.50 | 0.06 | 71.2 | 57.68 |
| 1.00 | 0.06 | 62.6 | 51.81 |
| 0.90 | 0.06 | (not tested directly) | |
| 0.70 | 0.03 | 68.2 | 55.65 |
| 0.70 | 0.10 | 70.5 | 57.21 |

### Doublewell sweep (gaussmix fixed at b=3.0, alpha=0.74)

| a | b | alpha | tau_dw | METRIC |
|---|---|-------|--------|--------|
| 1.00 | 4.00 | 3.00 | 30.5 | 48.45 | **Best** |
| 0.70 | 4.00 | 3.00 | 32.0 | 48.89 |
| 0.50 | 4.00 | 3.00 | 32.9 | 49.15 |
| 1.50 | 4.00 | 3.00 | 39.6 | 51.14 |
| 1.00 | 3.00 | 3.00 | 35.7 | 49.99 |
| 1.00 | 5.00 | 3.00 | 36.2 | 50.13 |
| 1.00 | 6.00 | 3.00 | 40.2 | 51.32 |
| 1.00 | 4.00 | 2.00 | 51.4 | 54.60 |
| 1.00 | 4.00 | 4.00 | 49.1 | 53.93 |

### Final per-potential breakdown (best config)

| Potential | Seed 42 | Seed 137 | Seed 2024 | Mean |
|-----------|---------|----------|-----------|------|
| harmonic_1d | 6.72 | 7.01 | 8.13 | 7.29 |
| doublewell_2d | 29.17 | 28.73 | 33.55 | 30.48 |
| gaussmix_2d | 64.53 | 60.59 | 47.86 | 57.66 |

**Weighted METRIC:** 0.024 * 7.29 + 0.294 * 30.48 + 0.682 * 57.66 = 0.17 + 8.96 + 39.32 = 48.45

## Prior Art and Novelty

### What is already known
- Effective-Q scaling via driving function (Martyna et al. 1992)
- Pade friction optimization for Nose-Hoover (orbits 003/004)
- Alpha < 1 optimal for multi-modal gaussmix (orbit-014)
- b=1.0 optimal for gaussmix with standard driving (orbit-012)

### What this orbit adds
- **Negative result:** demonstrates that independently optimized parameters interact destructively when combined
- Physical explanation: both changes weaken thermostat coupling through different mechanisms, and the combined effect overshoots the optimal coupling strength
- Exhaustive sweep confirms orbit-014's (b=3.0, alpha=0.74) as a local optimum for gaussmix
- Exhaustive sweep confirms orbit-012's doublewell params (a=1.0, b=4.0, alpha=3.0) as locally optimal

### Honest positioning
This orbit produced no improvement over orbit-014. Its value is purely as a negative result: the natural hypothesis that two orthogonal-looking improvements would compose was tested rigorously and rejected. The final solution is identical to orbit-014's. Future optimization efforts should look for fundamentally different approaches rather than combining existing parameter tweaks.

## References

- Martyna, G.J., Klein, M.L., Tuckerman, M. (1992). J. Chem. Phys. 97, 2635.
- Orbit-012: CMA-ES joint optimization, discovered b=1.0 for gaussmix
- Orbit-014: Alpha sweep, discovered alpha=0.74 for gaussmix

## Glossary

- **NH**: Nose-Hoover — deterministic thermostat maintaining canonical temperature
- **KL**: Kullback-Leibler divergence — measures deviation from target distribution
- **tau_int**: Integrated autocorrelation time — measures effective sample independence
- **Pade**: Rational function form for friction: g(xi) = xi*(a+b*xi^2)/(1+c*xi^2)
- **Effective-Q**: Virtual thermostat mass Q_eff = Q/alpha from driving function scaling
- **VVEF**: Velocity Verlet with Exact exponential Friction — the integrator scheme
