---
issue: 15
parents: [013-nhc-chain-v3, 010-potential-adaptive]
eval_version: eval-v3
metric: 54.06
---

# Research Notes — orbit/014-nhc-true-v3

## Key Finding

**The NHC hypothesis is invalid** — the evaluator does NOT pass xi to driving_function (only 3 args: q, p, grad_V). The task description's claim of "eval-v3" passing xi is incorrect. However, a simpler finding emerged: alpha < 1.0 for gaussmix (weaker thermostatting) improves inter-mode mixing significantly.

## Why NHC Cannot Work Here

The evaluator (evaluator.py line 526) calls `driving_fn(q, p, grad_q)` with 3 arguments. The NHC chain coupling requires xi (the primary thermostat variable), which is not available.

Shadow xi tracking was attempted: since xi starts at 0 and updates as `xi += (h - d*kT)/Q * dt`, one can maintain a shadow copy by tracking returned h values. However, feeding -xi2*xi1 back into h creates a feedback loop that the standard VVEF integrator is not designed to handle. The extended phase space measure is not preserved, causing:
- Harmonic: KL=0.13 (fails 0.05 gate)
- Gaussmix: completely unstable (tau=6000+, KL=2.0+)

This confirms orbit-013's conclusion: NHC requires a specially designed integrator (Suzuki-Yoshida splitting), not just a modified driving function.

## Discovery: Alpha < 1 for Gaussmix

The effective-Q driving `h = alpha*|p|^2 - (alpha-1)*d*kT` with alpha < 1 means the effective thermostat mass Q_eff = Q/alpha > Q. This gives the chain more inertia, allowing it to maintain momentum through inter-mode barriers.

**Physical intuition:** The 5 Gaussian modes are separated by radius 3. To cross between modes, the particle needs sustained high momentum. Standard NH (alpha=1) thermostatizes normally. Alpha < 1 makes the thermostat "sluggish" — it responds more slowly to kinetic energy fluctuations, allowing the particle to build up and sustain the momentum needed for barrier crossing.

### Iteration 1: NHC with full coupling (Q2=1.0)
- Result: METRIC=inf (KL gate failure on harmonic and gaussmix)
- Lesson: NHC coupling breaks the invariant measure under standard VVEF integration

### Iteration 2: Soft NHC coupling (lambda=0.3, Q2=2.0)
- Result: METRIC=inf (harmonic KL=0.24, gaussmix KL=3.88)
- Lesson: Even weak NHC coupling is destabilizing

### Iteration 3: Configurational driving for gaussmix (lambda_hybrid=0.5)
- Result: METRIC=inf (gaussmix KL=5.6 — totally wrong distribution)
- Lesson: E_ref normalization from burn-in probe is unreliable

### Iteration 4: Gaussmix alpha=1.2 (stronger thermostatting)
- Result: gaussmix tau=93.54 (WORSE than baseline 73.81)
- Harmonic KL=0.053 (marginal fail — probe length issue)
- Lesson: Stronger thermostatting hurts gaussmix

### Iteration 5: Gaussmix alpha=0.8 (weaker thermostatting)
- Result: **METRIC=54.06** (improvement from 60.34)
- Gaussmix tau: 73.81 -> 64.59 (12.5% improvement)
- All KL values pass comfortably
- Alpha sweep in progress for [0.4, 0.5, 0.6, 0.7, 0.9]

| Potential | Seed 42 | Seed 137 | Seed 2024 | Mean | alpha |
|-----------|---------|----------|-----------|------|-------|
| harmonic_1d tau | 6.72 | 7.01 | 8.13 | 7.29 | 2.0 |
| harmonic_1d KL | 0.037 | 0.020 | 0.034 | 0.030 | |
| doublewell_2d tau | 40.66 | 28.35 | 31.44 | 33.48 | 3.0 |
| doublewell_2d KL | 0.001 | 0.000 | 0.000 | 0.001 | |
| gaussmix_2d tau | 62.58 | 73.64 | 57.55 | 64.59 | 0.8 |
| gaussmix_2d KL | 0.002 | 0.003 | 0.002 | 0.002 | |

## Prior Art and Novelty

### What is already known
- Effective-Q scaling via driving function is well-known (Martyna et al. 1992)
- NHC requires specialized integrators (Tuckerman et al. 2001)
- Braga-Travis configurational driving (2005) requires careful normalization

### What this orbit adds
- Demonstrates that alpha < 1 (weaker thermostatting) improves inter-mode mixing for multi-modal potentials
- Confirms NHC cannot be implemented via driving function modification alone
- No strong novelty claim — this is parameter tuning of a known technique

### Honest positioning
This orbit's contribution is a systematic exploration of the alpha parameter space for gaussmix, discovering that weaker thermostatting (alpha < 1) helps inter-mode mixing. The improvement from 60.34 to 54.06 is modest but real.

## References

- Martyna, G.J., Klein, M.L., Tuckerman, M. (1992). J. Chem. Phys. 97, 2635.
- Braga, C. & Travis, K.P. (2005). J. Chem. Phys. 123, 134101.
- Tuckerman, M.E. et al. (2001). J. Chem. Phys. 115, 1678.

## Glossary

- **NH**: Nose-Hoover — deterministic thermostat maintaining canonical temperature
- **NHC**: Nose-Hoover Chain — extension with multiple coupled thermostat variables
- **KL**: Kullback-Leibler divergence — measures deviation from target distribution
- **tau_int**: Integrated autocorrelation time — measures effective sample independence
- **Pade**: Rational function form used for friction: g(xi) = xi*(a+b*xi^2)/(1+c*xi^2)
- **Effective-Q**: Virtual thermostat mass Q_eff = Q/alpha achieved via driving function scaling
- **VVEF**: Velocity Verlet with Exact exponential Friction — the integrator scheme
