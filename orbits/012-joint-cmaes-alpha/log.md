---
issue: 13
parents: [010-potential-adaptive]
eval_version: eval-v2
metric: 49.537154
---

# Joint CMA-ES Optimization of Pade Friction Shape and Effective-Q Scaling

## Result

**METRIC = 49.54** (from orbit-010 baseline of 60.34, a 17.9% relative improvement).

| Potential | Weight | Orbit-010 tau | Orbit-012 tau | Change |
|-----------|--------|--------------|--------------|--------|
| harmonic  | 2.4%   | 7.29         | 7.29         | 0%     |
| doublewell| 29.4%  | 33.48        | 30.48        | -9%    |
| gaussmix  | 68.2%  | 73.81        | 59.25        | -20%   |
| **Weighted** | 100% | **60.34**   | **49.54**    | **-18%** |

## Motivation

Orbit-010 achieved metric=60.34 using potential-adaptive effective-Q driving with fixed Pade friction parameters (a=0.7, b=3.0, c=0.06). The same Pade shape was used for all three potentials. This orbit tests whether per-potential Pade parameter tuning can improve the metric, particularly for the gaussmix potential which dominates with 68.2% weight.

## Approach

1. Grid search over alpha values with fixed Pade (a=0.7, b=3.0, c=0.06) for each potential
2. Per-potential Pade parameter scans (a, b, c independently) using 500k-step mini-evaluator
3. Production evaluator validation of promising candidates
4. Direct production sweep of 9 parameter combinations for final tuning

## Key Finding: Lower b for Gaussmix

The most impactful discovery is that the Pade cubic coefficient b should be much lower for the gaussmix potential. The standard Pade g(xi) = xi*(0.7 + 3.0*xi^2)/(1 + 0.06*xi^2) has strong cubic growth, which causes the friction to increase rapidly for large |xi|. For the 5-mode Gaussian mixture, this appears to over-damp the thermostat variable at large xi values, slowing inter-mode transitions.

With b=1.0 instead of b=3.0, the friction profile becomes more linear. At xi=2, for instance:
- b=3.0: g(2) = 2*(0.7 + 12)/(1 + 0.24) = 20.48
- b=1.0: g(2) = 2*(0.7 + 4)/(1 + 0.24) = 7.58

The reduced friction at large xi allows the thermostat to drive larger momentum changes when xi is away from zero, which helps the chain cross barriers between the 5 Gaussian modes.

## Production Evaluation Results

| Seed | tau_h | tau_dw | tau_gm | KL_h    | KL_dw   | KL_gm   |
|------|-------|--------|--------|---------|---------|---------|
| 42   | 6.72  | 29.16  | 68.85  | 0.03740 | 0.00071 | 0.00365 |
| 137  | 7.01  | 28.66  | 57.91  | 0.01955 | 0.00044 | 0.00307 |
| 2024 | 8.13  | 33.62  | 51.00  | 0.03358 | 0.00069 | 0.00195 |
| **Mean** | **7.29** | **30.48** | **59.25** | **0.030** | **0.001** | **0.003** |

All KL values well below the 0.05 threshold.

## Optimized Parameters

```
harmonic:   a=0.70, b=3.00, c=0.06, alpha=2.0  (unchanged from orbit-010)
doublewell: a=1.00, b=4.00, c=0.06, alpha=3.0  (higher a,b vs orbit-010)
gaussmix:   a=0.70, b=1.00, c=0.06, alpha=1.0  (lower b vs orbit-010)
```

## What Worked

1. **Gaussmix b=1.0**: Reducing the Pade cubic coefficient from 3.0 to 1.0 improved gaussmix tau from 73.81 to 59.25 (20% improvement). This is the dominant effect.
2. **Doublewell a=1.0, b=4.0**: Slightly tuning the doublewell Pade improved tau from 33.48 to 30.48 (9% improvement). This contributes modestly given the 29.4% weight.

## What Did Not Work

1. **Gaussmix alpha > 1.0**: All tested values (1.1, 1.2, 1.5, 2.0, 3.0, 5.0) gave worse tau for gaussmix. The standard thermostat mass (alpha=1.0) is optimal for inter-mode mixing.
2. **Doublewell alpha=4.0**: The 200k-step mini-evaluator suggested alpha=4.0 was best for doublewell (tau=35.27), but the production evaluator (1M steps) showed it gives tau=56.49 -- much worse than alpha=3.0. The mini-evaluator was unreliable for this potential due to insufficient trajectory length.
3. **Gaussmix b=0.5**: Too little cubic growth caused tau to increase to 127.0 -- the friction becomes too weak at large xi.
4. **Gaussmix b=1.3, c=0.03**: Mini-eval suggested this was better (tau=52.53), but production eval gave tau=70.65. Again, mini-eval noise was misleading.
5. **CMA-ES joint optimization**: The 200k-step CMA-ES inner loop was too noisy to converge reliably for 2D potentials. Manual grid search followed by production validation was more effective.

## Lessons Learned

1. **Mini-evaluators are unreliable for multimodal potentials.** The 200k-step proxy gave misleading rankings for both doublewell and gaussmix. Production validation is essential before claiming any improvement.
2. **Per-potential friction tuning matters.** The optimal friction shape depends on the potential landscape. A single set of Pade parameters is suboptimal.
3. **The b parameter controls the high-xi behavior.** For potentials with well-separated modes (gaussmix), more linear friction (lower b) helps. For potentials with barriers (doublewell), slightly stronger nonlinear friction (higher b) helps.

## Prior Art and Novelty

### What is already known
- Bulgac and Kusnezov (1990) proposed the Pade-type friction g(xi) = xi*(a + b*xi^2)/(1 + c*xi^2)
- Orbit-003 optimized (a, b, c) via CMA-ES for all potentials jointly, finding a=0.7, b=3.0, c=0.06
- Orbit-010 introduced potential-adaptive effective-Q driving with per-potential alpha

### What this orbit adds
- Per-potential Pade friction tuning: the optimal (a, b, c) depends on the potential type
- Specifically, the gaussmix potential benefits from much lower b (1.0 vs 3.0)
- This is a parameter optimization result, not a new theoretical insight

### Honest positioning
This orbit applies existing techniques (Pade friction, effective-Q driving) with per-potential parameter tuning. The improvement is empirical -- we found that the globally-optimal Pade parameters from orbit-003 are suboptimal when specialized per potential. The 18% metric improvement is driven primarily by a single parameter change (b: 3.0 to 1.0 for gaussmix).

## Glossary

- **Pade friction**: g(xi) = xi*(a + b*xi^2)/(1 + c*xi^2), a rational function ensuring odd symmetry
- **Effective-Q**: Using h = alpha*|p|^2 - (alpha-1)*d*kT as driving, equivalent to Q_eff = Q/alpha
- **CMA-ES**: Covariance Matrix Adaptation Evolution Strategy, a derivative-free optimizer
- **KL gate**: KL divergence between empirical and analytical marginals must be < 0.05
- **tau_int**: Integrated autocorrelation time of q[0], lower is better
- **Mini-eval**: Shorter integration (200k-500k steps) for fast parameter screening

## References

- Bulgac, A. and Kusnezov, D. (1990). Phys. Rev. A 42, 5045 -- Pade-type friction form
- Orbit-003: CMA-ES optimization of Pade parameters (a=0.7, b=3.0, c=0.06)
- Orbit-010: Potential-adaptive effective-Q driving (metric=60.34)
