---
issue: 9
parents: [003-cmaes-3pot]
eval_version: eval-v2
metric: inf
---

# Research Notes

## Result: DEAD-END (METRIC=inf, KL gate failed)

Pure Braga-Travis h=|∇V|²·d/E_ref fails ALL potentials. KL gate failed on harmonic (mean KL=15.6).

| Potential | mean τ_int | mean KL |
|---|---|---|
| harmonic_1d | ~900 | 15.6 |
| doublewell_2d | ~471 | 8.9 |
| gaussmix_2d | ~3166 | 9.2 |

## Root Cause

|∇V|²=0 at ALL local minima AND saddle points. BT coupling is zero exactly where the
particle lives and where barrier crossings must traverse. Thermostat decouples at critical
points → quasi-periodic dynamics → massive KL failure.

See orbit/010-potential-adaptive for the successful effective-Q approach instead.
