---
issue: 7
parents: [003-cmaes-3pot]
eval_version: eval-v1
metric: 84.142123
---

# Non-monotonic Effective Coupling for Barrier Crossing

## Motivation

The parent orbit (003-cmaes-3pot) achieved metric = 84.14 with the Pade form
g(xi) = xi * (0.7 + 3.0*xi^2) / (1 + 0.06*xi^2). The effective coupling
gamma(xi) = g(xi)/xi is monotonically increasing from 0.7 at the origin to
b/c = 50 at large |xi|. This means friction grows steadily with thermostat
displacement.

The Gaussian mixture potential (5 modes on a circle of radius 3, weight 68.2%
of the metric) requires barrier crossing between modes separated by distance
~3.5. The bottleneck is accumulating enough kinetic energy to cross these
barriers. A monotonically increasing friction profile fights this: as the
thermostat variable grows (storing energy), friction increases proportionally,
damping the momentum before it can propel the particle over a barrier.

The hypothesis: design gamma(xi) = g(xi)/xi with a dip at intermediate |xi|
(around 2-3), creating a "window" where friction is temporarily reduced. This
lets kinetic energy build up when the thermostat is in this range, enabling
more frequent barrier crossings. At large |xi|, friction rises again to
prevent runaway.

## Parametric Form

g(xi) = xi * (a + b*xi^2) / (1 + c*xi^2 + d*xi^4)

The effective coupling is:
  gamma(xi) = (a + b*xi^2) / (1 + c*xi^2 + d*xi^4)

For d > 0, the denominator grows as xi^4, so gamma -> 0 for large |xi|.
This is qualitatively different from the parent (d=0) where gamma -> b/c.

gamma(0) = a (coupling at origin, controls KAM tori breaking)
gamma(xi) ~ b/(d*xi^2) for large |xi| (decays to zero)

The dip location: d/dxi [gamma(xi)] = 0 at xi = xi*. With nonzero d, gamma
first rises from a, then bends back down as the xi^4 term dominates.

For a non-monotonic profile, we need gamma to have a local maximum before
descending. The peak is where the numerator growth (b*xi^2) balances the
denominator growth (d*xi^4). After the peak, gamma decreases — this IS the
"reduced friction window" we want.

Actually, the non-monotonicity in gamma means: gamma rises to a peak, then
falls. The "dip" below the initial value a happens only if gamma eventually
goes below a. This requires:
  a + b*xi^2 < a*(1 + c*xi^2 + d*xi^4)
  b < a*c + a*d*xi^2
For large xi: always true if d > 0.

So the profile is: starts at a, rises to peak, then decays back through a
and down toward 0. The crossing point where gamma = a occurs at:
  b*xi^2 = a*c*xi^2 + a*d*xi^4
  (b - a*c) = a*d*xi^2
  xi_cross = sqrt((b - a*c)/(a*d))

For the parent's a=0.7, b=3.0, c=0.06:
  b - a*c = 3.0 - 0.042 = 2.958
  xi_cross = sqrt(2.958 / (0.7*d))

To have xi_cross ~ 2: d ~ 2.958/(0.7*4) = 1.056
To have xi_cross ~ 3: d ~ 2.958/(0.7*9) = 0.470

## Strategy

1. Start with the parent's (a, b, c) = (0.7, 3.0, 0.06) and sweep d
2. Use CMA-ES to optimize all 4 parameters jointly
3. Enforce d > 0 via exp-reparameterization

## Iteration Log
