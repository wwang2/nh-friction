---
issue: 6
parents: [003-cmaes-3pot]
eval_version: eval-v1
metric: null
---

# Hessian Landscape Analysis of Pade Friction Parameters

## Hypothesis

The parent orbit (003-cmaes-3pot) established that (a=0.7, b=3.0, c=0.06) is a local optimum of the Pade friction function g(xi) = xi*(a + b*xi^2)/(1 + c*xi^2), achieving metric=84.14. But "local optimum" was established by grid search with discrete perturbations. The Hessian matrix of the metric surface tells us something the grid cannot: whether the basin is deep (all eigenvalues large, nowhere to go) or whether there exist soft directions (near-zero eigenvalues) along which the metric barely changes -- and along which we might find a lower-energy valley by walking further.

## Approach

1. Compute the 3x3 Hessian H_ij = d^2(metric)/d(theta_i)*d(theta_j) via central finite differences at the known optimum theta* = (0.7, 3.0, 0.06).
2. Step sizes: delta_i ~ 1-2% of each parameter value, with special care for c=0.06 (use absolute step 0.005).
3. Eigen-decompose H to find eigenvalues and eigenvectors.
4. If any eigenvalue < 5 (soft direction), hill-climb along that eigenvector using adaptive step sizes.
5. Report: is the basin deep or exploitable?

## Glossary

- **Hessian**: Matrix of second partial derivatives of the metric with respect to parameters
- **Soft direction**: Eigenvector of the Hessian with near-zero eigenvalue, indicating the metric is nearly flat along that direction
- **Pade friction**: Rational function g(xi) = xi*(a+b*xi^2)/(1+c*xi^2) for the Nose-Hoover thermostat
- **tau_int**: Integrated autocorrelation time -- the metric being minimized
- **KAM tori**: Kolmogorov-Arnold-Moser invariant surfaces that trap trajectories in phase space
