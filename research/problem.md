# Optimal Friction Functions for Deterministic Canonical Samplers

## Problem Statement

The Nosé-Hoover thermostat couples Hamiltonian dynamics to an auxiliary variable ξ through a friction function g(ξ). The standard choice g(ξ)=ξ (linear) is known to fail ergodicity on the 1D harmonic oscillator due to KAM tori. Nosé-Hoover Chains (NHC) fix this by chaining multiple thermostats, but an orthogonal approach is to design better g(ξ) while keeping a single auxiliary variable.

The modified Nosé-Hoover equations are:
  dq/dt = p/m
  dp/dt = -∇V(q) - g(ξ)·p/Q
  dξ/dt = (|p|²/m - d·kT) / Q

where g(ξ) must satisfy: g is odd, continuous, and the system preserves exp(-βH - ξ²/2) as invariant measure (Liouville verification required).

Deterministic thermostats are fundamental in molecular dynamics for sampling the canonical ensemble without stochastic noise. The Nosé-Hoover thermostat (Nosé 1984, Hoover 1985) is the simplest but suffers ergodicity failures. NHC (Martyna et al. 1992) chains M auxiliary variables. An underexplored direction is optimizing the nonlinear friction function g(ξ) itself.

Key insight: the derivative g'(0) controls coupling strength near equilibrium, while tail behavior g(ξ→∞) controls far-from-equilibrium dynamics. These two regimes may have competing optimal designs.

## Solution Interface

`solution.py` must define:

```python
def friction_function(xi: np.ndarray) -> np.ndarray:
    """g(ξ) — must be odd, bounded or controlled growth."""

def friction_derivative(xi: np.ndarray) -> np.ndarray:
    """g'(ξ) — needed for Jacobian verification."""
```

## Success Metric

**Metric:** Mean integrated autocorrelation time τ_int of the position observable, averaged over benchmark potentials (1D harmonic oscillator, 2D double-well, 2D Gaussian mixture). Lower is better.

- **Direction:** minimize
- **Baseline:** NHC (M=3) with Q=kT — typical τ_int ≈ 200–500 depending on potential
- **Target:** τ_int < 150 (30%+ improvement over NHC baseline)

Evaluation requirements:
- Integrate for 10⁶ steps with velocity Verlet (dt=0.01)
- 3 seeds per evaluation, report mean
- Verify canonical measure preservation: KL divergence to analytical Boltzmann < 0.05 (disqualify if violated)
- Report ergodicity score on 1D HO (KS test + variance matching)
- Benchmark potentials:
  1. 1D harmonic: V(x) = x²/2
  2. 2D double-well: V(x,y) = (x²-1)² + y²/2
  3. 2D Gaussian mixture: 5 modes, unit variance, means on circle radius 3

## Known Approaches

Candidate families to explore:
- tanh(α·ξ) with sweep over α
- 2ξ/(1+ξ²) (log-oscillator form, from Bulgac & Kusnezov 1990)
- ξ·exp(-ξ²/σ²) (Gaussian-damped)
- Piecewise linear with tunable slopes
- Learnable parametric families (e.g., rational functions, neural basis)

## References

- Nosé, S. (1984). J. Chem. Phys. 81, 511.
- Hoover, W.G. (1985). Phys. Rev. A 31, 1695.
- Martyna, G.J., Klein, M.L., Tuckerman, M. (1992). J. Chem. Phys. 97, 2635.
- Bulgac, A. & Kusnezov, D. (1990). Phys. Rev. A 42, 5045.
- Ceriotti, M. et al. (2010). J. Chem. Phys. 133, 124104.
- Leimkuhler, B. & Reich, S. (2004). Simulating Hamiltonian Dynamics. Cambridge University Press.
