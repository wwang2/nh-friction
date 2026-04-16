# Optimal Friction Functions for Deterministic Canonical Samplers (Pivot B: Configurational Driving)

## Problem Statement

The Nosé-Hoover thermostat couples Hamiltonian dynamics to an auxiliary variable ξ through a friction function g(ξ) and a thermostat driving term h(q,p). In the original formulation, h = |p|²/m (kinetic energy), which is insensitive to the particle's position in configuration space — causing slow mixing near barrier tops and mode boundaries.

This extended formulation allows h(q,p) to be any function satisfying E_canonical[h] = d·kT, enabling configuration-aware thermostat driving that couples strongly precisely where slow modes live.

The extended Nosé-Hoover equations are:
  dq/dt = p/m
  dp/dt = -∇V(q) - g(ξ)·p/Q
  dξ/dt = (h(q, p, ∇V(q)) - d·kT) / Q

where:
- g(ξ) must be odd and continuous (same as before)
- h(q, p, ∇V) must satisfy E_canonical[h] = d·kT (zero-mean condition under Boltzmann)
- The system preserves exp(-βH - ξ²/2) as invariant measure (Liouville: ∂ξ̇/∂ξ = 0 since h is independent of ξ)

**Braga-Travis configurational driving (primary approach):**
  h(q, p) = |∇V(q)|² / E_ref    where E_ref = E_canonical[|∇V|²]

This provides strong thermostat coupling near barrier tops and inter-mode regions (large forces),
and weak coupling in potential wells (small forces) — anticipating barrier crossings rather than
responding to them after the fact.

**Hybrid interpolation (optional):**
  h(q, p) = λ·|p|²/m + (1−λ)·|∇V(q)|²/E_ref    with λ ∈ [0,1]

At λ=1: standard Nosé-Hoover. At λ=0: pure configurational.

## Campaign History

eval-v1 (orbits 001–007): Optimized g(ξ) alone with kinetic driving h=|p|²/m.
Best achieved: τ_int = 84.14 using g(ξ) = ξ·(0.7 + 3.0·ξ²)/(1 + 0.06·ξ²).
Dead-end: ξ confined to [-1.6, 1.6]; KL gate locks g'(0)≈0.7; Hessian basin is deep (H_bb=1552).

## Solution Interface

`solution.py` must define:

```python
def friction_function(xi: np.ndarray) -> np.ndarray:
    """g(ξ) — must be odd, continuous. Same as eval-v1."""

def friction_derivative(xi: np.ndarray) -> np.ndarray:
    """g'(ξ) — needed for Jacobian verification."""

def driving_function(q: np.ndarray, p: np.ndarray, grad_V: np.ndarray) -> float:
    """h(q, p, ∇V) — driving term for dξ/dt = (h - d·kT) / Q.
    
    Must satisfy: E_canonical[h(q,p)] = d·kT  (zero-mean condition).
    
    Standard NH fallback (if this function is absent): return float(np.dot(p, p))
    Configurational (Braga-Travis): return float(np.dot(grad_V, grad_V)) / E_ref
    Hybrid: return lam * np.dot(p,p) + (1-lam) * np.dot(grad_V, grad_V) / E_ref
    
    Args:
        q:      position, shape (dim,)
        p:      momentum, shape (dim,)
        grad_V: force = -∂V/∂q evaluated at q, shape (dim,) — pre-computed by evaluator
    Returns:
        float — scalar driving value (units: kT)
    """
```

**Backward compatibility:** If `driving_function` is absent, the evaluator falls back to
kinetic energy driving h = |p|²/m, reproducing eval-v1 behavior exactly.

## Success Metric

**Metric:** Difficulty-weighted mean integrated autocorrelation time τ_int. Lower is better.

- **Direction:** minimize
- **eval-v1 best:** 84.14 (Padé CMA-ES, orbits 003/004)
- **New target:** τ_int < 65 (23% improvement over eval-v1 best; 51% over NHC M=3 baseline of 132.1)

Weights (unchanged from eval-v1):
- 1D harmonic: 2.4%
- 2D double-well: 29.4%
- 2D Gaussian mixture: 68.2%

## Normalization Requirements for h(q,p)

The zero-mean condition E_canonical[h] = d·kT must hold for the invariant measure to be preserved.

| Driving term | E_canonical[h] | Condition |
|---|---|---|
| \|p\|²/m (kinetic) | d·kT | Satisfied exactly (equipartition) |
| \|∇V\|²/d (harmonic approx) | kT·E[∇²V]/d | Exact for harmonic; approximate for others |
| \|∇V\|²/E_ref | d·kT by construction | Must compute E_ref = E[\|∇V\|²]/d |

Solutions are responsible for correct normalization. The KL gate (< 0.05) is the authoritative check.

**Pre-computed E_ref values (kT=1, canonical measure):**
- 1D harmonic V=q²/2: E[\|∇V\|²] = E[q²] = 1 → E_ref = 1.0
- 2D double-well V=(x²-1)²+y²/2: E[\|∇V\|²] ≈ 8.0 (numerical) → E_ref ≈ 4.0 per dimension
- 2D Gaussian mixture: E[\|∇V\|²] ≈ 2.0 (numerical, depends on inter-mode gradients) → E_ref ≈ 1.0

Solutions must estimate E_ref empirically during setup() if needed.

## Known Approaches

**Configurational driving (new in eval-v2):**
- Pure Braga-Travis: h = |∇V|²/E_ref (Braga & Travis 2005, J. Chem. Phys. 123, 134101)
- Hybrid kinetic+configurational: h = λ·|p|² + (1−λ)·|∇V|²/E_ref
- Potential-adaptive: different λ per potential (estimated from potential type during setup)

**Friction function (g) optimization (from eval-v1):**
- Padé form g(ξ) = ξ·(0.7 + 3.0·ξ²)/(1 + 0.06·ξ²) — best known (orbit 003)
- Joint optimization of both g(ξ) and h(q,p) parameters

## References

- Nosé, S. (1984). J. Chem. Phys. 81, 511.
- Hoover, W.G. (1985). Phys. Rev. A 31, 1695.
- Martyna, G.J., Klein, M.L., Tuckerman, M. (1992). J. Chem. Phys. 97, 2635.
- Braga, C. & Travis, K.P. (2005). J. Chem. Phys. 123, 134101. — Configurational thermostat
- Bulgac, A. & Kusnezov, D. (1990). Phys. Rev. A 42, 5045.
- Leimkuhler, B. & Reich, S. (2004). Simulating Hamiltonian Dynamics. Cambridge University Press.
- Tuckerman, M.E. et al. (2001). J. Chem. Phys. 115, 1678. — Non-Hamiltonian MD framework
