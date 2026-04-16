---
issue: 12
parents: [010-potential-adaptive]
eval_version: eval-v2
metric: inf
---

# Research Notes

## Result: DEAD-END — KL gate failure for all β

Cross-term h = |p|² + β·(p·∇V) breaks canonical invariance for all β ≠ 0.

### Empirical results
- β=0.3: KL≈1.7 (gaussmix_2d) → METRIC=inf
- β=1.0: KL≈1.7 (gaussmix_2d) → METRIC=inf
- β=0.0: recovers orbit 010 (METRIC=60.34)

### Theoretical root cause (Liouville condition)
The Nosé-Hoover equations preserve canonical measure iff div(F·μ)=0 where
μ ∝ exp(-βH - G(ξ)). This requires h - d·kT = α·(|p|²/m - d·kT) for
constant α — i.e. h must be an affine function of kinetic energy only.

Any q or ∇V dependence in h introduces position-coupled terms that violate
the zero-divergence condition. The cross-term p·∇V = dV/dt contains q
implicitly (through grad_V) → Liouville violation → non-canonical measure.

### Conclusion
The only valid driving functions within the eval-v2 interface are of the form
h = α·|p|² - (α-1)·d·kT for scalar α. Orbit 010's per-potential α detection
already exploits this optimally for harmonic (α=2.0) and doublewell (α=3.0).
Gaussmix remains at α=1.0 (standard kinetic) with τ_int≈73.81.
