# Hardware Inference

## Problem signals
> "Integrate for 10⁶ steps with velocity Verlet (dt=0.01) ... 3 seeds per evaluation ... benchmark potentials: 1D harmonic, 2D double-well, 2D Gaussian mixture"
> "Deterministic thermostats ... sampling the canonical ensemble"

The evaluation is pure numerical ODE integration on low-dimensional systems (1D and 2D).
No neural network inference, no large matrix operations. Parallelism is across seeds and
potentials, not within a single integration run.

## Inferred needs
- Evaluation: CPU — velocity Verlet integration in 1D/2D is CPU-bound; no GPU needed
- Experiments: CPU — even parametric/learnable solutions are evaluated via ODE integration,
  not trained during eval; parameter sweeps benefit from multiple CPU cores not GPU
- Estimated eval duration: ~30–120 seconds per candidate (9M integrator steps total,
  spread across 3 potentials × 3 seeds, each ~1–2s on modern CPU)

## Config
compute.gpu: null
