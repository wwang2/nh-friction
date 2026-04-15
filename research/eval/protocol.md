# Evaluation Protocol

## What We Measure

A solution is "better" if it mixes faster — i.e., the position observable becomes decorrelated in fewer timesteps. We measure this via the **difficulty-weighted mean integrated autocorrelation time (τ_int)** of the position q[0] across three canonical benchmark targets, averaged over 3 random seeds.

Lower τ_int means the thermostat produces more effectively independent samples per integration step. A τ_int of 100 means consecutive samples are nearly independent every 100 steps; a τ_int of 500 means every 500 steps. Non-ergodic thermostats (like standard NH on the 1D harmonic oscillator) get τ_int → ∞, capped at N/2 = 50,000.

Difficulty weights are frozen at baseline time — benchmark potentials where NHC M=3 struggles most (highest baseline τ_int) count proportionally more.

## How to Measure

1. **Load and validate the solution:**
   - Import `solution.py`; verify `friction_function` and `friction_derivative` are callable
   - Oddness check: max |g(ξ) + g(−ξ)| < 1e-6 on 100 log-spaced ξ ∈ [0.01, 10.0]; disqualify if violated
   - Derivative consistency check: finite-difference check at 10 points; warn but don't disqualify

2. **For each benchmark potential × seed (9 runs total):**
   - Seed NumPy RNG with seed value
   - Initialize: q ~ N(0,1), p ~ N(0,1), ξ = 0.0
   - Call `solution.setup(seed)` if defined (may run online optimizer)
   - Run velocity Verlet + exact-exponential friction for 10,010,000 steps, dt=0.01, discard first 10,000 (burn-in)
   - Record q[0] every 10 steps → 100,000 samples
   - Check canonical fidelity: bin q[0] into 100 bins on [−6, 6], compare to analytical marginal; compute KL divergence

3. **Compute τ_int per run:**
   - Sokal automatic windowing on the 100,000 samples (FFT-based autocorrelation)
   - Cutoff: W = first t where C(t)/C(0) < 0.05 OR W > 6 × running τ_int estimate
   - τ_int = 1 + 2 · Σ_{t=1}^{W} C(t)/C(0); cap at 50,000 if non-convergent

4. **Disqualification check:**
   - If mean KL (across 3 seeds) > 0.05 for ANY potential: METRIC = inf, disqualified
   - If oddness violated: METRIC = inf, disqualified

5. **Compute final metric:**
   - `weighted_tau_int = Σ_k w_k · mean_s(τ_int(k, s))`
   - Weights w_k stored in `research/eval/config.yaml` after baseline run

6. Report: `print(f"METRIC={weighted_tau_int:.6f}")`
   Also print auxiliary: `ESS_per_second`, `ergodicity_score_1d_HO`, `kl_per_potential`, `tau_int_per_potential`

## Acceptance Criteria

- Metric direction: **minimize**
- Deterministic: same `solution.py` + same seed → same METRIC (NumPy RNG seeded identically)
- Seeds: [42, 137, 2024] — fixed, not passed as CLI argument (seeded internally per run)
- Timeout: 600 seconds total (setup + 9 integration runs)

## What Counts as a Solution

A Python file `solution.py` that exposes:
- `friction_function(xi: np.ndarray) -> np.ndarray` — the function g(ξ), must be odd
- `friction_derivative(xi: np.ndarray) -> np.ndarray` — g'(ξ)
- (optional) `setup(seed: int) -> None` — called once before integration; may run an optimizer

The functions must handle arbitrary NumPy array shapes (scalar, 1D, batched). No external network calls during evaluation. No file I/O to fixed paths (evaluator runs in Modal sandbox).

Parametric solutions may use any Python/NumPy/SciPy — parameters must be either hardcoded or set via `setup()`. No external data files are permitted (parameters must be embedded in solution.py).

## Known Pitfalls

- **KAM tori / non-ergodicity:** Standard NH g(ξ)=ξ is non-ergodic on the 1D HO. Guard: τ_int is capped (not infinite) and KL gate catches distribution failure.
- **Sokal windowing variance:** τ_int estimates have high variance for slowly-mixing chains. Guard: 3 seeds per potential; difficulty-weighted average reduces variance from outlier runs.
- **Online optimizer overhead:** `setup()` may be slow (gradient descent, etc.). Guard: 600s total wall-time timeout; ESS/s auxiliary captures efficiency tradeoff.
- **BK thermostat non-ergodicity:** Bulgac-Kusnezov g(ξ)=2ξ/(1+ξ²) is known to be non-ergodic for some potentials (BK-NHC 2010 paper). Guard: 2D benchmarks are included specifically to catch this.
- **Logistic thermostat energy drift:** g(ξ)=tanh(ξ) (Tapias 2016) has poor energy conservation. Guard: KL gate on canonical distribution catches systematic drift.
- **Seed exploitation:** Evaluator seeds are fixed simple integers. Guard: KL gate on canonical distribution would catch any memorization attempt.
