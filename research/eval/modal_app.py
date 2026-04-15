"""
Shared Modal app definition for this campaign.

This file is the single source of truth for:
  - The Modal image (built from pyproject.toml via uv — add packages there)
  - Hardware config (CPU, memory, timeout)
  - The Modal app name

Both the evaluator (research/eval/evaluator.py) and experiment solutions
(orbits/*/solution.py) import from here so hardware config stays in one place.

Hardware: CPU-only (velocity Verlet integration in 1D/2D, no GPU needed).

To add a package: `uv add <package>` at the project root.
The Modal image re-installs from pyproject.toml on next build — no manual sync.
"""

import modal
from pathlib import Path

# ── Hardware config ────────────────────────────────────────────────────────────
# From research/config.yaml compute.*
GPU_TYPE     = None   # CPU-only: ODE integration in 1D/2D needs no GPU
CPU_COUNT    = 4      # 4 CPUs: parallelise across seeds/potentials if desired
MEMORY_MB    = 8192   # 8 GB: ample for 100k-sample autocorrelation buffers
TIMEOUT_SECS = 3600   # Modal hard timeout; per-eval timeout enforced in evaluator

# ── Image built from pyproject.toml via uv ────────────────────────────────────
_repo_root = Path(__file__).parent.parent.parent  # campaign repo root

image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("pip install uv --quiet")
    .copy_local_file(str(_repo_root / "pyproject.toml"), "/app/pyproject.toml")
    .run_commands("cd /app && uv pip install --system .")
)

# ── Modal app ──────────────────────────────────────────────────────────────────
app = modal.App("bath-nh-friction", image=image)
