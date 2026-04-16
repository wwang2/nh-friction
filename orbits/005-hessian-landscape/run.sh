#!/bin/bash
# Reproduce the Hessian landscape analysis experiment
# orbit/005-hessian-landscape
#
# Usage: cd to the worktree root, then:
#   bash orbits/005-hessian-landscape/run.sh

set -e

ORBIT_DIR="orbits/005-hessian-landscape"

echo "=== Step 1: Diagonal Hessian (7 evaluations) ==="
uv run python3 "$ORBIT_DIR/hessian_diag.py"

echo ""
echo "=== Step 2: Hill-climb along soft directions ==="
uv run python3 "$ORBIT_DIR/hill_climb.py"

echo ""
echo "=== Step 3: Generate figures ==="
uv run python3 "$ORBIT_DIR/plot_hessian.py"

echo ""
echo "=== Step 4: Evaluate final solution (3 seeds) ==="
for SEED in 1 2 3; do
    uv run python3 research/eval/evaluator.py --solution "$ORBIT_DIR/solution.py" --seed $SEED
done
