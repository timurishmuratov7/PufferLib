#!/usr/bin/env bash
set -euo pipefail

STEPS=(
  30000000
  100000000
  300000000
  1000000000
)

AGENTS=(
  1024
  2048
  4096
  8192
)

source .venv/bin/activate
export CCACHE_DIR="${CCACHE_DIR:-/tmp/ccache}"
mkdir -p "$CCACHE_DIR"

NVCC_ARCH="${NVCC_ARCH:-compute_90}" ./build.sh flappy

for steps in "${STEPS[@]}"; do
  for agents in "${AGENTS[@]}"; do
    echo
    echo "=== flappy: steps=${steps} agents=${agents} ==="
    puffer train flappy \
      --train.total-timesteps "$steps" \
      --vec.total-agents "$agents"
  done
done
