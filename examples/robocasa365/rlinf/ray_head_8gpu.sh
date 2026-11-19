#!/usr/bin/env bash
set -euo pipefail

# Ray environment workers inherit these settings from each node's Ray runtime.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STRUCTRL_ROOT="${STRUCTRL_ROOT:-${HERE}/../../..}"
export STRUCTRL_PROJECT_ROOT="$STRUCTRL_ROOT"
export STRUCTRL_REWARD_VARIANT="${STRUCTRL_REWARD_VARIANT:-structrl}"
export STRUCTRL_TIMING_JSON="${STRUCTRL_TIMING_JSON:-${STRUCTRL_ROOT}/structrl/demo_timing_composite_seen.json}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" USE_TF=0
export PYTHONPATH="${STRUCTRL_ROOT}:${RLINF_ROOT:-${STRUCTRL_ROOT}/RLinf}:${STRUCTRL_ROOT}/robocasa:${STRUCTRL_ROOT}/robosuite:${PYTHONPATH:-}"
if [[ "$STRUCTRL_REWARD_VARIANT" != refocus && ! -f "$STRUCTRL_TIMING_JSON" ]]; then
  echo "Missing timing data: $STRUCTRL_TIMING_JSON" >&2
  exit 2
fi

RAY_BIN="${RAY_BIN:-ray}"
RAY_PORT="${RAY_PORT:-29500}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_NODE="${CPUS_PER_NODE:-128}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-robocasa365-head}"

if "$RAY_BIN" status --address="127.0.0.1:${RAY_PORT}" >/dev/null 2>&1; then
  echo "Ray is already running at 127.0.0.1:${RAY_PORT}; leaving it unchanged."
  exit 0
fi

mkdir -p "$RAY_TMPDIR"
exec "$RAY_BIN" start --head \
  --port="$RAY_PORT" \
  --num-gpus="$GPUS_PER_NODE" \
  --num-cpus="$CPUS_PER_NODE" \
  --include-dashboard=true \
  --disable-usage-stats \
  --temp-dir="$RAY_TMPDIR"
