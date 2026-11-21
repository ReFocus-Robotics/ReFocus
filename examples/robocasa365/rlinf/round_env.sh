#!/usr/bin/env bash
# Source this file from the ReFocus repository root on every Ray node.
# Override any default by exporting the variable before sourcing this file.

export STRUCTRL_ROOT="${STRUCTRL_ROOT:-$PWD}"
export RLINF_ROOT="${RLINF_ROOT:-$STRUCTRL_ROOT/RLinf}"
export VENV="${VENV:-$STRUCTRL_ROOT/.venv-rlinf-openpi-train}"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$STRUCTRL_ROOT:$RLINF_ROOT:$STRUCTRL_ROOT/robocasa:$STRUCTRL_ROOT/robosuite:$STRUCTRL_ROOT/openpi:${PYTHONPATH:-}"

export STRUCTRL_ROUNDS_ROOT="${STRUCTRL_ROUNDS_ROOT:-$STRUCTRL_ROOT/runs/refocus}"
export STRUCTRL_PROJECT_ROOT="$STRUCTRL_ROOT"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export REPO_PATH="$RLINF_ROOT"
export STRUCTRL_REWARD_VARIANT="${STRUCTRL_REWARD_VARIANT:-refocus}"
export STRUCTRL_TIMING_JSON="${STRUCTRL_TIMING_JSON:-$STRUCTRL_ROOT/structrl/demo_timing_composite_seen.json}"

export RAY_BIN="${RAY_BIN:-$VENV/bin/ray}"
export RAY_PORT="${RAY_PORT:-29500}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NUM_NODES="${NUM_NODES:-2}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export USE_TF=0
export SCHEDULER_MODEL="${SCHEDULER_MODEL:-gpt-6-astra}"
export SCHEDULER_REASONING_EFFORT="${SCHEDULER_REASONING_EFFORT:-low}"
