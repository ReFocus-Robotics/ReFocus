#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STRUCTRL_ROOT="${STRUCTRL_ROOT:-${HERE}/../../..}"
RLINF_ROOT="${RLINF_ROOT:-${STRUCTRL_ROOT}/RLinf}"
VENV="${VENV:-${STRUCTRL_ROOT}/.venv-rlinf-openpi-train}"
PI05_SFT_CKPT="${PI05_SFT_CKPT:?Run Composite-Seen SFT first, then export PI05_SFT_CKPT to its converted PyTorch checkpoint}"
RAY_ADDRESS="${RAY_ADDRESS:?Start the 16-GPU Ray cluster and set RAY_ADDRESS first}"
NUM_NODES="${NUM_NODES:-2}"
[[ "$NUM_NODES" == 2 ]] || { echo "This launcher requires two 8-GPU nodes." >&2; exit 2; }
RUN_NAME="${RUN_NAME:-robocasa365_composite_seen_ppo_16xh100_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${STRUCTRL_ROOT}/logs}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
PY="${VENV}/bin/python"
RAY="${VENV}/bin/ray"
EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"

for required in "$PY" "$RAY" "$EMBODIED_PATH/train_embodied_agent.py" "$PI05_SFT_CKPT/robocasa365/norm_stats.json" "$STRUCTRL_ROOT/configs/robocasa365_ppo_openpi_pi05_structrl_16gpu.yaml" "${STRUCTRL_TIMING_JSON:-${STRUCTRL_ROOT}/structrl/demo_timing_composite_seen.json}"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 2; }
done

export RAY_ADDRESS PYTHONUNBUFFERED=1
export EMBODIED_PATH REPO_PATH="$RLINF_ROOT" STRUCTRL_PROJECT_ROOT="$STRUCTRL_ROOT"
export PI05_SFT_CKPT
export STRUCTRL_REWARD_VARIANT=structrl
export STRUCTRL_TIMING_JSON="${STRUCTRL_TIMING_JSON:-${STRUCTRL_ROOT}/structrl/demo_timing_composite_seen.json}"
export RLINF_CUDA_IPC_MODE="${RLINF_CUDA_IPC_MODE:-host_staging}"
export RAY_num_prestart_python_workers="${RAY_num_prestart_python_workers:-16}"
export RAY_worker_maximum_startup_concurrency="${RAY_worker_maximum_startup_concurrency:-16}"
export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-600}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" USE_TF=0
export PYTHONPATH="${STRUCTRL_ROOT}:${RLINF_ROOT}:${STRUCTRL_ROOT}/robocasa:${STRUCTRL_ROOT}/robosuite:${STRUCTRL_ROOT}/openpi:${PYTHONPATH:-}"

RAY_STATUS="$("$RAY" status --address="$RAY_ADDRESS")"
printf '%s\n' "$RAY_STATUS" | tee /tmp/"${RUN_NAME}"_ray_status.txt
TOTAL_RAY_GPUS="$(printf '%s\n' "$RAY_STATUS" | sed -nE 's/.*[0-9.]+\/[[:space:]]*([0-9.]+)[[:space:]]+GPU.*/\1/p' | head -n 1)"
[[ "$TOTAL_RAY_GPUS" == "16.0" || "$TOTAL_RAY_GPUS" == "16" ]] || {
  echo "Expected 16 GPUs in the Ray cluster, found '${TOTAL_RAY_GPUS:-unknown}'. Check the worker nodes before launch." >&2
  exit 2
}
mkdir -p "$RUN_DIR"

"$PY" "$HERE/export_tensorboard_metrics.py" \
  --log-dir "$RUN_DIR" \
  --output "$RUN_DIR/checkpoints/metrics/per_step_metrics.jsonl" \
  >"$RUN_DIR/metrics_exporter.log" 2>&1 &
METRICS_EXPORTER_PID=$!
cleanup_metrics_exporter() {
  kill "$METRICS_EXPORTER_PID" 2>/dev/null || true
  wait "$METRICS_EXPORTER_PID" 2>/dev/null || true
}
trap cleanup_metrics_exporter EXIT
cp "$HERE/train_ppo_16xh100_full.sh" "$RUN_DIR/launch.sh"
cp "$HERE/export_tensorboard_metrics.py" "$RUN_DIR/export_tensorboard_metrics.py"

cd "$RUN_DIR"
TRAIN_ARGS=(
  --config-path "$STRUCTRL_ROOT/configs/"
  --config-name robocasa365_ppo_openpi_pi05_structrl_16gpu
  "cluster.num_nodes=${NUM_NODES}"
  actor.micro_batch_size=32
  runner.max_epochs=100
  runner.max_steps=100
  runner.save_interval=10
  "runner.logger.log_path=${RUN_ROOT}"
  "env.eval.video_cfg.video_base_dir=${RUN_DIR}/video/eval"
  runner.logger.project_name=structrl_robocasa365
  "runner.logger.experiment_name=${RUN_NAME}"
  "actor.model.model_path=${PI05_SFT_CKPT}"
  "rollout.model.model_path=${PI05_SFT_CKPT}"
)
export HYDRA_FULL_ERROR=1
"$PY" "$EMBODIED_PATH/train_embodied_agent.py" "${TRAIN_ARGS[@]}" \
  --cfg job --resolve > "$RUN_DIR/effective_config.yaml"
"$PY" "$EMBODIED_PATH/train_embodied_agent.py" "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${RUN_DIR}/train.log"
