#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STRUCTRL_ROOT="${STRUCTRL_ROOT:-${HERE}/../../..}"
RLINF_ROOT="${RLINF_ROOT:-${STRUCTRL_ROOT}/RLinf}"
VENV="${VENV:-${STRUCTRL_ROOT}/.venv-rlinf-openpi-train}"
PI05_SFT_CKPT="${PI05_SFT_CKPT:?Export the converted Composite-Seen SFT checkpoint path}"
RAY_ADDRESS="${RAY_ADDRESS:?Start the 16-GPU Ray cluster and set RAY_ADDRESS first}"
NUM_NODES="${NUM_NODES:-2}"
[[ "$NUM_NODES" == 2 ]] || { echo "This profile requires two 8-GPU nodes." >&2; exit 2; }

TRAIN_ENVS=128
GLOBAL_BATCH_SIZE=256
MICRO_BATCH_SIZE=32
EVAL_INTERVAL=5
SAVE_INTERVAL=5
ACTOR_WORLD_SIZE=8
EVAL_ENVS=80
(( TRAIN_ENVS % 16 == 0 )) || { echo "TRAIN_ENVS must remain divisible by all 16 tasks." >&2; exit 2; }
[[ "$MICRO_BATCH_SIZE" == 32 ]] || { echo "This profile keeps MICRO_BATCH_SIZE=32." >&2; exit 2; }
[[ "$GLOBAL_BATCH_SIZE" == "$((MICRO_BATCH_SIZE * ACTOR_WORLD_SIZE))" ]] || {
  echo "GLOBAL_BATCH_SIZE must equal MICRO_BATCH_SIZE * ACTOR_WORLD_SIZE ($((MICRO_BATCH_SIZE * ACTOR_WORLD_SIZE)))." >&2
  exit 2
}
[[ "$EVAL_INTERVAL" == 5 && "$SAVE_INTERVAL" == 5 ]] || {
  echo "This profile evaluates and saves every 5 steps." >&2; exit 2;
}

RUN_NAME="${RUN_NAME:-robocasa365_composite_seen_ppo_16gpu_fast_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${STRUCTRL_ROOT}/logs}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
PY="${VENV}/bin/python"
RAY="${VENV}/bin/ray"
EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
CONFIG="${STRUCTRL_ROOT}/configs/robocasa365_ppo_openpi_pi05_structrl_16gpu_fast.yaml"

for required in "$PY" "$RAY" "$EMBODIED_PATH/train_embodied_agent.py" \
  "$PI05_SFT_CKPT/robocasa365/norm_stats.json" "$CONFIG" \
  "${STRUCTRL_TIMING_JSON:-${STRUCTRL_ROOT}/structrl/demo_timing_composite_seen.json}"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 2; }
done

export RAY_ADDRESS PYTHONUNBUFFERED=1 EMBODIED_PATH
export REPO_PATH="$RLINF_ROOT" STRUCTRL_PROJECT_ROOT="$STRUCTRL_ROOT" PI05_SFT_CKPT
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
  echo "Expected 16 GPUs in the Ray cluster, found '${TOTAL_RAY_GPUS:-unknown}'." >&2; exit 2;
}

mkdir -p "$RUN_DIR"
cp "$HERE/train_ppo_16xh100_fast.sh" "$RUN_DIR/launch.sh"
"$PY" "$HERE/export_tensorboard_metrics.py" --log-dir "$RUN_DIR" \
  --output "$RUN_DIR/checkpoints/metrics/per_step_metrics.jsonl" \
  >"$RUN_DIR/metrics_exporter.log" 2>&1 &
METRICS_EXPORTER_PID=$!
"$PY" "$HERE/retain_ppo_checkpoints.py" \
  --checkpoint-root "$RUN_DIR/checkpoints" \
  --metrics "$RUN_DIR/checkpoints/metrics/per_step_metrics.jsonl" \
  >"$RUN_DIR/checkpoint_retention.log" 2>&1 &
CHECKPOINT_RETENTION_PID=$!
cleanup_background_helpers() {
  kill "$CHECKPOINT_RETENTION_PID" "$METRICS_EXPORTER_PID" 2>/dev/null || true
  wait "$CHECKPOINT_RETENTION_PID" 2>/dev/null || true
  wait "$METRICS_EXPORTER_PID" 2>/dev/null || true
}
trap cleanup_background_helpers EXIT
cp "$HERE/export_tensorboard_metrics.py" "$RUN_DIR/export_tensorboard_metrics.py"
cp "$HERE/retain_ppo_checkpoints.py" "$RUN_DIR/retain_ppo_checkpoints.py"
cd "$RUN_DIR"

TRAIN_ARGS=(
  --config-path "$STRUCTRL_ROOT/configs/"
  --config-name robocasa365_ppo_openpi_pi05_structrl_16gpu_fast
  "cluster.num_nodes=${NUM_NODES}"
  "env.train.total_num_envs=${TRAIN_ENVS}"
  "env.eval.total_num_envs=${EVAL_ENVS}"
  "actor.micro_batch_size=${MICRO_BATCH_SIZE}"
  "actor.global_batch_size=${GLOBAL_BATCH_SIZE}"
  runner.max_epochs=100 runner.max_steps=100
  "runner.val_check_interval=${EVAL_INTERVAL}"
  "runner.save_interval=${SAVE_INTERVAL}"
  "runner.logger.log_path=${RUN_ROOT}"
  "env.eval.video_cfg.video_base_dir=${RUN_DIR}/video/eval"
  runner.logger.project_name=structrl_robocasa365
  "runner.logger.experiment_name=${RUN_NAME}"
  "actor.model.model_path=${PI05_SFT_CKPT}"
  "rollout.model.model_path=${PI05_SFT_CKPT}"
)
export HYDRA_FULL_ERROR=1
"$PY" "$EMBODIED_PATH/train_embodied_agent.py" "${TRAIN_ARGS[@]}" --cfg job --resolve > "$RUN_DIR/effective_config.yaml"
"$PY" - "$RUN_DIR/effective_config.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
assert cfg['env']['train']['total_num_envs'] % 16 == 0
assert cfg['actor']['micro_batch_size'] == 32
assert cfg['actor']['global_batch_size'] == 256
assert cfg['runner']['val_check_interval'] == 5
assert cfg['runner']['save_interval'] == 5
assert cfg['cluster']['component_placement']['actor'] == '0-7'
assert cfg['env']['train']['total_num_envs'] == 128
assert cfg['env']['eval']['total_num_envs'] == 80
assert cfg['env']['eval']['total_num_envs'] % 16 == 0
print('Resolved fast profile: 128 train envs, 80 eval episodes (5/task), actor world size 8, global batch 256, microbatch 32, eval/save every 5.')
PY
"$PY" "$EMBODIED_PATH/train_embodied_agent.py" "${TRAIN_ARGS[@]}" 2>&1 | tee "$RUN_DIR/train.log"
"$PY" "$HERE/retain_ppo_checkpoints.py" \
  --checkpoint-root "$RUN_DIR/checkpoints" \
  --metrics "$RUN_DIR/checkpoints/metrics/per_step_metrics.jsonl" \
  --once || echo "Checkpoint retention final pass failed; inspect checkpoint_retention.log."
