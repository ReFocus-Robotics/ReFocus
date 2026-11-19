#!/bin/bash
# pi0.5 SFT on RoboCasa365 Composite-Seen (openpi fork, JAX, 2 nodes).
#SBATCH --job-name=sft-pi05
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=86:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
OPENPI_ROOT="${OPENPI_ROOT:?export OPENPI_ROOT=<RoboCasa OpenPI checkout>}"
SFT_EXP_NAME="${SFT_EXP_NAME:-robocasa365_composite_seen_sft}"
export OPENPI_ROOT SFT_EXP_NAME
INITIAL_PI05_WEIGHTS="${OPENPI_ROOT}/checkpoints/openpi_pretrain/pi05_pretrain_human300/multitask_learning/75000/params"
[[ -d "${INITIAL_PI05_WEIGHTS}" ]] || { echo "Missing StructRL π0.5 Human300 initialization: ${INITIAL_PI05_WEIGHTS}" >&2; exit 2; }
cd "${OPENPI_ROOT}"

# JAX multi-node: srun starts 1 process per node; train.py initializes jax.distributed from SLURM_* vars
export JAX_PLATFORMS=""
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
# NIC names differ per node (e.g. enp75s0/enp72s0); the prefix "enp" matches all of them
export NCCL_SOCKET_IFNAME=enp
echo "=== pi05 2-node | nodes: $SLURM_JOB_NODELIST | nnodes=$SLURM_NNODES ==="
echo "initial weights: ${INITIAL_PI05_WEIGHTS}"
echo "SFT data: RoboCasa365 target_composite_seen (16 tasks)"

# pi0.5 full fine-tuning on the 16 RoboCasa365 Composite-Seen tasks, 2 nodes x 8 GPUs with FSDP
# pi05=True max_token_len=200 (knowledge-insulated dual head); everything else follows the official pi0 recipe
srun --ntasks-per-node=1 bash -c '
export JAX_PLATFORMS="" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0 NCCL_SOCKET_IFNAME=enp
echo "task PROCID=$SLURM_PROCID NODE=$(hostname) CVD=$CUDA_VISIBLE_DEVICES"
${OPENPI_ROOT}/.venv/bin/python ${OPENPI_ROOT}/scripts/train.py \
  pi05_robocasa_finetune_target_composite_seen \
  --exp_name=${SFT_EXP_NAME} \
  --fsdp_devices=16 --batch_size=64 \
  --no-wandb_enabled --overwrite
'
echo "PI05_2N EXIT: $?"
