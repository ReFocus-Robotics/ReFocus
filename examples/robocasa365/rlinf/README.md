# RoboCasa365 StructRL on 16 H100 GPUs

Use the [repository setup and recipe documentation](../../../README.md). The entrypoints are:

- `train_ppo_16xh100.sh`: default fast profile (`configs/robocasa365_ppo_openpi_pi05_structrl_16gpu_fast.yaml`): 128 train envs, global batch 256, microbatch 32, 80 eval episodes (5/task), eval/save every 5 steps.
- `train_ppo_16xh100_full.sh`: canonical full-sample recipe (`configs/robocasa365_ppo_openpi_pi05_structrl_16gpu.yaml`): 256 train envs, global batch 512, microbatch 32, save every 10 steps.
- `train_grpo_16xh100.sh`: `configs/robocasa365_grpo_openpi_pi05_structrl_16gpu.yaml`

The fast PPO launcher evaluates every 5 steps and keeps only two numeric checkpoint directories: the highest `eval/success_once` checkpoint (`checkpoints/best`) and the newest complete checkpoint (`checkpoints/last`). “Best” uses the mean success rate across the 80 evaluation episodes (16 target tasks × 5 episodes); because each task has the same episode count, this weights tasks equally. The two names are relative symlinks to the retained `global_step_N` directories. A checkpoint is eligible only after its full model weights and DCP training-state files are present and stable, and the matching 80-episode evaluation metric has been exported. The retention helper updates both links and its manifest before deleting other complete step directories. It fails closed and leaves checkpoints intact if the metric is missing or conflicting.

Both PPO launchers require exactly two 8-GPU nodes and an explicit `PI05_SFT_CKPT` pointing to the Composite-Seen SFT export. The default PPO entrypoint uses the faster-step profile; use `_full.sh` for the original full-sample recipe. Run PPO first, release its allocation, then run GRPO independently from that SFT export. Training stops at iteration 100. During-training evaluation for the full recipe stays disabled.

The dedicated recipes include the original author launch overrides: 50 executed actions, 10 denoising steps, actor LR 5e-6, PPO value LR 1e-4 and VLM-prefix critic, PPO intermediate StructRL rewards, and GRPO final-only StructRL rewards with zero-variance-group filtering. They load directly from this repository's `configs/`, avoiding a stale copied primary config in RLinf. Backend/model defaults still use the pinned patched RLinf installation.

On every node, set the same `STRUCTRL_ROOT`, `RLINF_ROOT`, `STRUCTRL_TIMING_JSON`, `RAY_BIN`, and head address before starting the helpers. The helpers export the paper reward variant before Ray starts. Existing runtime environments are not changed automatically.

```bash
export RAY_HEAD_IP=10.0.0.10  # replace with the actual head address
export RAY_ADDRESS="$RAY_HEAD_IP:29500"
export GPUS_PER_NODE=8 NUM_NODES=2
./examples/robocasa365/rlinf/ray_head_8gpu.sh
# Run on each of the other nodes:
./examples/robocasa365/rlinf/ray_worker_8gpu.sh
# On the head node, after validating the checkpoint and smoke test:
./examples/robocasa365/rlinf/train_ppo_16xh100.sh
# After PPO finishes and resources are released:
# ./examples/robocasa365/rlinf/train_grpo_16xh100.sh
```

TensorBoard events and `train.log` remain in `RUN_DIR`; emitted scalar metrics are also mirrored into `RUN_DIR/checkpoints/metrics/per_step_metrics.jsonl`.

For paper comparisons, use the author's offline `launch/eval_pi05_composite_seen.sbatch` (16 tasks × 100 episodes, target split, replan 50, denoise 10 and original task horizons), not the training YAML's 32 evaluation environments. Record the user's fixed per-task evaluation seeds 7–11 separately from the author's reset policy.
