# Reproducing the paper results

## ReFocus rounds

The coordinator runs Global and Local PPO phases in round order. Start it with
`scripts/run_rounds.sh`; the [README](../README.md#training) documents the
cluster setup, and [REFOCUS.md](REFOCUS.md) documents the rewards, restoration
and environment variables.

- **RoboCasa365.** 16 Composite-Seen tasks, 50-action chunks, a 2900-step
  horizon, three rounds of 25 Global and 25 Local updates on two 8-GPU nodes.
- **LIBERO-Long.** Ten `libero_10` tasks, 10-action chunks, a 1024-step episode
  limit, the same round schedule on one 8-GPU node.

Before a full run, check the LIBERO integration on a single GPU host with a
short job. The coordinator rejects an unpatched runtime, so pass a runtime that
`scripts/setup_upstream.sh --libero` prepared:

```bash
export STRUCTRL_ROUNDS_ROOT=/tmp/rounds-smoke
export LIBERO_RL_ROOT=/path/to/libero_rl
export CKPT_ROOT=/path/to/ckpts
export LIBERO_DENSE_REWARD=1
export LIBERO_REWARD_VARIANT=refocus
export STRUCTRL_REWARD_VARIANT=refocus

# 1. Configuration resolves and the round plan is valid.
python -m refocus.rounds.coordinator --benchmark libero_10 \
  --rlinf-root "$LIBERO_RL_ROOT/RLinf" --run-root "$STRUCTRL_ROUNDS_ROOT" --dry-run

# 2. Two updates with two environments exercise the reward transport and the
#    recording path, and write a decodable training video.
python "$LIBERO_RL_ROOT/RLinf/examples/embodiment/train_embodied_agent.py" \
  --config-path "$PWD/configs" --config-name libero_10_ppo_openpi_pi05_refocus \
  runner.max_steps=2 runner.val_check_interval=-1 runner.save_interval=-1 \
  env.train.total_num_envs=2 env.train.max_episode_steps=40 \
  env.train.max_steps_per_rollout_epoch=40 env.train.video_cfg.save_video=True
```

The environment worker logs `[dense_reward] swap chunk_rewards` each chunk; a
randomly initialized policy keeps `inc_max=0.000` because it does not complete a
subtask, which is expected. A controlled simulator test that toggles a BDDL
predicate verifies that the reward fires exactly once per subtask.

After the smoke passes, run the full round schedule:

```bash
bash scripts/run_rounds.sh --benchmark libero_10 \
  --rlinf-root "$LIBERO_RL_ROOT/RLinf" --run-root "$STRUCTRL_ROUNDS_ROOT" \
  --scheduler-provider openai-compatible \
  --scheduler-base-url "$SCHEDULER_BASE_URL" --scheduler-model "$SCHEDULER_MODEL"
```

The final benchmark numbers use a separate evaluation run on the last
checkpoint with the official protocol; the coordinator's per-round evaluation
is a development signal and is not the reported score.

## Inherited StructRL baselines

These launchers use RoboCasa365 Composite-Seen and the original StructRL
target-split evaluation procedure. They provide the baseline entry points
that are relevant to this repository. See [installation](INSTALL.md) for
the pinned dependencies and [RLinf examples](../examples/robocasa365/rlinf/README.md)
for the two-node launchers.

| Policy | Sparse-RL | StructRL |
| --- | --- | --- |
| π0.5 | [PPO](../launch/train_pi05_sparse_rl_ppo.sbatch) | [PPO](../launch/train_pi05_structrl_ppo.sbatch), [GRPO](../launch/train_pi05_structrl_grpo.sbatch) |
| GR00T-N1.5 | [PPO](../launch/train_gr00t_sparse_rl_ppo.sbatch) | [PPO](../launch/train_gr00t_structrl_ppo.sbatch), [GRPO](../launch/train_gr00t_structrl_grpo.sbatch) |

To compare the π0.5 training variants, use the same Composite-Seen
PyTorch SFT initialization. The [SFT launcher](../launch/sft_pi05.sh)
produces the checkpoint referenced by `PI05_SFT_CKPT`. Use independent run
directories for each configuration. Baseline launchers include cluster settings
that should be adapted to the available Ray or Slurm deployment.

LIBERO-Long baselines use the same policy and reward constants with 10-step
action chunks and a 1024-step episode limit; run them with
`LIBERO_REWARD_VARIANT=structrl`. The ReFocus path, including LIBERO stage
tables and the round coordinator, is documented in
[method and reproducibility notes](REFOCUS.md).
