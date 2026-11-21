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

Run the round schedule:

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
