# Reproducing the paper results

## ReFocus rounds

Start the coordinator with `scripts/run_rounds.sh`; cluster setup is in the
[README](../README.md#training) and rewards, restoration and environment
variables are in [REFOCUS.md](REFOCUS.md).

```bash
bash scripts/run_rounds.sh --benchmark libero_10 \
  --rlinf-root "$LIBERO_RL_ROOT/RLinf" --run-root "$STRUCTRL_ROUNDS_ROOT" \
  --scheduler-provider openai-compatible \
  --scheduler-base-url "$SCHEDULER_BASE_URL" --scheduler-model "$SCHEDULER_MODEL"
```

Use `--benchmark robocasa365` for RoboCasa365. `train_reset_state_ids` /
`eval_reset_state_ids` and all training parameters are in the configs.

The reported numbers come from a separate evaluation run on the final
checkpoint with the official protocol. The coordinator's per-round evaluation
is a development signal, not the reported score.

## Inherited StructRL baselines

For RoboCasa365 Composite-Seen, these launchers use the original StructRL
target-split evaluation and the same π<sub>0.5</sub> SFT initialization as
`PI05_SFT_CKPT`.

| Policy | Sparse-RL | StructRL |
| --- | --- | --- |
| π0.5 | [PPO](../launch/train_pi05_sparse_rl_ppo.sbatch) | [PPO](../launch/train_pi05_structrl_ppo.sbatch), [GRPO](../launch/train_pi05_structrl_grpo.sbatch) |
| GR00T-N1.5 | [PPO](../launch/train_gr00t_sparse_rl_ppo.sbatch) | [PPO](../launch/train_gr00t_structrl_ppo.sbatch), [GRPO](../launch/train_gr00t_structrl_grpo.sbatch) |

Use an independent run directory per configuration and adapt the cluster
settings to your Ray or Slurm deployment. For LIBERO-Long baselines, run them
with `LIBERO_REWARD_VARIANT=structrl`.
