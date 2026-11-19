# Inherited StructRL baselines

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

The original baseline-specific density studies, precedence ablations,
and LIBERO reproduction instructions are available from the pinned
[StructRL source snapshot](https://github.com/amazon-science/StructRL/tree/6caba1f9538167e0752a7338663e45aac8df965d).
For the ReFocus path and result boundaries see the
[method and reproducibility notes](REFOCUS.md).
