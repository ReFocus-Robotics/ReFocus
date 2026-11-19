# Source snapshot

The StructRL source files in this repository are copied from the official
[amazon-science/StructRL](https://github.com/amazon-science/StructRL) repository
at commit `6caba1f9538167e0752a7338663e45aac8df965d` (2026-10-20).

The original π0.5 PPO/GRPO and Composite-Seen environment configs match that upstream snapshot. Dedicated `*_structrl_16gpu.yaml` recipes add the overrides from the original author launch scripts, adapt nodes/microbatch to 2/32, and stop at the paper reporting iteration 100. The RoboCasa365 16-GPU example launchers adapt
the cluster from 4x8 to 2x8 GPUs and set the actor microbatch to 32, equal to
the full per-rank batch for a global batch of 512. They preserve the upstream
10-step checkpoint interval. The SFT launcher adds input checks and a
configurable experiment name while keeping the upstream recipe, initialization,
two-node 16-GPU allocation, and batch size.

## ReFocus integration

The ReFocus reward engine lives in `refocus/` and is independent of the
simulator. The RoboCasa365 integration extends the recorder, coordinator, and
RLinf patches with stage-balanced rewards, practice windows, and direct
restoration. Upstream notices and license texts are retained.
