# ReFocus implementation notes

## Rewards

The RoboCasa365 stage partition comes from `structrl/stage_config.py`. Only the
stage active before a reward evaluation can earn progress credit, and each
predicate earns credit once per episode. The simulator evaluates predicates
at each control step and RLinf aggregates their rewards over the action chunk.
The candidate uses an undiscounted progress budget of 2 and a separate
one-time full-task success reward of 2.

The Scheduler records milestone proposals as JSON for review. The coordinator
activates milestones only when a supplied `--refinements` file names a bundled
simulator check and the selected window targets the relevant stage. The only
bundled refinement targets `bowl_in_fridge` in `StoreLeftoversInBowl` and
uses RoboCasa's existing grasp check for the bowl. It assigns half the original
subtask reward to each of the grasp and placement events. Already-satisfied
grasp state at entry earns no new credit. If placement occurs without a new
eligible grasp event, it earns its own share. Original stage progression and
full-task success still use the original predicates.

## Interaction and restoration

| Local mode | Start | Boundary |
| --- | --- | --- |
| Normal Window | Restored training entry | Target-stage completion or window budget, subject to remaining horizon and episode termination. |
| Prefix Window | Normal task start | Target-stage completion, full horizon, or episode termination. |
| Suffix Window | Restored training entry | Remaining full-task horizon or episode termination. |

The launch argument `--local-mix` takes probabilities in Normal Window, Suffix
Window, Prefix Window order, with defaults 0.50, 0.25, and 0.25. The modes are
sampled at reset. A task without an eligible window falls back to a
normal-start full-task rollout.

The entry bank is populated by Global training and scoped to the committed
Global phase attempt. Normal and Suffix Windows restore a snapshot directly.
The worker rebuilds the recorded scene from source XML and restores MuJoCo
integration state, controller and fixture state, observables, RNG state,
and the progress reward state. It checks the rebuilt scene fingerprint and
simulator state before practice. Prefix actions are neither stored nor
replayed. The entry's elapsed controls reduce the remaining task horizon,
while restoration wall time and fresh controls are logged separately.

Start a new run directory when switching from the archived prefix replay
implementation. The earlier entry format lacks direct snapshots. The runtime
entry bank uses pickle and should contain only entries from the trusted run.
The RLinf hooks preserve final observations for truncation and exclude true
terminations from the time-limit bootstrap.

The Scheduler uses normal-start validation to diagnose failures and select
windows. The history provided to future rounds includes the recent selected
windows, pre/post normal-start success, and recorded fresh controls and
restoration wall time. The final benchmark evaluation is a separate run.

## Validation

The CPU suite covers reward allocation, milestone replacement, direct
restoration without action steps, truncation, and the RLinf worker hooks.
A headless MuJoCo 3.3.1 test reproduced 50 continuation steps in a synthetic
scene. The distributed training command is documented in the
[README](../README.md).

Set `STRUCTRL_REWARD_VARIANT=structrl` to use the StructRL reward in the
coordinator. This variant uses normal-start full-task rollouts in slot three
of the local mixture. Use a separate run directory for it.
