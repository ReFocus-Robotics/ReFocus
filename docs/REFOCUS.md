# ReFocus implementation notes

## Rewards

ReFocus uses the 16 RoboCasa365 stage definitions from the paper's appendix
in [`refocus/robocasa.py`](../refocus/robocasa.py). The StructRL baseline uses its
original stage table. On a nonterminal step, only the stage active before
reward evaluation can earn progress credit. Each predicate earns credit once
per episode. The simulator evaluates predicates
at each control step and RLinf aggregates their rewards over the action chunk.
On the successful terminal step, the reward also settles newly eligible
completions observed in that same state, in prerequisite order.
ReFocus uses an undiscounted progress budget of 2 and a separate
one-time full-task success reward of 2.

### Scheduler milestone interface

The Scheduler receives the stage predicates, available simulator object and fixture
names, and the supported checks. Visual rollout evidence guides the choice of
milestones. During training, the simulator evaluates their completion.

Each selected window can include a `milestone_refinement` object. Its `target`
is a subtask in the window's end stage. Its `milestones` array contains one to
three intermediate checks, each with `name`, `check`, `object`, `reference`,
and `share`. Use `object_grasped` with a null reference,
`object_in_receptacle` with an object reference, or `object_inside_fixture` with
a fixture reference. The final milestone uses the original target predicate.
All shares, including `final_share`, are fractions of the target's original
reward and sum to one. A null refinement uses the original progress reward.

The coordinator records the raw response in `advisor_response_round<R>.json`,
the validation results in `milestone_proposals_round<R>.json`, and the accepted
windows in `windows_round<R>.json`. Local workers receive the accepted windows
through `control.json` and verify the checks against each reset scene. An invalid
refinement leaves that rollout's original progress reward in effect.

Valid milestones can earn one-time credit in all three Local rollout modes
after the target's prerequisites hold. Entry conditions already satisfied at
reset earn no credit at reset. A final completion earns its own share. Original
stage progression and full-task success use the task predicates.

For an existing RLinf checkout with version 3 hooks, rerun the hook installer
before starting Ray. Use a fresh ReFocus run directory to collect training
entries with the appendix stage definitions and Scheduler-driven milestones.

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

## LIBERO

ReFocus supports LIBERO-10 (the standard `libero_10` suite) alongside
RoboCasa365. The stage definitions use BDDL goal conjunct tuples as subtask
keys instead of named predicates, so the same stage-balanced reward logic
applies without modification.

Apply `patches/libero_addon.patch` to a separate RLinf checkout and set the
following environment variables before training:

| Variable | Required value | Purpose |
| --- | --- | --- |
| `LIBERO_DENSE_REWARD` | `1` | Enable the BDDL milestone reward |
| `LIBERO_REWARD_VARIANT` | `refocus` | Select the stage-balanced variant |
| `STRUCTRL_REWARD_VARIANT` | `refocus` | Match the RSI coordinator |

The two-stage BDDL tasks (KITCHEN4: bowl then close cabinet; KITCHEN6: mug
then close microwave) are defined in `dense_reward.py` under `TASK_STAGES`.
All other tasks have a single parallel stage covering their goal conjuncts.
Conjuncts already satisfied at episode reset earn no reward and do not block
stage completion.

### Seed separation

LIBERO has 50 official initial states per task. To prevent training states from
appearing during dev evaluation, set `train_reset_state_ids` and
`eval_reset_state_ids` in the env config to non-overlapping index lists (e.g.
indices 0–39 for training and 40–49 for dev eval). The final test eval uses all
50 states (or a dedicated `test_reset_state_ids` subset). These fields are
optional; omitting them uses the full pool, which is the default RLinf behavior.
