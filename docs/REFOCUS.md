# ReFocus implementation notes

## Rewards

ReFocus uses the 16 RoboCasa365 stage definitions from the paper's appendix in
[`refocus/robocasa.py`](../refocus/robocasa.py) and the 10 LIBERO-Long stage
definitions in [`refocus/libero.py`](../refocus/libero.py). The StructRL
baseline uses its original stage table. On a nonterminal step, only the stage
active before reward evaluation can earn progress credit. Each predicate earns
credit once per episode. The simulator evaluates predicates at each control
step and RLinf aggregates their rewards over the action chunk. On the
successful terminal step, the reward also settles newly eligible completions
observed in that same state, in prerequisite order. ReFocus uses an
undiscounted progress budget of 2 and a separate one-time full-task success
reward of 2.

A milestone refinement divides the target subtask's share of that budget among
one to three intermediate checks. Shares always sum to the target's original
credit: checks satisfied before the milestone stage opens, or never satisfied
at all, are claimed together with the target so no budget is lost.

### Scheduler milestone interface

The Scheduler receives the stage predicates, the available simulator entity
names, the supported checks, and the task's StructRL checker source. Visual
rollout evidence guides the choice of milestones, and the simulator evaluates
their completion during training. Each benchmark exposes its own checks in
`MILESTONE_CHECKS`:

| Benchmark | Checks |
| --- | --- |
| RoboCasa365 | `object_grasped`, `object_in_receptacle`, `object_inside_fixture` |
| LIBERO-Long | `object_grasped`, `object_on`, `object_in` |

Each selected window can include a `milestone_refinement` object. Its `target`
is a subtask in the window's end stage. Its `milestones` array contains one to
three intermediate checks, each with `name`, `check`, `object`, `reference`
and `share`. The final milestone uses the original target predicate and only
needs `final_share`. A null refinement uses the original progress reward.

The coordinator records the raw response in `scheduler_response_round<R>.json`,
the validation results in `milestone_proposals_round<R>.json`, and the accepted
windows in `windows_round<R>.json`. Local workers receive the accepted windows
through `control.json` and verify the checks against each reset scene. An
invalid refinement leaves that rollout's original progress reward in effect.

Valid milestones can earn one-time credit in all three Local rollout modes
after the target's prerequisites hold. Conditions already satisfied at reset
earn no credit. Original stage progression and full-task success use the task
predicates.

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
Global phase attempt. On RoboCasa365, Normal and Suffix Windows restore a
snapshot directly: the worker rebuilds the recorded scene from source XML and
restores MuJoCo integration state, controller and fixture state, observables,
RNG state, and the progress reward state, then checks the rebuilt scene and
simulator state before practice. On LIBERO the scene is rebuilt from its
initial state and the recorded action prefix is replayed through the reward
before practice; the replay reproduces the progress state, and restoration is
therefore proportional to the prefix length. In both benchmarks the entry's
elapsed controls reduce the remaining task horizon, while restoration wall
time and fresh controls are logged separately.

The runtime entry bank uses pickle and should contain only entries from the
trusted run. The RLinf hooks preserve final observations for truncation and
exclude true terminations from the time-limit bootstrap.

The Scheduler uses normal-start validation to diagnose failures and select
windows. The history provided to future rounds includes the recent selected
windows, pre/post normal-start success, and recorded fresh controls and
restoration wall time. The final benchmark evaluation is a separate run.

## Validation

The CPU suite covers reward allocation, milestone replacement, entry capture
and restoration on both benchmarks, truncation, Scheduler validation, and the
RLinf worker hooks. A headless MuJoCo 3.3.1 test reproduced 50 continuation
steps in a synthetic scene on RoboCasa365. The distributed training commands
are documented in the [README](../README.md).

Set `STRUCTRL_REWARD_VARIANT=structrl` (RoboCasa365) or
`LIBERO_REWARD_VARIANT=structrl` (LIBERO) to use the StructRL reward in the
coordinator. That variant uses normal-start full-task rollouts in slot three of
the local mixture. Use a separate run directory for it.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `STRUCTRL_ROUNDS_ROOT` | Absolute run root; every Ray worker must inherit it before cluster start. |
| `STRUCTRL_REWARD_VARIANT` | `refocus` or `structrl`; must match the runtime reward. |
| `LIBERO_REWARD_VARIANT` | `refocus`, `structrl`/`smooth_band`, or `flat` for LIBERO. |
| `LIBERO_DENSE_REWARD` | `1` enables the LIBERO dense reward inside the env worker. |
| `STRUCTRL_ROUND_VISUAL_FEEDBACK` | `0` disables evaluation montages. |

## LIBERO

`patches/libero_addon.patch` contains the LIBERO integration: the flat and
StructRL dense rewards, the ReFocus stage reward selected by
`LIBERO_REWARD_VARIANT=refocus`, entry recording and replay, and the optional
`train_reset_state_ids` / `eval_reset_state_ids` pools. The stage table lives in
[`refocus/libero.py`](../refocus/libero.py) and keys each subtask by its BDDL
goal conjunct, so the same stage-balanced reward logic applies unchanged.

The two sequential tasks are the cabinet task (bowl, then close) and the
microwave task (mug, then close). Every other task has a single stage whose
subtasks may be completed in any order. The stove in
`KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` is already switched on in the
initial state, so its goal conjunct is a standing condition rather than a
subtask. Goal conjuncts that are already true at reset are marked satisfied and
do not block stage completion.

LIBERO has 50 official initial states per task. To keep training states out of
development evaluation, set `train_reset_state_ids` and `eval_reset_state_ids`
in the env config to non-overlapping index lists (for example 0–39 for training
and 40–49 for development). Both fields are optional; omitting them uses the
full pool, which is the default RLinf behavior.
