# Implementation notes

## Rewards

Each benchmark has its own stage table: 16 RoboCasa365 stages in
[`refocus/robocasa.py`](../refocus/robocasa.py) and 10 LIBERO-Long stages in
[`refocus/libero.py`](../refocus/libero.py). A stage is a set of subtask
predicates; each predicate earns credit once per episode and only while its
stage is active. Terminal success settles any completions observed in the same
state, in prerequisite order.

A milestone refinement splits the target subtask's share of the stage budget
across one to three intermediate checks. Shares always sum to the original
credit, so a milestone that is never reached cannot lose budget.

## Practice windows

The Scheduler picks one window per task and may attach a `milestone_refinement`
whose `target` is a subtask of the window's end stage. Each milestone carries
`name`, `check`, `object`, `reference` and `share`; the last one keeps the
target predicate and only needs `final_share`. A null refinement falls back to
the original progress reward. Supported checks come from `MILESTONE_CHECKS`:
`object_grasped`, `object_in_receptacle`, `object_inside_fixture` on
RoboCasa365; `object_grasped`, `object_on`, `object_in` on LIBERO. Invalid
refinements are rejected and the original reward stays in effect.

Accepted windows are written to `windows_round<R>.json`; validation output and
raw Scheduler replies are kept alongside it.

## Local rollouts and restoration

| Mode | Start | Boundary |
| --- | --- | --- |
| Normal | Restored training entry | Target stage done or window budget |
| Prefix | Normal task start | Target stage done or full horizon |
| Suffix | Restored training entry | Remaining full-task horizon |

`--local-mix` sets the probabilities in Normal, Suffix, Prefix order.
A task with no eligible window falls back to a normal-start full-task rollout.

RoboCasa365 and LIBERO restore saved MuJoCo integration state, controller
state, observables, and RNG. The worker rebuilds the entry's scene using the
source seed. Restoring a snapshot executes no prefix actions. The entry's
elapsed controls still shorten the remaining episode horizon.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `STRUCTRL_ROUNDS_ROOT` | Run root; every Ray worker must inherit it before cluster start. |
| `STRUCTRL_REWARD_VARIANT` | `refocus` or `structrl`. |
| `LIBERO_REWARD_VARIANT` | `refocus`, `structrl` / `smooth_band`, or `flat`. |
| `LIBERO_DENSE_REWARD` | `1` enables the LIBERO dense reward in the env worker. |
| `STRUCTRL_ROUND_VISUAL_FEEDBACK` | `0` disables evaluation montages. |
| `SCHEDULER_API_KEY`, `SCHEDULER_BASE_URL`, `SCHEDULER_MODEL` | Scheduler endpoint. |

Use a separate run root per reward variant and per Scheduler.
