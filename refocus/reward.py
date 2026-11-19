"""Stage-balanced ReFocus rewards and budget-preserving local refinements."""

import math


def valid_reward_weight(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class StageBalancedReward:
    """Stateful reward with the same worker interface as StructRL."""

    def __init__(self, stages, *, success_reward=2.0):
        self.stages = [list(stage) for stage in stages]
        subtasks = [subtask for stage in self.stages for subtask in stage]
        if not self.stages or any(not stage for stage in self.stages):
            raise ValueError("Each task requires at least one nonempty stage")
        if len(set(subtasks)) != len(subtasks):
            raise ValueError("Each subtask must belong to exactly one stage")
        if not valid_reward_weight(success_reward) or success_reward == 0:
            raise ValueError("Expected a finite positive success reward")
        self.K = len(self.stages)
        self.success_reward = success_reward
        self.reset()

    def reset(self):
        self.stage_idx = 0
        self.latched = {subtask: False for stage in self.stages for subtask in stage}
        self.outcome_fired = False
        self.cumulative = 0.0
        self.cumulative_progress = 0.0
        self.cumulative_outcome = 0.0
        self.refined_target = None
        self.milestone_shares = {}
        self.milestones_paid = set()
        self.previous_milestones = set()
        self.milestone_stage = None

    def configure_window(self, start, end, target, shares, signals):
        """Enable one reviewed refinement after reset or verified restoration."""
        stage = next((idx for idx, group in enumerate(self.stages) if target in group), None)
        if stage is None:
            raise ValueError(f"Unknown refinement target: {target}")
        if not start <= stage <= end or self.latched[target]:
            return
        if (target not in shares or len(shares) < 2
                or not all(valid_reward_weight(share) for share in shares.values())
                or not math.isclose(sum(shares.values()), 1.0, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("Milestone shares must include the target and sum to one")
        if set(shares) - {target} != signals.keys():
            raise ValueError("Milestones require a simulator signal for each intermediate check")
        self.refined_target = target
        self.milestone_stage = stage
        self.milestone_shares = dict(shares)
        self.previous_milestones = {name for name, satisfied in signals.items() if satisfied}

    def step(self, grouped_states, success, sim_t=None, *, signals=None):
        if self.outcome_fired:
            return self._report(0.0, 0.0)
        # Collect against the preceding stage, then advance the DAG once.
        stage = self.stage_idx
        active = self.stages[stage] if stage < self.K else ()
        progress = grouped_states["progress"]
        events = [name for name in active if progress.get(name, False) and not self.latched[name]]
        per_subtask = 2.0 / self.K / len(active) if active else 0.0
        reward = per_subtask * len(events)

        if self.refined_target is not None:
            required = self.milestone_shares.keys() - {self.refined_target}
            if signals is None or signals.keys() != required:
                raise ValueError("Missing verified milestone signals")
            current = {name for name, satisfied in signals.items() if satisfied}
            if stage == self.milestone_stage and not self.latched[self.refined_target]:
                new = current - self.previous_milestones - self.milestones_paid
                if self.refined_target in events:
                    new.add(self.refined_target)
                    reward -= per_subtask
                reward += per_subtask * sum(self.milestone_shares[name] for name in sorted(new))
                self.milestones_paid.update(new)
            self.previous_milestones = current

        for name in events:
            self.latched[name] = True
        if active and all(self.latched[name] for name in active):
            self.stage_idx += 1

        outcome = self.success_reward if success and not self.outcome_fired else 0.0
        self.outcome_fired |= bool(success)
        self.cumulative_progress += reward
        self.cumulative_outcome += outcome
        self.cumulative = self.cumulative_progress + self.cumulative_outcome
        return self._report(reward, outcome)

    def _report(self, reward, outcome):
        return {
            "stage_idx_after": self.stage_idx,
            "r_event": reward,
            "r_persistent": 0.0,
            "r_outcome": outcome,
            "r_total": reward + outcome,
            "cumulative": self.cumulative,
            "cumulative_progress": self.cumulative_progress,
            "cumulative_outcome": self.cumulative_outcome,
        }
