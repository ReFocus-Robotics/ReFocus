"""Bind the ReFocus reward to RoboCasa365 and the RSI worker interface."""

import math

from refocus.reward import StageBalancedReward, valid_reward_weight
from structrl.stage_config import get_stage_def


# Configuration can select only bundled simulator checks. The final milestone
# always uses the target's original, prerequisite-gated completion predicate.
def _bowl_grasped(env):
    from robocasa.utils.object_utils import check_obj_grasped

    return bool(check_obj_grasped(env, "bowl"))


_CHECKS = {
    "StoreLeftoversInBowl": ("bowl_in_fridge", {"bowl_grasped": _bowl_grasped}),
}


def validate_refinements(refinements, tasks):
    if not isinstance(refinements, dict) or refinements.keys() - set(tasks):
        raise ValueError("Refinements must be keyed by selected task names")
    for task, refinement in refinements.items():
        if task not in _CHECKS:
            raise ValueError(f"No bundled refinement checks for {task}")
        target, checks = _CHECKS[task]
        if not isinstance(refinement, dict) or set(refinement) != {"target", "shares"}:
            raise ValueError(f"Expected a target and shares for {task}")
        shares = refinement["shares"]
        if (refinement["target"] != target or not isinstance(shares, dict)
                or shares.keys() - {target} != checks.keys() or target not in shares
                or not all(valid_reward_weight(share) for share in shares.values())
                or not math.isclose(sum(shares.values()), 1.0, rel_tol=0, abs_tol=1e-9)):
            raise ValueError(f"Invalid target or milestone shares for {task}")
    return refinements


def milestone_signals(env):
    _, checks = _CHECKS[type(env).__name__]
    return {name: check(env) for name, check in checks.items()}


def build_robocasa_reward(task, *, success_reward=2.0):
    return StageBalancedReward(get_stage_def(task), success_reward=success_reward)
