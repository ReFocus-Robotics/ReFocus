"""Validate Scheduler refinements and bind them to RoboCasa365 predicates."""

import math

from refocus.reward import StageBalancedReward
from refocus.robocasa import TASK_STAGES


MILESTONE_CHECKS = {
    "object_grasped": "The gripper holds object. Set reference to null.",
    "object_in_receptacle": "Object is in the receptacle named by reference (another object).",
    "object_inside_fixture": "Object is inside the fixture named by reference.",
}


def milestone_entities(env):
    """Expose only exact simulator names accepted by the predicate interface."""
    return {
        "objects": sorted(getattr(env, "objects", {})),
        "fixtures": sorted(
            name for name, fixture in getattr(env, "fixtures", {}).items()
            if getattr(fixture, "get_int_sites", None) and fixture.get_int_sites()
        ),
    }


def _valid_share(value):
    return type(value) in (int, float) and 0 < value <= 1


def validate_refinement(refinement, stage, entities):
    """Return a validated refinement for the selected window's end stage."""
    if not isinstance(refinement, dict) or refinement.keys() != {"target", "milestones", "final_share"}:
        raise ValueError("Expected target, milestones, and final_share")
    target = refinement["target"]
    milestones = refinement["milestones"]
    final_share = refinement["final_share"]
    if not isinstance(target, str) or target not in stage:
        raise ValueError("Target must be a subtask in the window's end stage")
    if not isinstance(milestones, list) or not 1 <= len(milestones) <= 3:
        raise ValueError("Expected one to three intermediate milestones")
    if not _valid_share(final_share):
        raise ValueError("Final share must be in (0, 1]")
    names = set(stage)
    predicates = set()
    weights = [final_share]
    for milestone in milestones:
        if not isinstance(milestone, dict) or milestone.keys() != {"name", "check", "object", "reference", "share"}:
            raise ValueError("Expected name, check, object, reference, and share for each milestone")
        name, check, obj, ref, share = (milestone[key] for key in ("name", "check", "object", "reference", "share"))
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Milestone names must be distinct from each other and the stage subtasks")
        if not isinstance(check, str) or check not in MILESTONE_CHECKS:
            raise ValueError("Unsupported milestone check")
        if not isinstance(obj, str) or obj not in entities["objects"]:
            raise ValueError("Unknown milestone object")
        if check == "object_grasped":
            valid_reference = ref is None
        else:
            reference_names = entities["objects"] if check == "object_in_receptacle" else entities["fixtures"]
            valid_reference = isinstance(ref, str) and ref != obj and ref in reference_names
        if not valid_reference:
            raise ValueError("Invalid milestone reference")
        if (check, obj, ref) in predicates:
            raise ValueError("Duplicate milestone predicate")
        if not _valid_share(share):
            raise ValueError("Milestone share must be in (0, 1]")
        names.add(name)
        predicates.add((check, obj, ref))
        weights.append(share)
    if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Milestone shares must sum to one")
    return refinement


def milestone_signals(env, checks):
    """Evaluate only the supported simulator functions on validated entities."""
    from robocasa.utils import object_utils

    signals = {}
    for item in checks:
        check, obj, ref = item["check"], item["object"], item["reference"]
        if check == "object_grasped":
            satisfied = object_utils.check_obj_grasped(env, obj)
        elif check == "object_in_receptacle":
            satisfied = object_utils.check_obj_in_receptacle(env, obj, ref)
        elif check == "object_inside_fixture":
            satisfied = object_utils.obj_inside_of(env, obj, env.fixtures[ref], partial_check=False)
        else:
            raise ValueError("Unsupported milestone check")
        signals[item["name"]] = bool(satisfied)
    return signals


def build_robocasa_reward(task, *, success_reward=2.0):
    return StageBalancedReward(TASK_STAGES[task], success_reward=success_reward)
