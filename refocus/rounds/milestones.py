"""Validate Scheduler milestone refinements against benchmark simulator checks."""

import math


def _valid_share(value):
    return type(value) in (int, float) and 0 < value <= 1


def validate_refinement(refinement, stage, entities, checks):
    """Return a refinement whose target lies in ``stage`` and whose checks are verifiable.

    ``checks`` maps a check name to ``(description, reference_kind)``; a check whose
    reference kind is None takes a null reference, otherwise the reference must be an
    entity of that kind.
    """
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
        if not isinstance(check, str) or check not in checks:
            raise ValueError("Unsupported milestone check")
        if not isinstance(obj, str) or obj not in entities["objects"]:
            raise ValueError("Unknown milestone object")
        kind = checks[check][1]
        if kind is None:
            valid_reference = ref is None
        else:
            valid_reference = isinstance(ref, str) and ref != obj and ref in entities[kind]
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


def shares_of(refinement):
    """Map each milestone and the target to its fraction of the target's budget."""
    shares = {item["name"]: item["share"] for item in refinement["milestones"]}
    shares[refinement["target"]] = refinement["final_share"]
    return shares
