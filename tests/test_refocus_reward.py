import json
from pathlib import Path

import pytest

from refocus.reward import StageBalancedReward
from structrl.rsi.reward import build_robocasa_reward, validate_refinements


def advance(reward, *subtasks, success=False, signals=None):
    grouped = {"progress": dict.fromkeys(subtasks, True)}
    return reward.step(grouped, success, signals=signals)


def test_stage_budget_and_first_valid_completion():
    reward = build_robocasa_reward("StoreLeftoversInBowl")
    assert advance(reward, "bowl_in_fridge")["r_total"] == 0
    assert advance(reward, "vegetable_in_bowl")["r_total"] == 0.5
    assert advance(reward, "vegetable_in_bowl")["r_total"] == 0
    # The fridge predicate cannot be credited in the evaluation that unlocks it.
    assert advance(reward, "chicken_in_bowl", "bowl_in_fridge")["r_total"] == 0.5
    assert reward.stage_idx == 1
    assert advance(reward, "bowl_in_fridge", success=True)["r_total"] == 3
    assert advance(reward, "bowl_in_fridge", success=True)["r_total"] == 0
    assert reward.cumulative_progress == 2
    assert reward.cumulative_outcome == 2


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
def test_unordered_subtasks_leave_the_stage_budget_unchanged(order):
    reward = StageBalancedReward([order, ["c"]])
    assert advance(reward, "a", "b", "c")["r_total"] == 1
    assert reward.stage_idx == 1
    assert advance(reward, "c")["r_total"] == 1


def test_milestones_preserve_the_target_budget_and_other_progress():
    reward = StageBalancedReward([["a"], ["target", "sibling"], ["later"]])
    shares = {"grasp": 0.25, "target": 0.75}
    reward.configure_window(1, 1, "target", shares, {"grasp": True})
    assert advance(reward, "a", signals={"grasp": True})["r_total"] == pytest.approx(2 / 3)
    assert advance(reward, "sibling", signals={"grasp": True})["r_total"] == pytest.approx(1 / 3)
    assert advance(reward, signals={"grasp": False})["r_total"] == 0
    assert advance(reward, signals={"grasp": True})["r_total"] == pytest.approx(1 / 12)
    assert advance(reward, signals={"grasp": True})["r_total"] == 0
    assert advance(reward, "target", signals={"grasp": True})["r_total"] == pytest.approx(1 / 4)
    assert reward.latched["target"]
    assert advance(reward, "later", signals={"grasp": True})["r_total"] == pytest.approx(2 / 3)
    assert reward.cumulative_progress == pytest.approx(2)
    reward.reset()
    assert reward.refined_target is None
    assert reward.stage_idx == 0
    assert advance(reward, "a")["r_total"] == pytest.approx(2 / 3)


def test_milestones_stay_gated_by_the_target_prerequisites():
    reward = StageBalancedReward([["prep"], ["target"]])
    reward.configure_window(0, 1, "target", {"grasp": 0.5, "target": 0.5}, {"grasp": False})
    assert advance(reward, "prep", signals={"grasp": True})["r_total"] == 1
    assert advance(reward, "target", signals={"grasp": True})["r_total"] == 0.5


def test_refinement_config_accepts_only_bundled_checks():
    config = json.loads(Path("examples/robocasa365/milestone_refinements.json").read_text())
    assert validate_refinements(config, ["StoreLeftoversInBowl"]) == config
    config["StoreLeftoversInBowl"]["shares"]["arbitrary_llm_expression"] = 0
    with pytest.raises(ValueError, match="Invalid target or milestone"):
        validate_refinements(config, ["StoreLeftoversInBowl"])
