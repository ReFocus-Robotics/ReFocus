import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from refocus.reward import StageBalancedReward
from refocus.robocasa import MILESTONE_CHECKS, build_reward, get_subtask_states, milestone_signals
from refocus.rounds.milestones import validate_refinement


def advance(reward, *subtasks, success=False, signals=None):
    grouped = {"progress": dict.fromkeys(subtasks, True)}
    return reward.step(grouped, success, signals=signals)


def test_stage_budget_and_first_valid_completion():
    reward = build_reward("StoreLeftoversInBowl")
    assert advance(reward, "bowl_in_fridge")["r_total"] == 0
    assert advance(reward, "vegetable_in_bowl")["r_total"] == pytest.approx(1 / 3)
    assert advance(reward, "vegetable_in_bowl")["r_total"] == 0
    # The fridge predicate cannot be credited in the evaluation that unlocks it.
    assert advance(reward, "chicken_in_bowl", "grasp_bowl")["r_total"] == pytest.approx(1 / 3)
    assert reward.stage_idx == 1
    assert advance(reward, "grasp_bowl")["r_total"] == pytest.approx(2 / 3)
    assert advance(reward, "bowl_in_fridge", success=True)["r_total"] == pytest.approx(2 + 2 / 3)
    assert advance(reward, "bowl_in_fridge", success=True)["r_total"] == 0
    assert reward.cumulative_progress == pytest.approx(2)
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
    # grasp became True before the milestone stage opened, so its credit was not
    # issued in stage 0. On target completion the budget is preserved by
    # claiming the uncredited grasp share together with the target share.
    assert advance(reward, "target", signals={"grasp": True})["r_total"] == 1


def test_pre_satisfied_milestone_claims_share_on_target_completion():
    reward = StageBalancedReward([["prep"], ["target"]])
    reward.configure_window(0, 1, "target", {"grasp": 0.5, "target": 0.5}, {"grasp": True})
    assert advance(reward, "prep", signals={"grasp": True})["r_total"] == 1
    # grasp was pre-satisfied at window entry and never triggered in-episode.
    # Its share is claimed on target completion, preserving the full budget.
    assert advance(reward, "target", signals={"grasp": True})["r_total"] == 1


def test_untriggered_milestone_share_claimed_on_target_completion():
    reward = StageBalancedReward([["prep"], ["target"]])
    reward.configure_window(0, 1, "target", {"grasp": 0.5, "target": 0.5}, {"grasp": False})
    assert advance(reward, "prep", signals={"grasp": False})["r_total"] == 1
    # grasp never became True; its share is still claimed on target completion
    # so the total progress budget is preserved.
    assert advance(reward, "target", signals={"grasp": False})["r_total"] == 1


@pytest.mark.parametrize("first_stage, expected_progress", [
    (["prep"], 2.0),
    (["prep", "missing"], 0.5),
])
def test_success_settles_only_observed_and_eligible_stages(first_stage, expected_progress):
    reward = StageBalancedReward([first_stage, ["target"]])
    report = advance(reward, "prep", "target", success=True)
    assert report["r_event"] == expected_progress
    assert report["r_outcome"] == 2.0
    assert advance(reward, "target", success=True)["r_total"] == 0


def test_terminal_settlement_honors_milestone_shares():
    reward = StageBalancedReward([["prep"], ["target"]])
    reward.configure_window(0, 1, "target", {"grasp": 0.3, "target": 0.7}, {"grasp": False})
    report = advance(reward, "prep", "target", success=True, signals={"grasp": True})
    assert report["stage_idx_after"] == 2
    assert report["r_event"] == pytest.approx(2.0)
    assert reward.milestones_paid == {"grasp", "target"}


@pytest.mark.parametrize("edits, error", [
    ({"check": "unknown"}, "Unsupported"),
    ({"object": "missing"}, "Unknown milestone object"),
    ({"reference": "missing"}, "Invalid milestone reference"),
    ({"share": True}, "Milestone share"),
    ({"share": 0.4}, "sum to one"),
])
def test_rejects_unverified_milestone_output(edits, error):
    config = dict(target="place", final_share=0.5, milestones=[
        dict(name="pick", check="object_grasped", object="item", reference=None, share=0.5),
    ])
    config["milestones"][0].update(edits)
    with pytest.raises(ValueError, match=error):
        validate_refinement(config, ["place"], {"objects": ["item"], "fixtures": []}, MILESTONE_CHECKS)


def test_milestone_signals_bind_to_named_simulator_entities(monkeypatch):
    object_utils = SimpleNamespace(
        check_obj_grasped=Mock(return_value=True),
        check_obj_in_receptacle=Mock(return_value=False),
        obj_inside_of=Mock(return_value=True),
    )
    monkeypatch.setitem(sys.modules, "robocasa.utils", SimpleNamespace(object_utils=object_utils))
    fixture = object()
    env = SimpleNamespace(fixtures={"fixture": fixture})
    checks = [
        dict(name="pick", check="object_grasped", object="item", reference=None, share=0.25),
        dict(name="insert", check="object_in_receptacle", object="item", reference="tray", share=0.25),
        dict(name="store", check="object_inside_fixture", object="tray", reference="fixture", share=0.25),
    ]
    entities = {"objects": ["item", "tray"], "fixtures": ["fixture"]}
    assert validate_refinement(dict(target="place", milestones=checks, final_share=0.25),
                               ["place"], entities, MILESTONE_CHECKS)
    assert milestone_signals(env, checks) == {"pick": True, "insert": False, "store": True}
    object_utils.check_obj_grasped.assert_called_once_with(env, "item")
    object_utils.check_obj_in_receptacle.assert_called_once_with(env, "item", "tray")
    object_utils.obj_inside_of.assert_called_once_with(env, "tray", fixture, partial_check=False)


def test_fixture_milestones_take_fixture_objects():
    door = dict(name="half_closed", check="door_half_closed", object="dishwasher", reference=None, share=0.5)
    entities = {"objects": ["dish"], "fixtures": [], "door_fixtures": ["dishwasher"]}
    assert validate_refinement(dict(target="closed", milestones=[door], final_share=0.5),
                               ["closed"], entities, MILESTONE_CHECKS)
    with pytest.raises(ValueError, match="Unknown milestone object"):
        validate_refinement(dict(target="closed", milestones=[dict(door, object="dish")], final_share=0.5),
                            ["closed"], entities, MILESTONE_CHECKS)


def test_articulation_and_contact_milestones_read_simulator_state(monkeypatch):
    object_utils = SimpleNamespace(check_obj_grasped=Mock(return_value=True),
                                   check_obj_any_counter_contact=Mock(return_value=False))
    monkeypatch.setitem(sys.modules, "robocasa.utils", SimpleNamespace(object_utils=object_utils))
    dishwasher = SimpleNamespace(door_joint_names=["door"], get_joint_state=lambda env, names: {"door": 0.3})
    stove = SimpleNamespace(get_knobs_state=lambda env: {"front_left": 0.2, "rear_left": 0.0},
                            is_burner_on=lambda env, location: False)
    machine = SimpleNamespace(get_state=lambda: {"turned_on": True})
    gripper = object()
    env = SimpleNamespace(
        fixtures={"dishwasher": dishwasher, "stove": stove, "coffee_machine": machine},
        objects={"spatula": "spatula", "carrot": "carrot"},
        robots=[SimpleNamespace(gripper={"right": gripper})],
        check_contact=lambda a, b: (a, b) == ("spatula", "carrot") or (a is gripper and b is stove),
    )
    checks = [
        dict(name="half_open", check="door_half_open", object="dishwasher", reference=None),
        dict(name="half_closed", check="door_half_closed", object="dishwasher", reference=None),
        dict(name="knob", check="knob_turned", object="stove", reference=None),
        dict(name="burner", check="burner_on", object="stove", reference=None),
        dict(name="machine", check="machine_on", object="coffee_machine", reference=None),
        dict(name="stir", check="object_contact", object="spatula", reference="carrot"),
        dict(name="reach", check="gripper_fixture_contact", object="stove", reference=None),
        dict(name="lift", check="object_lifted", object="spatula", reference=None),
    ]
    assert milestone_signals(env, checks) == {
        "half_open": False, "half_closed": True, "knob": True, "burner": False,
        "machine": True, "stir": True, "reach": True, "lift": True,
    }


def test_appendix_predicates_use_simulator_checks_and_native_completion(monkeypatch):
    object_utils = SimpleNamespace(
        check_obj_grasped=Mock(return_value=False),
        check_obj_in_receptacle=Mock(side_effect=lambda env, obj, ref: (obj, ref) == ("meat0", "tupperware1")),
    )
    monkeypatch.setitem(sys.modules, "robocasa.utils", SimpleNamespace(object_utils=object_utils))
    monkeypatch.setitem(sys.modules, "structrl.subtask_checker", SimpleNamespace(
        get_subtask_states=lambda env: {"progress": {"bowl_in_fridge": True, "task_complete": False}},
    ))
    fridge = type("StoreLeftoversInBowl", (), {})()
    assert get_subtask_states(fridge)["progress"]["grasp_bowl"] is False
    object_utils.check_obj_grasped.assert_called_once_with(fridge, "bowl")

    lunches = type("PackIdenticalLunches", (), {})()
    assert get_subtask_states(lunches)["progress"] == {
        "vegetable0_packed": False, "vegetable1_packed": False, "meat0_packed": True, "meat1_packed": False,
    }

    wash = type("WashLettuce", (), {})()
    wash.washed_time = 10
    wash.sink = SimpleNamespace(check_obj_under_water=Mock(return_value=True))
    progress = get_subtask_states(wash)["progress"]
    assert progress["lettuce_under_water"] and progress["wash_t10"]
    assert not progress["task_complete"]

    stir = type("StirVegetables", (), {})()
    stir.success_time = 1
    progress = get_subtask_states(stir)["progress"]
    assert progress["stir_t1"] and not progress["task_complete"]
