"""RoboCasa365 progress stages, subtask predicates, and simulator milestone checks."""

import ast
import math
from pathlib import Path

from refocus.reward import StageBalancedReward


TASK_STAGES = {
    "DeliverStraw": (("grasp_straw",), ("straw_in_glass_cup",)),
    "GetToastedBread": (("toaster_on",), ("toast_on_plate",)),
    "KettleBoiling": (("grasp_kettle",), ("kettle_on_stove",), ("kettle_on_active_burner",)),
    "LoadDishwasher": (("dish0_on_rack", "dish1_on_rack"), ("door_half_closed",), ("dishwasher_closed",)),
    "PackIdenticalLunches": (("vegetable0_packed", "vegetable1_packed"), ("meat0_packed", "meat1_packed")),
    "PreSoakPan": (("water_on", "pan_in_sink", "sponge_in_sink"),),
    "PrepareCoffee": (("grasp_mug",), ("mug_at_machine",), ("machine_turned_on",)),
    "RinseSinkBasin": (("water_on",), ("washed_left", "washed_center", "washed_right")),
    "ScrubCuttingBoard": (("grasp_sponge",), ("contact_5steps", "sweep_range_0p1m")),
    "SearingMeat": (("meat_in_pan",), ("grasp_pan",), ("pan_on_active_knob",)),
    "SetUpCuttingStation": (("meat_on_board", "knife_on_board"),),
    "StackBowlsCabinet": (("grasp_any_bowl",), ("bowls_stacked",), ("any_bowl_in_cabinet",)),
    "SteamInMicrowave": (("veg_in_bowl",), ("bowl_in_micro",), ("door_closed",), ("turn_on_microwave",)),
    "StirVegetables": (("veg1_in_pot", "veg2_in_pot"), ("spatula_grasped",), ("stir_t1",), ("task_complete",)),
    "StoreLeftoversInBowl": (("chicken_in_bowl", "vegetable_in_bowl"), ("grasp_bowl",), ("bowl_in_fridge",)),
    "WashLettuce": (("water_on",), ("lettuce_under_water",), ("wash_t10",), ("task_complete",)),
}

_GRASP_PREDICATES = {
    "DeliverStraw": ("grasp_straw", ("straw",)),
    "KettleBoiling": ("grasp_kettle", ("obj",)),
    "PrepareCoffee": ("grasp_mug", ("obj",)),
    "ScrubCuttingBoard": ("grasp_sponge", ("sponge",)),
    "SearingMeat": ("grasp_pan", ("pan",)),
    "StackBowlsCabinet": ("grasp_any_bowl", ("bowl1", "bowl2")),
    "StoreLeftoversInBowl": ("grasp_bowl", ("bowl",)),
}


def task_name(env):
    return type(env).__name__


def get_subtask_states(env):
    """Evaluate the appendix DAG with the shared simulator progress checks."""
    from robocasa.utils import object_utils
    from structrl.subtask_checker import get_subtask_states as base_subtask_states

    task = task_name(env)
    if task == "PackIdenticalLunches":
        def packed(food, container):
            return any(object_utils.check_obj_in_receptacle(env, f"{food}{i}", container) for i in (0, 1))
        # Items of one kind are interchangeable. The second subtask of each
        # stage requires one item in each tupperware, as the success check does.
        vegetables = [packed("vegetable", c) for c in ("tupperware0", "tupperware1")]
        meats = [packed("meat", c) for c in ("tupperware0", "tupperware1")]
        progress = {
            "vegetable0_packed": any(vegetables),
            "vegetable1_packed": all(vegetables),
            "meat0_packed": any(meats),
            "meat1_packed": all(meats),
        }
        return {"progress": progress}

    grouped = base_subtask_states(env)
    progress = grouped["progress"]
    if task in _GRASP_PREDICATES:
        name, objects = _GRASP_PREDICATES[task]
        progress[name] = any(object_utils.check_obj_grasped(env, obj) for obj in objects)
    elif task == "LoadDishwasher":
        progress["door_half_closed"] = env.dishwasher.is_closed(env, th=0.3)
    elif task == "RinseSinkBasin":
        progress["water_on"] = bool(env.sink.get_handle_state(env=env)["water_on"])
    elif task == "SteamInMicrowave":
        progress["turn_on_microwave"] = progress["microwave_on"]
    elif task == "StirVegetables":
        progress["stir_t1"] = env.success_time >= 1
    elif task == "WashLettuce":
        progress["lettuce_under_water"] = bool(env.sink.check_obj_under_water(env, "lettuce"))
        progress["wash_t10"] = env.washed_time >= 10
    return grouped


def build_reward(task, *, success_reward=2.0):
    return StageBalancedReward(TASK_STAGES[task], success_reward=success_reward)


def task_evidence(task):
    """Source of the task's StructRL subtask checker, shown to the Scheduler as evidence."""
    path = Path(__file__).resolve().parents[1] / "structrl" / "subtask_checker.py"
    source = path.read_text()
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "cls_name"
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == task):
            return ast.get_source_segment(source, node)
    return None


MILESTONE_CHECKS = {
    "object_grasped": ("The gripper holds object. Set reference to null.", None),
    "object_lifted": ("The gripper holds object and it no longer touches any counter. Set reference to null.", None),
    "object_in_receptacle": ("Object is in the receptacle named by reference (another object).", "objects"),
    "object_inside_fixture": ("Object is inside the fixture named by reference.", "fixtures"),
    "object_partially_inside_fixture": ("Part of object is inside the fixture named by reference.", "fixtures"),
    "object_contact": ("Object touches the object named by reference.", "objects"),
    "object_fixture_contact": ("Object touches the fixture named by reference (from contact_fixtures).",
                               "contact_fixtures"),
    "gripper_fixture_contact": ("The gripper touches the fixture named by object (from contact_fixtures). "
                                "Set reference to null.", None, "contact_fixtures"),
    "door_half_open": ("Every door of the fixture named by object (from door_fixtures) is at least half open. "
                       "Set reference to null.", None, "door_fixtures"),
    "door_half_closed": ("Every door of the fixture named by object (from door_fixtures) is at most half open. "
                         "Set reference to null.", None, "door_fixtures"),
    "knob_turned": ("A knob of the stove named by object (from stoves) has left its off position. "
                    "Set reference to null.", None, "stoves"),
    "burner_on": ("A burner of the stove named by object (from stoves) is on. Set reference to null.",
                  None, "stoves"),
    "machine_on": ("The coffee machine named by object (from coffee_machines) is turned on. "
                   "Set reference to null.", None, "coffee_machines"),
}

# Normalized door position separating half-open from half-closed, and the knob
# angle (radians) beyond which a knob counts as turned away from off.
_DOOR_HALF = 0.5
_KNOB_TURNED = 0.1


def _door_fixture(fixture):
    try:
        return bool(fixture.door_joint_names)
    except AttributeError:
        return False


def milestone_entities(env):
    """Expose only exact simulator names accepted by the predicate interface."""
    from robocasa.models.fixtures import CoffeeMachine, Stove, Wall

    fixtures = getattr(env, "fixtures", {})
    return {
        "objects": sorted(getattr(env, "objects", {})),
        "fixtures": sorted(
            name for name, fixture in fixtures.items()
            if getattr(fixture, "get_int_sites", None) and fixture.get_int_sites()
        ),
        "contact_fixtures": sorted(name for name, fixture in fixtures.items() if not isinstance(fixture, Wall)),
        "door_fixtures": sorted(name for name, fixture in fixtures.items() if _door_fixture(fixture)),
        "stoves": sorted(name for name, fixture in fixtures.items() if isinstance(fixture, Stove)),
        "coffee_machines": sorted(name for name, fixture in fixtures.items() if isinstance(fixture, CoffeeMachine)),
    }


def milestone_signals(env, checks):
    from robocasa.utils import object_utils

    def doors(name):
        fixture = env.fixtures[name]
        return fixture.get_joint_state(env, fixture.door_joint_names).values()

    signals = {}
    for item in checks:
        check, obj, ref = item["check"], item["object"], item["reference"]
        if check == "object_grasped":
            satisfied = object_utils.check_obj_grasped(env, obj)
        elif check == "object_lifted":
            satisfied = (object_utils.check_obj_grasped(env, obj)
                         and not object_utils.check_obj_any_counter_contact(env, obj))
        elif check == "object_in_receptacle":
            satisfied = object_utils.check_obj_in_receptacle(env, obj, ref)
        elif check == "object_inside_fixture":
            satisfied = object_utils.obj_inside_of(env, obj, env.fixtures[ref], partial_check=False)
        elif check == "object_partially_inside_fixture":
            satisfied = object_utils.obj_inside_of(env, obj, env.fixtures[ref], partial_check=True)
        elif check == "object_contact":
            satisfied = env.check_contact(env.objects[obj], env.objects[ref])
        elif check == "object_fixture_contact":
            satisfied = env.check_contact(env.objects[obj], env.fixtures[ref])
        elif check == "gripper_fixture_contact":
            satisfied = env.check_contact(env.robots[0].gripper["right"], env.fixtures[obj])
        elif check == "door_half_open":
            satisfied = all(value >= _DOOR_HALF for value in doors(obj))
        elif check == "door_half_closed":
            satisfied = all(value <= _DOOR_HALF for value in doors(obj))
        elif check == "knob_turned":
            angles = env.fixtures[obj].get_knobs_state(env).values()
            satisfied = any(_KNOB_TURNED < angle < 2 * math.pi - _KNOB_TURNED for angle in angles)
        elif check == "burner_on":
            stove = env.fixtures[obj]
            satisfied = any(stove.is_burner_on(env, location) for location in stove.get_knobs_state(env))
        elif check == "machine_on":
            satisfied = env.fixtures[obj].get_state()["turned_on"]
        else:
            raise ValueError("Unsupported milestone check")
        signals[item["name"]] = bool(satisfied)
    return signals

