"""LIBERO-Long progress stages, BDDL subtask predicates, and simulator milestone checks."""

from pathlib import Path

from refocus.reward import StageBalancedReward

# Each stage maps subtask names to the BDDL goal conjunct that verifies them.
# Subtasks within a stage are unordered; stages are completed in order. The
# stove in KITCHEN_SCENE8 is switched on in the initial state, so its goal
# conjunct is a standing condition rather than a subtask.
_STAGES = {
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": (
        {"alphabet_soup_in_basket": ("in", "alphabet_soup_1", "basket_1_contain_region"),
         "tomato_sauce_in_basket": ("in", "tomato_sauce_1", "basket_1_contain_region")},
    ),
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": (
        {"cream_cheese_in_basket": ("in", "cream_cheese_1", "basket_1_contain_region"),
         "butter_in_basket": ("in", "butter_1", "basket_1_contain_region")},
    ),
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": (
        {"stove_on": ("turnon", "flat_stove_1"),
         "moka_pot_on_stove": ("on", "moka_pot_1", "flat_stove_1_cook_region")},
    ),
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": (
        {"bowl_in_drawer": ("in", "akita_black_bowl_1", "white_cabinet_1_bottom_region")},
        {"drawer_closed": ("close", "white_cabinet_1_bottom_region")},
    ),
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": (
        {"white_mug_on_left_plate": ("on", "porcelain_mug_1", "plate_1"),
         "yellow_white_mug_on_right_plate": ("on", "white_yellow_mug_1", "plate_2")},
    ),
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": (
        {"book_in_caddy": ("in", "black_book_1", "desk_caddy_1_back_contain_region")},
    ),
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": (
        {"white_mug_on_plate": ("on", "porcelain_mug_1", "plate_1"),
         "pudding_right_of_plate": ("on", "chocolate_pudding_1", "living_room_table_plate_right_region")},
    ),
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": (
        {"alphabet_soup_in_basket": ("in", "alphabet_soup_1", "basket_1_contain_region"),
         "cream_cheese_in_basket": ("in", "cream_cheese_1", "basket_1_contain_region")},
    ),
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": (
        {"moka_pot_1_on_stove": ("on", "moka_pot_1", "flat_stove_1_cook_region"),
         "moka_pot_2_on_stove": ("on", "moka_pot_2", "flat_stove_1_cook_region")},
    ),
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": (
        {"mug_in_microwave": ("in", "white_yellow_mug_1", "microwave_1_heating_region")},
        {"microwave_closed": ("close", "microwave_1")},
    ),
}

# Task names in LIBERO's libero_10 suite order (task_id 0..9).
TASKS = tuple(_STAGES)
TASK_STAGES = {task: tuple(tuple(stage) for stage in stages) for task, stages in _STAGES.items()}
SUBTASK_PREDICATES = {task: {name: predicate for stage in stages for name, predicate in stage.items()}
                      for task, stages in _STAGES.items()}

MILESTONE_CHECKS = {
    "object_grasped": ("The gripper holds object. Set reference to null.", None),
    "object_on": ("Object rests on the object, fixture, or region named by reference.", "supports"),
    "object_in": ("Object is inside the region named by reference.", "regions"),
    "half_open": ("The articulated object or region named by object (from articulated) is at least half way "
                  "from closed to open. Set reference to null.", None, "articulated"),
    "half_closed": ("The articulated object or region named by object (from articulated) is at least half way "
                    "from open to closed. Set reference to null.", None, "articulated"),
    "turned_on": ("The fixture named by object (from switchable) is turned on. Set reference to null.",
                  None, "switchable"),
}

# Open fraction separating half-open from half-closed.
_HALF = 0.5
_HINGE_OR_SLIDE = (2, 3)  # mjtJoint: slide, hinge


def _articulation(env, name):
    """Return the joints of an articulated entity with its open and closed joint positions."""
    state = env.object_states_dict[name]
    owner = env.get_object(getattr(state, "parent_name", name))
    ranges = getattr(owner, "object_properties", {}).get("articulation", {})
    opened, closed = ranges.get("default_open_ranges"), ranges.get("default_close_ranges")
    if not opened or not closed:
        return None
    joints = env.object_sites_dict[name].joints if name in env.object_sites_dict else owner.joints
    model = env.sim.model
    joints = [joint for joint in joints or ()
              if model.jnt_type[model.joint_name2id(joint)] in _HINGE_OR_SLIDE]
    if not joints:
        return None
    return joints, sum(opened) / 2, sum(closed) / 2


def _open_fractions(env, name):
    joints, opened, closed = _articulation(env, name)
    qpos = [env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(joint)] for joint in joints]
    return [(value - closed) / (opened - closed) for value in qpos]


def task_name(env):
    """Return the libero_10 task name of a LIBERO problem environment."""
    return Path(env.bddl_file_name).stem


def get_subtask_states(env, task):
    """Evaluate each subtask with the simulator's own BDDL predicate."""
    return {"progress": {name: bool(env._eval_predicate(list(predicate)))
                         for name, predicate in SUBTASK_PREDICATES[task].items()}}


def build_reward(task, *, success_reward=2.0):
    return StageBalancedReward(TASK_STAGES[task], success_reward=success_reward)


def task_evidence(task):
    """The BDDL goal of each subtask, shown to the Scheduler as checker evidence."""
    return {name: " ".join(predicate) for name, predicate in SUBTASK_PREDICATES[task].items()}


def milestone_entities(env):
    objects = sorted(env.objects_dict)
    regions = sorted(env.object_sites_dict)
    states = getattr(env, "object_states_dict", {})
    return {
        "objects": objects,
        "regions": regions,
        "supports": sorted({*objects, *env.fixtures_dict, *regions}),
        "articulated": sorted(name for name in states if _articulation(env, name)),
        "switchable": sorted(name for name in env.fixtures_dict
                             if hasattr(env.get_object(name), "turn_on")),
    }


def milestone_signals(env, checks):
    signals = {}
    for item in checks:
        check, obj, ref = item["check"], item["object"], item["reference"]
        if check == "object_grasped":
            satisfied = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.get_object(obj))
        elif check == "object_on":
            satisfied = env._eval_predicate(["on", obj, ref])
        elif check == "object_in":
            satisfied = env._eval_predicate(["in", obj, ref])
        elif check == "half_open":
            satisfied = all(fraction >= _HALF for fraction in _open_fractions(env, obj))
        elif check == "half_closed":
            satisfied = all(fraction <= _HALF for fraction in _open_fractions(env, obj))
        elif check == "turned_on":
            satisfied = env._eval_predicate(["turnon", obj])
        else:
            raise ValueError("Unsupported milestone check")
        signals[item["name"]] = bool(satisfied)
    return signals
