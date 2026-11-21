"""RoboCasa365 progress stages and predicates from the ReFocus appendix."""


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


def get_subtask_states(env):
    """Evaluate the appendix DAG with the shared simulator progress checks."""
    from robocasa.utils import object_utils
    from structrl.subtask_checker import get_subtask_states as base_subtask_states

    task = type(env).__name__
    # Packing uses per-object placements, so it can reuse the same two
    # receptacle checks without computing the baseline container flags.
    if task == "PackIdenticalLunches":
        veg_in_0 = [v for v in ("vegetable0", "vegetable1")
                    if object_utils.check_obj_in_receptacle(env, v, "tupperware0")]
        veg_in_1 = [v for v in ("vegetable0", "vegetable1")
                    if object_utils.check_obj_in_receptacle(env, v, "tupperware1")]
        meat_in_0 = [m for m in ("meat0", "meat1")
                     if object_utils.check_obj_in_receptacle(env, m, "tupperware0")]
        meat_in_1 = [m for m in ("meat0", "meat1")
                     if object_utils.check_obj_in_receptacle(env, m, "tupperware1")]
        # Vegetables (and meats) are interchangeable, so progress is tracked by
        # count across both tupperwares, not by object identity: stage 1 needs
        # both vegetables packed (any tupperware), stage 2 then needs both
        # meats. The env's _check_success enforces the balanced per-tupperware
        # constraint at termination; these are intermediate progress signals.
        veg_count = len(veg_in_0) + len(veg_in_1)
        meat_count = len(meat_in_0) + len(meat_in_1)
        progress = {
            "vegetable0_packed": veg_count >= 1,
            "vegetable1_packed": veg_count >= 2,
            "meat0_packed": meat_count >= 1,
            "meat1_packed": meat_count >= 2,
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
