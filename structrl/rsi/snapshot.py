"""Direct snapshots for the pinned RoboCasa365 and robosuite runtime."""

import copy
import random

import numpy as np


ENTRY_SNAPSHOT_VERSION = 1
_OBSERVABLE_FIELDS = (
    "_time_since_last_sample", "_current_delay", "_current_observed_value", "_sampled",
)
_INTERPOLATORS = ("interpolator_pos", "interpolator_ori", "interpolator")
_HISTORY_BUFFERS = (
    "recent_qpos", "recent_actions", "recent_torques", "recent_ee_forcetorques",
    "recent_ee_pose", "recent_ee_vel", "recent_ee_vel_buffer", "recent_ee_acc",
)


def _plain(value):
    if isinstance(value, np.ndarray):
        return not value.dtype.hasobject
    if type(value) in (bool, int, float, str, type(None)) or isinstance(value, np.generic):
        return True
    if isinstance(value, dict):
        return all(isinstance(key, (str, int)) and _plain(item) for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return all(_plain(item) for item in value)
    return False


def _values(obj, names=None, *, exclude=()):
    fields = vars(obj)
    return {key: copy.deepcopy(fields[key]) for key in (fields if names is None else names)
            if key in fields and key not in exclude and _plain(fields[key])}


def _state_objects(env):
    # Keep fresh runtime objects and their model/sim references. Restore only
    # their data, including the histories used by controllers and observables.
    objects = {("fixture", name): fixture for name, fixture in env.fixtures.items()}
    for index, robot in enumerate(env.robots):
        key = ("robot", index)
        objects[key] = robot
        controller = robot.composite_controller
        objects[key + ("controller",)] = controller
        for name, part in controller.part_controllers.items():
            part_key = key + ("part", name)
            objects[part_key] = part
            for attr in _INTERPOLATORS:
                interpolator = getattr(part, attr, None)
                if interpolator is not None:
                    objects[part_key + (attr,)] = interpolator
        for name, gripper in robot.gripper.items():
            objects[key + ("gripper", name)] = gripper
        for attr in _HISTORY_BUFFERS:
            buffer = getattr(robot, attr, None)
            buffers = buffer if isinstance(buffer, dict) else {"": buffer}
            for name, value in buffers.items():
                if value is not None:
                    objects[key + ("buffer", attr, name)] = value
    return objects


def _integration_state(sim):
    import mujoco

    model, data = sim.model._model, sim.data._data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return state


def capture(env):
    return {
        # Use source XML. Re-exporting a compiled model can change continuation
        # despite matching the flattened state at entry.
        "source_xml": env.model.get_xml(),
        "episode_meta": copy.deepcopy(env.get_ep_meta()),
        "integration": _integration_state(env.sim),
        "env": _values(env, exclude=("_ep_meta", "_obs_cache")),
        "objects": {key: _values(obj) for key, obj in _state_objects(env).items()},
        "observables": {name: _values(obs, _OBSERVABLE_FIELDS)
                        for name, obs in env._observables.items()},
        "obs_cache": copy.deepcopy(env._obs_cache),
        "env_rng": copy.deepcopy(env.rng.bit_generator.state),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def restore(env, snapshot):
    """Install a saved state after the initial reset has recreated task objects."""
    import mujoco

    env.reset_from_xml_string(snapshot["source_xml"])
    sim = env.sim
    model, data = sim.model._model, sim.data._data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    saved = snapshot["integration"]
    if saved.shape != (mujoco.mj_stateSize(model, spec),):
        raise RuntimeError("Direct snapshot has a different MuJoCo state size")
    mujoco.mj_setState(model, data, saved, spec)
    sim.forward()
    # Forward reconstructs derived quantities and may update warm-start state.
    mujoco.mj_setState(model, data, saved, spec)
    if not np.array_equal(_integration_state(sim), saved):
        raise RuntimeError("Direct snapshot did not reproduce MuJoCo integration state")

    objects = _state_objects(env)
    if objects.keys() != snapshot["objects"].keys() or env._observables.keys() != snapshot["observables"].keys():
        raise RuntimeError("Direct snapshot has a different controller, fixture, or observation layout")
    vars(env).update(copy.deepcopy(snapshot["env"]))
    for key, state in snapshot["objects"].items():
        vars(objects[key]).update(copy.deepcopy(state))
    for key, state in snapshot["observables"].items():
        vars(env._observables[key]).update(copy.deepcopy(state))
    env._obs_cache = copy.deepcopy(snapshot["obs_cache"])
    env.rng.bit_generator.state = copy.deepcopy(snapshot["env_rng"])
    np.random.set_state(snapshot["numpy_rng"])
    random.setstate(snapshot["python_rng"])
