"""LIBERO codec: direct MuJoCo snapshots for mid-trajectory restore.

LIBERO scenes are deterministic from their BDDL initial state, so restore only
needs to replay the MuJoCo physics and Python-side controller/observable state
after a normal seeded reset — no XML reload or episode-meta injection required.
"""

import copy
import random

import numpy as np


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
    # Keep fresh runtime objects; restore only their data fields.
    # LIBERO uses standard robosuite single-controller robots (no composite_controller
    # or part_controllers); interpolators are attributes directly on robot.controller,
    # and gripper is a single object rather than a dict.
    objects = {}
    for index, robot in enumerate(env.robots):
        key = ("robot", index)
        objects[key] = robot
        controller = robot.controller
        objects[key + ("controller",)] = controller
        for attr in _INTERPOLATORS:
            interpolator = getattr(controller, attr, None)
            if interpolator is not None:
                objects[key + (attr,)] = interpolator
        gripper = getattr(robot, "gripper", None)
        if gripper is not None:
            if isinstance(gripper, dict):
                for name, g in gripper.items():
                    objects[key + ("gripper", name)] = g
            else:
                objects[key + ("gripper",)] = gripper
        for attr in _HISTORY_BUFFERS:
            buffer = getattr(robot, attr, None)
            buffers = buffer if isinstance(buffer, dict) else {"": buffer}
            for bname, value in buffers.items():
                if value is not None:
                    objects[key + ("buffer", attr, bname)] = value
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
        "integration": _integration_state(env.sim),
        "env": _values(env, exclude=("_obs_cache",)),
        "objects": {key: _values(obj) for key, obj in _state_objects(env).items()},
        "observables": {name: _values(obs, _OBSERVABLE_FIELDS)
                        for name, obs in env._observables.items()},
        "obs_cache": copy.deepcopy(env._obs_cache),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def restore(env, snapshot):
    """Install a saved state after the BDDL initial reset has recreated task objects."""
    import mujoco
    sim = env.sim
    model, data = sim.model._model, sim.data._data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    saved = snapshot["integration"]
    if saved.shape != (mujoco.mj_stateSize(model, spec),):
        raise RuntimeError("LIBERO snapshot has a different MuJoCo state size")
    mujoco.mj_setState(model, data, saved, spec)
    sim.forward()
    # forward() may update warm-start; re-apply to guarantee exact state.
    mujoco.mj_setState(model, data, saved, spec)
    if not np.array_equal(_integration_state(sim), saved):
        raise RuntimeError("LIBERO snapshot did not reproduce MuJoCo integration state")

    objects = _state_objects(env)
    if (objects.keys() != snapshot["objects"].keys()
            or env._observables.keys() != snapshot["observables"].keys()):
        raise RuntimeError("LIBERO snapshot has a different controller or observation layout")
    vars(env).update(copy.deepcopy(snapshot["env"]))
    for key, state in snapshot["objects"].items():
        vars(objects[key]).update(copy.deepcopy(state))
    for key, state in snapshot["observables"].items():
        vars(env._observables[key]).update(copy.deepcopy(state))
    env._obs_cache = copy.deepcopy(snapshot["obs_cache"])
    np.random.set_state(snapshot["numpy_rng"])
    random.setstate(snapshot["python_rng"])
