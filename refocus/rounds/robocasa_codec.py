"""RoboCasa365 codec: direct MuJoCo snapshots and simulator milestone checks."""

from refocus import robocasa
from refocus.rounds import robocasa_snapshot
from refocus.rounds.recorder import canonical_scene_xml, leaf

import numpy as np

CAMERAS = (
    ("agentview_left", ("video.robot0_agentview_left", "robot0_agentview_left", "video.res256_image_side_0", "main_images")),
    ("agentview_right", ("video.robot0_agentview_right", "robot0_agentview_right", "video.res256_image_side_1", "extra_view_images")),
    ("wrist", ("video.robot0_eye_in_hand", "robot0_eye_in_hand", "video.res256_image_wrist_0", "wrist_images")),
)
CAMERA_NAMES = tuple(label for label, _ in CAMERAS)
default_horizon = 2900
direct_restore = True
milestone_checks = robocasa.MILESTONE_CHECKS


def views(observation):
    from refocus.rounds.visual_feedback import observation_views
    return observation_views(observation, CAMERAS)


def capture(env):
    return robocasa_snapshot.capture(env)


def restore(env, snapshot):
    robocasa_snapshot.restore(env, snapshot)
    if scene_id(env) != _expected_scene:
        raise RuntimeError('Direct snapshot did not reproduce the MuJoCo state')


_expected_scene = None


def scene_id(env):
    inner = leaf(env)
    xml = canonical_scene_xml(inner.sim.model.get_xml())
    state = np.array(inner.sim.get_state().flatten(), copy=True)
    return (xml, state)


def observation(env):
    wrapper = env.env
    return wrapper.get_groot_observation(wrapper.get_basic_observation(leaf(env)._get_observations()))


def subtask_states(env):
    return robocasa.get_subtask_states(env)


def step_time(env):
    """RoboCasa dense rewards ignore the step clock."""
    return None


def milestone_entities(env):
    return robocasa.milestone_entities(env)


def milestone_signals(env, checks):
    return robocasa.milestone_signals(env, checks)
