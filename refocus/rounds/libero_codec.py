"""LIBERO codec: scene identity from the initial state, replay-based restoration."""

from pathlib import Path

import numpy as np

from refocus import libero
from refocus.rounds.recorder import leaf
from refocus.rounds.visual_feedback import observation_views

CAMERAS = (
    ("agentview", ("agentview_image", "video.agentview_image")),
    ("wrist", ("robot0_eye_in_hand_image", "video.robot0_eye_in_hand_image")),
)
CAMERA_NAMES = tuple(label for label, _ in CAMERAS)
default_horizon = 1024
direct_restore = False
milestone_checks = libero.MILESTONE_CHECKS


def views(observation):
    return observation_views(observation, CAMERAS)


def seed(env, seed):
    """LIBERO resets from stored initial states; seed its placement generator."""
    np.random.seed(seed)


def reset_payload(kwargs, seed):
    """LIBERO rejects extra reset keywords; its scene comes from stored initial states."""
    return dict(kwargs)


def capture(env):
    """LIBERO rebuilds its scene on reset, so entries replay their actions instead."""
    return None


def restore(env, snapshot):
    raise RuntimeError('LIBERO entries are restored by action replay')


def scene_id(env):
    inner = leaf(env)
    return Path(inner.bddl_file_name).stem


def observation(env):
    return leaf(env)._get_observations()


def subtask_states(env):
    inner = leaf(env)
    return libero.get_subtask_states(inner, libero.task_name(inner))


def step_time(env):
    # Simulation step counter used by pacing and time-based subtasks.
    return int(getattr(leaf(env), 'timestep', 0))


def milestone_entities(env):
    return libero.milestone_entities(env)


def milestone_signals(env, checks):
    return libero.milestone_signals(env, checks)
