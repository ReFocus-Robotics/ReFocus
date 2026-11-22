"""LIBERO codec: direct MuJoCo snapshots for mid-trajectory restore."""

from pathlib import Path

import numpy as np

from refocus import libero
from refocus.rounds import libero_snapshot
from refocus.rounds.recorder import leaf
from refocus.rounds.visual_feedback import observation_views

CAMERAS = (
    ("agentview", ("agentview_image", "video.agentview_image")),
    ("wrist", ("robot0_eye_in_hand_image", "video.robot0_eye_in_hand_image")),
)
CAMERA_NAMES = tuple(label for label, _ in CAMERAS)
default_horizon = 1024
direct_restore = True
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
    return libero_snapshot.capture(leaf(env))


def restore(env, snapshot):
    # env is already the leaf (recorder passes leaf(self.env))
    libero_snapshot.restore(env, snapshot)


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
