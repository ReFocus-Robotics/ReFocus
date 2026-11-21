"""Simulator-process hooks that record and restore Global training entries."""

import copy
import hashlib
import json
import os
import pickle
import random
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from refocus.reward import StageBalancedReward
from refocus.rounds.milestones import shares_of, validate_refinement
from refocus.rounds.protocol import entry_pattern, write_json
from refocus.rounds.visual_feedback import (encode_snapshot, extract_instruction,
                                            observation_views, save_episode_montage,
                                            should_capture)


ENTRY_SNAPSHOT_VERSION = 2


def leaf(env):
    for _ in range(8):
        if 'sim' in getattr(env, '__dict__', {}):
            return env
        if not hasattr(env, 'env'):
            break
        env = env.env
    raise RuntimeError('Cannot identify simulator leaf')


def canonical_scene_xml(xml):
    # MuJoCo's mesh cache may omit the MIME type inferred from an OBJ filename
    # when exporting the same compiled scene in a fresh process. Normalize only
    # this equivalent spelling; retain all geometry, assets and physical fields.
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return xml
    for mesh in root.findall('./asset/mesh'):
        if mesh.get('file', '').lower().endswith('.obj') and mesh.get('content_type') == 'model/obj':
            del mesh.attrib['content_type']
    return ET.tostring(root, encoding='unicode')


def state_equal(left, right):
    """Compare state values, not pickle bytes (set/dict order varies by process)."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return bool(left == right)


class Recorder:
    """Wrap one environment so Global rollouts bank entries and Local rollouts restore them.

    The codec supplies benchmark-specific observation access, scene identity, and
    optional direct state restore. Without direct restore, an entry records its
    executed actions and restoration replays them from the same initial state.
    """

    def __init__(self, env, reset, step, reward, codec):
        self.env, self.raw_reset, self.raw_step, self.reward = env, reset, step, reward
        self.codec = codec
        self.root = Path(os.environ['STRUCTRL_ROUNDS_ROOT'])
        self.eval = bool(getattr(env, '_round_eval', False))
        self.visual_enabled = self.eval and os.environ.get('STRUCTRL_ROUND_VISUAL_FEEDBACK', '1') != '0'
        self.task = env._round_task
        self.slot = env._round_slot
        self.replicas = env._round_replicas
        self.resets = 0
        self.active = False
        self.has_restored_meta = False
        self.rng = np.random.default_rng(1234 + self.slot)

    def seeded_reset(self, kwargs):
        # Seed through the codec so reset kwargs stay benchmark-native.
        self.codec.seed(self.env, int(self.seed))
        return self.raw_reset(copy.deepcopy(kwargs))

    def _sample_entry(self, window):
        folder = self.root / 'entries' / str(self.control['round']) / self.task / str(window['start_stage'])
        paths = sorted(folder.glob(entry_pattern(self.entry_attempt_id)))
        if not paths:
            raise RuntimeError('Selected window lost its entry bank')
        scenes = {}
        for path in paths:
            _, seed, steps, _ = path.stem.rsplit('_', 3)
            seed, steps = int(seed), int(steps)
            if 0 < steps < self.horizon:
                scenes.setdefault(seed, []).append((path, steps))
        if not scenes:
            raise RuntimeError('Selected window has no remaining task horizon')
        seed = int(self.rng.choice(sorted(scenes)))
        pool = scenes[seed]
        path, steps = pool[int(self.rng.integers(len(pool)))]
        entry = pickle.loads(path.read_bytes())
        if entry.get('snapshot_version') != ENTRY_SNAPSHOT_VERSION:
            raise RuntimeError('Unsupported entry format; start a new run to collect entries')
        if (entry['mode'] != 'train' or entry['task'] != self.task
                or entry['stage'] != window['start_stage'] or entry['round'] != self.control['round']
                or entry['seed'] != seed or entry['entry_steps'] != steps):
            raise RuntimeError('Invalid training entry provenance')
        if self.entry_attempt_id is not None and entry.get('phase_attempt_id') != self.entry_attempt_id:
            raise RuntimeError('Training entry came from a different Global attempt')
        return entry

    def _restore_entry(self, entry):
        data = entry['snapshot']
        if data is not None:
            inner = leaf(self.env)
            inner.set_ep_meta(copy.deepcopy(data['episode_meta']))
            try:
                _, info = self.seeded_reset(self.reset_kwargs)
                self.codec.restore(inner, data)
                saved = entry['scene']
                if self.codec.scene_id(self.env) != saved:
                    raise RuntimeError('Direct snapshot did not reproduce the training scene')
                self._install_reward_state(entry)
                self.has_restored_meta = True
                return self.codec.observation(self.env), info
            except Exception:
                inner.unset_ep_meta()
                raise
        # Replay path: recreate the initial state, then re-execute the prefix. The
        # wrapped step already advances the reward, so replay must not step it again.
        obs, info = self.seeded_reset(self.reset_kwargs)
        if self.codec.scene_id(self.env) != entry['scene']:
            raise RuntimeError('Replayed entry did not reproduce the training scene')
        result = None
        for action in entry['actions']:
            result = self.raw_step(copy.deepcopy(action))
        rw = self.reward()
        if not state_equal(rw.stage_idx, entry['reward_state']['stage_idx']):
            raise RuntimeError('Replayed prefix did not reproduce reward progress')
        self._install_reward_state(entry)
        if not state_equal(rw.stage_idx, entry['reward_state']['stage_idx']):
            raise RuntimeError('Replayed prefix did not reproduce reward progress')
        return result[0], result[-1] if isinstance(result[-1], dict) else info

    def _install_reward_state(self, entry):
        rw = self.reward()
        reward_state = entry['reward_state']
        if (type(rw).__name__ != entry['reward_type']
                or rw.stages != reward_state['stages']
                or reward_state['stage_idx'] != entry['stage']):
            raise RuntimeError('Entry has a different progress reward state')
        vars(rw).clear()
        vars(rw).update(copy.deepcopy(reward_state))

    def reset(self, kwargs):
        self.finish()
        if self.has_restored_meta and hasattr(leaf(self.env), 'unset_ep_meta'):
            leaf(self.env).unset_ep_meta()
            self.has_restored_meta = False
        self.control = json.loads((self.root / 'control.json').read_text())
        variant = self.control.get('reward_variant')
        if variant is not None and variant != os.environ.get('STRUCTRL_REWARD_VARIANT'):
            raise RuntimeError('Ray worker and coordinator use different reward variants')
        self.phase_attempt_id = self.control.get('phase_attempt_id')
        self.entry_attempt_id = self.control.get('entry_attempt_id')
        self.refined = False
        self.milestone_fallback = None
        self.closed = False
        self.entries = {}
        self.steps = 0
        self.success = self.local = self.cut = self.complete = self.active = False
        self.entry_steps = 0
        self.restoration_seconds = 0.0
        self.action_prefix = []
        self.chunk = int(self.control.get('chunk', 50))
        self.horizon = int(self.control.get('horizon', self.codec.default_horizon))
        self.remaining = self.horizon
        self.sample_mode = self.requested_mode = 'global'
        self.entry_update = None
        self.visual_snapshots = []
        self.task_instruction = None
        self.window = window = self.control.get('windows', {}).get(self.task)
        self.fallback = None
        if not self.eval and self.control['mode'] == 'local':
            weights = self.control.get('local_mix', [.5, .25, .25])
            third_mode = 'prefix' if variant == 'refocus' else 'global'
            self.requested_mode = str(self.rng.choice(['window', 'suffix', third_mode], p=weights))
            if window and self.requested_mode != 'global':
                self.local = True
                self.sample_mode = self.requested_mode
            elif not window and self.requested_mode != 'global':
                self.fallback = 'no_eligible_window'

        entry = None
        if self.local and self.sample_mode in ('window', 'suffix'):
            restoration_started = time.perf_counter()
            entry = self._sample_entry(window)
        if self.eval:
            # Preserve the benchmark evaluator's original reset-state selection.
            self.reset_kwargs = copy.deepcopy(kwargs)
            self.seed = kwargs.get('seed')
            obs, info = self.raw_reset(self.reset_kwargs)
            self.task_instruction = extract_instruction(obs)
        else:
            seed = 100 + self.slot + (self.resets + 10000 * self.control['update']) * self.replicas
            self.resets += 1
            self.reset_kwargs = copy.deepcopy(entry['reset_kwargs']) if entry else dict(kwargs)
            self.seed = seed
            if entry:
                obs, info = self._restore_entry(entry)
                self.restoration_seconds = time.perf_counter() - restoration_started
                self.entry_steps = entry['entry_steps']
                self.remaining -= self.entry_steps
                self.entry_update = entry['source_update']
            else:
                obs, info = self.seeded_reset(self.reset_kwargs)
        rw = self.reward()
        if rw is None:
            raise RuntimeError('Rounds require a working dense reward')
        self.signature = hashlib.sha256(json.dumps(rw.stages, sort_keys=True).encode()).hexdigest()[:16]
        if (window and window.get('stage_signature', self.signature) != self.signature
                or entry and entry['stage_signature'] != self.signature):
            raise RuntimeError('Window or entry stage schema differs from the current task')
        if self.visual_enabled:
            self._capture_visual(obs, 0, int(rw.stage_idx))
        refinement = window.get('milestone_refinement') if self.local and variant == 'refocus' else None
        if refinement is not None:
            inner = leaf(self.env)
            try:
                validate_refinement(refinement, rw.stages[window['end_stage']],
                                    self.codec.milestone_entities(inner), self.codec.milestone_checks)
            except ValueError as exc:
                self.milestone_fallback = str(exc)
            else:
                target = refinement['target']
                checks = refinement['milestones']
                rw.configure_window(window['start_stage'], window['end_stage'], target,
                                    shares_of(refinement), self.codec.milestone_signals(inner, checks),
                                    checks=checks)
                self.refined = rw.refined_target is not None
        self.active = True
        return obs, info

    def step(self, action):
        result = self.raw_step(action)
        obs, rew, term, trunc, info = result
        self.steps += 1
        rw = self.reward()
        if rw is None:
            raise RuntimeError('Rounds require a working dense reward')
        if not self.eval and not self.local:
            self.action_prefix.append(copy.deepcopy(action))
        if self.visual_enabled and should_capture(self.steps, self.horizon):
            self._capture_visual(obs, self.steps, int(rw.stage_idx))
        self.success |= bool(info.get('success', False) or info.get('dense_reward_latched_outcome', 0))
        stage = rw.stage_idx
        if (not self.closed and not self.eval and self.control['mode'] == 'global'
                and self.steps % self.chunk == 0 and self.steps < self.horizon
                and not (self.success or term or trunc)):
            if stage < rw.K and stage not in self.entries:
                folder = self.root / 'entries' / str(self.control['round']) / self.task / str(stage)
                if sum(1 for _ in folder.glob(entry_pattern(self.phase_attempt_id, self.slot))) < 4:
                    self.entries[stage] = dict(
                        mode='train', task=self.task, stage=stage, round=self.control['round'],
                        source_update=self.control['update'], stage_signature=self.signature,
                        phase_attempt_id=self.phase_attempt_id, seed=self.seed,
                        reset_kwargs=copy.deepcopy(self.reset_kwargs),
                        entry_steps=self.steps, actions=copy.deepcopy(self.action_prefix),
                        scene=self.codec.scene_id(self.env), snapshot=self.codec.capture(self.env),
                        reward_type=type(rw).__name__, reward_state=copy.deepcopy(vars(rw)),
                        snapshot_version=ENTRY_SNAPSHOT_VERSION,
                    )
        if self.local and self.steps % self.chunk == 0:
            self.cut |= self.steps >= self.remaining
            if self.sample_mode in ('window', 'prefix'):
                self.cut |= stage > self.window['end_stage']
            if self.sample_mode == 'window':
                self.cut |= self.steps >= self.window['max_steps']
        info['rounds_truncated'] = bool(self.cut and not (self.success or term))
        if self.success or term or self.cut or trunc or self.steps >= self.horizon:
            self.complete = True
            self.finish()
        return obs, rew, term, trunc, info

    def _capture_visual(self, obs, control_step, stage_index):
        views = self.codec.views(obs)
        expected = set(self.codec.camera_names)
        if set(views) != expected:
            raise RuntimeError(f'Visual feedback requires the {", ".join(sorted(expected))} policy views')
        self.visual_snapshots.append(encode_snapshot(views, control_step, stage_index))

    def finish(self):
        if not self.active or self.closed:
            return
        self.closed = True
        rw = self.reward()
        if rw is None or not self.steps:
            return
        token = uuid.uuid4().hex
        row = dict(task=self.task, mode='eval' if self.eval else 'train',
                   round=self.control['round'], update=json.loads((self.root / 'control.json').read_text())['update'] + (1 if self.eval else 0),
                   stage=rw.stage_idx, stages=[list(s) for s in rw.stages], success=self.success,
                   seed=self.seed, local=self.local, window_steps=self.steps,
                   entry_steps=self.entry_steps, restoration_seconds=self.restoration_seconds,
                   fallback=self.fallback,
                   stage_signature=self.signature, complete=self.complete,
                   requested_mode=self.requested_mode, sample_mode=self.sample_mode,
                   remaining_horizon=self.remaining, entry_update=self.entry_update,
                   window_complete=bool(self.local and rw.stage_idx > self.window['end_stage']),
                   reward_scope='milestone_refinement' if self.refined else 'original_task')
        if self.eval:
            row['milestone_entities'] = self.codec.milestone_entities(leaf(self.env))
        if self.milestone_fallback is not None:
            row['milestone_fallback'] = self.milestone_fallback
        if self.phase_attempt_id is not None:
            row['phase_attempt_id'] = self.phase_attempt_id
        if self.visual_enabled and self.visual_snapshots:
            row['visual_montage'] = save_episode_montage(
                self.root, row['round'], self.task, row['update'], self.slot,
                self.visual_snapshots, self.success, control_count=self.steps)
            row['visual_snapshot_count'] = len(self.visual_snapshots)
            if self.task_instruction:
                row['task_instruction'] = self.task_instruction
        row['slot'] = self.slot
        write_json(self.root / 'traces' / f'{token}.json', row)
        for stage, entry in self.entries.items():
            folder = self.root / 'entries' / str(self.control['round']) / self.task / str(stage)
            folder.mkdir(parents=True, exist_ok=True)
            # Bounded, per-worker bank; no cross-worker check/write race.
            if len(list(folder.glob(entry_pattern(self.phase_attempt_id, self.slot)))) >= 4:
                continue
            prefix = f'{self.slot}_{self.phase_attempt_id}' if self.phase_attempt_id else str(self.slot)
            target = folder / f"{prefix}_{entry['seed']}_{entry['entry_steps']}_{token}.pkl"
            tmp = target.with_suffix('.tmp')
            tmp.write_bytes(pickle.dumps(entry))
            os.replace(tmp, target)
