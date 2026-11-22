import copy
import io
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from refocus.reward import StageBalancedReward
from refocus.rounds import libero_codec, robocasa_codec
from refocus.rounds.benchmarks import BENCHMARKS
from refocus.rounds.protocol import phase_plan, validate_windows, write_json
from refocus.rounds.recorder import ENTRY_SNAPSHOT_VERSION, Recorder, canonical_scene_xml, state_equal
from refocus.rounds.scheduler import _save_selection, diagnostics, history_for_task, select
from refocus.rounds.visual_feedback import (build_montage, encode_snapshot, observation_views,
                                            should_capture)
from scripts.install_round_hooks import transform_worker


class Sim:
    def __init__(self):
        self.value = 0
        self.model = SimpleNamespace(get_xml=lambda: 'fixed_xml')

    def get_state(self):
        return np.array([self.value])


class Toy:
    def __init__(self, evaluation=False, slot=0):
        self.sim = Sim()
        self.env = SimpleNamespace(get_basic_observation=lambda value: value,
                                   get_groot_observation=lambda value: value)
        self.episode_meta = {}
        self.seed_value = 0
        self._round_eval = evaluation
        self._round_task = 'Toy'
        self._round_slot = slot
        self._round_replicas = 5
        self.rw = SimpleNamespace(stage_idx=0, K=3, stages=[['a'], ['b'], ['c']], ticks=0)

    def set_ep_meta(self, data):
        self.episode_meta = data

    def unset_ep_meta(self):
        self.episode_meta = {}

    def _get_observations(self):
        return self.sim.value

    def reset(self, kwargs):
        self.sim.value = kwargs.get('seed', kwargs.get('reset_state_id', self.seed_value))
        self.rw.stage_idx = 0
        self.rw.ticks = 0
        return self.sim.value, {}

    def step(self, action):
        self.sim.value += action
        self.rw.ticks += 1
        self.rw.stage_idx = min(self.rw.ticks // 100, 2)
        return self.sim.value, 1., False, False, {'success': False}


CAMERAS = (
    ('agentview_left', ('video.res256_image_side_0',)),
    ('agentview_right', ('video.res256_image_side_1',)),
    ('wrist', ('video.res256_image_wrist_0',)),
)


class ToyCodec:
    """Direct-restore codec mirroring the RoboCasa interface."""

    default_horizon = 2900
    direct_restore = True
    milestone_checks = {'object_grasped': ('grasp', None)}

    def scene_id(self, env):
        return env.sim.model.get_xml()

    def capture(self, env):
        return {'episode_meta': {'scene': env.sim.value}, 'value': env.sim.value}

    def restore(self, env, saved):
        env.sim.value = saved['value']

    def observation(self, env):
        outer = env.env
        return outer.get_groot_observation(outer.get_basic_observation(env._get_observations()))

    def seed(self, env, seed):
        env.seed_value = seed

    def reset_payload(self, kwargs, seed):
        return dict(kwargs, seed=seed)

    def views(self, obs):
        return observation_views(obs, CAMERAS)

    def subtask_states(self, env):
        return {'progress': {}}

    def step_time(self, env):
        return None

    def milestone_entities(self, env):
        return {'objects': sorted(getattr(env, 'objects', {})), 'fixtures': []}

    def milestone_signals(self, env, checks):
        raise AssertionError('patched per test')


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old = os.environ.get('STRUCTRL_ROUNDS_ROOT')
        os.environ['STRUCTRL_ROUNDS_ROOT'] = str(self.root)
        os.environ.pop('STRUCTRL_REWARD_VARIANT', None)
        write_json(self.root / 'control.json', dict(mode='global', round=0, update=4, windows={}))

    def tearDown(self):
        if self.old is None:
            os.environ.pop('STRUCTRL_ROUNDS_ROOT', None)
        else:
            os.environ['STRUCTRL_ROUNDS_ROOT'] = self.old
        os.environ.pop('STRUCTRL_REWARD_VARIANT', None)
        self.tmp.cleanup()

    def recorder(self, evaluation=False, slot=0, codec=None):
        env = Toy(evaluation, slot)
        return Recorder(env, env.reset, env.step, lambda: env.rw, codec or ToyCodec())

    def test_reward_state_semantics(self):
        self.assertTrue(state_equal({'a': 1, 'b': {'x', 'y'}}, {'b': {'y', 'x'}, 'a': 1}))
        self.assertFalse(state_equal({'ticks': 50}, {'ticks': 51}))
        self.assertFalse(state_equal({'latched': {'a': True}}, {'latched': {'a': False}}))

    def test_benchmark_profiles_match_their_configs(self):
        import yaml

        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        rc = yaml.safe_load((config_dir / f'{BENCHMARKS["robocasa365"].default_config}.yaml').read_text())
        self.assertEqual(rc['cluster']['num_nodes'], 2)
        self.assertEqual(rc['cluster']['component_placement'], {'actor': '0-15', 'env,rollout': 'all'})
        self.assertEqual(rc['env']['train']['total_num_envs'], 256)
        self.assertEqual(rc['algorithm']['rollout_epoch'], 2)
        self.assertEqual(rc['actor']['micro_batch_size'], 32)
        self.assertEqual(rc['actor']['global_batch_size'], 512)
        self.assertEqual(rc['actor']['micro_batch_size'] * 16, 512)
        self.assertEqual(rc['env']['train']['total_num_envs'] * rc['algorithm']['rollout_epoch'], 512)
        self.assertEqual(rc['env']['eval']['total_num_envs'], 80)
        self.assertEqual(rc['runner']['val_check_interval'], 5)
        self.assertEqual(rc['runner']['max_steps'], phase_plan()[-1]['stop'])

        lb = yaml.safe_load((config_dir / f'{BENCHMARKS["libero_10"].default_config}.yaml').read_text())
        self.assertEqual(BENCHMARKS['libero_10'].chunk, 10)
        self.assertEqual(BENCHMARKS['libero_10'].horizon, 1024)
        self.assertEqual(lb['actor']['model']['num_action_chunks'], 10)
        self.assertEqual(lb['actor']['model']['num_steps'], 10)
        self.assertEqual(lb['env']['train']['max_episode_steps'], 1024)
        self.assertEqual(lb['env']['train']['max_steps_per_rollout_epoch'] % 10, 0)
        self.assertEqual(lb['env']['train']['total_num_envs'] * lb['algorithm']['rollout_epoch'],
                         lb['actor']['global_batch_size'])
        self.assertTrue(lb['env']['train']['dense_reward']['enabled'])
        self.assertTrue(lb['env']['eval']['use_ordered_reset_state_ids'])
        self.assertEqual(lb['runner']['logger']['experiment_name'], BENCHMARKS['libero_10'].experiment_name)

    def test_libero_stage_table_covers_every_goal_conjunct(self):
        from refocus import libero

        self.assertEqual(len(libero.TASKS), 10)
        for task in libero.TASKS:
            stages = libero.TASK_STAGES[task]
            self.assertTrue(all(stage for stage in stages), task)
            names = [name for stage in stages for name in stage]
            self.assertEqual(len(names), len(set(names)), task)
            self.assertEqual(set(names), set(libero.SUBTASK_PREDICATES[task]), task)

    def test_libero_subtask_states_use_the_bddl_predicate(self):
        from refocus import libero

        task = 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it'
        inner = SimpleNamespace(_eval_predicate=lambda predicate: predicate == ['turnon', 'flat_stove_1'])
        self.assertEqual(libero.get_subtask_states(inner, task),
                         {'progress': {'stove_on': True, 'moka_pot_on_stove': False}})

    def test_libero_refinement_checks_bind_to_regions(self):
        from refocus import libero
        from refocus.rounds.milestones import validate_refinement

        env = SimpleNamespace(objects_dict={'moka_pot_1': 1},
                              object_sites_dict={'flat_stove_1_cook_region': 1},
                              fixtures_dict={'flat_stove_1': 1})
        entities = libero.milestone_entities(env)
        self.assertIn('flat_stove_1_cook_region', entities['regions'])
        refinement = dict(target='moka_pot_on_stove', final_share=0.6, milestones=[
            dict(name='pot_on_cook_region', check='object_on', object='moka_pot_1',
                 reference='flat_stove_1_cook_region', share=0.4)])
        self.assertEqual(validate_refinement(refinement, ['moka_pot_on_stove'], entities,
                                             libero.MILESTONE_CHECKS), refinement)
        bad = copy.deepcopy(refinement)
        bad['milestones'][0]['reference'] = 'basket_1_contain_region'
        with self.assertRaises(ValueError):
            validate_refinement(bad, ['moka_pot_on_stove'], entities, libero.MILESTONE_CHECKS)

    def test_visual_capture_uses_the_recorded_camera_views(self):
        base = np.zeros((24, 32, 3), dtype=np.uint8)
        obs = {'video.res256_image_side_0': base,
               'video.res256_image_side_1': base + 30,
               'video.res256_image_wrist_0': base + 60}
        self.assertEqual(set(observation_views(obs, CAMERAS)), {'agentview_left', 'agentview_right', 'wrist'})
        self.assertEqual(set(robocasa_codec.views(obs)), {'agentview_left', 'agentview_right', 'wrist'})
        self.assertEqual(set(libero_codec.views({'agentview_image': base,
                                                 'robot0_eye_in_hand_image': base + 40})),
                         {'agentview', 'wrist'})
        self.assertTrue(should_capture(0, 2900))
        self.assertTrue(should_capture(193, 2900))
        self.assertFalse(should_capture(2, 2900))
        self.assertTrue(should_capture(2900, 2900))
        snapshots = [encode_snapshot(observation_views(obs, CAMERAS), step * 50, step // 3)
                     for step in range(1, 17)]
        montage = build_montage(snapshots, task='Toy', update=50, success=False,
                                episode_slot=0, control_count=2900)
        self.assertTrue(montage.startswith(b'\xff\xd8'))
        from PIL import Image
        self.assertEqual(Image.open(io.BytesIO(montage)).size, (1024, 1140))

    def test_visual_scheduler_sends_25_montages_and_never_sends_stage_sha(self):
        write_json(self.root / 'tasks.json', ['Toy'])
        image = build_montage([], task='Toy', update=25, success=False)
        signature = 'a' * 64
        for update in (5, 10, 15, 20, 25):
            folder = self.root / 'visual_feedback' / 'round_0' / f'update_{update:03d}' / 'Toy'
            folder.mkdir(parents=True, exist_ok=True)
            for slot in range(5):
                relative = (Path('visual_feedback') / 'round_0' / f'update_{update:03d}' / 'Toy' / f'episode_{slot:04d}.jpg')
                (self.root / relative).write_bytes(image)
                write_json(self.root / 'traces' / f'{update}_{slot}.json', dict(
                    task='Toy', mode='eval', round=0, update=update, complete=True,
                    success=False, stage=0, stages=[['stage predicate']], stage_signature=signature,
                    slot=slot, visual_montage=relative.as_posix(), task_instruction='Place the item.'))
        folder = self.root / 'entries' / '0' / 'Toy' / '0'
        folder.mkdir(parents=True)
        (folder / '0_1_50_token.pkl').write_bytes(b'entry')

        def response(payload):
            item = Mock()
            item.__enter__ = Mock(return_value=item)
            item.__exit__ = Mock(return_value=False)
            item.read = Mock(return_value=json.dumps({'choices': [{'message': {'content': json.dumps(payload)}}]}).encode())
            return item
        opener = Mock()
        opener.open.side_effect = [
            response({'task': 'Toy', 'visible_progress': 'reaches the first subtask',
                      'recurring_stall_stage': 1, 'evidence': ['episode images'], 'uncertainty': 'limited sample'}),
            response({'windows': {'Toy': {'start_stage': 0, 'end_stage': 0, 'max_steps': 100,
                                          'reason': 'repeated stall', 'milestone_refinement': None}}})]
        with patch.dict(os.environ, {'SCHEDULER_API_KEY': 'test-secret'}), \
                patch('refocus.rounds.scheduler.urllib.request.build_opener', return_value=opener), \
                patch('refocus.rounds.scheduler.time.sleep'):
            selected = select(self.root, 0, [5, 10, 15, 20, 25], benchmark=BENCHMARKS['robocasa365'],
                              base_url='https://api.example/v1', model='test-model', evidence='visual')
        first_payload = json.loads(opener.open.call_args_list[0].args[0].data)
        self.assertEqual(sum(part.get('type') == 'image_url' for part in first_payload['messages'][0]['content']), 25)
        self.assertEqual(selected['Toy']['start_stage'], 0)
        prompt_manifest = (self.root / 'visual_feedback/round_0/scheduler_request_Toy.json').read_text()
        window_manifest = (self.root / 'scheduler_request_round0.json').read_text()
        self.assertNotIn(signature, prompt_manifest + window_manifest)
        self.assertNotIn('test-secret', prompt_manifest + window_manifest)

    def test_scene_xml_inferred_mime(self):
        a = '<mujoco><asset><mesh file="a.obj" content_type="model/obj" scale="1 1 1" /></asset></mujoco>'
        b = a.replace(' content_type="model/obj"', '')
        self.assertEqual(canonical_scene_xml(a), canonical_scene_xml(b))
        self.assertNotEqual(canonical_scene_xml(a), canonical_scene_xml(b.replace('1 1 1', '2 1 1')))
        self.assertNotEqual(canonical_scene_xml(a), canonical_scene_xml(a.replace('model/obj', 'model/stl')))

    def test_schedule(self):
        self.assertEqual([(p['start'], p['stop']) for p in phase_plan()],
                         [(0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 150)])
        self.assertEqual([(p['start'], p['stop']) for p in phase_plan(1, 5)], [(0, 5), (5, 10)])

    def test_window_validation(self):
        candidates = {'Toy': dict(available_starts=[1], stages=['a', 'b', 'c'])}
        window = {'Toy': dict(start_stage=1, end_stage=2, max_steps=100, reason='failure')}
        self.assertEqual(validate_windows(window, candidates), window)
        for key, value in [('start_stage', 0), ('end_stage', 3), ('max_steps', 51)]:
            bad = copy.deepcopy(window)
            bad['Toy'][key] = value
            with self.assertRaises(ValueError):
                validate_windows(bad, candidates)
        libero_window = {'Toy': dict(start_stage=1, end_stage=2, max_steps=105, reason='failure')}
        with self.assertRaises(ValueError):
            validate_windows(libero_window, candidates, chunk=10)

    def test_eval_preserves_reset_state_and_writes_no_entries(self):
        with patch.dict(os.environ, {'STRUCTRL_ROUND_VISUAL_FEEDBACK': '0'}):
            for slot in range(5):
                r = self.recorder(True, slot)
                r.reset({'reset_state_id': 100 + slot})
                r.step(1)
                r.finish()
            rows = [json.loads(p.read_text()) for p in (self.root / 'traces').glob('*')]
            self.assertEqual({r['seed'] for r in rows}, {None})
            self.assertEqual({self.recorder(True, slot).reset({'reset_state_id': 200 + slot})[0]
                              for slot in range(5)}, set(range(200, 205)))
        self.assertEqual({r['update'] for r in rows}, {5})
        self.assertFalse((self.root / 'entries').exists())

    def make_entry(self):
        recorder = self.recorder()
        recorder.reset({})
        for _ in range(100):
            recorder.step(1)
        recorder.finish()
        write_json(self.root / 'control.json', dict(mode='local', round=0, update=25, local_mix=[1, 0, 0],
                                                   windows={'Toy': dict(start_stage=1, end_stage=2, max_steps=50,
                                                                        reason='failure')}))

    def test_restore_and_truncation(self):
        self.make_entry()
        recorder = self.recorder()
        with patch.object(recorder, 'raw_step', wraps=recorder.raw_step) as step, \
                patch.object(recorder, 'raw_reset', wraps=recorder.raw_reset) as reset:
            obs, _ = recorder.reset({})
            step.assert_not_called()
            reset.assert_called_once_with({'seed': 200100})
        self.assertEqual(obs, 200200)
        self.assertEqual(recorder.entry_steps, 100)
        self.assertEqual(recorder.reward().ticks, 100)
        self.assertEqual(recorder.env.episode_meta, {'scene': 200200})
        for _ in range(49):
            self.assertFalse(recorder.step(1)[4]['rounds_truncated'])
        result = recorder.step(1)
        self.assertTrue(result[4]['rounds_truncated'])
        self.assertFalse(result[2])
        self.assertEqual(result[1], 1.)
        rows = [json.loads(p.read_text()) for p in (self.root / 'traces').glob('*')]
        local = next(row for row in rows if row['local'])
        self.assertEqual(local['window_steps'], 50)
        self.assertEqual(local['entry_steps'], 100)
        self.assertGreater(local['restoration_seconds'], 0)
        self.assertFalse(local['success'])
        write_json(self.root / 'control.json', dict(mode='global', round=1, update=50, windows={}))
        recorder.reset({})
        self.assertEqual(recorder.env.episode_meta, {})

    def test_entry_bank_stores_actions_and_snapshot(self):
        self.make_entry()
        path = next((self.root / 'entries/0/Toy/1').glob('*.pkl'))
        data = pickle.loads(path.read_bytes())
        self.assertEqual(data['entry_steps'], 100)
        self.assertEqual(data['snapshot_version'], ENTRY_SNAPSHOT_VERSION)
        self.assertEqual(len(data['actions']), 100)
        self.assertEqual(data['snapshot']['value'], 200200)
        self.assertEqual(data['scene'], 'fixed_xml')

    def test_entry_replay_restores_when_the_codec_has_no_snapshot(self):
        class ReplayCodec(ToyCodec):
            direct_restore = False

            def capture(self, env):
                return None

        def record(env, reset_kwargs=None):
            probe = dict(reward=env.rw)
            recorder = Recorder(env, env.reset, env.step, lambda: probe['reward'], ReplayCodec())
            return recorder

        env = Toy()
        global_recorder = record(env)
        global_recorder.reset({})
        for _ in range(100):
            global_recorder.step(1)
        global_recorder.finish()

        write_json(self.root / 'control.json', dict(
            mode='local', round=0, update=25, local_mix=[1, 0, 0],
            windows={'Toy': dict(start_stage=1, end_stage=2, max_steps=50, reason='failure')}))

        env2 = Toy()
        replay = record(env2)
        replay.reset({})
        self.assertEqual(replay.entry_steps, 100)
        self.assertEqual(env2.rw.stage_idx, 1)

    def test_refocus_prefix_starts_normally_and_ends_at_window_target(self):
        self.make_entry()
        path = self.root / 'control.json'
        control = json.loads(path.read_text())
        control.update(local_mix=[0, 0, 1], reward_variant='refocus')
        control['windows']['Toy']['end_stage'] = 1
        write_json(path, control)
        with patch.dict(os.environ, {'STRUCTRL_REWARD_VARIANT': 'refocus'}):
            recorder = self.recorder()
            recorder.reset({})
            self.assertEqual(recorder.sample_mode, 'prefix')
            self.assertEqual(recorder.entry_steps, 0)
            for _ in range(199):
                self.assertFalse(recorder.step(1)[4]['rounds_truncated'])
            self.assertTrue(recorder.step(1)[4]['rounds_truncated'])
            self.assertEqual(recorder.steps, 200)

    def test_restore_mismatch_fails(self):
        self.make_entry()
        recorder = self.recorder()
        recorder.env.sim.model.get_xml = lambda: 'different'
        with self.assertRaisesRegex(RuntimeError, 'scene'):
            recorder.reset({})

    def test_incomplete_diagnostics_fails(self):
        write_json(self.root / 'tasks.json', ['Toy'])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            diagnostics(self.root, 0, 0)

    def test_short_smoke_checkpoint_list(self):
        write_json(self.root / 'tasks.json', ['Toy'])
        for step in (1, 2, 3):
            for episode in range(5):
                write_json(self.root / 'traces' / f'{step}_{episode}.json', dict(
                    task='Toy', mode='eval', round=0, update=step, complete=True,
                    success=False, stage=0, stages=[['a'], ['b'], ['c']], stage_signature='toy'))
        folder = self.root / 'entries/0/Toy/0'
        folder.mkdir(parents=True)
        (folder / 'entry.pkl').write_bytes(b'index-only')
        self.assertEqual(set(diagnostics(self.root, 0, [1, 2, 3])), {'Toy'})
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            diagnostics(self.root, 0, [1, 2, 4])

    def test_scheduler_settings_are_explicit_and_key_is_not_persisted(self):
        write_json(self.root / 'tasks.json', ['Toy'])
        for episode in range(5):
            write_json(self.root / 'traces' / f'eval_{episode}.json', dict(
                task='Toy', mode='eval', round=0, update=5, complete=True,
                success=False, stage=0, stages=[['a'], ['b'], ['c']], stage_signature='schema',
                milestone_entities={'objects': ['item'], 'fixtures': []}))
        folder = self.root / 'entries/0/Toy/0'
        folder.mkdir(parents=True)
        (folder / '0_1_50_token.pkl').write_bytes(b'test-entry')
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        refinement = dict(target='b', final_share=0.75, milestones=[
            dict(name='pick', check='object_grasped', object='item', reference=None, share=0.25)])
        response.read = Mock(return_value=json.dumps({'choices': [{'message': {'content': json.dumps({
            'windows': {'Toy': dict(start_stage=0, end_stage=1, max_steps=50, reason='recurring failure',
                                    milestone_refinement=refinement)}})}}]}).encode())
        opener = Mock()
        opener.open.return_value = response
        benchmark = BENCHMARKS['robocasa365']
        with patch.dict(os.environ, {'SCHEDULER_API_KEY': 'test-secret'}), patch(
                'refocus.rounds.scheduler.urllib.request.build_opener', return_value=opener):
            windows = select(self.root, 0, [5], benchmark=benchmark, base_url='https://api.example/v1',
                             model='test-model', reasoning_effort='low', evidence='text')
        request = opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, 'https://api.example/v1/chat/completions')
        self.assertEqual(payload['model'], 'test-model')
        self.assertEqual(payload['reasoning_effort'], 'low')
        self.assertEqual(windows['Toy']['start_stage'], 0)
        self.assertEqual(windows['Toy']['milestone_refinement'], refinement)
        proposal = json.loads((self.root / 'milestone_proposals_round0.json').read_text())['Toy']
        self.assertEqual(proposal['status'], 'accepted')
        self.assertIn('restored entry state', payload['messages'][0]['content'][0]['text'])
        self.assertNotIn('test-secret', (self.root / 'scheduler_request_round0.json').read_text())
        self.assertNotIn('test-secret', (self.root / 'scheduler_response_round0.json').read_text())
        self.assertEqual(payload['response_format']['type'], 'json_schema')

        # The selected window is handed to the worker through control.json.
        windows['Toy'].pop('stage_signature')
        write_json(self.root / 'control.json', dict(mode='local', round=0, update=25,
                                                    windows=windows, local_mix=[0, 0, 1], reward_variant='refocus'))
        env = Toy()
        env.objects = {'item': object()}
        env.rw = StageBalancedReward([['a'], ['b'], ['c']])
        with patch.dict(os.environ, {'STRUCTRL_REWARD_VARIANT': 'refocus'}), patch.object(
                ToyCodec, 'milestone_signals', return_value={'pick': False}) as read_signals:
            recorder = Recorder(env, env.reset, env.step, lambda: env.rw, ToyCodec())
            recorder.reset({})
            self.assertTrue(recorder.refined)
            read_signals.assert_called_once_with(env, refinement['milestones'])
            env.rw.step({'progress': {'a': True}}, False, signals={'pick': False})
            report = env.rw.step({'progress': {}}, False, signals={'pick': True})
            self.assertAlmostEqual(report['r_event'], (2 / 3) * 0.25)

            env.objects = {}
            env.rw.reset()
            recorder.reset({})
            self.assertFalse(recorder.refined)
            self.assertEqual(recorder.milestone_fallback, 'Unknown milestone object')

    def test_rejected_scheduler_refinement_keeps_valid_window(self):
        benchmark = BENCHMARKS['robocasa365']
        candidates = {'Toy': dict(available_starts=[0], stages=[['a']],
                                  milestone_entities={'objects': [], 'fixtures': []})}
        window = dict(start_stage=0, end_stage=0, max_steps=50, reason='repeated stall')
        response = {'windows': {'Toy': dict(window, milestone_refinement={'target': 'a'})}}
        self.assertEqual(_save_selection(self.root, 0, 'test-model', response, candidates, benchmark),
                         {'Toy': window})
        proposal = json.loads((self.root / 'milestone_proposals_round0.json').read_text())['Toy']
        self.assertEqual(proposal['status'], 'rejected')

    def test_task_evidence_is_task_specific_and_missing_is_explicit(self):
        benchmark = BENCHMARKS['robocasa365']
        evidence = benchmark.task_evidence('StoreLeftoversInBowl')
        self.assertIn('check_rack_contact', evidence)
        self.assertIn('chicken_in_bowl', evidence)
        self.assertNotIn('dishwasher_closed', evidence)
        self.assertIsNone(benchmark.task_evidence('UnknownTask'))
        libero_evidence = BENCHMARKS['libero_10'].task_evidence(
            'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it')
        self.assertEqual(libero_evidence['mug_in_microwave'], 'in white_yellow_mug_1 microwave_1_heating_region')

    def test_suffix_keeps_original_remaining_horizon(self):
        self.make_entry()
        control = json.loads((self.root / 'control.json').read_text())
        control.update(local_mix=[0, 1, 0], horizon=200)
        write_json(self.root / 'control.json', control)
        recorder = self.recorder()
        recorder.reset({})
        for _ in range(50):
            last = recorder.step(1)
        self.assertFalse(last[4]['rounds_truncated'])
        for _ in range(50):
            last = recorder.step(1)
        self.assertTrue(last[4]['rounds_truncated'])
        self.assertEqual(recorder.sample_mode, 'suffix')
        self.assertEqual(recorder.remaining, 100)

    def test_global_mix_and_eval_never_restore(self):
        self.make_entry()
        control = json.loads((self.root / 'control.json').read_text())
        control['local_mix'] = [0, 0, 1]
        write_json(self.root / 'control.json', control)
        recorder = self.recorder()
        recorder.reset({})
        self.assertFalse(recorder.local)
        control['local_mix'] = [1, 0, 0]
        write_json(self.root / 'control.json', control)
        with patch.dict(os.environ, {'STRUCTRL_ROUND_VISUAL_FEEDBACK': '0'}):
            recorder = self.recorder(True)
            recorder.reset({})
            self.assertFalse(recorder.local)
        self.assertIsNone(recorder.seed)

    def test_schema_mismatch_fails(self):
        self.make_entry()
        control = json.loads((self.root / 'control.json').read_text())
        control['windows']['Toy']['stage_signature'] = 'wrong'
        write_json(self.root / 'control.json', control)
        with self.assertRaisesRegex(RuntimeError, 'schema'):
            self.recorder().reset({})

    def test_missing_window_records_fallback(self):
        write_json(self.root / 'control.json', dict(mode='local', round=0, update=25,
                                                    windows={}, local_mix=[1, 0, 0]))
        recorder = self.recorder()
        recorder.reset({})
        self.assertFalse(recorder.local)
        self.assertEqual(recorder.fallback, 'no_eligible_window')

    def test_diagnostics_requires_five_eval_episodes_per_checkpoint(self):
        write_json(self.root / 'tasks.json', ['Toy'])
        for update in [5, 10, 15, 20, 25]:
            for episode in range(5):
                write_json(self.root / 'traces' / f'{update}_{episode}.json', dict(
                    task='Toy', mode='eval', round=0, update=update,
                    complete=True, success=False, stage=0, stages=[['a']], stage_signature='schema'))
        folder = self.root / 'entries/0/Toy/0'
        folder.mkdir(parents=True)
        (folder / 'entry.pkl').write_bytes(b'unused-test-entry')
        self.assertEqual(set(diagnostics(self.root, 0, 0)), {'Toy'})
        row = json.loads((self.root / 'traces/5_0.json').read_text())
        write_json(self.root / 'traces/duplicate.json', row)
        with self.assertRaisesRegex(ValueError, 'expected 5 episodes'):
            diagnostics(self.root, 0, 0)
        (self.root / 'traces/duplicate.json').unlink()
        row['complete'] = False
        write_json(self.root / 'traces/5_0.json', row)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            diagnostics(self.root, 0, 0)

    def test_diagnostics_uses_only_committed_phase_attempt(self):
        write_json(self.root / 'tasks.json', ['Toy'])
        # A failed attempt leaves complete rows behind; they remain auditable but
        # must not be counted with the successful retry.
        for attempt, slots in [('failed', range(5)), ('successful', range(5))]:
            for update in [5, 10, 15, 20, 25]:
                for episode in slots:
                    write_json(self.root / 'traces' / f'{attempt}_{update}_{episode}.json', dict(
                        task='Toy', mode='eval', round=0, update=update, slot=episode,
                        phase_attempt_id=attempt, complete=True, success=False, stage=0,
                        stages=[['a']], stage_signature='schema'))
        folder = self.root / 'entries/0/Toy/0'
        folder.mkdir(parents=True)
        (folder / '0_failed_entry.pkl').write_bytes(b'test-entry')
        self.assertEqual(diagnostics(self.root, 0, 0, 'successful'), {})
        (folder / '0_successful_entry.pkl').write_bytes(b'test-entry')
        self.assertEqual(set(diagnostics(self.root, 0, 0, 'successful')), {'Toy'})
        with self.assertRaisesRegex(ValueError, 'expected 5 episodes'):
            diagnostics(self.root, 0, 0, 'missing')
        duplicate = dict(task='Toy', mode='eval', round=0, update=5, slot=0,
                         phase_attempt_id='successful', complete=True, success=True, stage=0,
                         stages=[['a']], stage_signature='schema')
        write_json(self.root / 'traces' / 'duplicate_current_attempt.json', duplicate)
        selected = diagnostics(self.root, 0, 0, 'successful')
        update_five = next(row for row in selected['Toy']['evaluations'] if row['update'] == 5)
        self.assertTrue(update_five['success'])

    def test_recorder_tags_trace_with_phase_attempt_id(self):
        write_json(self.root / 'control.json', dict(mode='global', round=0, update=4,
                                                    phase_attempt_id='attempt-123', windows={}))
        recorder = self.recorder()
        recorder.reset({})
        for _ in range(2):
            recorder.step(1)
        recorder.finish()
        rows = [json.loads(p.read_text()) for p in (self.root / 'traces').glob('*.json')]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['phase_attempt_id'], 'attempt-123')

    def test_history_includes_global_baseline_and_both_local_points(self):
        write_json(self.root / 'r0_global' / 'complete.json', dict(
            round=0, mode='global', start=0, stop=25, phase_attempt_id='global-ok'))
        write_json(self.root / 'r0_local' / 'complete.json', dict(
            round=0, mode='local', start=25, stop=50, phase_attempt_id='local-ok'))
        image = build_montage([], task='Toy', update=25, success=False)
        points = [(25, 'global-ok', 'global_end_before_local'),
                  (30, 'local-ok', 'local_after_5_updates'),
                  (50, 'local-ok', 'local_end_after_25_updates')]
        for update, attempt, point in points:
            for slot in range(5):
                relative = (Path('visual_feedback') / 'round_0' / f'update_{update:03d}' / 'Toy' / f'episode_{slot:04d}.jpg')
                target = self.root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(image)
                write_json(self.root / 'traces' / f'{update}_{slot}.json', dict(
                    task='Toy', mode='eval', round=0, update=update, slot=slot,
                    phase_attempt_id=attempt, complete=True, success=slot == 0, stage=0,
                    stages=[['a']], stage_signature='schema', visual_montage=relative.as_posix()))
        history = history_for_task(self.root, 1, 'Toy')
        self.assertEqual(len(history), 1)
        self.assertEqual([item['point'] for item in history[0]['intervention_timeline']],
                         [point for _, _, point in points])
        self.assertEqual([item['update'] for item in history[0]['intervention_timeline']], [25, 30, 50])
        self.assertEqual(len(history[0]['images']), 15)

    def test_real_worker_patch_compiles(self):
        lines = Path('patches/rlinf_core.patch').read_text().splitlines()
        start = lines.index('+++ b/rlinf/envs/robocasa365/venv.py') + 1
        source = []
        for line in lines[start:]:
            if line.startswith('diff --git'):
                break
            if line.startswith(('+', ' ')):
                source.append(line[1:])
        result = transform_worker('\n'.join(source) + '\n')
        compile(result, 'patched_worker', 'exec')
        self.assertIn('recorder.step(data)', result)
        self.assertIn('if os.environ.get("STRUCTRL_ROUNDS_ROOT"):\n                raise', result)


if __name__ == '__main__':
    unittest.main()
