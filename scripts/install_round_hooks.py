#!/usr/bin/env python3
"""Apply the round hooks to an isolated, already StructRL-patched RLinf checkout.

LIBERO runtimes receive the same integration through ``patches/libero_addon.patch``
at setup time and need no installer. This script covers RoboCasa365. No GPU needed.
"""
import argparse
import os
from pathlib import Path

MARKER = '.refocus_round_hooks_v1'


def once(s, old, new):
    if s.count(old) != 1:
        raise ValueError('Unsupported runtime anchor: ' + old[:100])
    return s.replace(old, new, 1)


def route_progress_checks(source, indent):
    grouped = '''if _reward_variant == "refocus":
    from refocus.robocasa import get_subtask_states as get_refocus_subtask_states
    grouped = get_refocus_subtask_states(inner)
else:
    grouped = get_subtask_states(inner)
'''
    return once(source, '\n' + indent + 'grouped = get_subtask_states(inner)\n',
                '\n' + ''.join(indent + line + '\n' for line in grouped.splitlines()))


def transform_worker(s):
    start = s.index('                # GrootRoboCasaEnv.step')
    stop = s.index('                if obs_bufs is not None:', start)
    body = s[start:stop]
    body = '\n'.join(line[12:] if line.startswith(' ' * 12) else line for line in body.splitlines()) + '\n'
    # Strict reward errors under rounds; the original non-round fallback remains intact.
    body = body.replace('    except Exception as _e:\n',
                        '    except Exception as _e:\n        if os.environ.get("STRUCTRL_ROUNDS_ROOT"):\n            raise\n')
    body = route_progress_checks(body, ' ' * 8)
    body = once(body, '\n        tinfo = timing_reward.step(grouped, success, sim_t=sim_t)\n', '''
        if getattr(timing_reward, "refined_target", None) is not None:
            from refocus.rounds.robocasa_codec import milestone_signals
            tinfo = timing_reward.step(grouped, success, sim_t=sim_t,
                                       signals=milestone_signals(inner, timing_reward.milestone_checks))
        else:
            tinfo = timing_reward.step(grouped, success, sim_t=sim_t)
''')
    helper = '''    def _step_once(data):
        nonlocal timing_reward, timing_data, timing_task_name, latched_n_done, latched_outcome
''' + ''.join('    ' + line + '\n' for line in body.splitlines()) + '        return env_return\n\n'
    reset_start = s.index('                # GrootRoboCasaEnv.reset')
    reset_stop = s.index('                if obs_bufs is not None:', reset_start)
    reset_body = s[reset_start:reset_stop]
    reset_body = '\n'.join(line[12:] if line.startswith(' ' * 12) else line for line in reset_body.splitlines()) + '\n'
    helper += '''    def _reset_once(data):
        nonlocal latched_n_done, latched_outcome, timing_reward
''' + ''.join('    ' + line + '\n' for line in reset_body.splitlines()) + '''        if timing_reward is None and os.environ.get("STRUCTRL_ROUNDS_ROOT"):
            from refocus.rounds.recorder import leaf
            task = type(leaf(env)).__name__
            variant = os.environ.get("STRUCTRL_REWARD_VARIANT")
            if variant == "refocus":
                from refocus.robocasa import build_reward
                timing_reward = build_reward(task)
            elif variant == "structrl":
                from structrl.timing_reward_builder import load_timing_data
                from structrl.smooth_band_reward import build_smooth_band_reward
                timing_reward = build_smooth_band_reward(task, load_timing_data())
            else:
                raise RuntimeError("Reward variant must be refocus or structrl")
        return obs, info

'''
    helper += '''    import os
    recorder = None
    if os.environ.get("STRUCTRL_ROUNDS_ROOT"):
        from refocus.rounds import robocasa_codec
        from refocus.rounds.recorder import Recorder
        recorder = Recorder(env, _reset_once, _step_once, lambda: timing_reward, robocasa_codec)
'''
    s = s[:reset_start] + '                obs, info = recorder.reset(data) if recorder else _reset_once(data)\n' + s[reset_stop:]
    start = s.index('                # GrootRoboCasaEnv.step')
    stop = s.index('                if obs_bufs is not None:', start)
    s = s[:start] + '                env_return = recorder.step(data) if recorder else _step_once(data)\n' + s[stop:]
    s = once(s, '    timing_task_name = None\n', '    timing_task_name = None\n' + helper)
    return s


def install(root):
    root = Path(root)
    marker = root / MARKER
    if marker.exists():
        raise ValueError('Already patched; use a fresh isolated runtime')
    paths = {
        'worker': root / 'rlinf/envs/robocasa365/venv.py',
        'env': root / 'rlinf/envs/robocasa365/robocasa365_env.py',
        'base': root / 'rlinf/envs/robocasa/robocasa_env.py',
        'runner': root / 'rlinf/runners/embodied_runner.py',
        'bootstrap': root / 'rlinf/workers/env/env_worker.py',
    }
    outputs = {'worker': transform_worker(paths['worker'].read_text())}
    s = paths['env'].read_text()
    s = once(s, '    def get_env_fns(self):\n        env_fns = []',
             '    def get_env_fns(self):\n        env_fns = []\n'
             '        rounds_eval = bool(self.cfg.is_eval)\n'
             '        if self.group_size != 1 or self.cfg.get("per_rank_single_task", False):\n'
             '            raise ValueError("Rounds require group_size=1 and round-robin tasks")')
    s = once(s, '            def env_fn(env_gym_id=env_gym_id, seed=env_seed, scene_kwargs=scene_kwargs):',
             '''            slot = (self.seed_offset * self.num_envs + env_id) // self.num_tasks
            replicas = int(self.cfg.total_num_envs) // self.num_tasks
            def env_fn(env_gym_id=env_gym_id, seed=env_seed, scene_kwargs=scene_kwargs,
                       task_name=task_name, slot=slot, replicas=replicas):''')
    s = once(s, '                return _ActionDictAdapter(env)', '''                adapter = _ActionDictAdapter(env)
                adapter._round_eval = rounds_eval
                adapter._round_task = task_name
                adapter._round_slot = slot
                adapter._round_replicas = replicas
                return adapter''')
    outputs['env'] = s
    s = paths['base'].read_text()
    s = once(s, '        truncations = self._elapsed_steps >= self.cfg.max_episode_steps', '''        truncations = self._elapsed_steps >= self.cfg.max_episode_steps
        if "rounds_truncated" in infos:
            truncations |= np.asarray(infos["rounds_truncated"], dtype=bool)''')
    outputs['base'] = s
    s = paths['runner'].read_text()
    s = once(s, '            # set global step', '''            if os.environ.get("STRUCTRL_ROUNDS_ROOT"):
                from refocus.rounds.protocol import write_json
                import json
                from pathlib import Path
                control_path = Path(os.environ["STRUCTRL_ROUNDS_ROOT"]) / "control.json"
                control = json.loads(control_path.read_text())
                control["update"] = self.global_step
                write_json(control_path, control)
            # set global step''')
    # Every checkpoint is kept while rounds own the run directory.
    if 'if save_model and improved_checkpoint:' in s:
        s = once(s, 'if save_model and improved_checkpoint:',
                 'if save_model and (improved_checkpoint or os.environ.get("STRUCTRL_ROUNDS_ROOT")):')
        s = once(s, '                    self._prune_non_best_checkpoints()',
                 '                    if not os.environ.get("STRUCTRL_ROUNDS_ROOT"):\n'
                 '                        self._prune_non_best_checkpoints()')
    elif 'if save_model:' not in s:
        raise ValueError('Unsupported checkpoint save semantics')
    outputs['runner'] = s
    s = paths['bootstrap'].read_text()
    s = once(s, 'last_step_truncations = env_output.truncations[:, -1]',
             'last_step_truncations = env_output.truncations[:, -1] & ~env_output.terminations[:, -1]')
    outputs['bootstrap'] = s
    # Validate everything before mutating any file.
    for key, source in outputs.items():
        compile(source, str(paths[key]), 'exec')
    for key, source in outputs.items():
        paths[key].write_text(source)
    marker.write_text('v1\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rlinf_root', help='isolated RLinf checkout for RoboCasa365')
    install(parser.parse_args().rlinf_root)
