"""Run Global and Local PPO phases in round order, sharing actor, critic and optimizer checkpoints."""

import argparse
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path

from refocus.rounds.benchmarks import BENCHMARKS
from refocus.rounds.protocol import phase_plan, write_json
from refocus.rounds.recorder import ENTRY_SNAPSHOT_VERSION
from refocus.rounds.scheduler import provider_settings, read_key, select


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=sorted(BENCHMARKS), required=True)
    p.add_argument('--rlinf-root', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--updates-per-phase', type=int, default=25)
    p.add_argument('--eval-interval', type=int, default=5)
    p.add_argument('--save-interval', type=int, default=None)
    p.add_argument('--config-path', type=Path, default=None)
    p.add_argument('--config-name', default=None,
                   help='Hydra profile to launch (default: the benchmark default)')
    p.add_argument('--tasks', nargs='+', default=None)
    p.add_argument('--train-envs', type=int, default=None)
    p.add_argument('--eval-envs', type=int, default=None)
    p.add_argument('--horizon', type=int, default=None)
    p.add_argument('--local-mix', nargs=3, type=float, default=[.5, .25, .25],
                   metavar=('WINDOW', 'SUFFIX', 'PREFIX'),
                   help='Rollout probabilities. The StructRL baseline uses normal starts in slot three.')
    p.add_argument('--hydra-override', action='append', default=[])
    p.add_argument('--scheduler-provider', choices=('openai-compatible', 'anthropic'), default='openai-compatible')
    p.add_argument('--scheduler-base-url', default=None)
    p.add_argument('--scheduler-model', default=None)
    p.add_argument('--scheduler-reasoning-effort', choices=('none', 'low', 'medium', 'high'), default=None)
    p.add_argument('--scheduler-json-mode', choices=('schema', 'prompt'), default='schema')
    p.add_argument('--scheduler-max-tokens', type=int, default=4096,
                   help='Maximum response tokens for the native Anthropic adapter')
    p.add_argument('--scheduler-evidence', choices=('visual', 'text'), default='visual')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    benchmark = BENCHMARKS[args.benchmark]
    chunk, horizon = benchmark.chunk, args.horizon or benchmark.horizon
    config_name = args.config_name or benchmark.default_config
    repo = Path(__file__).resolve().parents[2]
    root = args.run_root.resolve()
    runtime = args.rlinf_root.resolve()
    if args.config_path is None:
        args.config_path = repo / 'configs'
    if args.save_interval is None:
        args.save_interval = args.eval_interval
    if args.eval_interval < 1 or args.updates_per_phase % args.eval_interval:
        raise ValueError('updates-per-phase must be divisible by eval-interval')
    if args.save_interval < 1 or args.updates_per_phase % args.save_interval:
        raise ValueError('updates-per-phase must be divisible by save-interval')
    if horizon < chunk or horizon % chunk:
        raise ValueError(f'horizon must be a positive multiple of the {chunk}-action chunk')
    if any(not math.isfinite(weight) or weight < 0 for weight in args.local_mix) or abs(sum(args.local_mix) - 1.0) > 1e-6:
        raise ValueError('local-mix must be three nonnegative probabilities summing to one')
    plan = phase_plan(args.rounds, args.updates_per_phase)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    base, model, effort = provider_settings(args.scheduler_base_url, args.scheduler_model,
                                            args.scheduler_reasoning_effort, args.scheduler_provider)
    read_key(args.scheduler_provider)
    if args.scheduler_max_tokens < 1:
        raise ValueError('scheduler-max-tokens must be positive')
    if not (runtime / '.refocus_round_hooks_v1').exists():
        raise RuntimeError('Install hooks in an isolated RLinf runtime first')
    # All Ray workers must inherit these variables before cluster startup.
    if os.environ.get('STRUCTRL_ROUNDS_ROOT') != str(root):
        raise RuntimeError('Export absolute STRUCTRL_ROUNDS_ROOT before starting Ray')
    variant = os.environ.get(benchmark.reward_variant_env)
    if variant not in ('refocus', 'structrl'):
        raise RuntimeError(f'Set {benchmark.reward_variant_env} to refocus or structrl before Ray startup')
    config_dir = Path(args.config_path).resolve()
    tasks = benchmark.tasks(config_dir)
    if args.tasks is not None:
        unknown = set(args.tasks) - set(tasks)
        if unknown:
            raise ValueError(f'Unknown {benchmark.title} tasks: ' + ', '.join(sorted(unknown)))
        tasks = args.tasks
        if not tasks:
            raise ValueError('At least one task is required')
    release_config = dict(benchmark=benchmark.name, reward_variant=variant, tasks=tasks,
                          entry_snapshot_version=ENTRY_SNAPSHOT_VERSION,
                          local_mix=args.local_mix, horizon=horizon, phases=plan,
                          eval_interval=args.eval_interval, save_interval=args.save_interval,
                          train_envs=args.train_envs, eval_envs=args.eval_envs,
                          config_path=str(config_dir), config_name=config_name,
                          hydra_overrides=args.hydra_override)
    if variant == 'refocus':
        release_config['milestone_source'] = 'scheduler'
        release_config['task_stages'] = {task: [list(stage) for stage in benchmark.task_stages[task]]
                                         for task in tasks}
    else:
        release_config['refinements'] = {}
    release_config_path = root / 'release_config.json'
    if release_config_path.exists():
        previous_config = json.loads(release_config_path.read_text())
        if previous_config.get('entry_snapshot_version') != ENTRY_SNAPSHOT_VERSION:
            raise RuntimeError('Use a new run root to collect entries')
        if previous_config != release_config:
            raise RuntimeError('Method settings changed on resume; use a separate run root')
    if not release_config_path.exists() and any(root.glob('r*_*/complete.json')):
        raise RuntimeError('Use a new run root for runs predating the release configuration')
    write_json(release_config_path, release_config)
    # Refuse accidental changes on resume: each Scheduler comparison uses its own run root.
    scheduler_config = dict(provider=args.scheduler_provider, base_url=base, model=model,
                            reasoning_effort=effort, evidence=args.scheduler_evidence,
                            json_mode=('prompt' if args.scheduler_provider == 'anthropic' else args.scheduler_json_mode),
                            max_tokens=args.scheduler_max_tokens)
    scheduler_config_path = root / 'scheduler_config.json'
    if scheduler_config_path.exists() and json.loads(scheduler_config_path.read_text()) != scheduler_config:
        raise RuntimeError('Scheduler settings changed on resume; use a separate run root')
    write_json(scheduler_config_path, scheduler_config)
    write_json(root / 'tasks.json', tasks)
    prior = None
    for phase in plan:
        name = f"r{phase['round']}_{phase['mode']}"
        phase_dir = root / name
        checkpoint = phase_dir / benchmark.experiment_name / 'checkpoints' / f"global_step_{phase['stop']}"
        if (phase_dir / 'complete.json').exists():
            if not (checkpoint / 'actor').is_dir():
                raise RuntimeError('Completed phase lost checkpoint')
            prior = checkpoint
            continue
        windows = {}
        global_attempt_id = None
        phase_attempt_id = uuid.uuid4().hex
        if phase['mode'] == 'local':
            write_json(root / 'status.json', dict(state='selecting_windows', phase=name))
            try:
                global_phase = plan[plan.index(phase) - 1]
                checkpoints = range(global_phase['start'] + args.eval_interval,
                                    global_phase['stop'] + 1, args.eval_interval)
                completed_global = root / f"r{global_phase['round']}_global" / 'complete.json'
                if not completed_global.is_file():
                    raise RuntimeError(f'Missing completed Global phase marker: {completed_global}')
                global_attempt_id = json.loads(completed_global.read_text()).get('phase_attempt_id')
                windows = select(root, phase['round'], checkpoints, benchmark=benchmark,
                                 base_url=base, model=model, reasoning_effort=effort,
                                 provider=args.scheduler_provider, json_mode=args.scheduler_json_mode,
                                 max_tokens=args.scheduler_max_tokens, evidence=args.scheduler_evidence,
                                 phase_attempt_id=global_attempt_id)
            except Exception as exc:
                write_json(root / 'status.json', dict(state='failed', phase=name,
                    operation='scheduler', error_type=type(exc).__name__))
                raise
        write_json(root / 'control.json', dict(phase, phase_attempt_id=phase_attempt_id,
            update=phase['start'], windows=windows, entry_attempt_id=global_attempt_id,
            chunk=chunk, horizon=horizon, local_mix=args.local_mix, reward_variant=variant))
        phase_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(runtime / 'examples/embodiment/train_embodied_agent.py'),
               '--config-path', str(config_dir), '--config-name', config_name,
               f"runner.max_epochs={plan[-1]['stop']}", f"runner.max_steps={phase['stop']}",
               f'runner.val_check_interval={args.eval_interval}', f'runner.save_interval={args.save_interval}',
               f'runner.logger.log_path={phase_dir}',
               f"env.train.auto_reset={'True' if phase['mode'] == 'local' else 'False'}"]
        if args.tasks is not None:
            cmd.extend(benchmark.task_overrides(args.tasks))
        if args.train_envs is not None:
            cmd.append(f'env.train.total_num_envs={args.train_envs}')
        if args.eval_envs is not None:
            cmd.append(f'env.eval.total_num_envs={args.eval_envs}')
        if horizon != benchmark.horizon:
            cmd.extend([f'env.train.max_episode_steps={horizon}',
                        f'env.train.max_steps_per_rollout_epoch={horizon}',
                        f'env.eval.max_episode_steps={horizon}',
                        f'env.eval.max_steps_per_rollout_epoch={horizon}'])
        cmd.extend(args.hydra_override)
        if prior:
            cmd.append(f'runner.resume_dir={prior}')
        write_json(phase_dir / 'command.json', cmd)
        child_env = os.environ.copy()
        # Only the coordinator talks to the Scheduler; do not pass its keys to workers.
        for key in ('SCHEDULER_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
            child_env.pop(key, None)
        with (phase_dir / 'effective_config.yaml').open('w') as out:
            subprocess.run(cmd + ['--cfg', 'job', '--resolve'], stdout=out, check=True, env=child_env)
        import yaml
        cfg = yaml.safe_load((phase_dir / 'effective_config.yaml').read_text())
        if cfg['actor']['model']['num_action_chunks'] != chunk:
            raise ValueError(f'Config must use {chunk} action chunks')
        write_json(root / 'status.json', dict(state='running', phase=name, config=str(phase_dir / 'effective_config.yaml')))
        with (phase_dir / 'job.log').open('a') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=child_env)
        if result.returncode:
            write_json(root / 'status.json', dict(state='failed', phase=name, exit_code=result.returncode))
            raise RuntimeError('Training failed; inspect ' + str(phase_dir / 'job.log'))
        weights = checkpoint / 'actor/model_state_dict/full_weights.pt'
        if not weights.is_file():
            raise RuntimeError(f'Missing final weights: {weights}')
        import torch
        torch.load(weights, map_location='meta', weights_only=True)
        write_json(phase_dir / 'complete.json', dict(phase, phase_attempt_id=phase_attempt_id,
                                                    eval_interval=args.eval_interval))
        prior = checkpoint
    write_json(root / 'status.json', dict(state='complete', updates=plan[-1]['stop']))


if __name__ == '__main__':
    main()
