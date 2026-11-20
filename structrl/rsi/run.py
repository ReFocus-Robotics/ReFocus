"""Bounded Global/Local PPO jobs sharing actor, critic and optimizer checkpoints."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid
from refocus.robocasa import TASK_STAGES
from .protocol import phase_plan, write_json
from .advisor import select, _provider_settings, read_key
from .snapshot import ENTRY_SNAPSHOT_VERSION

DEFAULT_RSI_CONFIG_NAME = 'robocasa365_ppo_openpi_pi05_refocus_16gpu'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rlinf-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--updates-per-phase',type=int,default=25)
    p.add_argument('--eval-interval',type=int,default=5)
    p.add_argument('--save-interval',type=int,default=None)
    p.add_argument('--config-path',type=Path,default=None)
    p.add_argument('--config-name', default=DEFAULT_RSI_CONFIG_NAME,
                   help='Hydra profile to launch (default: %(default)s)')
    p.add_argument('--tasks',nargs='+',default=None)
    p.add_argument('--train-envs',type=int,default=None)
    p.add_argument('--eval-envs',type=int,default=None)
    p.add_argument('--horizon',type=int,default=2900)
    p.add_argument('--local-mix', nargs=3, type=float, default=[.5, .25, .25],
                   metavar=('WINDOW', 'SUFFIX', 'PREFIX'),
                   help='Rollout probabilities. The StructRL baseline uses normal starts in slot three.')
    p.add_argument('--hydra-override',action='append',default=[])
    p.add_argument('--advisor-provider',choices=('openai-compatible','anthropic'),default='openai-compatible')
    p.add_argument('--advisor-base-url','--astra-base-url',dest='advisor_base_url',default=None)
    p.add_argument('--advisor-model','--astra-model',dest='advisor_model',default=None)
    p.add_argument('--advisor-reasoning-effort','--astra-reasoning-effort',dest='advisor_reasoning_effort',
                   choices=('none','low','medium','high'),default=None)
    p.add_argument('--advisor-json-mode',choices=('schema','prompt'),default='schema')
    p.add_argument('--advisor-max-tokens',type=int,default=4096,
                   help='Maximum response tokens for the native Anthropic adapter')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--advisor-mode',choices=('visual','text'),default='visual')
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[2]
    root=args.run_root.resolve(); runtime=args.rlinf_root.resolve()
    if args.config_path is None: args.config_path=repo/'configs'
    if args.save_interval is None: args.save_interval=args.eval_interval
    if args.eval_interval < 1 or args.updates_per_phase % args.eval_interval:
        raise ValueError('updates-per-phase must be divisible by eval-interval')
    if args.save_interval < 1 or args.updates_per_phase % args.save_interval:
        raise ValueError('updates-per-phase must be divisible by save-interval')
    if args.horizon < 50 or args.horizon % 50:
        raise ValueError('horizon must be a positive multiple of the 50-action chunk')
    if any(not math.isfinite(weight) or weight < 0 for weight in args.local_mix) or abs(sum(args.local_mix)-1.0)>1e-6:
        raise ValueError('local-mix must be three nonnegative probabilities summing to one')
    plan=phase_plan(args.rounds,args.updates_per_phase)
    if args.dry_run:
        print(json.dumps(plan,indent=2));return
    base, model, effort = _provider_settings(args.advisor_base_url, args.advisor_model,
                                              args.advisor_reasoning_effort, args.advisor_provider)
    read_key(args.advisor_provider)
    if args.advisor_max_tokens < 1:
        raise ValueError('advisor-max-tokens must be positive')
    if not (runtime/'.structrl_rsi_hooks_v4').exists():
        raise RuntimeError('Install hooks in an isolated RLinf runtime first')
    # All Ray workers must inherit these variables before cluster startup.
    if os.environ.get('STRUCTRL_RSI_ROOT')!=str(root):
        raise RuntimeError('Export absolute STRUCTRL_RSI_ROOT before starting Ray')
    variant = os.environ.get('STRUCTRL_REWARD_VARIANT')
    if variant not in ('refocus', 'structrl'):
        raise RuntimeError('Set STRUCTRL_REWARD_VARIANT to refocus or structrl before Ray startup')
    import yaml
    config_dir=Path(args.config_path).resolve()
    tasks=yaml.safe_load((config_dir/'env/robocasa365_composite_seen.yaml').read_text())['task_names']
    if args.tasks is not None:
        unknown=set(args.tasks)-set(tasks)
        if unknown: raise ValueError('Unknown RoboCasa tasks: '+', '.join(sorted(unknown)))
        tasks=args.tasks
        if not tasks: raise ValueError('At least one task is required')
    release_config = dict(reward_variant=variant, tasks=tasks,
                          entry_snapshot_version=ENTRY_SNAPSHOT_VERSION,
                          local_mix=args.local_mix, horizon=args.horizon, phases=plan,
                          eval_interval=args.eval_interval, save_interval=args.save_interval,
                          train_envs=args.train_envs, eval_envs=args.eval_envs,
                          config_path=str(config_dir), config_name=args.config_name,
                          hydra_overrides=args.hydra_override)
    if variant == 'refocus':
        release_config['milestone_source'] = 'scheduler'
        release_config['task_stages'] = {task: [list(stage) for stage in TASK_STAGES[task]] for task in tasks}
    else:
        release_config['refinements'] = {}
    release_config_path = root / 'release_config.json'
    if release_config_path.exists():
        previous_config = json.loads(release_config_path.read_text())
        if previous_config.get('entry_snapshot_version') != ENTRY_SNAPSHOT_VERSION:
            raise RuntimeError('Use a new run root to collect direct snapshots')
        if previous_config != release_config:
            raise RuntimeError('Method settings changed on resume; use a separate run root')
    if not release_config_path.exists() and any(root.glob('r*_*/complete.json')):
        raise RuntimeError('Use a new run root for runs predating the release configuration')
    write_json(release_config_path, release_config)
    # Refuse accidental changes on resume: each advisor comparison uses its own run root.
    advisor_config = dict(provider=args.advisor_provider, base_url=base, model=model, reasoning_effort=effort,
                          mode=args.advisor_mode, json_mode=('prompt' if args.advisor_provider == 'anthropic' else args.advisor_json_mode),
                          max_tokens=args.advisor_max_tokens)
    advisor_config_path = root/'advisor_config.json'
    if advisor_config_path.exists() and json.loads(advisor_config_path.read_text()) != advisor_config:
        raise RuntimeError('Advisor settings changed on resume; use a separate run root')
    write_json(advisor_config_path, advisor_config)
    write_json(root/'tasks.json',tasks)
    prior=None
    for phase in plan:
        name=f"r{phase['round']}_{phase['mode']}"
        phase_dir=root/name
        checkpoint=phase_dir/'robocasa_rsi/checkpoints'/f"global_step_{phase['stop']}"
        if (phase_dir/'complete.json').exists():
            if not (checkpoint/'actor').is_dir():raise RuntimeError('Completed phase lost checkpoint')
            prior=checkpoint;continue
        windows={}
        global_attempt_id = None
        phase_attempt_id=uuid.uuid4().hex
        if phase['mode']=='local':
            write_json(root/'status.json',dict(state='selecting_windows',phase=name))
            try:
                global_phase=plan[plan.index(phase)-1]
                checkpoints=range(global_phase['start']+args.eval_interval,
                                  global_phase['stop']+1,args.eval_interval)
                completed_global=root/f"r{global_phase['round']}_global"/'complete.json'
                if not completed_global.is_file():
                    raise RuntimeError(f'Missing completed Global phase marker: {completed_global}')
                global_attempt_id=json.loads(completed_global.read_text()).get('phase_attempt_id')
                windows=select(root,phase['round'],checkpoints,
                    base_url=base,model=model,reasoning_effort=effort,
                    provider=args.advisor_provider,json_mode=args.advisor_json_mode,
                    max_tokens=args.advisor_max_tokens,
                    mode=args.advisor_mode,phase_attempt_id=global_attempt_id)
            except Exception as exc:
                write_json(root/'status.json',dict(state='failed',phase=name,
                    operation='advisor',error_type=type(exc).__name__))
                raise
        write_json(root/'control.json',dict(phase,phase_attempt_id=phase_attempt_id,
            update=phase['start'],windows=windows,entry_attempt_id=global_attempt_id,
            chunk=50,horizon=args.horizon,local_mix=args.local_mix,
            reward_variant=variant))
        phase_dir.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(runtime/'examples/embodiment/train_embodied_agent.py'),
             '--config-path',str(config_dir),'--config-name',args.config_name,
             f"runner.max_epochs={plan[-1]['stop']}",f"runner.max_steps={phase['stop']}",
             f'runner.val_check_interval={args.eval_interval}',f'runner.save_interval={args.save_interval}',
             f'runner.logger.log_path={phase_dir}',
             f"env.train.auto_reset={'True' if phase['mode']=='local' else 'False'}"]
        if args.tasks is not None:
            names='['+','.join(args.tasks)+']'
            cmd.extend([f'env.train.task_names={names}',f'env.eval.task_names={names}'])
        if args.train_envs is not None:cmd.append(f'env.train.total_num_envs={args.train_envs}')
        if args.eval_envs is not None:cmd.append(f'env.eval.total_num_envs={args.eval_envs}')
        if args.horizon != 2900:
            cmd.extend([f'env.train.max_episode_steps={args.horizon}',
                        f'env.train.max_steps_per_rollout_epoch={args.horizon}',
                        f'env.eval.max_episode_steps={args.horizon}',
                        f'env.eval.max_steps_per_rollout_epoch={args.horizon}'])
        cmd.extend(args.hydra_override)
        if prior:cmd.append(f'runner.resume_dir={prior}')
        write_json(phase_dir/'command.json',cmd)
        child_env=os.environ.copy()
        # Only the coordinator talks to the advisor; do not pass its keys to workers.
        for key in ('OPENAI_API_KEY', 'ADVISOR_API_KEY', 'ANTHROPIC_API_KEY'):
            child_env.pop(key, None)
        with (phase_dir/'effective_config.yaml').open('w') as out:
            subprocess.run(cmd+['--cfg','job','--resolve'],stdout=out,check=True,env=child_env)
        cfg=yaml.safe_load((phase_dir/'effective_config.yaml').read_text())
        assert cfg['actor']['model']['num_action_chunks']==50
        assert cfg['actor']['model']['num_steps']==10
        expected_eval_envs=args.eval_envs if args.eval_envs is not None else 80
        assert cfg['env']['eval']['total_num_envs']==expected_eval_envs
        write_json(root/'status.json',dict(state='running',phase=name,config=str(phase_dir/'effective_config.yaml')))
        with (phase_dir/'job.log').open('a') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=child_env)
        if result.returncode:
            write_json(root/'status.json',dict(state='failed',phase=name,exit_code=result.returncode))
            raise RuntimeError('Training failed; inspect '+str(phase_dir/'job.log'))
        weights=checkpoint/'actor/model_state_dict/full_weights.pt'
        if not weights.is_file():raise RuntimeError(f'Missing final weights: {weights}')
        import torch
        torch.load(weights,map_location='meta',weights_only=True)
        write_json(phase_dir/'complete.json',dict(phase, phase_attempt_id=phase_attempt_id,
                                                   eval_interval=args.eval_interval))
        prior=checkpoint
    write_json(root/'status.json',dict(state='complete',updates=plan[-1]['stop']))

if __name__=='__main__': main()
