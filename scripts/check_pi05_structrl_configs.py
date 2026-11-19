#!/usr/bin/env python3
"""Compare resolved 16-GPU recipes to the original StructRL launch commands.

Run using the installed RLinf training Python, with EMBODIED_PATH pointing to
RLinf/examples/embodiment. Does not load weights, start Ray, or use a GPU.
"""
import json
import os
from pathlib import Path
import re

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    'cluster.num_nodes', 'actor.micro_batch_size',
    'runner.max_epochs', 'runner.max_steps',
    'runner.logger.experiment_name', 'runner.logger.log_path',
}


def flatten(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(flatten(item, f'{prefix}.{key}' if prefix else key))
    else:
        result[prefix] = value
    return result


def main():
    os.environ.setdefault('PI05_SFT_CKPT', '/config-audit/same-sft-checkpoint')
    if 'EMBODIED_PATH' not in os.environ:
        raise SystemExit('Set EMBODIED_PATH to RLinf/examples/embodiment first.')
    for algo in ('ppo', 'grpo'):
        launcher = (ROOT / f'launch/train_pi05_structrl_{algo}.sbatch').read_text()
        overrides = []
        for line in launcher.splitlines():
            match = re.match(r'^\s+(\+?[\w.]+)=([^\s\\]+)', line)
            if match and '$' not in match[2] and not match[1].startswith('runner.logger.'):
                overrides.append(match[1] + '=' + match[2])
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'configs')):
            reference = compose(
                config_name=f'robocasa365_{algo}_openpi_pi05_composite_seen',
                overrides=overrides,
            )
            adapted = compose(config_name=f'robocasa365_{algo}_openpi_pi05_structrl_16gpu')
        expected = flatten(OmegaConf.to_container(reference, resolve=True))
        actual = flatten(OmegaConf.to_container(adapted, resolve=True))
        diffs = {key: [expected.get(key), actual.get(key)]
                 for key in expected.keys() | actual.keys()
                 if expected.get(key) != actual.get(key)}
        assert set(diffs) <= ALLOWED, (algo, diffs)
        assert actual['cluster.num_nodes'] == 2
        assert actual['actor.micro_batch_size'] == 32
        assert actual['actor.global_batch_size'] == 512
        assert actual['runner.max_steps'] == actual['runner.max_epochs'] == 100
        assert actual['actor.model.openpi.action_chunk'] == 50
        assert actual['actor.model.openpi.num_steps'] == 10
        assert actual['actor.optim.lr'] == 5e-6
        assert actual['env.train.total_num_envs'] == 256
        assert actual['algorithm.rollout_epoch'] == 2
        assert actual['env.eval.max_steps_per_rollout_epoch'] == 2900
        if algo == 'ppo':
            assert actual['env.train.dense_reward.enabled']
            assert not actual['env.train.final_only_reward.enabled']
            assert actual['actor.model.openpi.value_after_vlm']
            assert actual['actor.model.openpi.detach_critic_input']
            assert actual['actor.optim.value_lr'] == 1e-4
        else:
            assert actual['env.train.final_only_reward.enabled']
            assert not actual['env.train.dense_reward.enabled']
            assert actual['algorithm.filter_zero_var_groups']
            assert not actual['algorithm.filter_rewards']
        print(json.dumps({'algorithm': algo, 'status': 'RESOLVED_CONFIG_PARITY_PASS',
                          'intentional_differences': diffs}, indent=2))


if __name__ == '__main__':
    main()
