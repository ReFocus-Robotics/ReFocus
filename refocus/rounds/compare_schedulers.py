"""Re-run the Scheduler on recorded evaluation evidence without training."""

import argparse
import json
from pathlib import Path

from .benchmarks import BENCHMARKS
from .protocol import write_json
from .scheduler import provider_settings, read_key, select


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=sorted(BENCHMARKS), required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True,
                   help='New, separate directory for this model and repeat')
    p.add_argument('--round', type=int, required=True)
    p.add_argument('--checkpoints', type=int, nargs='+', required=True)
    p.add_argument('--phase-attempt-id', default=None,
                   help='Override the committed Global phase attempt')
    p.add_argument('--scheduler-provider', choices=('openai-compatible', 'anthropic'), default='openai-compatible')
    p.add_argument('--scheduler-base-url', default=None)
    p.add_argument('--scheduler-model', required=True)
    p.add_argument('--scheduler-reasoning-effort', choices=('none', 'low', 'medium', 'high'))
    p.add_argument('--scheduler-json-mode', choices=('schema', 'prompt'), default='schema')
    p.add_argument('--scheduler-max-tokens', type=int, default=4096)
    p.add_argument('--scheduler-evidence', choices=('visual', 'text'), default='visual')
    a = p.parse_args()
    benchmark = BENCHMARKS[a.benchmark]
    source = a.source_root.resolve()
    output = a.output_root.resolve()
    if output == source or source in output.parents:
        raise ValueError('Output must be outside the original run root')
    marker = source / f'r{a.round}_global' / 'complete.json'
    attempt = a.phase_attempt_id
    if attempt is None:
        if not marker.is_file():
            raise ValueError('Global completion marker missing; specify the phase attempt explicitly')
        attempt = json.loads(marker.read_text()).get('phase_attempt_id')
        if not attempt:
            raise ValueError('Global completion marker has no attempt id; specify it explicitly')
    base, model, effort = provider_settings(a.scheduler_base_url, a.scheduler_model,
                                            a.scheduler_reasoning_effort, a.scheduler_provider)
    read_key(a.scheduler_provider)
    if a.scheduler_max_tokens < 1:
        raise ValueError('scheduler-max-tokens must be positive')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'comparison_config.json', {
        'benchmark': benchmark.name,
        'source_root': str(source), 'round': a.round, 'checkpoints': a.checkpoints,
        'phase_attempt_id': attempt, 'provider': a.scheduler_provider, 'base_url': base,
        'model': model, 'reasoning_effort': effort, 'evidence': a.scheduler_evidence,
        'json_mode': 'prompt' if a.scheduler_provider == 'anthropic' else a.scheduler_json_mode,
        'max_tokens': a.scheduler_max_tokens,
    })
    windows = select(source, a.round, a.checkpoints, benchmark=benchmark, base_url=base, model=model,
                     reasoning_effort=effort, evidence=a.scheduler_evidence, phase_attempt_id=attempt,
                     provider=a.scheduler_provider, json_mode=a.scheduler_json_mode,
                     max_tokens=a.scheduler_max_tokens, output_root=output)
    print(json.dumps({'output_root': str(output), 'windows': windows}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
