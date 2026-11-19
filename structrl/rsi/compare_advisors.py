"""Run one advisor on previously recorded RSI evidence without training."""
import argparse
import json
from pathlib import Path

from .advisor import _provider_settings, read_key, select
from .protocol import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True,
                   help='New, separate directory for this model and repeat')
    p.add_argument('--round', type=int, required=True)
    p.add_argument('--checkpoints', type=int, nargs='+', required=True)
    p.add_argument('--phase-attempt-id', default=None,
                   help='Override committed Global phase attempt for legacy or audited runs')
    p.add_argument('--advisor-provider', choices=('openai-compatible', 'anthropic'), default='openai-compatible')
    p.add_argument('--advisor-base-url', default=None)
    p.add_argument('--advisor-model', required=True)
    p.add_argument('--advisor-reasoning-effort', choices=('none', 'low', 'medium', 'high'))
    p.add_argument('--advisor-json-mode', choices=('schema', 'prompt'), default='schema')
    p.add_argument('--advisor-max-tokens', type=int, default=4096)
    p.add_argument('--advisor-mode', choices=('visual', 'text'), default='visual')
    a = p.parse_args()
    source = a.source_root.resolve()
    output = a.output_root.resolve()
    if output == source or source in output.parents:
        raise ValueError('Output must be outside the original run root')
    marker = source / f'r{a.round}_global' / 'complete.json'
    attempt = a.phase_attempt_id
    if attempt is None:
        if not marker.is_file():
            raise ValueError('Global completion marker missing; specify the audited phase attempt explicitly')
        attempt = json.loads(marker.read_text()).get('phase_attempt_id')
        if not attempt:
            raise ValueError('Global completion marker has no attempt id; specify it explicitly')
    base, model, effort = _provider_settings(a.advisor_base_url, a.advisor_model,
                                              a.advisor_reasoning_effort, a.advisor_provider)
    read_key(a.advisor_provider)
    if a.advisor_max_tokens < 1:
        raise ValueError('advisor-max-tokens must be positive')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'comparison_config.json', {
        'source_root': str(source), 'round': a.round, 'checkpoints': a.checkpoints,
        'phase_attempt_id': attempt, 'provider': a.advisor_provider, 'base_url': base,
        'model': model, 'reasoning_effort': effort, 'mode': a.advisor_mode,
        'json_mode': 'prompt' if a.advisor_provider == 'anthropic' else a.advisor_json_mode,
        'max_tokens': a.advisor_max_tokens,
    })
    windows = select(source, a.round, a.checkpoints, base_url=base, model=model,
        reasoning_effort=effort, mode=a.advisor_mode, phase_attempt_id=attempt,
        provider=a.advisor_provider, json_mode=a.advisor_json_mode,
        max_tokens=a.advisor_max_tokens, output_root=output)
    print(json.dumps({'output_root': str(output), 'windows': windows}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
