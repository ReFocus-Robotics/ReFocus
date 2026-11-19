import json
import os
import uuid
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)


def entry_pattern(attempt_id=None, slot=None):
    prefix = '*' if slot is None else str(slot)
    if attempt_id is not None:
        return f'{prefix}_{attempt_id}_*.pkl'
    return f'{prefix}_*.pkl' if slot is not None else '*.pkl'


def phase_plan(rounds=3, updates=25):
    if rounds < 1 or updates < 1 or updates % 5:
        raise ValueError('Positive rounds and updates divisible by five required')
    return [dict(round=i//2, mode='global' if i%2==0 else 'local',
                 start=i*updates, stop=(i+1)*updates) for i in range(rounds*2)]


def validate_windows(windows, candidates, chunk=50):
    if not isinstance(windows, dict) or set(windows) != set(candidates):
        raise ValueError('Return exactly one window per eligible task')
    for task, w in windows.items():
        c=candidates[task]
        if not isinstance(w,dict): raise ValueError('Window must be an object')
        for k in ('start_stage','end_stage','max_steps'):
            if type(w.get(k)) is not int: raise ValueError('Integer window fields required')
        if w['start_stage'] not in c['available_starts']:
            raise ValueError('Window has no training entry')
        if not 0 <= w['start_stage'] <= w['end_stage'] < len(c['stages']):
            raise ValueError('Invalid stage bounds')
        if not chunk <= w['max_steps'] <= 1000 or w['max_steps'] % chunk:
            raise ValueError('Window budget must be a chunk multiple, at most 1000')
        if not isinstance(w.get('reason'),str) or not w['reason'].strip():
            raise ValueError('Missing rationale')
    return windows
