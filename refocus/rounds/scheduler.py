"""Scheduler: diagnose bottlenecks from evaluation rollouts and choose practice windows."""
import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .milestones import validate_refinement
from .protocol import entry_pattern, validate_windows, write_json
from .visual_feedback import image_cases


def _checkpoint_steps(checkpoints):
    if isinstance(checkpoints, int):
        return list(range(checkpoints + 5, checkpoints + 26, 5))
    steps = sorted(set(int(step) for step in checkpoints))
    if not steps or any(step <= 0 for step in steps):
        raise ValueError('At least one positive diagnostic checkpoint is required')
    return steps


def _read_trace_rows(root):
    rows = []
    for path in sorted((Path(root) / 'traces').glob('*.json')):
        row = json.loads(path.read_text())
        # Ties between retried episodes of one slot resolve to the latest write.
        row['_trace_mtime_ns'] = path.stat().st_mtime_ns
        rows.append(row)
    return rows


def _latest_episode_rows(rows):
    """Keep the latest trace for each task/update/worker slot."""
    selected = {}
    unkeyed = []
    for source in rows:
        row = dict(source)
        slot = row.get('slot')
        if slot is None:
            # Traces without a slot identity cannot be deduplicated; keep them.
            unkeyed.append(row)
            continue
        key = (row.get('task'), row.get('update'), slot)
        previous = selected.get(key)
        if previous is None or row['_trace_mtime_ns'] >= previous['_trace_mtime_ns']:
            selected[key] = row
    result = unkeyed + sorted(selected.values(), key=lambda row: (row.get('update', 0), row.get('task', ''),
                                                                  row.get('slot', -1), row['_trace_mtime_ns']))
    for row in result:
        row.pop('_trace_mtime_ns', None)
    return result


def _load_eval_rows(root, round_id, checkpoints, phase_attempt_id=None):
    root = Path(root)
    tasks = json.loads((root / 'tasks.json').read_text())
    rows = [row for row in _read_trace_rows(root) if row.get('mode') == 'eval' and row.get('complete')
            and row.get('round') == round_id and row.get('update') in checkpoints]
    # A failed phase can be retried from the preceding completed checkpoint.
    # Only the attempt recorded in complete.json is used.
    if phase_attempt_id is not None:
        rows = [row for row in rows if row.get('phase_attempt_id') == phase_attempt_id]
    rows = _latest_episode_rows(rows)
    for task in tasks:
        for update in checkpoints:
            count = sum(row.get('task') == task and row.get('update') == update for row in rows)
            if count != 5:
                raise ValueError(f'Incomplete evaluation: {task}, update {update}, expected 5 episodes, got {count}')
    return tasks, rows


def diagnostics(root, round_id, checkpoints, phase_attempt_id=None):
    root = Path(root)
    checkpoints = _checkpoint_steps(checkpoints)
    tasks, rows = _load_eval_rows(root, round_id, checkpoints, phase_attempt_id)
    candidates = {}
    for task in tasks:
        records = [row for row in rows if row['task'] == task]
        folder = root / 'entries' / str(round_id) / task
        starts = sorted(int(path.name) for path in folder.glob('*')
                        if path.is_dir() and any(path.glob(entry_pattern(phase_attempt_id))))
        signatures = {row['stage_signature'] for row in records}
        if len(signatures) != 1:
            raise ValueError('Inconsistent stage schema for ' + task)
        if starts and any(not row['success'] for row in records):
            kinds = set.intersection(*(set(row.get('milestone_entities', {})) for row in records))
            entities = {kind: sorted(set.intersection(*(set(row['milestone_entities'][kind]) for row in records)))
                        for kind in sorted(kinds)}
            candidates[task] = dict(available_starts=starts, stages=records[-1]['stages'],
                stage_signature=records[-1]['stage_signature'],
                milestone_entities=entities,
                evaluations=[{key: row[key] for key in ('update', 'success', 'stage')} for row in records])
    return candidates


def _public_candidates(candidates):
    """Project only model-facing evidence; stage hashes stay internal."""
    return {task: {key: candidate[key] for key in
            ('available_starts', 'stages', 'evaluations', 'milestone_entities')}
            for task, candidate in candidates.items()}


def _schema(name, properties, required):
    return {'type': 'json_schema', 'json_schema': {
        'name': name, 'strict': True,
        'schema': {'type': 'object', 'properties': properties,
                   'required': required, 'additionalProperties': False}}}


def _selection_schema(candidates, checks):
    milestone = {'type': 'object', 'properties': {
        'name': {'type': 'string'},
        'check': {'type': 'string', 'enum': list(checks)},
        'object': {'type': 'string'},
        'reference': {'type': ['string', 'null']},
        'share': {'type': 'number'}},
        'required': ['name', 'check', 'object', 'reference', 'share'], 'additionalProperties': False}
    refinement = {'type': ['object', 'null'], 'properties': {
        'target': {'type': 'string'},
        'milestones': {'type': 'array', 'items': milestone},
        'final_share': {'type': 'number'}},
        'required': ['target', 'milestones', 'final_share'], 'additionalProperties': False}
    window = {'type': 'object', 'properties': {
        'start_stage': {'type': 'integer'}, 'end_stage': {'type': 'integer'},
        'max_steps': {'type': 'integer'}, 'reason': {'type': 'string'},
        'milestone_refinement': refinement},
        'required': ['start_stage', 'end_stage', 'max_steps', 'reason', 'milestone_refinement'],
        'additionalProperties': False}
    return _schema('practice_window_selection', {
        'windows': {'type': 'object', 'properties': {task: window for task in candidates},
                    'required': list(candidates), 'additionalProperties': False}}, ['windows'])


def _check_descriptions(checks):
    return {name: spec[0] for name, spec in checks.items()}


_MILESTONE_INSTRUCTIONS = (
    ' For each selected window, set milestone_refinement to null unless the evidence '
    'supports refining one of the subtasks in end_stage. If it does, name that exact '
    'subtask as target and specify one to three distinct intermediate milestones with '
    'checks from supported_milestone_checks and exact names from that task\'s milestone_entities. '
    'Choose meaningful intermediate steps toward the target. Avoid duplicating any DAG '
    'subtask or a condition already satisfied in the restored entry state. '
    'The final milestone uses the original target predicate and only needs final_share. '
    'Choose positive shares (equal by default) including final_share that sum to one; '
    'these divide the original target reward budget. Explain the evidence in reason. '
    'Use null if the available simulator checks cannot express a meaningful refinement.'
)


def _save_selection(output_root, round_id, model, response, candidates, benchmark):
    proposed = response['windows']
    validate_windows(proposed, candidates, chunk=benchmark.chunk)
    windows = {}
    proposals = {}
    for task, proposal in proposed.items():
        window = {key: proposal[key] for key in ('start_stage', 'end_stage', 'max_steps', 'reason')}
        refinement = proposal.get('milestone_refinement')
        result = {'proposal': refinement, 'status': 'no_refinement'}
        if refinement is not None:
            candidate = candidates[task]
            try:
                window['milestone_refinement'] = validate_refinement(
                    refinement, candidate['stages'][window['end_stage']], candidate['milestone_entities'],
                    benchmark.milestone_checks)
                result['status'] = 'accepted'
            except ValueError as exc:
                result.update(status='rejected', reason=str(exc))
        windows[task] = window
        proposals[task] = result
    write_json(output_root / f'scheduler_response_round{round_id}.json',
               {'model': model, 'windows': proposed})
    write_json(output_root / f'milestone_proposals_round{round_id}.json', proposals)
    return windows


def provider_settings(base_url=None, model=None, reasoning_effort=None, provider='openai-compatible'):
    if provider not in ('openai-compatible', 'anthropic'):
        raise ValueError('Unsupported Scheduler provider')
    model = model or os.environ.get('SCHEDULER_MODEL')
    if not model:
        raise RuntimeError('Set --scheduler-model or SCHEDULER_MODEL')
    if reasoning_effort == 'none':
        reasoning_effort = None
    if reasoning_effort not in (None, 'low', 'medium', 'high'):
        raise ValueError('Invalid Scheduler reasoning effort')
    if provider == 'anthropic' and reasoning_effort is not None:
        raise ValueError('Reasoning effort is not supported by the Anthropic adapter')
    default_base = 'https://api.anthropic.com/v1' if provider == 'anthropic' else os.environ.get('OPENAI_BASE_URL', '')
    base = (base_url or os.environ.get('SCHEDULER_BASE_URL') or default_base).rstrip('/')
    if not base:
        raise RuntimeError('Set --scheduler-base-url, SCHEDULER_BASE_URL, or OPENAI_BASE_URL')
    parts = urllib.parse.urlsplit(base)
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('Scheduler base URL must be HTTP(S) without credentials, query, or fragment')
    return base, model, reasoning_effort


def structured_call(root, artifact, prompt, *, base, model, reasoning_effort,
                    response_schema, images=(), metadata=None,
                    provider='openai-compatible', json_mode='schema', max_tokens=4096):
    """Call the Scheduler model; credentials and encoded images are never written to disk."""
    if json_mode not in ('schema', 'prompt'):
        raise ValueError('Scheduler JSON mode must be schema or prompt')
    key = read_key(provider)
    if provider == 'anthropic' or json_mode == 'prompt':
        prompt += '\nReturn a JSON object matching this schema:\n' + json.dumps(response_schema['json_schema']['schema'])
    user_content = [{'type': 'text', 'text': prompt}]
    for case_id, image_path in images:
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode('ascii')
        if provider == 'anthropic':
            user_content.append({'type': 'image', 'source': {'type': 'base64',
                'media_type': 'image/jpeg', 'data': encoded}})
        else:
            user_content.append({'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + encoded, 'detail': 'high'}})
    body = {'model': model, 'messages': [{'role': 'user', 'content': user_content}]}
    if provider == 'anthropic':
        body['max_tokens'] = max_tokens
    else:
        if reasoning_effort is not None:
            body['reasoning_effort'] = reasoning_effort
        if json_mode == 'schema':
            body['response_format'] = response_schema
    write_json(Path(root) / artifact, {
        'model': model,
        'provider': provider,
        'json_mode': 'prompt' if provider == 'anthropic' else json_mode,
        'reasoning_effort': reasoning_effort,
        'prompt': prompt,
        'image_case_ids': [case_id for case_id, _ in images],
        'image_count': len(images),
        'metadata': metadata or {},
    })
    suffix = '/messages' if provider == 'anthropic' else '/chat/completions'
    endpoint = base if base.endswith(suffix) else base + suffix
    headers = {'Content-Type': 'application/json'}
    if provider == 'anthropic':
        headers.update({'x-api-key': key, 'anthropic-version': '2023-06-01'})
    else:
        headers['Authorization'] = 'Bearer ' + key
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode('utf-8'), headers=headers)
    opener = urllib.request.build_opener()
    attempts = []
    result = None
    for attempt in range(3):
        started = time.monotonic()
        try:
            with opener.open(request, timeout=180) as response:
                envelope = json.load(response)
            message = (envelope['content'] if provider == 'anthropic' else
                       envelope['choices'][0]['message']['content'])
            if isinstance(message, list):
                message = ''.join(item.get('text', '') for item in message if isinstance(item, dict))
            result = json.loads(message)
            if not isinstance(result, dict):
                raise ValueError('Scheduler JSON response must be an object')
            attempts.append({'attempt': attempt + 1, 'outcome': 'success',
                             'served_model': envelope.get('model'),
                             'usage': envelope.get('usage'),
                             'elapsed_seconds': round(time.monotonic() - started, 3)})
            break
        except urllib.error.HTTPError as exc:
            code = exc.code
            attempts.append({'attempt': attempt + 1, 'outcome': 'http_error', 'http_status': code})
            if code in (400, 401, 403, 404) or attempt == 2:
                write_json(Path(root) / (str(artifact) + '.attempts.json'), {'attempts': attempts})
                raise RuntimeError(f'Scheduler request failed: HTTP {code}') from None
        except Exception as exc:
            attempts.append({'attempt': attempt + 1, 'outcome': 'error',
                             'error_type': type(exc).__name__,
                             'elapsed_seconds': round(time.monotonic() - started, 3)})
            if attempt == 2:
                write_json(Path(root) / (str(artifact) + '.attempts.json'), {'attempts': attempts})
                raise RuntimeError(f'Scheduler request failed: {type(exc).__name__}') from None
        time.sleep(2 * (attempt + 1))
    write_json(Path(root) / (str(artifact) + '.attempts.json'), {'attempts': attempts})
    return result


def history_for_task(root, round_id, task, *, traces=None, include_images=True):
    """Summarize earlier practice windows: Global-end baseline and Local evaluations."""
    root = Path(root)
    if traces is None:
        traces = _read_trace_rows(root)
    history = []
    for prior_round in range(max(0, round_id - 2), round_id):
        global_complete = root / f'r{prior_round}_global' / 'complete.json'
        local_complete = root / f'r{prior_round}_local' / 'complete.json'
        if not global_complete.is_file() or not local_complete.is_file():
            continue
        global_phase = json.loads(global_complete.read_text())
        local_phase = json.loads(local_complete.read_text())
        local_start = int(local_phase['start'])
        local_stop = int(local_phase['stop'])
        eval_interval = int(local_phase.get('eval_interval', 5))
        local_attempt = local_phase.get('phase_attempt_id')
        points = [('global_end_before_local', int(global_phase['stop']),
                   global_phase.get('phase_attempt_id'))]
        if local_start + eval_interval < local_stop:
            points.append((f'local_after_{eval_interval}_updates',
                           local_start + eval_interval, local_attempt))
        points.append((f'local_end_after_{local_stop - local_start}_updates',
                       local_stop, local_attempt))
        images = []
        summaries = []
        complete_timeline = True
        for label, update, attempt_id in points:
            episode_rows = [row for row in traces if row.get('mode') == 'eval'
                            and row.get('complete') and row.get('round') == prior_round
                            and row.get('task') == task and row.get('update') == update]
            if attempt_id is not None:
                episode_rows = [row for row in episode_rows
                                if row.get('phase_attempt_id') == attempt_id]
            episode_rows = _latest_episode_rows(episode_rows)
            if len(episode_rows) != 5:
                # History is optional context for the next decision.
                complete_timeline = False
                break
            if include_images:
                try:
                    images.extend(image_cases(root, prior_round, task, [update], episode_rows))
                except ValueError:
                    pass
            summaries.append({'point': label, 'update': update,
                'successes': sum(bool(row['success']) for row in episode_rows),
                'episodes': len(episode_rows),
                'stages': [int(row['stage']) + 1 for row in sorted(episode_rows, key=lambda row: row.get('slot', 0))]})
        if not complete_timeline:
            continue
        window_path = root / f'windows_round{prior_round}.json'
        prior_window = json.loads(window_path.read_text()).get(task) if window_path.is_file() else None
        if prior_window:
            prior_window = {key: prior_window[key] for key in
                            ('start_stage', 'end_stage', 'max_steps', 'reason') if key in prior_window}
        local_rows = [row for row in traces if row.get('mode') == 'train'
                      and row.get('round') == prior_round and row.get('task') == task
                      and row.get('phase_attempt_id') == local_phase.get('phase_attempt_id')
                      and local_phase['start'] <= row.get('update', -1) < local_phase['stop']]
        before, after = summaries[0], summaries[-1]
        delta = after['successes'] / after['episodes'] - before['successes'] / before['episodes']
        history.append({'intervention_round': prior_round, 'window': prior_window,
                        'intervention_timeline': summaries, 'images': images,
                        'success_rate_change': delta,
                        'fresh_controls': sum(row.get('window_steps', 0) for row in local_rows),
                        'restoration_seconds': sum(row.get('restoration_seconds', 0.0) for row in local_rows)})
    return history


def _task_prompt(benchmark, task, records, stages, case_ids, instruction, history):
    task_data = {
        'task': task,
        'instruction': instruction,
        'instruction_source': 'policy observation' if instruction else 'not present in the observation; use task name and configured predicates only',
        'ordered_stage_subtasks': [{'stage': index + 1, 'predicates': stage} for index, stage in enumerate(stages)],
        'episode_outcomes': [{'update': row['update'], 'success': bool(row['success']),
                              'stage': int(row['stage']) + 1} for row in records],
        'image_case_ids_in_order': case_ids,
        'historical_local_interventions': [
            {key: value for key, value in item.items() if key != 'images'} for item in history],
    }
    return (
        'You are a visual robotics RL analyst. We are training one shared pi0.5 policy '
        f'for {benchmark.title}. Diagnose bottlenecks using normal-start validation '
        'rollouts from Global RL. Inspect the attached 4x4 timeline montages. Each timeline '
        f'cell is a 2x2 camera panel showing the {benchmark.cameras} views; its caption gives '
        'the control step and task stage. '
        'The episode outcome is explicitly labeled SUCCESS or FAILURE in the montage. Do not infer '
        'success from visual appearance when the recorded outcome says otherwise. Describe visible '
        'progress, the earliest stage where failures appear to stall, evidence, and uncertainty. '
        'For prior Local interventions, compare the Global-end pre-intervention baseline and the '
        'Local evaluations descriptively; do not claim causality. '
        'Do not report reward values. Return only JSON matching the supplied schema.\n'
        + json.dumps(task_data, ensure_ascii=False)
    )


def _diagnose_tasks(root, round_id, checkpoints, tasks, rows, *, benchmark, base, model, reasoning_effort,
                    provider, json_mode, max_tokens, output_root):
    root = Path(root)
    results = {}
    histories = {}
    traces = _read_trace_rows(root)
    diagnosis_schema = _schema('task_visual_diagnosis', {
        'task': {'type': 'string'},
        'visible_progress': {'type': 'string'},
        'recurring_stall_stage': {'type': ['integer', 'null']},
        'evidence': {'type': 'array', 'items': {'type': 'string'}},
        'uncertainty': {'type': 'string'},
    }, ['task', 'visible_progress', 'recurring_stall_stage', 'evidence', 'uncertainty'])
    for task in tasks:
        task_rows = [row for row in rows if row['task'] == task]
        if len({row['stage_signature'] for row in task_rows}) != 1:
            raise ValueError('Inconsistent stage schema for ' + task)
        ordered = sorted(task_rows, key=lambda row: (row['update'], row.get('slot', 0)))
        cases = image_cases(root, round_id, task, checkpoints, ordered)
        history = history_for_task(root, round_id, task, traces=traces)
        histories[task] = [{key: value for key, value in entry.items() if key != 'images'}
                           for entry in history]
        current_images = [(case['case_id'], case['image_path']) for case in cases]
        historical_images = []
        for intervention in history:
            for case in intervention['images']:
                case_id = f"history-r{intervention['intervention_round']}-{case['case_id']}"
                historical_images.append((case_id, case['image_path']))
        case_ids = [case_id for case_id, _ in current_images + historical_images]
        instruction = next((row.get('task_instruction') for row in ordered if row.get('task_instruction')), None)
        prompt = _task_prompt(benchmark, task, ordered, ordered[-1]['stages'], case_ids, instruction, history)
        response = structured_call(output_root,
            f'visual_feedback/round_{round_id}/scheduler_request_{task}.json', prompt,
            base=base, model=model, reasoning_effort=reasoning_effort,
            provider=provider, json_mode=json_mode, max_tokens=max_tokens,
            response_schema=diagnosis_schema,
            images=current_images + historical_images,
            metadata={'round': round_id, 'task': task, 'checkpoint_steps': checkpoints})
        if response.get('task') != task:
            raise ValueError(f'Scheduler diagnosis task mismatch: expected {task}')
        if (not isinstance(response.get('visible_progress'), str) or not isinstance(response.get('uncertainty'), str)
                or not isinstance(response.get('evidence'), list)
                or not all(isinstance(item, str) for item in response['evidence'])):
            raise ValueError(f'Invalid Scheduler diagnosis for {task}')
        stall = response.get('recurring_stall_stage')
        if stall is not None and (type(stall) is not int or not 1 <= stall <= len(ordered[-1]['stages'])):
            raise ValueError(f'Scheduler proposed an out-of-range stage for {task}')
        results[task] = response
        write_json(output_root / f'visual_feedback/round_{round_id}/scheduler_response_{task}.json', response)
    return results, histories


def _window_instructions(benchmark):
    return (
        'For every eligible task, recommend the next Local practice window: start_stage and '
        'end_stage are inclusive and zero-based, max_steps is a multiple of '
        f'{benchmark.chunk} from {benchmark.chunk} to 1000. Choose start_stage only from that '
        'task\'s available training entries, and keep the end inside its configured stage range. '
        'Use the earliest recurring stall when supported; do not recommend a window for an '
        'ineligible task. Give a concise evidence-based reason and state uncertainty when the '
        'evidence does not support a clear choice. Return exactly one window for each eligible '
        'task, no extra tasks, and JSON only.'
    )


def _select_windows(root, round_id, candidates, task_diagnoses, histories, *, benchmark, base, model,
                    reasoning_effort, provider, json_mode, max_tokens, output_root, metadata):
    if not candidates:
        return {}
    evidence = {'eligible_tasks': _public_candidates(candidates),
                'practice_history': {task: histories[task] for task in candidates},
                'supported_milestone_checks': _check_descriptions(benchmark.milestone_checks),
                'task_subtask_checkers': {task: benchmark.task_evidence(task) for task in candidates}}
    if task_diagnoses is not None:
        evidence['visual_diagnoses'] = task_diagnoses
    prompt = (
        f'You are coordinating the next Local RL phase for {benchmark.title}. '
        + ('The per-task diagnoses below were grounded in evaluation timeline montages. '
           if task_diagnoses is not None else
           'Use the checkpoint evaluation outcomes below; no images are provided. ')
        + 'Use prior windows, interaction costs, and normal-start outcomes to inform window selection. '
        + _window_instructions(benchmark) + _MILESTONE_INSTRUCTIONS + '\n'
        + json.dumps(evidence, ensure_ascii=False)
    )
    response = structured_call(output_root, f'scheduler_request_round{round_id}.json', prompt,
        base=base, model=model, reasoning_effort=reasoning_effort,
        provider=provider, json_mode=json_mode, max_tokens=max_tokens,
        response_schema=_selection_schema(candidates, benchmark.milestone_checks),
        metadata=metadata)
    return _save_selection(output_root, round_id, model, response, candidates, benchmark)


def select(root, round_id, checkpoints, *, benchmark, base_url=None, model=None,
           reasoning_effort=None, evidence='visual', phase_attempt_id=None,
           provider='openai-compatible', json_mode='schema', max_tokens=4096, output_root=None):
    """Choose one practice window per eligible task.

    ``evidence='visual'`` first diagnoses each task from evaluation montages;
    ``evidence='text'`` decides from evaluation outcomes alone.
    """
    if evidence not in ('visual', 'text'):
        raise ValueError('Scheduler evidence must be visual or text')
    root = Path(root)
    output_root = Path(output_root) if output_root is not None else root
    checkpoint_steps = _checkpoint_steps(checkpoints)
    tasks, rows = _load_eval_rows(root, round_id, checkpoint_steps, phase_attempt_id)
    candidates = diagnostics(root, round_id, checkpoint_steps, phase_attempt_id)
    base, model, reasoning_effort = provider_settings(base_url, model, reasoning_effort, provider)
    settings = dict(benchmark=benchmark, base=base, model=model, reasoning_effort=reasoning_effort,
                    provider=provider, json_mode=json_mode, max_tokens=max_tokens, output_root=output_root)
    if evidence == 'visual':
        task_diagnoses, histories = _diagnose_tasks(root, round_id, checkpoint_steps, tasks, rows, **settings)
    else:
        traces = _read_trace_rows(root)
        task_diagnoses = None
        histories = {task: history_for_task(root, round_id, task, traces=traces, include_images=False)
                     for task in candidates}
    windows = _select_windows(root, round_id, candidates, task_diagnoses, histories, **settings,
                              metadata={'round': round_id, 'checkpoint_steps': checkpoint_steps,
                                        'eligible_tasks': list(candidates), 'diagnosed_tasks': tasks})
    for task, window in windows.items():
        window['stage_signature'] = candidates[task]['stage_signature']
    write_json(output_root / f'windows_round{round_id}.json', windows)
    return windows


def read_key(provider='openai-compatible'):
    """Read credentials only in the coordinator process; never save or print them."""
    key = os.environ.get('SCHEDULER_API_KEY') or os.environ.get(
        'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'OPENAI_API_KEY')
    if not key:
        raise RuntimeError('Set SCHEDULER_API_KEY before starting the coordinator')
    return key
