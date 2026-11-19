"""Multimodal advisor feedback with bounded transport and deterministic gates."""
import base64
import ast
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

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
        # mtime provides a deterministic last-write-wins tie-break for legacy
        # traces that predate phase_attempt_id and for duplicate slots.
        row['_trace_mtime_ns'] = path.stat().st_mtime_ns
        rows.append(row)
    return rows


def _deduplicate_episode_rows(rows):
    """Keep the latest complete trace for each task/update/worker slot."""
    selected = {}
    unkeyed = []
    for source in rows:
        row = dict(source)
        slot = row.get('slot')
        if slot is None:
            # Older trace formats have no slot identity; retain them as-is.
            unkeyed.append(row)
            continue
        key = (row.get('task'), row.get('update'), slot)
        previous = selected.get(key)
        if previous is None or row['_trace_mtime_ns'] >= previous['_trace_mtime_ns']:
            selected[key] = row
    result = unkeyed + list(selected.values())
    result.sort(key=lambda row: (row.get('update', 0), row.get('task', ''),
                                 row.get('slot', -1), row['_trace_mtime_ns']))
    for row in result:
        row.pop('_trace_mtime_ns', None)
    return result


def _load_eval_rows(root, round_id, checkpoints, phase_attempt_id=None):
    root = Path(root)
    tasks = json.loads((root / 'tasks.json').read_text())
    rows = _read_trace_rows(root)
    rows = [row for row in rows if row.get('mode') == 'eval' and row.get('complete')
            and row.get('round') == round_id and row.get('update') in checkpoints]
    # A failed phase can be retried from the preceding completed checkpoint.
    # Keep its traces for audit, but only use the attempt committed by complete.json.
    if phase_attempt_id is not None:
        rows = [row for row in rows if row.get('phase_attempt_id') == phase_attempt_id]
    rows = _deduplicate_episode_rows(rows)
    for task in tasks:
        for update in checkpoints:
            count = sum(row.get('task') == task and row.get('update') == update for row in rows)
            if count != 5:
                raise ValueError(f'Incomplete StructRL-style evaluation: {task}, update {update}, expected 5 episodes, got {count}')
    return tasks, rows


def diagnostics(root, round_id, checkpoints, phase_attempt_id=None):
    root = Path(root)
    checkpoints = _checkpoint_steps(checkpoints)
    tasks, rows = _load_eval_rows(root, round_id, checkpoints, phase_attempt_id)
    candidates = {}
    for task in tasks:
        records = [row for row in rows if row['task'] == task]
        folder = root / 'entries' / str(round_id) / task
        starts = sorted(int(path.name) for path in folder.glob('*') if path.is_dir() and any(path.glob(entry_pattern(phase_attempt_id))))
        signatures = {row['stage_signature'] for row in records}
        if len(signatures) != 1:
            raise ValueError('Inconsistent stage schema for ' + task)
        if starts and any(not row['success'] for row in records):
            candidates[task] = dict(available_starts=starts, stages=records[-1]['stages'],
                stage_signature=records[-1]['stage_signature'],
                evaluations=[{key: row[key] for key in ('update', 'success', 'stage')} for row in records])
    return candidates


def _public_candidates(candidates):
    """Project only model-facing evidence; stage hashes stay internal."""
    return {task: {key: candidate[key] for key in
            ('available_starts', 'stages', 'evaluations')} for task, candidate in candidates.items()}


def _checker_evidence(task):
    """Expose the existing task-specific checker as evidence, not executable advice."""
    path = Path(__file__).resolve().parents[1] / 'subtask_checker.py'
    try:
        source = path.read_text()
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == 'cls_name'
                and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == task):
            return ast.get_source_segment(source, node)
    return None


def _selection_schema(candidates):
    milestone = {'type': 'object', 'properties': {
        'name': {'type': 'string'},
        'completion_condition': {'type': 'string'},
        'required_simulator_signals': {'type': 'array', 'items': {'type': 'string'}},
        'source_evidence': {'type': 'string'},
        'missing_information': {'type': 'string'}},
        'required': ['name', 'completion_condition', 'required_simulator_signals',
                     'source_evidence', 'missing_information'], 'additionalProperties': False}
    window = {'type': 'object', 'properties': {
        'start_stage': {'type': 'integer'}, 'end_stage': {'type': 'integer'},
        'max_steps': {'type': 'integer'}, 'reason': {'type': 'string'},
        'milestones': {'type': 'array', 'items': milestone}},
        'required': ['start_stage', 'end_stage', 'max_steps', 'reason', 'milestones'],
        'additionalProperties': False}
    return _schema('robo_local_window_selection', {
        'windows': {'type': 'object', 'properties': {task: window for task in candidates},
                    'required': list(candidates), 'additionalProperties': False}}, ['windows'])


_MILESTONE_INSTRUCTIONS = (
    ' For each selected window, also propose up to three fine-grained milestones within its '
    'target subtask, with names, completion conditions, required simulator signals, '
    'source evidence and missing information. Use the supplied task checker and stage '
    'definitions as evidence, not proof that a new milestone is implemented. Refer to exact '
    'existing signal or function names only when shown in the input; otherwise describe the '
    'required observable conceptually and identify the missing implementation. Never invent '
    'API names, thresholds, reward weights, or treat missing evidence as failure. '
    'A proposed milestone reward is a one-time event on a new valid completion, not a '
    'recurring reward while a condition remains true; account for the restored entry state '
    'and do not award milestones already satisfied on restore. A verified refinement may '
    'apply in the target stage of Normal, Prefix, and Suffix Windows. Its milestones share '
    'the original target subtask budget, and the final milestone uses its original condition. '
    'Output proposals for review only.'
)


def _save_selection(output_root, round_id, model, response, candidates):
    proposed = response['windows']
    windows = {task: {key: value for key, value in window.items() if key != 'milestones'}
               for task, window in proposed.items()}
    validate_windows(windows, candidates)
    proposals = {task: {'window': windows[task], 'milestones': window.get('milestones', []),
                        'status': 'proposed_not_validated'}
                 for task, window in proposed.items()}
    write_json(output_root / f'advisor_response_round{round_id}.json',
               {'model': model, 'windows': windows, 'milestone_proposals': proposals})
    write_json(output_root / f'milestone_proposals_round{round_id}.json', proposals)
    return windows


def _schema(name, properties, required):
    return {'type': 'json_schema', 'json_schema': {
        'name': name, 'strict': True,
        'schema': {'type': 'object', 'properties': properties,
                   'required': required, 'additionalProperties': False}}}


def _provider_settings(base_url=None, model=None, reasoning_effort=None, provider='openai-compatible'):
    if provider not in ('openai-compatible', 'anthropic'):
        raise ValueError('Unsupported advisor provider')
    model = model or os.environ.get('ADVISOR_MODEL') or os.environ.get('ASTRA_MODEL', 'gpt-6-astra')
    if model in ('gpt-6-astra', 'openai/gpt-6-astra'):
        if provider != 'openai-compatible':
            raise ValueError('GPT-6 Astra requires its OpenAI-compatible API')
        reasoning_effort = reasoning_effort or os.environ.get('ASTRA_REASONING_EFFORT', 'low')
        if reasoning_effort != 'low':
            raise ValueError('GPT-6 Astra tests require reasoning effort low')
    if reasoning_effort == 'none':
        reasoning_effort = None
    if reasoning_effort not in (None, 'low', 'medium', 'high'):
        raise ValueError('Invalid advisor reasoning effort')
    if provider == 'anthropic' and reasoning_effort is not None:
        raise ValueError('Reasoning effort is not supported by the Anthropic adapter')
    default_base = 'https://api.anthropic.com/v1' if provider == 'anthropic' else os.environ.get('OPENAI_BASE_URL', '')
    base = (base_url or os.environ.get('ADVISOR_BASE_URL') or default_base).rstrip('/')
    if not base:
        raise RuntimeError('Set ADVISOR_BASE_URL or OPENAI_BASE_URL before starting the RSI run')
    parts = urllib.parse.urlsplit(base)
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('Advisor base URL must be HTTP(S) without credentials, query, or fragment')
    return base, model, reasoning_effort


def _structured_call(root, artifact, prompt, *, base, model, reasoning_effort,
                     response_schema, images=(), metadata=None,
                     provider='openai-compatible', json_mode='schema', max_tokens=4096):
    """Call the advisor without persisting credentials or encoded images."""
    if json_mode not in ('schema', 'prompt'):
        raise ValueError('advisor JSON mode must be schema or prompt')
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
    safe_artifact = {
        'model': model,
        'provider': provider,
        'json_mode': 'prompt' if provider == 'anthropic' else json_mode,
        'reasoning_effort': reasoning_effort,
        'prompt': prompt,
        'image_case_ids': [case_id for case_id, _ in images],
        'image_count': len(images),
        'metadata': metadata or {},
    }
    write_json(Path(root) / artifact, safe_artifact)
    suffix = '/messages' if provider == 'anthropic' else '/chat/completions'
    endpoint = base if base.endswith(suffix) else base + suffix
    headers = {'Content-Type': 'application/json'}
    if provider == 'anthropic':
        headers.update({'x-api-key': key, 'anthropic-version': '2023-06-01'})
    else:
        headers['Authorization'] = 'Bearer ' + key
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode('utf-8'),
        headers=headers)
    proxy = os.environ.get('ASTRA_PROXY')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': proxy} if proxy else {}))
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
                raise ValueError('Advisor JSON response must be an object')
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
                raise RuntimeError(f'Advisor request failed: HTTP {code}') from None
        except Exception as exc:
            attempts.append({'attempt': attempt + 1, 'outcome': 'error',
                             'error_type': type(exc).__name__,
                             'elapsed_seconds': round(time.monotonic() - started, 3)})
            if attempt == 2:
                write_json(Path(root) / (str(artifact) + '.attempts.json'), {'attempts': attempts})
                raise RuntimeError(f'Advisor request failed: {type(exc).__name__}') from None
        time.sleep(2 * (attempt + 1))
    write_json(Path(root) / (str(artifact) + '.attempts.json'), {'attempts': attempts})
    return result


def _history_for_task(root, round_id, task, *, traces=None, include_images=True):
    """Include Global-end baseline and early/final Local evals for prior interventions."""
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
        # Baseline belongs to the completed Global attempt. Both Local points
        # belong to the completed Local attempt, not to a failed retry.
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
            episode_rows = _deduplicate_episode_rows(episode_rows)
            if len(episode_rows) != 5:
                # Historical evidence is optional context. Do not stop the next
                # training phase because an old intervention lacks one snapshot.
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
                        'replayed_controls': sum(row.get('replay_steps', 0) for row in local_rows),
                        'restoration_seconds': sum(row.get('restoration_seconds', 0.0) for row in local_rows)})
    return history


def _task_prompt(task, records, stages, case_ids, instruction, history):
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
        'for RoboCasa365 Composite-Seen with StructRL rewards. This is a development diagnostic '
        'for the Global phase of RSI Version C, not a final benchmark. Inspect the attached 4x4 '
        'timeline montages. Each timeline cell is a 2x2 camera panel labeled agentview_left, '
        'agentview_right, wrist, or Not recorded; its caption gives the control step and task stage. '
        'The episode outcome is explicitly labeled SUCCESS or FAILURE in the montage. Do not infer '
        'success from visual appearance when the recorded outcome says otherwise. Describe visible '
        'progress, the earliest stage where failures appear to stall, evidence, and uncertainty. '
        'For prior Local interventions, compare the Global-end pre-intervention baseline, Local +5, '
        'and Local +25 evaluations descriptively; do not claim causality. '
        'Do not report reward values. Return only JSON matching the supplied schema.\n'
        + json.dumps(task_data, ensure_ascii=False)
    )


def _diagnose_visual_tasks(root, round_id, checkpoints, tasks, rows, *, base, model, reasoning_effort,
                           provider, json_mode, max_tokens, output_root):
    root = Path(root)
    results = {}
    histories = {}
    traces = _read_trace_rows(root)
    diagnosis_schema = _schema('robo_task_visual_diagnosis', {
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
        history = _history_for_task(root, round_id, task, traces=traces)
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
        prompt = _task_prompt(task, ordered, ordered[-1]['stages'], case_ids,
                              instruction, history)
        response = _structured_call(output_root,
            f'visual_feedback/round_{round_id}/advisor_request_{task}.json', prompt,
            base=base, model=model, reasoning_effort=reasoning_effort,
            provider=provider, json_mode=json_mode, max_tokens=max_tokens,
            response_schema=diagnosis_schema,
            images=current_images + historical_images,
            metadata={'round': round_id, 'task': task, 'checkpoint_steps': checkpoints})
        if response.get('task') != task:
            raise ValueError(f'Advisor visual response task mismatch: expected {task}')
        if not isinstance(response.get('visible_progress'), str) or not isinstance(response.get('uncertainty'), str) or not isinstance(response.get('evidence'), list) or not all(isinstance(item, str) for item in response['evidence']):
            raise ValueError(f'Invalid advisor visual diagnosis for {task}')
        stage_count = len(ordered[-1]['stages'])
        stall = response.get('recurring_stall_stage')
        if stall is not None and (type(stall) is not int or not 1 <= stall <= stage_count):
            raise ValueError(f'Advisor proposed an out-of-range stage for {task}')
        results[task] = response
        write_json(output_root / f'visual_feedback/round_{round_id}/advisor_response_{task}.json', response)
    return results, histories


def select_visual(root, round_id, checkpoints, *, base_url=None, model=None, reasoning_effort=None,
                  phase_attempt_id=None, provider='openai-compatible', json_mode='schema', max_tokens=4096,
                  output_root=None):
    root = Path(root)
    output_root = Path(output_root) if output_root is not None else root
    checkpoint_steps = _checkpoint_steps(checkpoints)
    tasks, rows = _load_eval_rows(root, round_id, checkpoint_steps, phase_attempt_id)
    candidates = diagnostics(root, round_id, checkpoint_steps, phase_attempt_id)
    base, model, reasoning_effort = _provider_settings(base_url, model, reasoning_effort, provider)
    task_diagnoses, histories = _diagnose_visual_tasks(root, round_id, checkpoint_steps, tasks, rows,
                                             base=base, model=model,
                                             reasoning_effort=reasoning_effort, provider=provider,
                                             json_mode=json_mode, max_tokens=max_tokens, output_root=output_root)
    if not candidates:
        windows = {}
    else:
        eligible_for_prompt = _public_candidates(candidates)
        selection_schema = _selection_schema(candidates)
        prompt = (
            'You are coordinating the next Local RL phase for RoboCasa365. '
            'The per-task visual diagnoses below were grounded in 4x4, three-camera evaluation montages. '
            'Use prior windows, interaction costs, and normal-start outcomes to inform window selection. '
            'For every eligible task, explicitly recommend the next Local practice window: start_stage '
            'and end_stage are inclusive and zero-based, max_steps is a multiple of 50 from 50 to 1000. '
            'Choose start_stage only from that task\'s available training entries, and keep the end '
            'inside its configured stage range. Use the earliest recurring visual stall when supported; '
            'do not recommend a window for an ineligible task. Give a concise evidence-based reason and '
            'state uncertainty when the images do not support a clear choice. Return exactly one window '
            'for each eligible task, no extra tasks, and JSON only.'
            + _MILESTONE_INSTRUCTIONS + '\n'
            + json.dumps({'eligible_tasks': eligible_for_prompt, 'visual_diagnoses': task_diagnoses,
                          'practice_history': {task: histories[task] for task in candidates},
                          'existing_task_checkers': {task: _checker_evidence(task) for task in candidates}},
                         ensure_ascii=False)
        )
        response = _structured_call(output_root, f'advisor_request_round{round_id}.json', prompt,
            base=base, model=model, reasoning_effort=reasoning_effort,
            provider=provider, json_mode=json_mode, max_tokens=max_tokens,
            response_schema=selection_schema,
            metadata={'round': round_id, 'checkpoint_steps': checkpoint_steps,
                      'eligible_tasks': list(candidates), 'diagnosed_tasks': tasks})
        windows = _save_selection(output_root, round_id, model, response, candidates)
    for task, window in windows.items():
        window['stage_signature'] = candidates[task]['stage_signature']
    write_json(output_root / f'windows_round{round_id}.json', windows)
    return windows


def select_text(root, round_id, checkpoints, *, base_url=None, model=None, reasoning_effort=None,
                phase_attempt_id=None, provider='openai-compatible', json_mode='schema', max_tokens=4096,
                output_root=None):
    """Explicit legacy text path for diagnostic-only use; formal RSI defaults to images."""
    root = Path(root)
    output_root = Path(output_root) if output_root is not None else root
    checkpoints = _checkpoint_steps(checkpoints)
    candidates = diagnostics(root, round_id, checkpoints, phase_attempt_id)
    if not candidates:
        windows = {}
    else:
        base, model, reasoning_effort = _provider_settings(base_url, model, reasoning_effort, provider)
        traces = _read_trace_rows(root)
        histories = {task: _history_for_task(root, round_id, task, traces=traces, include_images=False)
                     for task in candidates}
        prompt = ('Select one practice window per eligible RoboCasa task from checkpoint evaluations. '
                  'Use prior windows, interaction costs, and normal-start outcomes to inform selection. '
                  'Return JSON matching the supplied schema. Stages are zero-based and end_stage is inclusive. '
                  'Choose an available_start at or before a recurring failure. max_steps must be a multiple '
                  'of 50, from 50 to 1000. Do not invent entries. This is development evaluation.'
                  + _MILESTONE_INSTRUCTIONS + '\n'
                  + json.dumps({'eligible_tasks': _public_candidates(candidates),
                                'practice_history': histories,
                                'existing_task_checkers': {task: _checker_evidence(task) for task in candidates}},
                               ensure_ascii=False))
        selection_schema = _selection_schema(candidates)
        response = _structured_call(output_root, f'advisor_request_round{round_id}.json', prompt,
            base=base, model=model, reasoning_effort=reasoning_effort,
            provider=provider, json_mode=json_mode, max_tokens=max_tokens,
            response_schema=selection_schema)
        windows = _save_selection(output_root, round_id, model, response, candidates)
    for task, window in windows.items():
        window['stage_signature'] = candidates[task]['stage_signature']
    write_json(output_root / f'windows_round{round_id}.json', windows)
    return windows


def select(root, round_id, checkpoints, *, base_url=None, model=None,
           reasoning_effort=None, mode='visual', phase_attempt_id=None,
           provider='openai-compatible', json_mode='schema', max_tokens=4096, output_root=None):
    if mode == 'visual':
        return select_visual(root, round_id, checkpoints, base_url=base_url,
                             model=model, reasoning_effort=reasoning_effort,
                             phase_attempt_id=phase_attempt_id, provider=provider,
                             json_mode=json_mode, max_tokens=max_tokens, output_root=output_root)
    if mode == 'text':
        return select_text(root, round_id, checkpoints, base_url=base_url,
                           model=model, reasoning_effort=reasoning_effort,
                           phase_attempt_id=phase_attempt_id, provider=provider,
                           json_mode=json_mode, max_tokens=max_tokens, output_root=output_root)
    raise ValueError('advisor mode must be visual or text')


def read_key(provider='openai-compatible'):
    """Read credentials only in the coordinator process; never save or print them."""
    key = os.environ.get('ADVISOR_API_KEY') or os.environ.get(
        'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'OPENAI_API_KEY')
    if not key:
        raise RuntimeError('Set the advisor API key in the environment before starting the RSI run')
    return key
