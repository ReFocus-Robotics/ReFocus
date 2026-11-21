# Comparing practice-window Schedulers

The Scheduler diagnoses Global rollout evidence and selects Local practice windows. It is **not** the π<sub>0.5</sub> actor. The adapter supports Anthropic Messages API models and any vision-capable model served by an OpenAI-compatible endpoint. Supply the exact model ID accepted by your provider; availability, image limits, pricing and credentials depend on that provider. Visual evidence sends rollout images to the selected endpoint, so only use an endpoint authorized to receive them.

## Compare on the same recorded evidence without running training

Use a completed Global phase with `tasks.json`, `traces/`, `entries/` and, for visual evidence, its recorded timeline montages. Each task and checkpoint must have exactly five completed evaluation episodes. Record the same source run, committed phase attempt, checkpoints, evidence mode, and prior history for every model. Keep the source immutable; `comparison_config.json`, request manifests, responses and selected windows are written to a **new output directory outside the source run**.

```bash
# Example 1: an OpenAI-compatible gateway. Do not put keys in arguments.
export OPENAI_BASE_URL="https://YOUR_ENDPOINT/v1"
export SCHEDULER_API_KEY="YOUR_KEY"
python -m refocus.rounds.compare_schedulers \
  --benchmark robocasa365 \
  --source-root /path/to/finished/run --output-root /path/to/comparison/gateway-01 \
  --round 0 --checkpoints 5 10 15 20 25 \
  --scheduler-model gpt-6-astra --scheduler-reasoning-effort low

# Example 2: Claude through the Anthropic Messages API.
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY"
python -m refocus.rounds.compare_schedulers \
  --benchmark robocasa365 \
  --source-root /path/to/finished/run --output-root /path/to/comparison/opus-01 \
  --round 0 --checkpoints 5 10 15 20 25 --scheduler-provider anthropic \
  --scheduler-model claude-opus-5-5

# Example 3: a vision model on a local OpenAI-compatible server.
export SCHEDULER_BASE_URL="http://YOUR_SERVER:8000/v1"
export SCHEDULER_API_KEY="YOUR_SERVER_KEY"
python -m refocus.rounds.compare_schedulers \
  --benchmark robocasa365 \
  --source-root /path/to/finished/run --output-root /path/to/comparison/qwen-01 \
  --round 0 --checkpoints 5 10 15 20 25 --scheduler-model Qwen/Qwen3.8-27B \
  --scheduler-json-mode prompt
```

`--benchmark libero_10` runs the same comparison on a LIBERO run, whose montages carry agentview and wrist panels instead of three camera views.

The default visual mode sends all 25 current episode montages per task and may include prior-intervention montages. The endpoint must support images, the input size, and the context length required. `--scheduler-evidence text` is a different evidence condition, not a direct visual-model comparison; report it as such. For an OpenAI-compatible service that supports strict JSON schemas, use `--scheduler-json-mode schema` (default); use `prompt` for servers without that feature. The native Anthropic adapter always uses prompt-based JSON with the same schema. Outputs and selected windows are validated locally, and invalid replies fail rather than being silently substituted. Reasoning effort is omitted unless explicitly supplied, and the Anthropic adapter does not expose an effort parameter.

Saved JSON files include the requested model and, when the service returns it, the served model, latency and usage. API keys and encoded images are not persisted. Set `SCHEDULER_API_KEY` (or the provider key) in the coordinator environment; subprocesses launched by the coordinator do not inherit it. Never pass credentials in a URL or a command-line argument.

Inspect diagnoses, evidence and validated window choices across models, ideally with independent human labels and repeated calls. This measures agreement on saved episodes only; it does **not** measure the effect on policy success rate. For that, run independent training with separate `--run-root` values, a matched initial checkpoint, tasks, evaluation protocol and interaction budget. Report completed runs, failures, model configuration, evaluation episodes and uncertainty. Do not equate repeated evaluations of one checkpoint with independent training runs.

## Selecting a Scheduler

`scripts/run_rounds.sh` accepts `--scheduler-provider`, `--scheduler-base-url`, `--scheduler-model`, `--scheduler-reasoning-effort`, `--scheduler-json-mode` and `--scheduler-max-tokens`. Use a separate run root per Scheduler. The coordinator records `scheduler_config.json` and rejects changing it on a resumed run.
