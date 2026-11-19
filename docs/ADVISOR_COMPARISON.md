# Comparing practice-window advisors

The advisor diagnoses Global rollout evidence and selects Local practice windows. It is **not** the pi0.5 actor. This adapter supports GPT-6 Astra (the existing default), Anthropic Messages API models, and other models offered by an OpenAI-compatible multimodal endpoint. Supply the exact model ID accepted by your provider; model availability, image limits, API pricing and credentials depend on that provider. Visual mode sends rollout images to the selected endpoint. Only use an endpoint authorized to receive them.

## Compare on the same recorded evidence without running training

Use a completed Global phase with `tasks.json`, `traces/`, `entries/` and, for visual mode, its recorded three-camera montage files. Each task and checkpoint must have exactly five completed evaluation episodes. Record the same source run, committed phase attempt, checkpoints, mode, and prior history for every model. Keep the source immutable during all comparisons; `comparison_config.json`, request manifests, responses and selected windows are written to a **new output directory outside the source run**.

```bash
# Example 1: Astra, through the existing gateway; do not put keys in arguments.
export OPENAI_BASE_URL="https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
export OPENAI_API_KEY="YOUR_KEY"
python -m structrl.rsi.compare_advisors \
  --source-root /path/to/finished/rsi-run --output-root /path/to/comparison/astra-01 \
  --round 0 --checkpoints 5 10 15 20 25 \
  --advisor-model gpt-6-astra --advisor-reasoning-effort low

# Example 2: Claude via the native Anthropic Messages API.
unset ADVISOR_API_KEY ADVISOR_BASE_URL
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY"
python -m structrl.rsi.compare_advisors \
  --source-root /path/to/finished/rsi-run --output-root /path/to/comparison/opus-01 \
  --round 0 --checkpoints 5 10 15 20 25 --advisor-provider anthropic \
  --advisor-model claude-opus-5-5

# Example 3: Qwen served by a vision-capable OpenAI-compatible endpoint.
export ADVISOR_BASE_URL="http://YOUR_QWEN_SERVER:8000/v1"
export ADVISOR_API_KEY="YOUR_SERVER_KEY"
python -m structrl.rsi.compare_advisors \
  --source-root /path/to/finished/rsi-run --output-root /path/to/comparison/qwen-01 \
  --round 0 --checkpoints 5 10 15 20 25 --advisor-model Qwen/Qwen3.8-27B \
  --advisor-json-mode prompt
```

Default visual mode sends all 25 current episode montages per task and may include prior-intervention montages. The endpoint must actually support images, the input size and context required. `--advisor-mode text` is a different evidence condition, not a direct visual-model comparison. For an OpenAI-compatible service that supports strict JSON schemas, use `--advisor-json-mode schema` (default); use `prompt` for servers without that feature. Native Anthropic uses prompt-based JSON with the same schema; output and selected windows are checked locally and invalid replies fail rather than being silently substituted. The modes have different format instructions and should be reported. Reasoning effort is omitted for generic models unless explicitly supplied; Astra uses `low`. The native Anthropic adapter does not expose an effort parameter.

The saved JSON files include requested and, if returned by the service, served model name, latency and usage. API keys and encoded images are not persisted. Configure `ADVISOR_API_KEY` or the provider-specific key (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) on the coordinator **after** starting Ray; subprocesses launched by the coordinator do not inherit these three variables. Never pass credentials in a URL or command-line argument.

Inspect visual diagnosis, evidence and validated window choices across models, ideally with independent human labels and repeated model calls. This only measures agreement on the saved episodes; it does **not** measure the effect on policy success rate. For that, run independent RSI training with separate `--run-root` values, matched initial checkpoint, tasks, training/evaluation protocol and interaction budgets. Report actual completed runs, failures, model configuration, evaluation episodes and uncertainty. Do not equate multiple evaluations of one checkpoint with independent training runs. Use the original StructRL reset/scenario protocol, and keep outcome videos where the evaluation system supports them.

## Selecting an advisor for RSI training

The existing `scripts/run_robocasa_rsi.sh` supports `--advisor-provider`, `--advisor-base-url`, `--advisor-model`, `--advisor-reasoning-effort`, `--advisor-json-mode` and `--advisor-max-tokens`. The old `--astra-*` arguments still work as aliases. Astra remains the default; use separate run roots for different advisors. The coordinator records `advisor_config.json` and rejects changing the advisor on a resumed run. Existing `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `ASTRA_MODEL`, and `ASTRA_REASONING_EFFORT` remain supported.
