# Comparing practice-window Schedulers

The Scheduler diagnoses Global rollout evidence and chooses Local practice
windows. It is **not** the π<sub>0.5</sub> actor. The adapter supports Anthropic
Messages API models and any vision-capable model behind an OpenAI-compatible
endpoint; availability, image limits and credentials depend on that provider.
Rollout images go to the selected endpoint, so only use one authorized to
receive them.

## Compare on recorded evidence without retraining

Point every run at the same finished Global phase (`tasks.json`, `traces/`,
`entries/`, and its timeline montages for visual evidence), with five
completed evaluation episodes per task and checkpoint. Results go to a **new
output directory outside the source run**, which stays untouched.

```bash
# OpenAI-compatible gateway
export OPENAI_BASE_URL="https://YOUR_ENDPOINT/v1"
export SCHEDULER_API_KEY="YOUR_KEY"
python -m refocus.rounds.compare_schedulers \
  --benchmark robocasa365 \
  --source-root /path/to/finished/run --output-root /path/to/comparison/gateway-01 \
  --round 0 --checkpoints 5 10 15 20 25 \
  --scheduler-model gpt-6-astra --scheduler-reasoning-effort low

# Anthropic Messages API
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY"
python -m refocus.rounds.compare_schedulers \
  --benchmark robocasa365 \
  --source-root /path/to/finished/run --output-root /path/to/comparison/opus-01 \
  --round 0 --checkpoints 5 10 15 20 25 --scheduler-provider anthropic \
  --scheduler-model claude-opus-5-5
```

`--benchmark libero_10` runs the same comparison on a LIBERO run. Visual mode
sends every current episode montage per task and requires an endpoint that
accepts images. `--scheduler-evidence text` is a separate evidence condition,
not a visual comparison; report it as such. Use `--scheduler-json-mode schema`
(default) when the endpoint supports strict JSON schemas, and `prompt`
otherwise. Invalid replies fail rather than being substituted.

Keys and encoded images are never written to disk. Set `SCHEDULER_API_KEY` in
the coordinator environment; subprocesses do not inherit it.

Agreement on saved episodes is not an effect on policy success rate. For that,
run independent training with separate `--run-root` values and matched
checkpoint, tasks, evaluation protocol and interaction budget.

## Selecting a Scheduler

`scripts/run_rounds.sh` accepts `--scheduler-provider`, `--scheduler-base-url`,
`--scheduler-model`, `--scheduler-reasoning-effort`, `--scheduler-json-mode`
and `--scheduler-max-tokens`. Use a separate run root per Scheduler;
`config.json` is recorded and cannot change on a resumed run.
