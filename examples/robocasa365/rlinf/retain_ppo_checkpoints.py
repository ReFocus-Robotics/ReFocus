#!/usr/bin/env python3
"""Keep the best evaluated and latest complete RLinf PPO checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

CHECKPOINT_RE = re.compile(r"^global_step_(\d+)$")


def read_metric_scores(metrics_path: Path, metric_tag: str, expected_episodes: int) -> dict[int, float]:
    values: dict[int, list[float]] = {}
    episodes: dict[int, list[float]] = {}
    if not metrics_path.exists():
        return {}
    with metrics_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                step = int(row["step"])
                tag = str(row["tag"])
                value = float(row["value"])
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid metrics record at {metrics_path}:{line_number}") from exc
            if tag == metric_tag:
                values.setdefault(step, []).append(value)
            elif tag == "eval/num_trajectories":
                episodes.setdefault(step, []).append(value)

    scores: dict[int, float] = {}
    for step, observations in values.items():
        # TensorBoard can be re-read after an exporter restart. Identical duplicate
        # scalars are harmless; conflicting values make best selection ambiguous.
        unique = set(observations)
        if len(unique) != 1:
            raise ValueError(f"Conflicting {metric_tag} values at step {step}: {sorted(unique)}")
        episode_values = set(episodes.get(step, []))
        if episode_values != {float(expected_episodes)}:
            continue
        score = unique.pop()
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Expected {metric_tag} in [0, 1], got {score} at step {step}")
        scores[step] = score
    return scores


def checkpoint_signature(checkpoint: Path) -> tuple[tuple[str, int, int], ...] | None:
    actor = checkpoint / "actor"
    full_weights = actor / "model_state_dict" / "full_weights.pt"
    dcp = actor / "dcp_checkpoint"
    metadata = dcp / ".metadata"
    if not checkpoint.is_dir() or not full_weights.is_file() or not metadata.is_file():
        return None
    files = [metadata, full_weights]
    shards = sorted(path for path in dcp.rglob("*") if path.is_file() and path.name.endswith(".distcp"))
    if not shards or any(path.stat().st_size == 0 for path in [*files, *shards]):
        return None
    files.extend(shards)
    return tuple(sorted((str(path.relative_to(checkpoint)), path.stat().st_size, path.stat().st_mtime_ns) for path in files))


def atomic_symlink(parent: Path, name: str, target_name: str) -> None:
    link = parent / name
    if link.exists() and not link.is_symlink():
        raise RuntimeError(f"Refusing to replace non-symlink retention path: {link}")
    temporary = parent / f".{name}.tmp.{os.getpid()}"
    temporary.unlink(missing_ok=True)
    os.symlink(target_name, temporary)
    os.replace(temporary, link)


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def retain_checkpoints(
    checkpoint_root: Path,
    metrics_path: Path,
    metric_tag: str = "eval/success_once",
    expected_episodes: int = 80,
    stable_observations: dict[Path, tuple[tuple[tuple[str, int, int], ...], int]] | None = None,
) -> dict | None:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    scores = read_metric_scores(metrics_path, metric_tag, expected_episodes)
    stable_observations = stable_observations if stable_observations is not None else {}
    complete: dict[int, Path] = {}
    for candidate in checkpoint_root.iterdir():
        match = CHECKPOINT_RE.match(candidate.name)
        if not match:
            continue
        step = int(match.group(1))
        signature = checkpoint_signature(candidate)
        if signature is None or step not in scores:
            stable_observations.pop(candidate, None)
            continue
        previous = stable_observations.get(candidate)
        observations = previous[1] + 1 if previous and previous[0] == signature else 1
        stable_observations[candidate] = (signature, observations)
        if observations >= 2:
            complete[step] = candidate

    if not complete:
        return None

    manifest_path = checkpoint_root / "retention.json"
    previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    old_best_step = previous_manifest.get("best_step")
    old_best_score = previous_manifest.get("best_score")

    # Include the currently retained best in the comparison even though all
    # intermediate numeric checkpoints may already have been pruned.
    old_best_path = checkpoint_root / f"global_step_{old_best_step}" if isinstance(old_best_step, int) else None
    if (
        isinstance(old_best_step, int)
        and old_best_score is not None
        and old_best_path is not None
        and old_best_path.is_dir()
        and checkpoint_signature(old_best_path) is not None
    ):
        best_step, best_score = old_best_step, float(old_best_score)
    else:
        best_step, best_score = min(complete), scores[min(complete)]

    for step, path in complete.items():
        score = scores[step]
        if score > best_score or (score == best_score and step > best_step):
            best_step, best_score = step, score

    last_step = max(complete)
    atomic_symlink(checkpoint_root, "best", f"global_step_{best_step}")
    atomic_symlink(checkpoint_root, "last", f"global_step_{last_step}")
    manifest = {
        "policy": "best_eval_success_once_and_latest_complete",
        "metric": metric_tag,
        "expected_eval_episodes": expected_episodes,
        "best_step": best_step,
        "best_score": best_score,
        "last_step": last_step,
        "last_score": scores[last_step],
    }
    atomic_json(manifest_path, manifest)

    keep = {best_step, last_step}
    eligible_names = {path.name for path in complete.values()}
    for candidate in checkpoint_root.iterdir():
        match = CHECKPOINT_RE.match(candidate.name)
        if match and candidate.name in eligible_names and int(match.group(1)) not in keep:
            shutil.rmtree(candidate)
            stable_observations.pop(candidate, None)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--metric-tag", default="eval/success_once")
    parser.add_argument("--expected-episodes", type=int, default=80)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--final-wait-seconds", type=float, default=60.0)
    args = parser.parse_args()
    stable: dict[Path, tuple[tuple[tuple[str, int, int], ...], int]] = {}
    deadline = time.monotonic() + args.final_wait_seconds if args.once else None
    while True:
        result = retain_checkpoints(
            args.checkpoint_root,
            args.metrics,
            args.metric_tag,
            args.expected_episodes,
            stable,
        )
        if result:
            print(json.dumps(result, sort_keys=True), flush=True)
        if args.once and (result is not None or time.monotonic() >= deadline):
            break
        delay = max(args.poll_seconds, 1.0)
        if deadline is not None:
            delay = min(delay, max(deadline - time.monotonic(), 0.1))
        time.sleep(delay)


if __name__ == "__main__":
    main()
