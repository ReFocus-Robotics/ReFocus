#!/usr/bin/env python3
"""Mirror TensorBoard scalar events into JSONL under the checkpoint directory."""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

stop = False


def request_stop(_signum: int, _frame: object) -> None:
    global stop
    stop = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    accumulators: dict[Path, EventAccumulator] = {}
    seen: dict[tuple[Path, str], int] = {}

    while True:
        for event_file in sorted(args.log_dir.rglob("events.out.tfevents.*")):
            try:
                acc = accumulators.get(event_file)
                if acc is None:
                    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
                    accumulators[event_file] = acc
                acc.Reload()
                for tag in acc.Tags().get("scalars", []):
                    events = acc.Scalars(tag)
                    key = (event_file, tag)
                    offset = seen.get(key, 0)
                    if offset >= len(events):
                        continue
                    with args.output.open("a", encoding="utf-8") as output:
                        for event in events[offset:]:
                            output.write(json.dumps({
                                "step": int(event.step),
                                "tag": tag,
                                "value": float(event.value),
                                "wall_time": float(event.wall_time),
                                "event_file": str(event_file.relative_to(args.log_dir)),
                            }, ensure_ascii=False, allow_nan=False) + "\n")
                        output.flush()
                    seen[key] = len(events)
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"Waiting for readable TensorBoard event file {event_file}: {exc}", flush=True)
        if stop:
            break
        time.sleep(max(args.poll_seconds, 0.5))


if __name__ == "__main__":
    main()
