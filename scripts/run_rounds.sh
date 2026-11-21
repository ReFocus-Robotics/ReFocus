#!/usr/bin/env bash
# Launch the round coordinator (Global/Local PPO phases with a Scheduler in between).
exec python -m refocus.rounds.coordinator "$@"
