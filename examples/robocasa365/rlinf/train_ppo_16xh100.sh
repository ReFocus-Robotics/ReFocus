#!/usr/bin/env bash
set -euo pipefail

# Default 16-GPU PPO entrypoint: use the faster-step profile.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/train_ppo_16xh100_fast.sh" "$@"
