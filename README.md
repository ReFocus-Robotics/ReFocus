# ReFocus

**Progress-Guided Adaptive Practice for Long-Horizon Robotic Manipulation Reinforcement Learning**

[Project website source](https://github.com/ReFocus-Robotics/refocus.github.io) · [Installation](docs/INSTALL.md) · [Training details](docs/REFOCUS.md)

<p align="center">
  <img src="docs/assets/refocus-overview.png" alt="ReFocus alternates global policy training and targeted practice, guided by subtask progress and practice history." width="860">
</p>

ReFocus alternates Global PPO and Local PPO on a shared π0.5 policy. Stage-balanced rewards provide feedback on verified subtask completions. A Scheduler uses progress and practice history to select bottleneck windows. The Local phase practices these windows from normal starts or restored simulation states.

This repository provides the training code, configuration, and launch scripts for RoboCasa365 Composite-Seen in simulation. Supply a compatible Composite-Seen SFT checkpoint to initialize training.

## Setup

1. Follow the [installation guide](docs/INSTALL.md) to prepare the training environment, simulator assets, and π0.5 SFT checkpoint. Install the pinned OpenPI and simulator sources with `bash scripts/setup_upstream.sh --openpi`.
2. From the repository root, configure the environment on each Ray node. Set these shell variables to the shared checkpoint path and the address of your Ray head node.

   ```bash
   export PI05_SFT_CKPT
   export RAY_HEAD_IP
   source examples/robocasa365/rlinf/rsi_env.sh
   export RAY_ADDRESS="$RAY_HEAD_IP:$RAY_PORT"
   ```

3. Install the ReFocus hooks once in the shared, StructRL-patched RLinf checkout.

   ```bash
   "$VENV/bin/python" scripts/install_rsi_hooks.py "$RLINF_ROOT"
   ```

   Start the [Ray head](examples/robocasa365/rlinf/ray_head_8gpu.sh) and the
   [Ray worker](examples/robocasa365/rlinf/ray_worker_8gpu.sh) using the same
   shared `STRUCTRL_RSI_ROOT` on both nodes.

## Training

| Profile | Training envs | Global batch | Microbatch per actor GPU |
| --- | ---: | ---: | ---: |
| [Paper-size](configs/robocasa365_ppo_openpi_pi05_refocus_16gpu.yaml), default | 256 | 512 | 32 |
| [Fast](configs/robocasa365_ppo_openpi_pi05_refocus_16gpu_fast.yaml) | 64 | 128 | 8 |

Both profiles use two nodes with eight actor GPUs each. The coordinator runs
three rounds of 25 Global and 25 Local PPO updates. It evaluates five
normal-start episodes per task at each diagnostic checkpoint.

Set `ADVISOR_API_KEY` on the coordinator after starting Ray. The visual
Scheduler sends evaluation montages to the configured endpoint, which must
accept image inputs.

```bash
scripts/run_robocasa_rsi.sh \
  --rlinf-root "$RLINF_ROOT" --run-root "$STRUCTRL_RSI_ROOT" \
  --advisor-provider openai-compatible \
  --advisor-base-url "$ADVISOR_BASE_URL" --advisor-model "$ADVISOR_MODEL"
```

Pass `--config-name robocasa365_ppo_openpi_pi05_refocus_16gpu_fast` to select the
fast profile and give it a separate run directory. See [Scheduler
configuration](docs/ADVISOR_COMPARISON.md) for provider-specific settings.
The [method notes](docs/REFOCUS.md) describe rollout modes, snapshot
restoration, and the optional milestone configuration.

## Local checks

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest
.venv/bin/python -m structrl.rsi.run \
  --rlinf-root ./RLinf --run-root ./runs/dry-run --dry-run
```

The dry run prints the training schedule. The [installation guide](docs/INSTALL.md)
specifies the simulator and training dependencies.

## Attribution and licenses

ReFocus builds on StructRL and RLinf. StructRL-derived material is distributed
under the upstream [CC BY-NC 4.0 license](LICENSE). See [NOTICE](NOTICE),
[third-party licenses](THIRD-PARTY-LICENSES), and [source provenance](SOURCE_SNAPSHOT.md)
for details.
