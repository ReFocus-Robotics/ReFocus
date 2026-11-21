<h1 align="center">ReFocus</h1>

<p align="center"><strong>Progress-Guided Adaptive Practice for Long-Horizon Robotic Manipulation Reinforcement Learning</strong></p>

<p align="center">
  <a href="https://github.com/ReFocus-Robotics/ReFocus"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&amp;logoColor=white" alt="arXiv Paper"></a>
  <a href="https://github.com/ReFocus-Robotics/ReFocus"><img src="https://img.shields.io/badge/Project-Website-0077b5" alt="Project Website"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC_BY--NC_4.0-green" alt="License CC BY-NC 4.0"></a>
  <a href="#environment-setup"><img src="https://img.shields.io/badge/Python-3.11-3776ab?logo=python&amp;logoColor=white" alt="Python 3.11"></a>
</p>

<p align="center">
  <img src="docs/assets/refocus-overview.png" alt="ReFocus alternates full-task training and targeted practice using subtask progress and practice history." width="860">
</p>

## Overview

Online reinforcement learning enables vision-language-action policies to improve through environmental interaction, but long-horizon manipulation tasks give sparse success/failure feedback that says little about *where* a policy fails. ReFocus addresses this with a progress-guided multi-round RL framework that alternates Global and Local RL. During Global RL, each task is decomposed into a subtask DAG and a DAG-based Progress Reward drives both training and bottleneck diagnosis. From the diagnosed bottlenecks, ReFocus selects Adaptive Practice Windows — task segments from an entry stage to a target stage — and spends extra interaction on them during Local RL while still updating the whole policy. The policy is then evaluated on the full task and the next round begins. On LIBERO-Long and RoboCasa365, ReFocus reaches 97.8% and 51.3% success versus 90.6% and 39.3% for π<sub>0.5</sub>, and it transfers to three real-world long-horizon tasks (95% vs 30% on BreadBaker).

## Environment setup

~~~bash
conda create -n refocus --override-channels -c conda-forge python=3.11 pip ffmpeg=7
conda activate refocus
~~~

Install the training stack and simulator assets by following [docs/INSTALL.md](docs/INSTALL.md).

Checkpoints:

- **RoboCasa365** — train a π<sub>0.5</sub> PyTorch SFT checkpoint with [launch/sft_pi05.sh](launch/sft_pi05.sh), starting from [pi05_pretrain_human300](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/pi05_pretrain_human300/multitask_learning/75000).
- **LIBERO-Long** — the [RLinf π<sub>0.5</sub> LIBERO SFT checkpoint](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT). Set `CKPT_ROOT` to the directory that holds it.

## Training

The coordinator runs Global RL, asks the Scheduler for one practice window per task, runs Local RL from that choice, and repeats. Actor, critic and optimizer are carried across phases.

### RoboCasa365 (two 8-GPU Ray nodes)

~~~bash
conda activate refocus
export VENV="$CONDA_PREFIX"
export PI05_SFT_CKPT=/path/to/robocasa365-pytorch-sft-checkpoint
export RAY_HEAD_IP=10.0.0.10
source examples/robocasa365/rlinf/round_env.sh
export RAY_ADDRESS="$RAY_HEAD_IP:$RAY_PORT"

python scripts/install_round_hooks.py "$RLINF_ROOT"   # once, before Ray starts
bash examples/robocasa365/rlinf/ray_head_8gpu.sh      # head node
bash examples/robocasa365/rlinf/ray_worker_8gpu.sh    # worker node
~~~

Then, after Ray is up and `SCHEDULER_*` is set on the head node:

~~~bash
bash scripts/run_rounds.sh \
  --benchmark robocasa365 \
  --rlinf-root "$RLINF_ROOT" --run-root "$STRUCTRL_ROUNDS_ROOT" \
  --scheduler-provider openai-compatible \
  --scheduler-base-url "$SCHEDULER_BASE_URL" --scheduler-model "$SCHEDULER_MODEL"
~~~

### LIBERO-Long (one 8-GPU host)

`scripts/setup_upstream.sh --libero` creates a second RLinf checkout in `$LIBERO_RL_ROOT`. Export these before starting Ray so every worker inherits them:

~~~bash
export STRUCTRL_ROUNDS_ROOT=/path/to/libero-run
export LIBERO_RL_ROOT=/path/to/libero_rl
export CKPT_ROOT=/path/to/ckpts
export LIBERO_DENSE_REWARD=1
export LIBERO_REWARD_VARIANT=refocus
export STRUCTRL_REWARD_VARIANT=refocus
~~~

~~~bash
bash scripts/run_rounds.sh \
  --benchmark libero_10 \
  --rlinf-root "$LIBERO_RL_ROOT/RLinf" --run-root "$STRUCTRL_ROUNDS_ROOT" \
  --scheduler-provider openai-compatible \
  --scheduler-base-url "$SCHEDULER_BASE_URL" --scheduler-model "$SCHEDULER_MODEL"
~~~

Training parameters live in [configs/](configs/); Scheduler options and environment variables are listed in [docs/REFOCUS.md](docs/REFOCUS.md) and [docs/SCHEDULER_COMPARISON.md](docs/SCHEDULER_COMPARISON.md).

## Attribution and license

StructRL-derived material is distributed under the upstream [CC BY-NC 4.0 license](LICENSE). See [NOTICE](NOTICE), [third-party licenses](THIRD-PARTY-LICENSES), and [source provenance](SOURCE_SNAPSHOT.md).

## Acknowledgments

We thank the [StructRL](https://github.com/amazon-science/StructRL) and [RLinf](https://github.com/RLinf/RLinf) teams for open-sourcing the training code that ReFocus builds on, and the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and [RoboCasa365](https://github.com/robocasa/robocasa) teams for their simulators.
