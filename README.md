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

## 📚 Contents

- [✨ Overview](#overview)
- [⚙️ Environment setup](#environment-setup)
- [🏋️ Training](#training)
- [📁 Repository structure](#repository-structure)
- [📄 Attribution and license](#attribution-and-license)
- [🙏 Acknowledgments](#acknowledgments)

<a name="overview"></a>

## ✨ Overview

Online reinforcement learning (RL) enables vision-language-action policies to improve through environmental interaction. However, in long-horizon manipulation, sparse success/failure feedback provides little information about where policies encounter bottlenecks. Moreover, repeated full-task rollouts also leave these bottlenecks weakly analyzed and insufficiently practiced. To this end, we introduce **ReFocus**, a progress-guided multi-round RL framework that alternates Global with Local RL to identify task-specific bottlenecks and allocate additional practice to them. Specifically, during Global RL, we decompose each task into a Directed Acyclic Graph (DAG) of subtasks. Then a DAG-based Progress Reward is leveraged for both policy training and bottleneck diagnosis. Based on the identified bottlenecks, ReFocus selects Adaptive Practice Windows, defining task segments from an entry stage to a target stage. It allocates additional interaction to these segments during Local RL while updating the entire policy. After that, it evaluates the policy to perform the full-task for the next round RL, forming an iteration between Global and Local RL. Experiments on LIBERO-Long and RoboCasa365 show that ReFocus achieves success rates of 97.8% and 51.3% respectively, compared with 90.6% and 39.3% for the π<sub>0.5</sub> baseline. Furthermore, we evaluate a real-world adaptation on three long-horizon tasks. On BreadBaker, ReFocus achieves 95% success, compared with 30% for the π<sub>0.5</sub>.

<a name="environment-setup"></a>

## ⚙️ Environment setup

~~~bash
conda create -n refocus --override-channels -c conda-forge python=3.11 pip ffmpeg=7
conda activate refocus
~~~

### Data and checkpoints

- **RoboCasa365.** Prepare the [simulation assets](docs/INSTALL.md#download-simulator-assets-and-prepare-the-checkpoint), the [demonstration dataset](https://robocasa.ai/docs/build/html/datasets/using_datasets.html) and pretrained [π<sub>0.5</sub> weights](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/pi05_pretrain_human300/multitask_learning/75000). RL starts from a π<sub>0.5</sub> PyTorch SFT checkpoint with `robocasa365/norm_stats.json`.
- **LIBERO-Long.** Prepare the [LIBERO simulator assets and benchmark initial states](https://github.com/RLinf/LIBERO) and the [RLinf π<sub>0.5</sub> LIBERO SFT checkpoint](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT). Set `CKPT_ROOT` to the directory that holds it.

<a name="training"></a>

## 🏋️ Training

### 1. SFT initialization

- **RoboCasa365.** Fine-tune π<sub>0.5</sub> on demonstrations using the [SFT launcher](launch/sft_pi05.sh) and [setup instructions](docs/INSTALL.md#3-data-and-sft-checkpoints). 
- **LIBERO-Long.** Initialize RL directly from the [π<sub>0.5</sub> SFT checkpoint released by RLinf](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT).

### 2. ReFocus post-training

The coordinator runs the first Global phase, asks the Scheduler for one practice window per task, runs the Local phase from that choice, then repeats. Every phase resumes the actor, critic, and optimizer of the previous one. The commands below cover both benchmarks.

#### RoboCasa365 on two 8-GPU Ray nodes

Run these commands from the repository root on each node. Replace the checkpoint and head IP with your own values.

~~~bash
conda activate refocus
export VENV="$CONDA_PREFIX"
export PI05_SFT_CKPT=/path/to/robocasa365-pytorch-sft-checkpoint
export RAY_HEAD_IP=10.0.0.10
source examples/robocasa365/rlinf/round_env.sh
export RAY_ADDRESS="$RAY_HEAD_IP:$RAY_PORT"
~~~

Install the hooks once in the shared RLinf checkout, before starting Ray.

~~~bash
python scripts/install_round_hooks.py "$RLINF_ROOT"
~~~

For Robocasa365, run `bash examples/robocasa365/rlinf/ray_head_8gpu.sh` on the head node and
`bash examples/robocasa365/rlinf/ray_worker_8gpu.sh` on the worker node.

Set `SCHEDULER_API_KEY`, `SCHEDULER_BASE_URL`, and `SCHEDULER_MODEL` on the head node after Ray starts. The endpoint needs to accept image inputs for the visual Scheduler. See the [Scheduler setup](docs/SCHEDULER_COMPARISON.md) for provider-specific settings.

~~~bash
bash scripts/run_rounds.sh \
  --benchmark robocasa365 \
  --rlinf-root "$RLINF_ROOT" --run-root "$STRUCTRL_ROUNDS_ROOT" \
  --scheduler-provider openai-compatible \
  --scheduler-base-url "$SCHEDULER_BASE_URL" --scheduler-model "$SCHEDULER_MODEL"
~~~

The [training configuration](configs/robocasa365_ppo_openpi_pi05_refocus_16gpu.yaml) uses **256 training environments**, a **global batch of 512**, and a **microbatch of 32 per GPU**. Three rounds of 25 Global and 25 Local PPO updates are run. At each diagnostic checkpoint the coordinator evaluates five normal-start episodes per task. Outputs are written to `$STRUCTRL_ROUNDS_ROOT`.

#### LIBERO-Long on one 8-GPU host

`scripts/setup_upstream.sh --libero` creates a second RLinf checkout with the LIBERO hooks, in `$LIBERO_RL_ROOT`. Export these variables before starting Ray so every worker inherits them:

~~~bash
export STRUCTRL_ROUNDS_ROOT=/path/to/libero-run
export LIBERO_RL_ROOT=/path/to/libero_rl
export CKPT_ROOT=/path/to/ckpts
export LIBERO_DENSE_REWARD=1
export LIBERO_REWARD_VARIANT=refocus
export STRUCTRL_REWARD_VARIANT=refocus
~~~

Then launch the coordinator against the LIBERO checkout:

~~~bash
bash scripts/run_rounds.sh \
  --benchmark libero_10 \
  --rlinf-root "$LIBERO_RL_ROOT/RLinf" --run-root "$STRUCTRL_ROUNDS_ROOT" \
  --scheduler-provider openai-compatible \
  --scheduler-base-url "$SCHEDULER_BASE_URL" --scheduler-model "$SCHEDULER_MODEL"
~~~

The [LIBERO configuration](configs/libero_10_ppo_openpi_pi05_refocus.yaml) uses **64 training environments**, a **global batch of 512**, 10-step action chunks, and a 1024-step episode limit. LIBERO entries are restored by replaying their recorded action prefix rather than by a state snapshot, so a Local rollout costs its prefix again. The optional `train_reset_state_ids` and `eval_reset_state_ids` config fields split the 50 official initial states of each task into non-overlapping training and development pools.

#### Practice windows and milestones

The Scheduler can return simulator-checkable milestones with each practice window. Local RL applies valid refinements automatically and otherwise keeps the original progress reward. See the [Scheduler milestone interface](docs/REFOCUS.md#scheduler-milestone-interface).

<a name="repository-structure"></a>

## 📁 Repository structure

| Path | Purpose |
| --- | --- |
| [`refocus/reward.py`](refocus/reward.py) | Stage-balanced progress reward and milestone budgets |
| [`refocus/robocasa.py`](refocus/robocasa.py), [`refocus/libero.py`](refocus/libero.py) | Per-benchmark stage tables, subtask predicates, and milestone checks |
| [`refocus/rounds/`](refocus/rounds/) | Round coordinator, Scheduler client, recorders, and entry restoration |
| [`configs/`](configs/) | Hydra training and simulator configurations |
| [`scripts/`](scripts/) | Upstream setup, hook installation, and the round launcher |
| [`examples/robocasa365/`](examples/robocasa365/) | Ray launch scripts and the worker environment file |
| [`docs/`](docs/) | [Installation guide](docs/INSTALL.md), [method notes](docs/REFOCUS.md), and [StructRL baselines](docs/REPRODUCE.md) |

<a name="attribution-and-license"></a>

## 📄 Attribution and license

StructRL-derived material is distributed under the upstream [CC BY-NC 4.0 license](LICENSE). See [NOTICE](NOTICE), [third-party licenses](THIRD-PARTY-LICENSES), and [source provenance](SOURCE_SNAPSHOT.md).

<a name="acknowledgments"></a>

## 🙏 Acknowledgments

We thank the [StructRL](https://github.com/amazon-science/StructRL) and [RLinf](https://github.com/RLinf/RLinf) teams for open-sourcing the training code that ReFocus builds on. We also thank the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and [RoboCasa365](https://github.com/robocasa/robocasa) teams for developing and maintaining their simulation benchmarks.
