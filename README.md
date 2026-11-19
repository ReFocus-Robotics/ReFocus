# ReFocus

**Progress-Guided Adaptive Practice for Long-Horizon Robotic Manipulation Reinforcement Learning**

[![arXiv Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://github.com/ReFocus-Robotics/ReFocus)
[![Project Website](https://img.shields.io/badge/Project-Website-0077b5)](https://github.com/ReFocus-Robotics/ReFocus)
[![License CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-green)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776ab?logo=python&logoColor=white)](#environment-setup)

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

Online reinforcement learning (RL) enables vision-language-action policies to improve through environmental interaction. However, in long-horizon manipulation, sparse success/failure feedback provides little information about where policies encounter bottlenecks. Moreover, repeated full-task rollouts also leave these bottlenecks weakly analyzed and insufficiently practiced. To this end, we introduce **ReFocus**, a progress-guided multi-round RL framework that alternates Global with Local RL to identify task-specific bottlenecks and allocate additional practice to them. Specifically, during Global RL, we decompose each task into a Directed Acyclic Graph (DAG) of subtasks. Then a DAG-based Progress Reward is leveraged for both policy training and bottleneck diagnosis. Based on the identified bottlenecks, ReFocus selects Adaptive Practice Windows, defining task segments from an entry stage to a target stage. It allocates additional interaction to these segments during Local RL while updating the entire policy. After that, it evaluates the policy to perform the full-task for the next round RL, forming an iteration between Global and Local RL.

Experiments on LIBERO-Long and RoboCasa365 show that ReFocus achieves success rates of 97.8% and 51.3% respectively, compared with 90.6% and 39.3% for the π<sub>0.5</sub> baseline. Furthermore, we evaluate a real-world adaptation on three long-horizon tasks. On BreadBaker, ReFocus achieves 95% success, compared with 30% for the π<sub>0.5</sub>.

<a name="environment-setup"></a>

## ⚙️ Environment setup

~~~bash
conda create -n refocus --override-channels -c conda-forge python=3.11 pip ffmpeg=7
conda activate refocus
~~~

### Data and checkpoints

- [RoboCasa365 simulation assets](docs/INSTALL.md#download-simulator-assets-and-prepare-the-checkpoint) for Composite-Seen tasks.
- A Composite-Seen π<sub>0.5</sub> PyTorch SFT checkpoint with `robocasa365/norm_stats.json`.

<a name="training"></a>

## 🏋️ Training

### 1. Configure both Ray nodes

Activate the environment and run these commands from the repository root on each node. Replace the checkpoint and head IP with your own values.

~~~bash
conda activate refocus
export VENV="$CONDA_PREFIX"
export PI05_SFT_CKPT=/path/to/composite-seen-pytorch-checkpoint
export RAY_HEAD_IP=10.0.0.10
source examples/robocasa365/rlinf/rsi_env.sh
export RAY_ADDRESS="$RAY_HEAD_IP:$RAY_PORT"
~~~

### 2. Install the hooks and start Ray

Install the hooks once in the shared RLinf checkout, before starting Ray.

~~~bash
python scripts/install_rsi_hooks.py "$RLINF_ROOT"
~~~

Run `bash examples/robocasa365/rlinf/ray_head_8gpu.sh` on the head node and
`bash examples/robocasa365/rlinf/ray_worker_8gpu.sh` on the worker node.

### 3. Launch ReFocus on the head node

Set `ADVISOR_API_KEY`, `ADVISOR_BASE_URL`, and `ADVISOR_MODEL` on the head node after Ray starts. The endpoint needs to accept image inputs for the visual Scheduler. See the [Scheduler setup](docs/ADVISOR_COMPARISON.md) for provider-specific settings.

~~~bash
bash scripts/run_robocasa_rsi.sh \
  --rlinf-root "$RLINF_ROOT" --run-root "$STRUCTRL_RSI_ROOT" \
  --advisor-provider openai-compatible \
  --advisor-base-url "$ADVISOR_BASE_URL" --advisor-model "$ADVISOR_MODEL"
~~~

The [training configuration](configs/robocasa365_ppo_openpi_pi05_refocus_16gpu.yaml) uses **256 training environments**, a **global batch of 512**, and a **microbatch of 32 per GPU**. The coordinator runs three rounds of 25 Global and 25 Local PPO updates. At each diagnostic checkpoint, it evaluates five normal-start episodes per task. Outputs are written to `$STRUCTRL_RSI_ROOT`.

<a name="repository-structure"></a>

## 📁 Repository structure

| Path | Purpose |
| --- | --- |
| [`refocus/`](refocus/) | Stage-balanced progress and milestone rewards |
| [`structrl/rsi/`](structrl/rsi/) | Training coordinator, Scheduler integration, and snapshot restoration |
| [`configs/`](configs/) | Hydra training and simulator configurations |
| [`scripts/`](scripts/) | Upstream setup, hook installation, and training entry points |
| [`examples/robocasa365/`](examples/robocasa365/) | Ray launch scripts and milestone configuration example |
| [`docs/`](docs/) | [Installation guide](docs/INSTALL.md), [method notes](docs/REFOCUS.md), and [StructRL baselines](docs/REPRODUCE.md) |

<a name="attribution-and-license"></a>

## 📄 Attribution and license

StructRL-derived material is distributed under the upstream [CC BY-NC 4.0 license](LICENSE). See [NOTICE](NOTICE), [third-party licenses](THIRD-PARTY-LICENSES), and [source provenance](SOURCE_SNAPSHOT.md).

<a name="acknowledgments"></a>

## 🙏 Acknowledgments

We thank the [StructRL](https://github.com/amazon-science/StructRL) and [RLinf](https://github.com/RLinf/RLinf) teams for open-sourcing the training code that ReFocus builds on. We also thank the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and [RoboCasa365](https://github.com/robocasa/robocasa) teams for developing and maintaining their simulation benchmarks.
