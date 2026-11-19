# Installation

## ReFocus on RoboCasa365

Run the setup on Linux x86-64 with an NVIDIA driver that supports CUDA 12.4
and EGL available for headless MuJoCo rendering. Start with the
[Conda environment in the README](../README.md#environment-setup).

### Install the training stack

From the ReFocus repository root, with the `refocus` environment active:

~~~bash
export PROJECT_ROOT="$PWD"
bash scripts/setup_upstream.sh
python -m pip install --no-deps -r envs/rlinf-openpi-train.txt
python -m pip install --no-deps -e ./robosuite -e ./robocasa

site_packages=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
cp -r "$site_packages/openpi/models_pytorch/transformers_replace/." "$site_packages/transformers/"
bash RLinf/requirements/embodied/download_assets.sh --assets openpi
~~~

Conda supplies Python and FFmpeg. The [training package list](../envs/rlinf-openpi-train.txt)
pins the CUDA-enabled PyTorch 2.6 stack, FlashAttention, Ray, MuJoCo, and
OpenPI. `--no-deps` uses the versions in the archived environment directly,
including the PyTorch version selected by RLinf. RLinf is imported from the
patched source checkout. The last two commands install OpenPI’s Transformers
overrides and tokenizer using the same procedure as the pinned
[RLinf installer](https://github.com/RLinf/RLinf/blob/499363965432843cdbae607a561d59b68f785d34/requirements/install.sh#L1089-L1100).
Install the environment on each node and run the upstream setup once for a
shared checkout.

### Download simulator assets and prepare the checkpoint

~~~bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
~~~

Download the assets once per simulator checkout. Set `PI05_SFT_CKPT` to a
compatible Composite-Seen **PyTorch** SFT export, including its
`robocasa365/norm_stats.json`. The checkpoint path must be readable from
both nodes. See the [SFT recipe](#3-data-and-sft-checkpoints) if you need
to produce the checkpoint.

Continue with [distributed training](../README.md#training).

## Inherited StructRL recipes

### 1. Upstream code

StructRL is a set of patches and add-on modules for the upstream projects below, pinned to the commits
used in the paper.

| Repository | Commit | Patch | Used for |
|---|---|---|---|
| [RLinf](https://github.com/RLinf/RLinf) | `4993639` | `rlinf_core.patch` | RL training on RoboCasa365 |
| [robocasa](https://github.com/robocasa/robocasa) | `8f3c96e` | `robocasa_gymutils.patch` | RoboCasa365 |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) | `232ce7d` (v1.5.2) | none | RoboCasa365 |
| [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | `4af2b62` | `isaacgroot_shim.patch` | GR00T-N1.5 |
| [openpi (RoboCasa fork)](https://github.com/robocasa-benchmark/openpi) | `5a6beda` | `openpi_rc365.patch` | π0.5 SFT and evaluation on RoboCasa365 |
| RLinf, second checkout | `4993639` | `rlinf_core.patch`, then `libero_addon.patch` | RL training on LIBERO |
| [LIBERO (RLinf fork)](https://github.com/RLinf/LIBERO) | `0c5e40c` | none | RL training on LIBERO |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `8f1084e` | none | LIBERO evaluation client |

`scripts/setup_upstream.sh` clones them next to the code, applies the patches, and installs the Hydra
configs into RLinf:

```bash
export PROJECT_ROOT=$PWD
bash scripts/setup_upstream.sh              # RoboCasa365 with GR00T-N1.5
bash scripts/setup_upstream.sh --openpi     # + π0.5 on RoboCasa365
bash scripts/setup_upstream.sh --libero     # + LIBERO (second RLinf in $PROJECT_ROOT/libero_rl)
export LIBERO_RL_ROOT=$PROJECT_ROOT/libero_rl CKPT_ROOT=$PROJECT_ROOT/ckpts
```

### 2. Python environments

The inherited launchers use the virtual environments below, under `$PROJECT_ROOT`. `envs/` has the frozen package
list of each one (Python 3.11 unless noted). Install the list, then the patched checkouts it names with
`pip install --no-deps -e`. RLinf itself is used from source through `PYTHONPATH`.

| Environment | Package list | Used by |
|---|---|---|
| `.venv-rlinf` | `envs/rlinf-gr00t.txt` | GR00T-N1.5 training on RoboCasa365 |
| `.venv-groot` | `envs/groot-eval.txt` | GR00T-N1.5 evaluation; policy server for LIBERO evaluation |
| `.venv-rlinf-openpi-train` | `envs/rlinf-openpi-train.txt` | π0.5 training on RoboCasa365 |
| `.venv-rlinf-libero` | `envs/rlinf-libero.txt` | GR00T-N1.5 training on LIBERO |
| `.venv-rlinf-openpi-libero` | `envs/rlinf-openpi-libero.txt` | π0.5 training on LIBERO |
| `.venv-libero` | `envs/libero-eval.txt` (Python 3.10) | LIBERO evaluation client |
| `openpi/.venv` | `uv sync` in the openpi fork | π0.5 SFT and policy server on RoboCasa365 |
| `.venv-rlinf-openpi-eval` | robocasa, robosuite, `openpi-client` | π0.5 evaluation client on RoboCasa365 |

### 3. Data and SFT checkpoints

| What | Source | Location |
|---|---|---|
| RoboCasa365 v1.0 datasets | [RoboCasa docs](https://robocasa.ai) | `$PROJECT_ROOT/datasets` (see `configs/env/`) |
| GR00T-N1.5, RoboCasa365 Composite-Seen SFT | [robocasa/robocasa365_checkpoints](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000) | `$PROJECT_ROOT/ckpts/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000` |
| π0.5, RoboCasa365 Composite-Seen SFT | initialize from `checkpoints/openpi_pretrain/pi05_pretrain_human300/multitask_learning/75000/params`, then run `launch/sft_pi05.sh` | use the run's `99999_pytorch` export as `PI05_SFT_CKPT` for both PPO and GRPO |
| GR00T-N1.5, LIBERO SFT | [youliangtan/gr00t-n1.5-libero-long-posttrain](https://huggingface.co/youliangtan/gr00t-n1.5-libero-long-posttrain) (and `-spatial`, `-object`, `-goal`) | `$CKPT_ROOT/gr00t-n1.5-libero-{long,spatial,object,goal}` |
| π0.5, LIBERO SFT | [RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT) | `$CKPT_ROOT/RLinf-Pi05-LIBERO-130-fullshot-SFT` |

LIBERO reads its asset paths from `$LIBERO_CONFIG_PATH/config.yaml` (default `~/.libero`), which LIBERO
creates on first import.

### 4. Cluster

The inherited Slurm batch scripts use 4 nodes with 8 GPUs each for training and 2 nodes with 8 GPUs each for evaluation. Submit from the repository root and pass the partition with
`sbatch --partition` when your cluster requires it. Slurm logs go to `logs/`.
Use the network interface prefix for your cluster in the NCCL settings.
