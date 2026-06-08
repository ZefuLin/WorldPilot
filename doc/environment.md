# Installation

This project uses four environments:

1. `WorldPilot` for policy training
2. `Cosmos-Policy` for Cosmos serving and cache precompute
3. `Libero` as the LIBERO baseline environment
4. `Libero-Plus` for public evaluation

The setup below is aligned with this repo's public scripts. Before following the local commands here,
prefer checking the upstream installation docs first:

- [ABot-Manipulation](https://github.com/amap-cvlab/ABot-Manipulation)
- [Cosmos-Policy](https://github.com/NVlabs/cosmos-policy)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus)

This repo does not use the Docker-first setup described in the official `cosmos-policy` README. Its
public scripts assume local `mamba` environments and invoke Cosmos with commands such as
`mamba run -n cosmos-policy python -m cosmos_bridge.cosmos_server`.

## System packages

`Libero-Plus` requires a few system-level dependencies. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git git-lfs curl wget ffmpeg \
  libgl1 libegl1-mesa-dev libgl1-mesa-dri libglib2.0-0 \
  libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev
```

## 1. WorldPilot

Official references:

- [ABot-Manipulation](https://github.com/amap-cvlab/ABot-Manipulation)

The official `ABot-Manipulation` setup uses Python 3.10. This repo keeps the same base.

```bash
mamba create -n WorldPilot python=3.10 -y
mamba activate WorldPilot

cd /path/to/WorldPilot
pip install torch torchvision --index-url <your-pytorch-cuda-wheel-index>
pip install -r requirements.txt
pip install -e .
```

`VGGT` is required by the current Libero config:

```bash
pip install -e /path/to/vggt
```

If your Qwen3-VL stack supports it, install FlashAttention as in the official `ABot-Manipulation` setup:

```bash
pip install flash-attn --no-build-isolation
```

## 2. Cosmos-Policy

Official references:

- [Cosmos-Policy](https://github.com/NVlabs/cosmos-policy)

The official `cosmos-policy` repo documents a Docker-first setup around `uv sync`. The public training
and evaluation flow in this repo does not use that Docker path. Use a local `mamba` environment instead,
because `cosmos_bridge/run_precompute.sh` and the evaluation scripts expect commands like
`mamba run -n cosmos-policy python ...`.

```bash
mamba create -n cosmos-policy python=3.10 -y
mamba activate cosmos-policy

git clone https://github.com/NVlabs/cosmos-policy.git
cd /path/to/cosmos-policy
pip install -e ".[cu128]"
pip install -r cosmos_policy/experiments/robot/libero/libero_requirements.txt
pip install websockets msgpack
```

## 3. Libero

Official reference: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

The official `LIBERO` repo uses Python `3.8.13` with the `torch==1.11.0+cu113` stack.

```bash
mamba create -n libero python=3.8.13 -y
mamba activate libero

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
```

This environment is not the main public entrypoint of this repo, but it is still useful if you need to
compare against the official `LIBERO` tooling directly.

## 4. Libero-Plus

Official reference: [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus)

`Libero-Plus` is the evaluation environment used by the public scripts in this repo.

```bash
mamba create -n libero-plus python=3.8 -y
mamba activate libero-plus

git clone https://github.com/sylvestf/LIBERO-plus.git
cd /path/to/LIBERO-plus
pip install -r requirements.txt
pip install -r extra_requirements.txt
pip install -e .

cd /path/to/WorldPilot
pip install -r examples/LIBERO-plus/eval_files/libero_plus_requirements.txt
```

Create the local Libero-Plus config:

```bash
mkdir -p ~/.libero-plus
cat > ~/.libero-plus/config.yaml <<'EOF'
assets: /path/to/LIBERO-plus/libero/libero/assets
bddl_files: /path/to/LIBERO-plus/libero/libero/bddl_files
benchmark_root: /path/to/LIBERO-plus/libero/libero
datasets: /path/to/libero_plus_datasets
init_states: /path/to/LIBERO-plus/libero/libero/init_files
EOF
```

Recommended runtime variables for evaluation:

```bash
export LIBERO_CONFIG_PATH=~/.libero-plus
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## Notes

- `cosmos_bridge/run_precompute.sh` and the Libero-Plus eval scripts assume your Cosmos checkpoint, statistics file, and T5 embedding pickle already exist.
- The public scripts in this repo are written so that you edit the top config block of each `.sh` file instead of exporting a long list of shell variables.
