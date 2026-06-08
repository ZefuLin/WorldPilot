#!/bin/bash
set -euo pipefail

###########################################################################################
# Edit these before running:
#
# 1. In examples/LIBERO/train_files/ABot_libero_cosmos.yaml:
#    - run_root_dir
#    - run_id
#    - framework.vggt_path
#    - framework.qwenvl.base_vlm
#    - datasets.vla_data.data_root_dir
#    - datasets.vla_data.cosmos_cache_dir
#    - trainer.pretrained_checkpoint
#
# 2. In this script:
#    - GPU_IDS
#    - CONFIG_YAML (if you want another config)
###########################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_YAML="${1:-${REPO_ROOT}/examples/LIBERO/train_files/ABot_libero_cosmos.yaml}" 
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-WorldPilot/config/deepseeds/deepspeed_zero2.yaml}"
WORLDPILOT_ACCELERATE="${WORLDPILOT_ACCELERATE:-accelerate}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-131072}"

require_dir() {
    if [ ! -d "$1" ]; then
        echo "[ERROR] Missing directory: $1" >&2
        exit 1
    fi
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "[ERROR] Missing file: $1" >&2
        exit 1
    fi
}

require_file "${CONFIG_YAML}"
cd "${REPO_ROOT}"

mapfile -t CFG_VALUES < <(
    python - "${CONFIG_YAML}" <<'PY'
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg["datasets"]["vla_data"]
trainer_cfg = cfg["trainer"]

values = [
    cfg["run_root_dir"],
    cfg["run_id"],
    data_cfg["data_root_dir"],
    trainer_cfg.get("pretrained_checkpoint", ""),
    data_cfg.get("cosmos_cache_dir", ""),
]
for value in values:
    print(value if value is not None else "")
PY
)

RUN_ROOT_DIR="${CFG_VALUES[0]}"
RUN_ID="${CFG_VALUES[1]}"
LIBERO_DATA_ROOT="${CFG_VALUES[2]}"
PRETRAIN_CKPT="${CFG_VALUES[3]}"
COSMOS_CACHE_DIR="${CFG_VALUES[4]}"

require_dir "${LIBERO_DATA_ROOT}"
require_file "${PRETRAIN_CKPT}"

for subset in \
    libero_10_no_noops_1.0.0_lerobot \
    libero_goal_no_noops_1.0.0_lerobot \
    libero_object_no_noops_1.0.0_lerobot \
    libero_spatial_no_noops_1.0.0_lerobot; do
    require_dir "${LIBERO_DATA_ROOT}/${subset}"
done

if [ -n "${COSMOS_CACHE_DIR}" ]; then
    require_dir "${COSMOS_CACHE_DIR}"
    for subset in \
        libero_10_no_noops_1.0.0_lerobot \
        libero_goal_no_noops_1.0.0_lerobot \
        libero_object_no_noops_1.0.0_lerobot \
        libero_spatial_no_noops_1.0.0_lerobot; do
        require_dir "${COSMOS_CACHE_DIR}/${subset}"
    done
fi

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

NUM_PROCESSES=$(python - "${GPU_IDS}" <<'PY'
import sys

gpu_ids = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
if not gpu_ids:
    raise SystemExit("GPU_IDS must contain at least one GPU id.")
print(len(gpu_ids))
PY
)

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

"${WORLDPILOT_ACCELERATE}" launch \
  --config_file "${DEEPSPEED_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  WorldPilot/training/train.py \
  --config_yaml "${CONFIG_YAML}"
