#!/bin/bash
set -euo pipefail

# Single-suite LIBERO-plus evaluation.
# Runs one Cosmos server, one ABot policy server, and one LIBERO-plus simulator.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

###########################################################################################
# Edit this block before running.
#
# Required:
# - WORLDPILOT_CKPT
# - WORLDPILOT_PYTHON
# - COSMOS_CKPT
# - COSMOS_STATS
# - COSMOS_T5
# - LIBERO_PLUS_ROOT
# - LIBERO_CONFIG_PATH
#
# Common knobs:
# - UNNORM_KEY
# - TASK_SUITE_NAME
# - GPU
# - COSMOS_PORT / POLICY_PORT
# - START_TASK_ID / SUCCESS_COUNT
# - USE_CHECKPOINT_SOURCE
###########################################################################################
WORLDPILOT_CKPT="/path/to/run_dir/checkpoints/steps_50000_pytorch_model.pt"
WORLDPILOT_PYTHON="/path/to/envs/WorldPilot/bin/python"

COSMOS_CKPT="/path/to/Cosmos-Policy-LIBERO-Predict2-2B.pt"
COSMOS_STATS="/path/to/libero_dataset_statistics.json"
COSMOS_T5="/path/to/libero_t5_embeddings.pkl"

LIBERO_PLUS_ROOT="/path/to/LIBERO-plus"
LIBERO_CONFIG_PATH="${HOME}/.libero-plus"
LIBERO_PYTHON="python"

OUTPUT_DIR="${REPO_ROOT}/outputs/libero_plus"
UNNORM_KEY="franka"
TASK_SUITE_NAME="libero_10"
GPU=0
COSMOS_PORT=8002
POLICY_PORT=9883

START_TASK_ID=0
SUCCESS_COUNT=0
USE_CHECKPOINT_SOURCE=false
###########################################################################################

export COSMOS_CKPT COSMOS_STATS COSMOS_T5

export LIBERO_CONFIG_PATH
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

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

validate_libero_config() {
    local cfg_file="${LIBERO_CONFIG_PATH}/config.yaml"
    require_file "${cfg_file}"
    CHECK_CFG="${cfg_file}" "${LIBERO_PYTHON}" - <<'PY'
import os
import sys
import yaml

cfg_path = os.environ["CHECK_CFG"]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

required = ["assets", "bddl_files", "benchmark_root", "init_states"]
missing = [key for key in required if not cfg.get(key)]
if missing:
    raise SystemExit(f"[ERROR] Missing keys in {cfg_path}: {', '.join(missing)}")

bad_paths = [f"{key}={cfg[key]}" for key in required if not os.path.exists(cfg[key])]
if bad_paths:
    raise SystemExit("[ERROR] Invalid LIBERO-plus config paths: " + "; ".join(bad_paths))
PY
}

require_file "${WORLDPILOT_CKPT}"
require_file "${WORLDPILOT_PYTHON}"
require_file "${COSMOS_CKPT}"
require_file "${COSMOS_STATS}"
require_file "${COSMOS_T5}"
[ -n "${LIBERO_PLUS_ROOT}" ] && require_dir "${LIBERO_PLUS_ROOT}"
validate_libero_config

join_by_colon() {
    local IFS=:
    echo "$*"
}

run_dir="$(dirname "$(dirname "${WORLDPILOT_CKPT}")")"
source_code_dir="${run_dir}/source_code"
PYTHONPATH_ITEMS=()
if [ -n "${LIBERO_PLUS_ROOT}" ]; then
    PYTHONPATH_ITEMS+=("${LIBERO_PLUS_ROOT}")
fi
if [ "${USE_CHECKPOINT_SOURCE}" = "true" ] && [ -d "${source_code_dir}" ]; then
    echo "[INFO] Using checkpoint source snapshot: ${source_code_dir}"
    PYTHONPATH_ITEMS+=("${source_code_dir}")
fi
PYTHONPATH_ITEMS+=("${REPO_ROOT}")
if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH_ITEMS+=("${PYTHONPATH}")
fi
export PYTHONPATH="$(join_by_colon "${PYTHONPATH_ITEMS[@]}")"

kill_tree() {
    local pid="$1"
    local child
    for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
        kill_tree "${child}"
    done
    kill "${pid}" 2>/dev/null || true
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local label="$3"
    local timeout_sec="${4:-300}"
    local waited=0

    while [ "${waited}" -lt "${timeout_sec}" ]; do
        if CHECK_HOST="${host}" CHECK_PORT="${port}" "${LIBERO_PYTHON}" - <<'PY' >/dev/null 2>&1
import os
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect((os.environ["CHECK_HOST"], int(os.environ["CHECK_PORT"])))
except Exception:
    sys.exit(1)
else:
    sock.close()
    sys.exit(0)
PY
        then
            echo "[INFO] ${label} ready at ${host}:${port}"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done

    echo "[ERROR] Timed out waiting for ${label} at ${host}:${port}" >&2
    return 1
}

COSMOS_PID=""
POLICY_PID=""
cleanup() {
    echo "[INFO] Cleaning up servers..."
    [ -n "${POLICY_PID}" ] && kill_tree "${POLICY_PID}"
    [ -n "${COSMOS_PID}" ] && kill_tree "${COSMOS_PID}"
}
trap cleanup EXIT

UNNORM_ARGS=()
if [ -n "${UNNORM_KEY}" ]; then
    UNNORM_ARGS=(--unnorm-key "${UNNORM_KEY}")
fi

echo "[INFO] Starting Cosmos server on GPU ${GPU}, port ${COSMOS_PORT}"
COSMOS_GPU="${GPU}" COSMOS_PORT="${COSMOS_PORT}" bash "${REPO_ROOT}/cosmos_bridge/run_cosmos_server.sh" &
COSMOS_PID=$!
wait_for_port 127.0.0.1 "${COSMOS_PORT}" "Cosmos server" "${COSMOS_START_TIMEOUT:-600}"

echo "[INFO] Starting ABot policy server on GPU ${GPU}, port ${POLICY_PORT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${WORLDPILOT_PYTHON}" "${REPO_ROOT}/deployment/model_server/server_policy.py" \
    --ckpt_path "${WORLDPILOT_CKPT}" \
    --port "${POLICY_PORT}" \
    --cosmos_port "${COSMOS_PORT}" \
    --use_bf16 &
POLICY_PID=$!
wait_for_port 127.0.0.1 "${POLICY_PORT}" "ABot policy server" "${POLICY_START_TIMEOUT:-600}"

ckpt_stem="$(basename "${WORLDPILOT_CKPT}" .pt)"
run_id="$(basename "$(dirname "$(dirname "${WORLDPILOT_CKPT}")")")"
run_root="$(basename "$(dirname "$(dirname "$(dirname "${WORLDPILOT_CKPT}")")")")"
folder_name="${run_root}_${run_id}_${ckpt_stem}"
LOG_DIR="${OUTPUT_DIR}/logs/${folder_name}_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${LOG_DIR}"
log_file="${LOG_DIR}/${TASK_SUITE_NAME}.log"

echo "[INFO] Starting LIBERO-plus eval: suite=${TASK_SUITE_NAME}, gpu=${GPU}, policy=:${POLICY_PORT}"
MUJOCO_EGL_DEVICE_ID="${GPU}" CUDA_VISIBLE_DEVICES="${GPU}" \
"${LIBERO_PYTHON}" "${REPO_ROOT}/examples/LIBERO-plus/eval_files/eval_libero.py" \
    --host 127.0.0.1 \
    --port "${POLICY_PORT}" \
    --task-suite-name "${TASK_SUITE_NAME}" \
    --pretrained-path "${WORLDPILOT_CKPT}" \
    "${UNNORM_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --log-path "${LOG_DIR}" \
    --start-task-id "${START_TASK_ID}" \
    --success-count "${SUCCESS_COUNT}" \
    2>&1 | tee "${log_file}"

echo "[INFO] Evaluation complete. Results in: ${LOG_DIR}"
