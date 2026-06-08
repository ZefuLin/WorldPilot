#!/bin/bash
set -euo pipefail

# Parallel LIBERO-plus evaluation.
# Starts one Cosmos server, one ABot policy server, and one simulator per suite.

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
# - SUITES / GPUS / COSMOS_PORTS / POLICY_PORTS
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
USE_CHECKPOINT_SOURCE=false

SUITES=(libero_goal libero_spatial libero_object libero_10)
GPUS=(0 1 2 3)
COSMOS_PORTS=(8002 8003 8004 8005)
POLICY_PORTS=(9883 9884 9885 9886)

declare -A START_TASK_ID=(
    [libero_goal]=0
    [libero_spatial]=0
    [libero_object]=0
    [libero_10]=0
)
declare -A SUCCESS_COUNT=(
    [libero_goal]=0
    [libero_spatial]=0
    [libero_object]=0
    [libero_10]=0
)
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

NUM_SUITES="${#SUITES[@]}"
if [ "${#GPUS[@]}" -ne "${NUM_SUITES}" ] || \
   [ "${#COSMOS_PORTS[@]}" -ne "${NUM_SUITES}" ] || \
   [ "${#POLICY_PORTS[@]}" -ne "${NUM_SUITES}" ]; then
    echo "[ERROR] SUITES, GPUS, COSMOS_PORTS, and POLICY_PORTS must have the same length." >&2
    exit 1
fi

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

COSMOS_PIDS=()
POLICY_PIDS=()
EVAL_PIDS=()
cleanup() {
    echo "[INFO] Cleaning up eval jobs and servers..."
    for pid in "${EVAL_PIDS[@]}"; do kill_tree "${pid}"; done
    for pid in "${POLICY_PIDS[@]}"; do kill_tree "${pid}"; done
    for pid in "${COSMOS_PIDS[@]}"; do kill_tree "${pid}"; done
}
trap cleanup EXIT

UNNORM_ARGS=()
if [ -n "${UNNORM_KEY}" ]; then
    UNNORM_ARGS=(--unnorm-key "${UNNORM_KEY}")
fi

echo "[INFO] Starting ${NUM_SUITES} Cosmos servers"
for i in "${!SUITES[@]}"; do
    gpu="${GPUS[$i]}"
    cosmos_port="${COSMOS_PORTS[$i]}"
    echo "[INFO]   ${SUITES[$i]}: Cosmos GPU ${gpu}, port ${cosmos_port}"
    COSMOS_GPU="${gpu}" COSMOS_PORT="${cosmos_port}" bash "${REPO_ROOT}/cosmos_bridge/run_cosmos_server.sh" &
    COSMOS_PIDS+=("$!")
done
for i in "${!SUITES[@]}"; do
    wait_for_port 127.0.0.1 "${COSMOS_PORTS[$i]}" "Cosmos server ${SUITES[$i]}" "${COSMOS_START_TIMEOUT:-600}"
done

echo "[INFO] Starting ${NUM_SUITES} ABot policy servers"
for i in "${!SUITES[@]}"; do
    gpu="${GPUS[$i]}"
    cosmos_port="${COSMOS_PORTS[$i]}"
    policy_port="${POLICY_PORTS[$i]}"
    echo "[INFO]   ${SUITES[$i]}: Policy GPU ${gpu}, policy :${policy_port}, cosmos :${cosmos_port}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${WORLDPILOT_PYTHON}" "${REPO_ROOT}/deployment/model_server/server_policy.py" \
        --ckpt_path "${WORLDPILOT_CKPT}" \
        --port "${policy_port}" \
        --cosmos_port "${cosmos_port}" \
        --use_bf16 &
    POLICY_PIDS+=("$!")
done
for i in "${!SUITES[@]}"; do
    wait_for_port 127.0.0.1 "${POLICY_PORTS[$i]}" "ABot policy server ${SUITES[$i]}" "${POLICY_START_TIMEOUT:-600}"
done

ckpt_stem="$(basename "${WORLDPILOT_CKPT}" .pt)"
run_id="$(basename "$(dirname "$(dirname "${WORLDPILOT_CKPT}")")")"
run_root="$(basename "$(dirname "$(dirname "$(dirname "${WORLDPILOT_CKPT}")")")")"
folder_name="${run_root}_${run_id}_${ckpt_stem}"
LOG_DIR="${OUTPUT_DIR}/logs/${folder_name}_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${LOG_DIR}"

echo "[INFO] Starting LIBERO-plus evaluations"
for i in "${!SUITES[@]}"; do
    suite="${SUITES[$i]}"
    gpu="${GPUS[$i]}"
    policy_port="${POLICY_PORTS[$i]}"
    log_file="${LOG_DIR}/${suite}.log"

    echo "[INFO]   Eval ${suite}: GPU ${gpu}, policy :${policy_port}"
    (
        set -o pipefail
        MUJOCO_EGL_DEVICE_ID="${gpu}" CUDA_VISIBLE_DEVICES="${gpu}" \
        "${LIBERO_PYTHON}" "${REPO_ROOT}/examples/LIBERO-plus/eval_files/eval_libero.py" \
            --host 127.0.0.1 \
            --port "${policy_port}" \
            --task-suite-name "${suite}" \
            --pretrained-path "${WORLDPILOT_CKPT}" \
            "${UNNORM_ARGS[@]}" \
            --output-dir "${OUTPUT_DIR}" \
            --log-path "${LOG_DIR}" \
            --start-task-id "${START_TASK_ID[$suite]:-0}" \
            --success-count "${SUCCESS_COUNT[$suite]:-0}" \
            2>&1 | tee "${log_file}"
    ) &
    EVAL_PIDS+=("$!")
done

echo "[INFO] Waiting for all evaluations to finish"
wait "${EVAL_PIDS[@]}"
echo "[INFO] All evaluations complete. Results in: ${LOG_DIR}"
