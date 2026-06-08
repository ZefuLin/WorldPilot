#!/bin/bash
set -euo pipefail

# Standalone ABot policy server for LIBERO-plus evaluation.
# Normal evaluation should use eval_libero_single.sh or eval_libero_batch.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

: "${WORLDPILOT_CKPT:?Set WORLDPILOT_CKPT to a WorldPilot checkpoint path}"

POLICY_PORT="${POLICY_PORT:-9883}"
COSMOS_PORT="${COSMOS_PORT:-8002}"
POLICY_GPU="${POLICY_GPU:-0}"
WORLDPILOT_PYTHON="${WORLDPILOT_PYTHON:-python}"
USE_CHECKPOINT_SOURCE="${USE_CHECKPOINT_SOURCE:-false}"

run_dir="$(dirname "$(dirname "${WORLDPILOT_CKPT}")")"
source_code_dir="${run_dir}/source_code"

PYTHONPATH_ITEMS=("${REPO_ROOT}")
if [ "${USE_CHECKPOINT_SOURCE}" = "true" ] && [ -d "${source_code_dir}" ]; then
    echo "[INFO] Using checkpoint source snapshot: ${source_code_dir}"
    PYTHONPATH_ITEMS=("${source_code_dir}" "${PYTHONPATH_ITEMS[@]}")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_ITEMS[*]}"):${PYTHONPATH:-}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${POLICY_GPU}" "${WORLDPILOT_PYTHON}" deployment/model_server/server_policy.py \
    --ckpt_path "${WORLDPILOT_CKPT}" \
    --port "${POLICY_PORT}" \
    --cosmos_port "${COSMOS_PORT}" \
    --use_bf16
