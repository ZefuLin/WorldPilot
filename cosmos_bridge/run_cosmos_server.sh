#!/bin/bash
set -euo pipefail

# Launch the Cosmos WebSocket server.
# Required inputs are supplied through environment variables so the script is
# portable across machines and installations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${COSMOS_CKPT:?Set COSMOS_CKPT to the Cosmos Policy checkpoint}"
: "${COSMOS_STATS:?Set COSMOS_STATS to the Cosmos dataset statistics JSON}"
: "${COSMOS_T5:?Set COSMOS_T5 to the Cosmos T5 embeddings pickle}"

COSMOS_GPU="${COSMOS_GPU:-0}"
COSMOS_PORT="${COSMOS_PORT:-8002}"
COSMOS_CONFIG="${COSMOS_CONFIG:-cosmos_predict2_2b_480p_libero__inference_only}"
COSMOS_SUITE="${COSMOS_SUITE:-auto}"
NUM_DENOISE_STEPS="${NUM_DENOISE_STEPS:-5}"
COSMOS_CHUNK_SIZE="${COSMOS_CHUNK_SIZE:-}"
COSMOS_ACTION_DIM="${COSMOS_ACTION_DIM:-}"
COSMOS_PROPRIO_DIM="${COSMOS_PROPRIO_DIM:-}"

if [ -n "${COSMOS_PYTHON:-}" ]; then
    PY_CMD=("${COSMOS_PYTHON}")
else
    COSMOS_ENV_NAME="${COSMOS_ENV_NAME:-cosmos-policy}"
    PY_CMD=(mamba run -n "${COSMOS_ENV_NAME}" python)
fi

EXTRA_ARGS=()
if [ -n "${COSMOS_CHUNK_SIZE}" ]; then
    EXTRA_ARGS+=(--chunk_size "${COSMOS_CHUNK_SIZE}")
fi
if [ -n "${COSMOS_ACTION_DIM}" ]; then
    EXTRA_ARGS+=(--action_dim "${COSMOS_ACTION_DIM}")
fi
if [ -n "${COSMOS_PROPRIO_DIM}" ]; then
    EXTRA_ARGS+=(--proprio_dim "${COSMOS_PROPRIO_DIM}")
fi

COSMOS_ROOT_ARGS=()
if [ -n "${COSMOS_ROOT:-}" ]; then
    COSMOS_ROOT_ARGS=(--cosmos_root "${COSMOS_ROOT}")
fi

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${COSMOS_GPU}" "${PY_CMD[@]}" -m cosmos_bridge.cosmos_server \
    --cosmos_config "${COSMOS_CONFIG}" \
    --cosmos_ckpt "${COSMOS_CKPT}" \
    --cosmos_dataset_stats "${COSMOS_STATS}" \
    --cosmos_t5_embeddings "${COSMOS_T5}" \
    --suite "${COSMOS_SUITE}" \
    --num_denoising_steps "${NUM_DENOISE_STEPS}" \
    "${COSMOS_ROOT_ARGS[@]}" \
    --port "${COSMOS_PORT}" \
    "${EXTRA_ARGS[@]}"
