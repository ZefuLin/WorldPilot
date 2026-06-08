#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${DATA_ROOT:?Set DATA_ROOT to the root directory containing LIBERO LeRobot datasets}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the target Cosmos cache directory}"
: "${COSMOS_CKPT:?Set COSMOS_CKPT to the Cosmos Policy checkpoint}"
: "${COSMOS_STATS:?Set COSMOS_STATS to the Cosmos dataset statistics JSON}"
: "${COSMOS_T5:?Set COSMOS_T5 to the Cosmos T5 embeddings pickle}"

COSMOS_CONFIG="${COSMOS_CONFIG:-cosmos_predict2_2b_480p_libero__inference_only}"
NUM_DENOISE_STEPS="${NUM_DENOISE_STEPS:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BASE_PORT="${BASE_PORT:-8002}"
COSMOS_START_WAIT="${COSMOS_START_WAIT:-30}"

if [ -n "${GPUS:-}" ]; then
    read -r -a GPUS_ARR <<< "${GPUS}"
else
    GPUS_ARR=(0 1 2 3)
fi

if [ -n "${DATASETS:-}" ]; then
    read -r -a DATASETS_ARR <<< "${DATASETS}"
else
    DATASETS_ARR=(
        libero_10_no_noops_1.0.0_lerobot
        libero_goal_no_noops_1.0.0_lerobot
        libero_object_no_noops_1.0.0_lerobot
        libero_spatial_no_noops_1.0.0_lerobot
    )
fi

NUM_DATASETS="${#DATASETS_ARR[@]}"
if [ "${#GPUS_ARR[@]}" -ne "${NUM_DATASETS}" ]; then
    echo "[ERROR] GPUS and DATASETS must have the same length." >&2
    exit 1
fi

if [ -n "${COSMOS_PYTHON:-}" ]; then
    COSMOS_CMD=("${COSMOS_PYTHON}")
else
    COSMOS_ENV_NAME="${COSMOS_ENV_NAME:-cosmos-policy}"
    COSMOS_CMD=(mamba run -n "${COSMOS_ENV_NAME}" python)
fi

if [ -n "${WORLDPILOT_PYTHON:-}" ]; then
    PRECOMPUTE_CMD=("${WORLDPILOT_PYTHON}")
else
    WORLDPILOT_ENV_NAME="${WORLDPILOT_ENV_NAME:-WorldPilot}"
    PRECOMPUTE_CMD=(mamba run -n "${WORLDPILOT_ENV_NAME}" python)
fi

SERVER_PIDS=()
PRECOMPUTE_PIDS=()

cleanup() {
    for pid in "${PRECOMPUTE_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
    for pid in "${SERVER_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT

cd "${REPO_ROOT}"
echo "[INFO] Starting ${NUM_DATASETS} Cosmos servers..."
for i in $(seq 0 $((NUM_DATASETS - 1))); do
    gpu="${GPUS_ARR[$i]}"
    port=$((BASE_PORT + i))
    dataset="${DATASETS_ARR[$i]}"
    echo "[INFO]   GPU ${gpu} | port ${port} | dataset ${dataset}"

    CUDA_VISIBLE_DEVICES="${gpu}" "${COSMOS_CMD[@]}" -m cosmos_bridge.cosmos_server \
        --cosmos_config "${COSMOS_CONFIG}" \
        --cosmos_ckpt "${COSMOS_CKPT}" \
        --cosmos_dataset_stats "${COSMOS_STATS}" \
        --cosmos_t5_embeddings "${COSMOS_T5}" \
        --num_denoising_steps "${NUM_DENOISE_STEPS}" \
        --port "${port}" &
    SERVER_PIDS+=("$!")
done

echo "[INFO] Waiting ${COSMOS_START_WAIT}s for Cosmos servers to initialize..."
sleep "${COSMOS_START_WAIT}"

echo "[INFO] Starting ${NUM_DATASETS} precompute workers..."
for i in $(seq 0 $((NUM_DATASETS - 1))); do
    port=$((BASE_PORT + i))
    dataset="${DATASETS_ARR[$i]}"

    "${PRECOMPUTE_CMD[@]}" -m cosmos_bridge.precompute \
        --data_root "${DATA_ROOT}" \
        --output_dir "${OUTPUT_DIR}" \
        --cosmos_host 127.0.0.1 \
        --cosmos_port "${port}" \
        --num_workers "${NUM_WORKERS}" \
        --datasets "${dataset}" &
    PRECOMPUTE_PIDS+=("$!")
done

echo "[INFO] Waiting for precompute workers to finish..."
for pid in "${PRECOMPUTE_PIDS[@]}"; do
    wait "${pid}"
done

echo "[INFO] Done. Cache saved to: ${OUTPUT_DIR}"
