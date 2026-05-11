#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ARCHIVE="${ARCHIVE:-/SSD4/yjjung/datasets/archive.zip}"
EXTRACT_DIR="${EXTRACT_DIR:-/SSD4/yjjung/datasets/ffhq1024_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-/SSD4/yjjung/datasets/ffhq1024_latent_128_shard}"
SHARD_PREFIX="${SHARD_PREFIX:-train}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VAE_MICRO_BATCH_SIZE="${VAE_MICRO_BATCH_SIZE:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
NUM_WORKERS_TOTAL="${NUM_WORKERS_TOTAL:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
DTYPE="${DTYPE:-fp16}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
EMPTY_CACHE="${EMPTY_CACHE:-false}"
STORE_IMAGES="${STORE_IMAGES:-true}"
VAE_MODEL="${VAE_MODEL:-stabilityai/sd-vae-ft-ema}"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "[prepare-zip] archive does not exist: ${ARCHIVE}" >&2
  exit 1
fi

mkdir -p "${EXTRACT_DIR}" "${OUTPUT_DIR}"

if [[ ! -d "${EXTRACT_DIR}/images1024x1024" ]]; then
  echo "[prepare-zip] extracting ${ARCHIVE} -> ${EXTRACT_DIR}"
  unzip -q "${ARCHIVE}" -d "${EXTRACT_DIR}"
else
  echo "[prepare-zip] using existing extraction: ${EXTRACT_DIR}/images1024x1024"
fi

INPUT_ROOT="${EXTRACT_DIR}/images1024x1024"

IFS=',' read -r -a DEVICE_LIST <<< "${CUDA_DEVICES}"
if [[ "${#DEVICE_LIST[@]}" -eq 0 ]]; then
  echo "[prepare-zip] CUDA_DEVICES is empty." >&2
  exit 1
fi

COMMON_ARGS=(
  "${REPO_ROOT}/scripts/build_facehq1024_latent_shards_from_folders.py"
  --input-root "${INPUT_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --shard-prefix "${SHARD_PREFIX}"
  --samples-per-shard "${SAMPLES_PER_SHARD}"
  --batch-size "${BATCH_SIZE}"
  --vae-micro-batch-size "${VAE_MICRO_BATCH_SIZE}"
  --image-size "${IMAGE_SIZE}"
  --num-workers-total "${NUM_WORKERS_TOTAL}"
  --dtype "${DTYPE}"
  --vae-model "${VAE_MODEL}"
  --extensions ".png"
)

if [[ -n "${NUM_SAMPLES}" ]]; then
  COMMON_ARGS+=(--num-samples "${NUM_SAMPLES}")
fi
if [[ "${EMPTY_CACHE}" == "true" ]]; then
  COMMON_ARGS+=(--empty-cache)
fi
if [[ "${STORE_IMAGES}" == "false" ]]; then
  COMMON_ARGS+=(--no-store-images)
fi

echo "[prepare-zip] input_root=${INPUT_ROOT}"
echo "[prepare-zip] output_dir=${OUTPUT_DIR}"
echo "[prepare-zip] workers=${NUM_WORKERS_TOTAL}, cuda_devices=${CUDA_DEVICES}, samples_per_shard=${SAMPLES_PER_SHARD}"

pids=()
for ((worker_id = 0; worker_id < NUM_WORKERS_TOTAL; worker_id++)); do
  device="${DEVICE_LIST[$((worker_id % ${#DEVICE_LIST[@]}))]}"
  echo "[prepare-zip] launching worker ${worker_id}/${NUM_WORKERS_TOTAL} on CUDA device ${device}"
  CUDA_VISIBLE_DEVICES="${device}" python "${COMMON_ARGS[@]}" --worker-id "${worker_id}" --device cuda &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "[prepare-zip] done. Generated shards:"
find "${OUTPUT_DIR}" -maxdepth 1 -type f -name "${SHARD_PREFIX}*.tar" -printf "%f\n" | sort
