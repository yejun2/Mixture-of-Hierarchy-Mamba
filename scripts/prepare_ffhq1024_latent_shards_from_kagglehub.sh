#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

KAGGLEHUB_DATASET="${KAGGLEHUB_DATASET:-rahulbhalley/ffhq-1024x1024}"
INPUT_ROOT="${INPUT_ROOT:-}"
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
EXTENSIONS="${EXTENSIONS:-.png,.jpg,.jpeg,.webp,.bmp}"
RECURSIVE="${RECURSIVE:-false}"

if [[ -z "${INPUT_ROOT}" ]]; then
  echo "[prepare] INPUT_ROOT not set; downloading/finding KaggleHub dataset: ${KAGGLEHUB_DATASET}"
  INPUT_ROOT="$(
    python - "${KAGGLEHUB_DATASET}" <<'PY'
import sys
from pathlib import Path

dataset = sys.argv[1]
try:
    import kagglehub
except ImportError as exc:
    raise SystemExit(
        "kagglehub is not installed. Install it or set INPUT_ROOT to an existing FFHQ image root."
    ) from exc

root = Path(kagglehub.dataset_download(dataset))
candidates = [
    root / "images1024x1024",
    root / "ffhq-dataset" / "images1024x1024",
    root,
]
for candidate in candidates:
    if candidate.is_dir() and any(p.is_dir() for p in candidate.iterdir()):
        print(candidate)
        break
else:
    raise SystemExit(f"Could not find numbered FFHQ image folders under {root}")
PY
  )"
fi

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "[prepare] INPUT_ROOT does not exist: ${INPUT_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

IFS=',' read -r -a DEVICE_LIST <<< "${CUDA_DEVICES}"
if [[ "${#DEVICE_LIST[@]}" -eq 0 ]]; then
  echo "[prepare] CUDA_DEVICES is empty." >&2
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
  --extensions "${EXTENSIONS}"
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
if [[ "${RECURSIVE}" == "true" ]]; then
  COMMON_ARGS+=(--recursive)
fi

echo "[prepare] input_root=${INPUT_ROOT}"
echo "[prepare] output_dir=${OUTPUT_DIR}"
echo "[prepare] workers=${NUM_WORKERS_TOTAL}, cuda_devices=${CUDA_DEVICES}, batch_size=${BATCH_SIZE}, micro_batch=${VAE_MICRO_BATCH_SIZE}, store_images=${STORE_IMAGES}"

pids=()
for ((worker_id = 0; worker_id < NUM_WORKERS_TOTAL; worker_id++)); do
  device="${DEVICE_LIST[$((worker_id % ${#DEVICE_LIST[@]}))]}"
  echo "[prepare] launching worker ${worker_id}/${NUM_WORKERS_TOTAL} on CUDA device ${device}"
  CUDA_VISIBLE_DEVICES="${device}" python "${COMMON_ARGS[@]}" --worker-id "${worker_id}" --device cuda &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "[prepare] done. Generated shards:"
find "${OUTPUT_DIR}" -maxdepth 1 -type f -name "${SHARD_PREFIX}*.tar" -printf "%f\n" | sort
echo "[prepare] train with DATA_TAR_BASE=${OUTPUT_DIR}/"
if [[ "${NUM_WORKERS_TOTAL}" -gt 1 ]]; then
  last_worker=$((NUM_WORKERS_TOTAL - 1))
  echo "[prepare] multi-worker shards are named ${SHARD_PREFIX}-w<worker>-<shard>.tar"
  echo "[prepare] example TRAIN_SHARDS='${SHARD_PREFIX}-w{0..${last_worker}}-{000000..000034}.tar' for 70k images with SAMPLES_PER_SHARD=1000 and ${NUM_WORKERS_TOTAL} workers"
else
  echo "[prepare] example TRAIN_SHARDS='${SHARD_PREFIX}-{000000..000069}.tar' for 70k images with SAMPLES_PER_SHARD=1000"
fi
