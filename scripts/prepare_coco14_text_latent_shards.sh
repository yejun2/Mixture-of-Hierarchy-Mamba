#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

usage() {
  cat <<'EOF'
Usage:
  RAW_DIR=/path/to/coco2014_raw \
  OUTPUT_DIR=/path/to/coco2014_sd15_wds \
  bash scripts/prepare_coco14_text_latent_shards.sh

Build COCO 2014 WebDataset shards for config/data/coco.yaml.

Outputs contain:
  image.jpg
  img_feature256.npy
  caption_feature.npy
  caption.json

Common overrides:
  CONDA_SH=/path/to/conda.sh
  CONDA_ENV=MoHmamba_128
  RAW_DIR=/SSD4/yjjung/datasets/coco2014_raw
  OUTPUT_DIR=/SSD4/yjjung/datasets/coco2014_sd15_wds
  SPLITS="train val"
  SAMPLES_PER_SHARD=1000
  BATCH_SIZE=8
  VAE_MICRO_BS=4
  TEXT_BS=64
  CUDA_DEVICE=cuda
  DTYPE=fp16
  MAX_SAMPLES_PER_SPLIT=100
  SKIP_DOWNLOAD=true
  LOCAL_FILES_ONLY=true
  OVERWRITE=true

Options:
  -h, --help  Show this help message.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CONDA_SH="${CONDA_SH:-/SSD4/yjjung/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-MoHmamba_128}"
if [[ -f "${CONDA_SH}" ]]; then
  export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
  set +u
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  set -u
else
  echo "[warn] Missing conda activation script: ${CONDA_SH}" >&2
  echo "[warn] Continuing with the current Python environment." >&2
fi

RAW_DIR="${RAW_DIR:-/SSD4/yjjung/datasets/coco2014_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-/SSD4/yjjung/datasets/coco2014_sd15_wds}"
SPLITS="${SPLITS:-train val}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VAE_MICRO_BS="${VAE_MICRO_BS:-4}"
TEXT_BS="${TEXT_BS:-64}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
CAPTIONS_PER_IMAGE="${CAPTIONS_PER_IMAGE:-5}"
MODEL_ID="${MODEL_ID:-runwayml/stable-diffusion-v1-5}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
MAX_SAMPLES_PER_SPLIT="${MAX_SAMPLES_PER_SPLIT:-}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-false}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-false}"
OVERWRITE="${OVERWRITE:-false}"

ARGS=(
  --raw-dir "${RAW_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --splits ${SPLITS}
  --samples-per-shard "${SAMPLES_PER_SHARD}"
  --batch-size "${BATCH_SIZE}"
  --vae-micro-batch-size "${VAE_MICRO_BS}"
  --text-batch-size "${TEXT_BS}"
  --image-size "${IMAGE_SIZE}"
  --captions-per-image "${CAPTIONS_PER_IMAGE}"
  --model-id "${MODEL_ID}"
  --device "${CUDA_DEVICE}"
  --dtype "${DTYPE}"
)

if [[ -n "${MAX_SAMPLES_PER_SPLIT}" ]]; then
  ARGS+=(--max-samples-per-split "${MAX_SAMPLES_PER_SPLIT}")
fi
if [[ "${SKIP_DOWNLOAD}" == "true" ]]; then
  ARGS+=(--skip-download)
fi
if [[ "${LOCAL_FILES_ONLY}" == "true" ]]; then
  ARGS+=(--local-files-only)
fi
if [[ "${OVERWRITE}" == "true" ]]; then
  ARGS+=(--overwrite)
fi

cat <<EOF
Preparing COCO 2014 text-latent shards
  repo: ${REPO_DIR}
  raw_dir: ${RAW_DIR}
  output_dir: ${OUTPUT_DIR}
  splits: ${SPLITS}
  samples/shard: ${SAMPLES_PER_SHARD}
  batch: ${BATCH_SIZE}, vae_micro: ${VAE_MICRO_BS}, text_bs: ${TEXT_BS}
  model: ${MODEL_ID}
  device/dtype: ${CUDA_DEVICE}/${DTYPE}
EOF

python scripts/prepare_coco14_text_latent_shards.py "${ARGS[@]}"
