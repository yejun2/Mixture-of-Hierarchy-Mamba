#!/usr/bin/env bash
set -euo pipefail

cd /SSD4/vipnu/hierarchical_zigma_v1/zigma

set +u
source /SSD4/vipnu/anaconda3/etc/profile.d/conda.sh
conda activate zigma_server_cuda124
set -u

MASTER_PORT="${MASTER_PORT:-8873}"
CUDA_DEVICES="${CUDA_DEVICES:-2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-32}"
VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_GPU}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
SAMPLE_FID_BS="${SAMPLE_FID_BS:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-600000}"

DATA_CONFIG="${DATA_CONFIG:-facehq_1024}"
DATA_TAR_BASE="${DATA_TAR_BASE:-/SSD4/vipnu/datasets/celeba_hq_256_hf_shard/}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"

MODEL_CONFIG="${MODEL_CONFIG:-zigzag8_b1_pe2}"
MODEL_IMG_DIM="${MODEL_IMG_DIM:-32}"
IN_CHANNELS="${IN_CHANNELS:-4}"
PATCH_SIZE="${PATCH_SIZE:-1}"
EMBED_DIM="${EMBED_DIM:-768}"
DEPTH="${DEPTH:-24}"
SCAN_TYPE="${SCAN_TYPE:-zigzagN8}"
USE_PE="${USE_PE:-2}"
HIERARCHICAL_CONTEXT="${HIERARCHICAL_CONTEXT:-false}"
USE_CHECKPOINT="${USE_CHECKPOINT:-false}"
RMS_NORM="${RMS_NORM:-true}"
FUSED_ADD_NORM="${FUSED_ADD_NORM:-true}"
USE_JIT="${USE_JIT:-true}"

CKPT_PATH="${CKPT:-}"

usage() {
  cat <<'EOF'
Usage: bash zigma/scripts/train_celebahq256_zigma_backbone_2gpu.sh [--ckpt PATH]

Train the hierarchy-free ZigMa backbone on the same CelebA-HQ 256 latent shards
used by the hierarchy_hybrid_local_v2 FaceHQ/CelebA-HQ 256 runs.

Options:
  --ckpt PATH   Resume from an experiment directory or a specific .pt checkpoint file.
  -h, --help    Show this help message.

Environment overrides:
  CUDA_DEVICES=0,1
  NUM_PROCESSES=2
  BATCH_SIZE_PER_GPU=32
  DATA_TAR_BASE=/SSD4/vipnu/datasets/celeba_hq_256_hf_shard/
  TRAIN_SHARDS='train-{000000..000029}.tar'
  MODEL_CONFIG=zigzag8_b1_pe2
  EMBED_DIM=768
  DEPTH=24
  USE_CHECKPOINT=false
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      if [[ $# -lt 2 ]]; then
        echo "--ckpt requires a path argument" >&2
        usage
        exit 1
      fi
      CKPT_PATH="$2"
      shift 2
      ;;
    --ckpt=*)
      CKPT_PATH="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${SCAN_TYPE}" == "zigzag8" ]]; then
  SCAN_TYPE="zigzagN8"
fi

if [[ -z "${TRAIN_SHARDS:-}" ]]; then
  TRAIN_SHARDS='train-{000000..000029}.tar'
fi

if [[ -z "${VAL_SHARDS:-}" ]]; then
  VAL_SHARDS="${TRAIN_SHARDS}"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

validate_shards() {
  local shard_spec="$1"
  local shard_name="$2"

  if [[ "${shard_spec}" != *.tar ]]; then
    echo "Invalid ${shard_name}: ${shard_spec}" >&2
    echo "${shard_name} must end with .tar" >&2
    exit 1
  fi

  if [[ "${shard_spec}" == *"{"* || "${shard_spec}" == *"}"* ]]; then
    if [[ "${shard_spec}" != *"}.tar" ]]; then
      echo "Invalid ${shard_name}: ${shard_spec}" >&2
      echo "Expected brace-expanded shard pattern like train-{000000..000029}.tar" >&2
      exit 1
    fi
  fi
}

validate_shards "${TRAIN_SHARDS}" "TRAIN_SHARDS"
validate_shards "${VAL_SHARDS}" "VAL_SHARDS"

if [[ -n "${CKPT_PATH}" ]]; then
  if [[ -f "${CKPT_PATH}" && "${CKPT_PATH}" == *.pt ]]; then
    CKPT_PATH="$(cd "$(dirname "${CKPT_PATH}")/.." && pwd)"
  elif [[ -d "${CKPT_PATH}" ]]; then
    CKPT_PATH="$(cd "${CKPT_PATH}" && pwd)"
  else
    echo "Invalid --ckpt path: ${CKPT_PATH}" >&2
    echo "Pass either an experiment directory or a specific .pt checkpoint file." >&2
    exit 1
  fi

  if [[ ! -d "${CKPT_PATH}/checkpoints" ]]; then
    echo "Resume directory is missing checkpoints/: ${CKPT_PATH}" >&2
    exit 1
  fi
fi

EXTRA_ARGS=()
if [[ -n "${CKPT_PATH}" ]]; then
  EXTRA_ARGS+=("ckpt=${CKPT_PATH}")
fi

LAUNCH_ARGS=(
  --num_processes "${NUM_PROCESSES}"
  --num_machines 1
  --mixed_precision fp16
  --main_process_ip 127.0.0.1
  --main_process_port "${MASTER_PORT}"
)

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  LAUNCH_ARGS+=(--multi_gpu)
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python -m accelerate.commands.launch \
  "${LAUNCH_ARGS[@]}" \
  train_acc.py \
  data="${DATA_CONFIG}" \
  model="${MODEL_CONFIG}" \
  gradient_accumulation_steps="${GRAD_ACCUM_STEPS}" \
  is_latent=true \
  use_latent=true \
  data.tar_base="${DATA_TAR_BASE}" \
  data.image_size="${IMAGE_SIZE}" \
  data.train_steps="${TRAIN_STEPS}" \
  data.train.shards="\"${TRAIN_SHARDS}\"" \
  data.validation.shards="\"${VAL_SHARDS}\"" \
  data.batch_size="${BATCH_SIZE_PER_GPU}" \
  data.val_batch_size="${VAL_BATCH_SIZE_PER_GPU}" \
  data.num_workers="${NUM_WORKERS}" \
  data.val_num_workers="${VAL_NUM_WORKERS}" \
  data.sample_fid_bs="${SAMPLE_FID_BS}" \
  model.params.img_dim="${MODEL_IMG_DIM}" \
  model.params.in_channels="${IN_CHANNELS}" \
  model.params.patch_size="${PATCH_SIZE}" \
  model.params.embed_dim="${EMBED_DIM}" \
  model.params.depth="${DEPTH}" \
  model.params.scan_type="${SCAN_TYPE}" \
  model.params.use_pe="${USE_PE}" \
  model.params.hierarchical_context="${HIERARCHICAL_CONTEXT}" \
  +model.params.use_checkpoint="${USE_CHECKPOINT}" \
  +model.params.rms_norm="${RMS_NORM}" \
  +model.params.fused_add_norm="${FUSED_ADD_NORM}" \
  +model.params.use_jit="${USE_JIT}" \
  "${EXTRA_ARGS[@]}"
