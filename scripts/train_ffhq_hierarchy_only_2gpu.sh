#!/usr/bin/env bash
set -euo pipefail

cd /SSD4/vipnu/hierarchical_zigma_v1/zigma

set +u
source /SSD4/vipnu/anaconda3/etc/profile.d/conda.sh
conda activate zigma_server_cuda124
set -u

MASTER_PORT="${MASTER_PORT:-8868}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-32}"
VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_GPU}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
SAMPLE_FID_BS="${SAMPLE_FID_BS:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
DATA_CONFIG="${DATA_CONFIG:-facehq_1024}"
DATA_TAR_BASE="${DATA_TAR_BASE:-/SSD4/vipnu/datasets/celeba_hq_256_hf_shard/}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
MODEL_IMG_DIM="${MODEL_IMG_DIM:-32}"
SCAN_TYPE="${SCAN_TYPE:-v2}"
WINDOW_SIZE="${WINDOW_SIZE:-2}"
FIRST_LAYER_STRIDE="${FIRST_LAYER_STRIDE:-2}"
WINDOW_STRIDE="${WINDOW_STRIDE:-2}"
STAGE_DEPTH="${STAGE_DEPTH:-1}"
EMBED_DIM="${EMBED_DIM:-192}"
CONTEXT_COMPRESS_TYPE="${CONTEXT_COMPRESS_TYPE:-last}"
SHARE_STAGE_PROCESSOR="${SHARE_STAGE_PROCESSOR:-false}"
USE_CHECKPOINT="${USE_CHECKPOINT:-true}"

if [[ -z "${TRAIN_SHARDS:-}" ]]; then
  TRAIN_SHARDS='train-{000000..000027}.tar'
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
      echo "Expected brace-expanded shard pattern like train-{000000..000027}.tar" >&2
      exit 1
    fi
  fi
}

validate_shards "${TRAIN_SHARDS}" "TRAIN_SHARDS"
validate_shards "${VAL_SHARDS}" "VAL_SHARDS"

CUDA_VISIBLE_DEVICES="0,1" python -m accelerate.commands.launch \
  --num_processes 2 \
  --num_machines 1 \
  --multi_gpu \
  --mixed_precision fp16 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${MASTER_PORT}" \
  train_acc.py \
  data="${DATA_CONFIG}" \
  model=hierarchy_only_local_v1 \
  gradient_accumulation_steps="${GRAD_ACCUM_STEPS}" \
  is_latent=true \
  use_latent=true \
  data.tar_base="${DATA_TAR_BASE}" \
  data.image_size="${IMAGE_SIZE}" \
  data.train.shards="\"${TRAIN_SHARDS}\"" \
  data.validation.shards="\"${VAL_SHARDS}\"" \
  data.batch_size="${BATCH_SIZE_PER_GPU}" \
  data.val_batch_size="${VAL_BATCH_SIZE_PER_GPU}" \
  data.num_workers="${NUM_WORKERS}" \
  data.val_num_workers="${VAL_NUM_WORKERS}" \
  data.sample_fid_bs="${SAMPLE_FID_BS}" \
  model.params.img_dim="${MODEL_IMG_DIM}" \
  model.params.in_channels=4 \
  model.params.out_channels=4 \
  model.params.patch_size=1 \
  model.params.embed_dim="${EMBED_DIM}" \
  model.params.scan_type="${SCAN_TYPE}" \
  model.params.use_pe=2 \
  model.params.hierarchy_window_size="${WINDOW_SIZE}" \
  model.params.first_layer_stride="${FIRST_LAYER_STRIDE}" \
  model.params.hierarchy_stride="${WINDOW_STRIDE}" \
  model.params.hierarchy_stage_depth="${STAGE_DEPTH}" \
  model.params.context_compress_type="${CONTEXT_COMPRESS_TYPE}" \
  model.params.share_stage_processor="${SHARE_STAGE_PROCESSOR}" \
  +model.params.use_checkpoint="${USE_CHECKPOINT}"
