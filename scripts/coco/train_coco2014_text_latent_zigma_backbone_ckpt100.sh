#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_SH="${CONDA_SH:-/SSD4/yjjung/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-MoHmamba_128}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" && -f "${CONDA_SH}" ]]; then
  export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
  set +u
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  set -u
fi

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

MASTER_PORT="${MASTER_PORT:-8878}"
CUDA_DEVICES="${CUDA_DEVICES:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

DATA_CONFIG="${DATA_CONFIG:-coco}"
DATA_TAR_BASE="${DATA_TAR_BASE:-/SSD4/yjjung/datasets/coco2014_text_latent_wds}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
if [[ -z "${TRAIN_SHARDS:-}" ]]; then
  TRAIN_SHARDS='train-{000000..000082}.tar'
fi
if [[ -z "${VAL_SHARDS:-}" ]]; then
  VAL_SHARDS='val-{000000..000040}.tar'
fi
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
CKPT_EVERY="${CKPT_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-1000}"
SAMPLE_FID_N="${SAMPLE_FID_N:-1}"
SAMPLE_FID_EVERY="${SAMPLE_FID_EVERY:-999999}"
SAMPLE_FID_BS="${SAMPLE_FID_BS:-1}"
SAMPLE_VIS_N="${SAMPLE_VIS_N:-1}"

MODEL_CONFIG="${MODEL_CONFIG:-zigzag8_b1_pe2}"
MODEL_IMG_DIM="${MODEL_IMG_DIM:-32}"
IN_CHANNELS="${IN_CHANNELS:-4}"
PATCH_SIZE="${PATCH_SIZE:-1}"
EMBED_DIM="${EMBED_DIM:-768}"
DEPTH="${DEPTH:-24}"
SCAN_TYPE="${SCAN_TYPE:-zigzagN8}"
USE_PE="${USE_PE:-2}"
HAS_TEXT="${HAS_TEXT:-true}"
TEXT_CONTEXT_DIM="${TEXT_CONTEXT_DIM:-768}"
HIERARCHICAL_CONTEXT="${HIERARCHICAL_CONTEXT:-false}"
USE_CHECKPOINT="${USE_CHECKPOINT:-false}"
RMS_NORM="${RMS_NORM:-true}"
FUSED_ADD_NORM="${FUSED_ADD_NORM:-true}"
USE_JIT="${USE_JIT:-true}"

USE_WANDB="${USE_WANDB:-false}"
NOTE="${NOTE:-coco2014_text_latent_zigma_backbone_ckpt100}"
TIMESTAMP="${TIMESTAMP:-coco2014_text_latent_zigma_backbone_ckpt100}"

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
      echo "Expected brace-expanded shard pattern like train-{000000..000082}.tar" >&2
      exit 1
    fi
  fi
}

validate_shards "${TRAIN_SHARDS}" "TRAIN_SHARDS"
validate_shards "${VAL_SHARDS}" "VAL_SHARDS"

LAUNCH_ARGS=(
  --num_processes "${NUM_PROCESSES}"
  --num_machines 1
  --mixed_precision "${MIXED_PRECISION}"
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
  mixed_precision="${MIXED_PRECISION}" \
  use_wandb="${USE_WANDB}" \
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
  data.sample_fid_n="${SAMPLE_FID_N}" \
  data.sample_fid_every="${SAMPLE_FID_EVERY}" \
  data.sample_fid_bs="${SAMPLE_FID_BS}" \
  data.sample_vis_n="${SAMPLE_VIS_N}" \
  model.params.img_dim="${MODEL_IMG_DIM}" \
  model.params.in_channels="${IN_CHANNELS}" \
  model.params.patch_size="${PATCH_SIZE}" \
  model.params.embed_dim="${EMBED_DIM}" \
  model.params.depth="${DEPTH}" \
  model.params.scan_type="${SCAN_TYPE}" \
  model.params.use_pe="${USE_PE}" \
  model.params.has_text="${HAS_TEXT}" \
  model.params.d_context="${TEXT_CONTEXT_DIM}" \
  model.params.hierarchical_context="${HIERARCHICAL_CONTEXT}" \
  +model.params.use_checkpoint="${USE_CHECKPOINT}" \
  +model.params.rms_norm="${RMS_NORM}" \
  +model.params.fused_add_norm="${FUSED_ADD_NORM}" \
  +model.params.use_jit="${USE_JIT}" \
  log_every="${LOG_EVERY}" \
  ckpt_every="${CKPT_EVERY}" \
  note="${NOTE}" \
  timestamp="${TIMESTAMP}"
