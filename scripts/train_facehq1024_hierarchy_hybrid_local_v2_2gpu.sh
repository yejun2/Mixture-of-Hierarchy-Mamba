#!/usr/bin/env bash
set -euo pipefail

cd /SSD4/vipnu/hierarchical_zigma_v1/zigma

set +u
source /SSD4/vipnu/anaconda3/etc/profile.d/conda.sh
conda activate zigma_server_cuda124
set -u

MASTER_PORT="${MASTER_PORT:-8872}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-32}"
VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_GPU}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
SAMPLE_FID_BS="${SAMPLE_FID_BS:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-600000}"
OPTIM_LR="${OPTIM_LR:-}"

DATA_CONFIG="${DATA_CONFIG:-facehq_1024}"
DATA_TAR_BASE="${DATA_TAR_BASE:-/SSD4/vipnu/datasets/celeba_hq_256_hf_shard/}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
MODEL_CONFIG="${MODEL_CONFIG:-hierarchy_hybrid_local_v2}"
MODEL_IMG_DIM="${MODEL_IMG_DIM:-32}"
EMBED_DIM="${EMBED_DIM:-192}"
SCAN_TYPE="${SCAN_TYPE:-v2}"
WINDOW_SIZE="${WINDOW_SIZE:-2}"
FIRST_LAYER_STRIDE="${FIRST_LAYER_STRIDE:-2}"
WINDOW_STRIDE="${WINDOW_STRIDE:-2}"
STAGE_DEPTH="${STAGE_DEPTH:-2}"
HIGHRES_STAGE_DEPTH="${HIGHRES_STAGE_DEPTH:-3}"
BOTTLENECK_STAGE_DEPTH="${BOTTLENECK_STAGE_DEPTH:-2}"
CONTEXT_COMPRESS_TYPE="${CONTEXT_COMPRESS_TYPE:-mean}"
SHARE_STAGE_PROCESSOR="${SHARE_STAGE_PROCESSOR:-false}"
USE_CHECKPOINT="${USE_CHECKPOINT:-false}"
DOWNSAMPLE_USE_PREMIX="${DOWNSAMPLE_USE_PREMIX:-true}"
DOWNSAMPLE_PREMIX_DEPTH="${DOWNSAMPLE_PREMIX_DEPTH:-0}"
DOWNSAMPLE_CONV_TYPE="${DOWNSAMPLE_CONV_TYPE:-standard}"
HIGHRES_LOCAL_CONV_DEPTH="${HIGHRES_LOCAL_CONV_DEPTH:-2}"
HIGHRES_LOCAL_CONV_TYPE="${HIGHRES_LOCAL_CONV_TYPE:-standard}"
PREDICTION_HEAD_CONV_DEPTH="${PREDICTION_HEAD_CONV_DEPTH:-2}"
PREDICTION_HEAD_CONV_TYPE="${PREDICTION_HEAD_CONV_TYPE:-standard}"
FINAL_SKIP_REFINER_DEPTH="${FINAL_SKIP_REFINER_DEPTH:-0}"
FINAL_SKIP_REFINER_CONV_TYPE="${FINAL_SKIP_REFINER_CONV_TYPE:-standard}"
FINAL_SKIP_REFINER_USE_CHANNEL_GATE="${FINAL_SKIP_REFINER_USE_CHANNEL_GATE:-true}"
FINAL_SKIP_REFINER_USE_SPATIAL_GATE="${FINAL_SKIP_REFINER_USE_SPATIAL_GATE:-true}"
USE_AUX_4X4_CONTEXT="${USE_AUX_4X4_CONTEXT:-false}"
AUX_4X4_INJECT_TARGET="${AUX_4X4_INJECT_TARGET:-decoder_anchor}"
AUX_4X4_CONTEXT_RESOLUTION="${AUX_4X4_CONTEXT_RESOLUTION:-4}"
AUX_4X4_TARGET_RESOLUTION="${AUX_4X4_TARGET_RESOLUTION:-auto}"
AUX_4X4_PREPOOL_DEPTH="${AUX_4X4_PREPOOL_DEPTH:-1}"
AUX_4X4_CONTEXT_DEPTH="${AUX_4X4_CONTEXT_DEPTH:-1}"
AUX_4X4_FUSION_DEPTH="${AUX_4X4_FUSION_DEPTH:-1}"
AUX_4X4_CONV_TYPE="${AUX_4X4_CONV_TYPE:-standard}"
AUX_4X4_UPSAMPLE_MODE="${AUX_4X4_UPSAMPLE_MODE:-bilinear}"
AUX_4X4_USE_TIMESTEP_GATE="${AUX_4X4_USE_TIMESTEP_GATE:-true}"
AUX_4X4_RESIDUAL_SCALE_MULTIPLIER="${AUX_4X4_RESIDUAL_SCALE_MULTIPLIER:-1.0}"

USE_MULTISCALE_FUSION_HEAD="${USE_MULTISCALE_FUSION_HEAD:-true}"
FUSION_MODE="${FUSION_MODE:-concat}"
FUSION_SELECTED_STAGES="${FUSION_SELECTED_STAGES:-32,16,8}"
FUSION_ANCHOR_RESOLUTION="${FUSION_ANCHOR_RESOLUTION:-4}"
DECODER_ANCHOR_RESOLUTION="${DECODER_ANCHOR_RESOLUTION:-8}"
FUSION_STAGE_DIM="${FUSION_STAGE_DIM:-320}"
FUSION_STAGE_DIM_OVERRIDES="${FUSION_STAGE_DIM_OVERRIDES:-8:256,16:320,32:384}"
FUSION_GATE_TYPE="${FUSION_GATE_TYPE:-stage_timestep}"
FUSION_POS_EMBED_TYPE="${FUSION_POS_EMBED_TYPE:-anchor_shared}"
FUSION_BLOCK_DEPTH="${FUSION_BLOCK_DEPTH:-3}"
FUSION_STAGE_DEPTH_OVERRIDES="${FUSION_STAGE_DEPTH_OVERRIDES:-16:2,32:4}"
FUSION_CHANNEL_GATE_STAGES="${FUSION_CHANNEL_GATE_STAGES:-8}"
FUSION_USE_SPATIAL_GATE="${FUSION_USE_SPATIAL_GATE:-true}"
FUSION_CONV_DEPTH="${FUSION_CONV_DEPTH:-1}"
FUSION_PRE_MAMBA_CONV_DEPTH="${FUSION_PRE_MAMBA_CONV_DEPTH:-1}"
FUSION_POST_MAMBA_CONV_DEPTH="${FUSION_POST_MAMBA_CONV_DEPTH:-1}"
ANCHOR_BUILDER_DEPTH="${ANCHOR_BUILDER_DEPTH:-3}"
STAGE_SCAN_TYPE_OVERRIDES="${STAGE_SCAN_TYPE_OVERRIDES:-}"
LOCAL_MAMBA_STAGE_RESOLUTIONS="${LOCAL_MAMBA_STAGE_RESOLUTIONS:-}"
LOCAL_MAMBA_WINDOW_SIZE_OVERRIDES="${LOCAL_MAMBA_WINDOW_SIZE_OVERRIDES:-}"
LOCAL_MAMBA_SHIFT_RESOLUTIONS="${LOCAL_MAMBA_SHIFT_RESOLUTIONS:-}"
FUSION_PREDICTION_HEAD_TYPE="${FUSION_PREDICTION_HEAD_TYPE:-anchor_conv_upsample}"
FUSION_LOGGING_VERBOSE="${FUSION_LOGGING_VERBOSE:-true}"

CKPT_PATH="${CKPT:-}"

usage() {
  cat <<'EOF'
Usage: bash zigma/scripts/train_facehq1024_hierarchy_hybrid_local_v2_2gpu.sh [--ckpt PATH]

Options:
  --ckpt PATH   Resume from an experiment directory or a specific .pt checkpoint file.
  -h, --help    Show this help message.

Environment:
  CKPT=PATH     Equivalent to --ckpt PATH.
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
if [[ -n "${OPTIM_LR}" ]]; then
  EXTRA_ARGS+=("optim.lr=${OPTIM_LR}")
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
  model.params.highres_stage_depth="${HIGHRES_STAGE_DEPTH}" \
  model.params.bottleneck_stage_depth="${BOTTLENECK_STAGE_DEPTH}" \
  model.params.context_compress_type="${CONTEXT_COMPRESS_TYPE}" \
  model.params.share_stage_processor="${SHARE_STAGE_PROCESSOR}" \
  model.params.use_multiscale_fusion_head="${USE_MULTISCALE_FUSION_HEAD}" \
  model.params.fusion_mode="${FUSION_MODE}" \
  model.params.fusion_selected_stages="\"${FUSION_SELECTED_STAGES}\"" \
  model.params.fusion_anchor_resolution="${FUSION_ANCHOR_RESOLUTION}" \
  model.params.decoder_anchor_resolution="${DECODER_ANCHOR_RESOLUTION}" \
  model.params.fusion_stage_dim="${FUSION_STAGE_DIM}" \
  model.params.fusion_stage_dim_overrides="\"${FUSION_STAGE_DIM_OVERRIDES}\"" \
  model.params.fusion_gate_type="${FUSION_GATE_TYPE}" \
  model.params.fusion_pos_embed_type="${FUSION_POS_EMBED_TYPE}" \
  model.params.fusion_block_depth="${FUSION_BLOCK_DEPTH}" \
  model.params.fusion_stage_depth_overrides="\"${FUSION_STAGE_DEPTH_OVERRIDES}\"" \
  model.params.fusion_channel_gate_stages="\"${FUSION_CHANNEL_GATE_STAGES}\"" \
  model.params.fusion_use_spatial_gate="${FUSION_USE_SPATIAL_GATE}" \
  model.params.fusion_conv_depth="${FUSION_CONV_DEPTH}" \
  model.params.fusion_pre_mamba_conv_depth="${FUSION_PRE_MAMBA_CONV_DEPTH}" \
  model.params.fusion_post_mamba_conv_depth="${FUSION_POST_MAMBA_CONV_DEPTH}" \
  model.params.anchor_builder_depth="${ANCHOR_BUILDER_DEPTH}" \
  ++model.params.stage_scan_type_overrides="\"${STAGE_SCAN_TYPE_OVERRIDES}\"" \
  ++model.params.local_mamba_stage_resolutions="\"${LOCAL_MAMBA_STAGE_RESOLUTIONS}\"" \
  ++model.params.local_mamba_window_size_overrides="\"${LOCAL_MAMBA_WINDOW_SIZE_OVERRIDES}\"" \
  ++model.params.local_mamba_shift_resolutions="\"${LOCAL_MAMBA_SHIFT_RESOLUTIONS}\"" \
  model.params.fusion_prediction_head_type="${FUSION_PREDICTION_HEAD_TYPE}" \
  model.params.fusion_logging_verbose="${FUSION_LOGGING_VERBOSE}" \
  model.params.downsample_use_premix="${DOWNSAMPLE_USE_PREMIX}" \
  model.params.downsample_premix_depth="${DOWNSAMPLE_PREMIX_DEPTH}" \
  model.params.downsample_conv_type="${DOWNSAMPLE_CONV_TYPE}" \
  model.params.highres_local_conv_depth="${HIGHRES_LOCAL_CONV_DEPTH}" \
  model.params.highres_local_conv_type="${HIGHRES_LOCAL_CONV_TYPE}" \
  model.params.prediction_head_conv_depth="${PREDICTION_HEAD_CONV_DEPTH}" \
  model.params.prediction_head_conv_type="${PREDICTION_HEAD_CONV_TYPE}" \
  model.params.final_skip_refiner_depth="${FINAL_SKIP_REFINER_DEPTH}" \
  model.params.final_skip_refiner_conv_type="${FINAL_SKIP_REFINER_CONV_TYPE}" \
  model.params.final_skip_refiner_use_channel_gate="${FINAL_SKIP_REFINER_USE_CHANNEL_GATE}" \
  model.params.final_skip_refiner_use_spatial_gate="${FINAL_SKIP_REFINER_USE_SPATIAL_GATE}" \
  ++model.params.use_aux_4x4_context="${USE_AUX_4X4_CONTEXT}" \
  ++model.params.aux_4x4_inject_target="${AUX_4X4_INJECT_TARGET}" \
  ++model.params.aux_4x4_context_resolution="${AUX_4X4_CONTEXT_RESOLUTION}" \
  ++model.params.aux_4x4_target_resolution="${AUX_4X4_TARGET_RESOLUTION}" \
  ++model.params.aux_4x4_prepool_depth="${AUX_4X4_PREPOOL_DEPTH}" \
  ++model.params.aux_4x4_context_depth="${AUX_4X4_CONTEXT_DEPTH}" \
  ++model.params.aux_4x4_fusion_depth="${AUX_4X4_FUSION_DEPTH}" \
  ++model.params.aux_4x4_conv_type="${AUX_4X4_CONV_TYPE}" \
  ++model.params.aux_4x4_upsample_mode="${AUX_4X4_UPSAMPLE_MODE}" \
  ++model.params.aux_4x4_use_timestep_gate="${AUX_4X4_USE_TIMESTEP_GATE}" \
  ++model.params.aux_4x4_residual_scale_multiplier="${AUX_4X4_RESIDUAL_SCALE_MULTIPLIER}" \
  +model.params.use_checkpoint="${USE_CHECKPOINT}" \
  "${EXTRA_ARGS[@]}"
