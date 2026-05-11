#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/ablation/tmp_coco.py [--ckpt PATH] [--dry-run]

This launches COCO 2014 text-conditioned latent training from WebDataset shards
with the expected keys:
  image.jpg
  img_feature256.npy
  caption_feature.npy
  caption.json

Defaults:
  DATA_TAR_BASE=/SSD4/yjjung/datasets/coco2014_text_latent_wds
  TRAIN_SHARDS=train-{000000..000082}.tar
  VAL_SHARDS=val-{000000..000040}.tar
  DATA_CONFIG=coco
  MODEL_CONFIG=hierarchy_hybrid_local_v2
  CUDA_DEVICES=0,1,2,3
  NUM_PROCESSES=4

Common overrides:
  CUDA_DEVICES=0,1
  NUM_PROCESSES=2
  BATCH_SIZE_PER_GPU=1
  GRAD_ACCUM_STEPS=1
  TRAIN_STEPS=600000
  USE_CHECKPOINT=true
  USE_WANDB=true
  SAMPLE_FID_EVERY=10000
  SAMPLE_FID_N=5000
  NOTE=coco2014_ablation_train_efficiency_bs1_workers1

Options:
  --ckpt PATH    Resume from an experiment directory or a .pt checkpoint.
  --dry-run      Print the launch command and exit.
  -h, --help     Show this help message.
EOF
}

die() {
  echo "$*" >&2
  exit 1
}

CKPT_PATH="${CKPT:-}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      [[ $# -ge 2 ]] || die "--ckpt requires a path argument"
      CKPT_PATH="$2"
      shift 2
      ;;
    --ckpt=*)
      CKPT_PATH="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "Unknown option: $1"
      ;;
  esac
done

# Runtime ---------------------------------------------------------------------
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
  echo "[warn] Continuing with the current shell environment." >&2
fi

export HF_HOME="${HF_HOME:-${REPO_DIR}/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_DIR}/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_DIR}/.cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

# Data ------------------------------------------------------------------------
DATA_TAR_BASE="${DATA_TAR_BASE:-/SSD4/yjjung/datasets/coco2014_text_latent_wds}"
DATA_CONFIG="${DATA_CONFIG:-coco}"
if [[ -z "${TRAIN_SHARDS:-}" ]]; then
  TRAIN_SHARDS='train-{000000..000082}.tar'
fi
if [[ -z "${VAL_SHARDS:-}" ]]; then
  VAL_SHARDS='val-{000000..000040}.tar'
fi
IMAGE_SIZE="${IMAGE_SIZE:-256}"

BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-600000}"
SAMPLE_FID_N="${SAMPLE_FID_N:-5000}"
SAMPLE_FID_EVERY="${SAMPLE_FID_EVERY:-10000}"
SAMPLE_FID_BS="${SAMPLE_FID_BS:-1}"
SAMPLE_VIS_N="${SAMPLE_VIS_N:-1}"

# Launch ----------------------------------------------------------------------
CUDA_DEVICES="${CUDA_DEVICES:-4}"
if [[ -z "${NUM_PROCESSES:-}" ]]; then
  IFS=',' read -r -a CUDA_DEVICE_LIST <<< "${CUDA_DEVICES}"
  NUM_PROCESSES="${#CUDA_DEVICE_LIST[@]}"
fi
MASTER_PORT="${MASTER_PORT:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
USE_WANDB="${USE_WANDB:-true}"
NOTE="${NOTE:-coco2014_ablation_train_efficiency_bs1_workers1}"
TIMESTAMP="${TIMESTAMP:-coco2014_ablation_train_efficiency_bs1_workers1}"
OPTIM_LR="${OPTIM_LR:-}"
EMA_RATE="${EMA_RATE:-}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
LOG_EVERY="${LOG_EVERY:-100}"

# Model -----------------------------------------------------------------------
MODEL_CONFIG="${MODEL_CONFIG:-hierarchy_hybrid_local_v2}"
MODEL_IMG_DIM="${MODEL_IMG_DIM:-32}"
IN_CHANNELS="${IN_CHANNELS:-4}"
OUT_CHANNELS="${OUT_CHANNELS:-4}"
PATCH_SIZE="${PATCH_SIZE:-1}"
TEXT_CONTEXT_DIM="${TEXT_CONTEXT_DIM:-768}"
USE_CHECKPOINT="${USE_CHECKPOINT:-false}"

# Ported from outputs/.../2026-04-28_15-08-48_only_compress_type_router.
# COCO text features are projected from 768 to EMBED_DIM once in the model.
# Decoder stages then project that text context to their own wider stage dims.
EMBED_DIM="${EMBED_DIM:-256}"
SCAN_TYPE="${SCAN_TYPE:-v2}"
USE_PE="${USE_PE:-2}"
HIERARCHY_WINDOW_SIZE="${HIERARCHY_WINDOW_SIZE:-2}"
FIRST_LAYER_STRIDE="${FIRST_LAYER_STRIDE:-2}"
HIERARCHY_STRIDE="${HIERARCHY_STRIDE:-2}"
HIERARCHY_STAGE_DEPTH="${HIERARCHY_STAGE_DEPTH:-3}"
HIERARCHY_MAX_STAGES="${HIERARCHY_MAX_STAGES:-null}"
HIERARCHY_ALLOW_PARTIAL="${HIERARCHY_ALLOW_PARTIAL:-false}"
CONTEXT_COMPRESS_TYPE="${CONTEXT_COMPRESS_TYPE:-mean}"
SHARE_STAGE_PROCESSOR="${SHARE_STAGE_PROCESSOR:-false}"
HIERARCHICAL_OUTPUT_MODE="${HIERARCHICAL_OUTPUT_MODE:-prediction}"
HIGHRES_STAGE_DEPTH="${HIGHRES_STAGE_DEPTH:-4}"
BOTTLENECK_STAGE_DEPTH="${BOTTLENECK_STAGE_DEPTH:-4}"

DOWNSAMPLE_USE_PREMIX="${DOWNSAMPLE_USE_PREMIX:-true}"
DOWNSAMPLE_PREMIX_DEPTH="${DOWNSAMPLE_PREMIX_DEPTH:-1}"
DOWNSAMPLE_CONV_TYPE="${DOWNSAMPLE_CONV_TYPE:-standard}"
HIGHRES_LOCAL_CONV_DEPTH="${HIGHRES_LOCAL_CONV_DEPTH:-3}"
HIGHRES_LOCAL_CONV_TYPE="${HIGHRES_LOCAL_CONV_TYPE:-standard}"
PREDICTION_HEAD_CONV_DEPTH="${PREDICTION_HEAD_CONV_DEPTH:-3}"
PREDICTION_HEAD_CONV_TYPE="${PREDICTION_HEAD_CONV_TYPE:-standard}"

USE_MULTISCALE_FUSION_HEAD="${USE_MULTISCALE_FUSION_HEAD:-true}"
FUSION_MODE="${FUSION_MODE:-gated_sum}"
FUSION_SELECTED_STAGES="${FUSION_SELECTED_STAGES:-32,16}"
FUSION_ANCHOR_RESOLUTION="${FUSION_ANCHOR_RESOLUTION:-8}"
DECODER_ANCHOR_RESOLUTION="${DECODER_ANCHOR_RESOLUTION:-8}"
FUSION_STAGE_DIM="${FUSION_STAGE_DIM:-320}"
FUSION_STAGE_DIM_OVERRIDES="${FUSION_STAGE_DIM_OVERRIDES:-16:448,32:512}"
FUSION_GATE_TYPE="${FUSION_GATE_TYPE:-stage_timestep}"
FUSION_POS_EMBED_TYPE="${FUSION_POS_EMBED_TYPE:-anchor_shared}"
FUSION_BLOCK_DEPTH="${FUSION_BLOCK_DEPTH:-4}"
FUSION_STAGE_DEPTH_OVERRIDES="${FUSION_STAGE_DEPTH_OVERRIDES:-16:4,32:6}"
FUSION_CHANNEL_GATE_STAGES="${FUSION_CHANNEL_GATE_STAGES:-16,32}"
FUSION_USE_SPATIAL_GATE="${FUSION_USE_SPATIAL_GATE:-false}"
FUSION_CONV_DEPTH="${FUSION_CONV_DEPTH:-1}"
FUSION_PRE_MAMBA_CONV_DEPTH="${FUSION_PRE_MAMBA_CONV_DEPTH:-1}"
FUSION_POST_MAMBA_CONV_DEPTH="${FUSION_POST_MAMBA_CONV_DEPTH:-1}"
ANCHOR_BUILDER_DEPTH="${ANCHOR_BUILDER_DEPTH:-3}"
FUSION_PREDICTION_HEAD_TYPE="${FUSION_PREDICTION_HEAD_TYPE:-anchor_conv_upsample}"
FUSION_LOGGING_VERBOSE="${FUSION_LOGGING_VERBOSE:-true}"

FINAL_SKIP_REFINER_DEPTH="${FINAL_SKIP_REFINER_DEPTH:-3}"
FINAL_SKIP_REFINER_CONV_TYPE="${FINAL_SKIP_REFINER_CONV_TYPE:-standard}"
FINAL_SKIP_REFINER_USE_CHANNEL_GATE="${FINAL_SKIP_REFINER_USE_CHANNEL_GATE:-true}"
FINAL_SKIP_REFINER_USE_SPATIAL_GATE="${FINAL_SKIP_REFINER_USE_SPATIAL_GATE:-false}"

USE_FACTORIZED_TOP4_ROUTER="${USE_FACTORIZED_TOP4_ROUTER:-false}"
ROUTED_STAGE_RESOLUTIONS="${ROUTED_STAGE_RESOLUTIONS:-32,16,8}"
ROUTED_STAGE_COUNT="${ROUTED_STAGE_COUNT:-3}"
INCLUDE_ANCHOR_IN_STAGE_ROUTER="${INCLUDE_ANCHOR_IN_STAGE_ROUTER:-true}"
STAGE_ROUTER_TOP_K="${STAGE_ROUTER_TOP_K:-3}"
STAGE_ROUTER_WEIGHT_FLOOR="${STAGE_ROUTER_WEIGHT_FLOOR:-0.0}"
STAGE_ROUTER_MAX_WEIGHT="${STAGE_ROUTER_MAX_WEIGHT:-2.0}"
STAGE_ROUTER_WEIGHT_MODE="${STAGE_ROUTER_WEIGHT_MODE:-equal_selection}"

USE_MAMBA_SIZE_ROUTER="${USE_MAMBA_SIZE_ROUTER:-false}"
MAMBA_SIZE_ROUTER_STAGES="${MAMBA_SIZE_ROUTER_STAGES:-auto}"
MAMBA_SIZE_PRESETS="${MAMBA_SIZE_PRESETS:-light:1:0.75,base:2:1.0,context:3:1.0}"
ENCODER_MAMBA_SIZE_PRESETS="${ENCODER_MAMBA_SIZE_PRESETS:-light:1:0.75,base:2:1.0,context:3:1.25}"
FUSION_MAMBA_SIZE_PRESETS="${FUSION_MAMBA_SIZE_PRESETS:-base:2:1.0,deep:4:1.25,xdeep:6:1.5}"
ENCODER_MAMBA_SIZE_PRESET_OVERRIDES="${ENCODER_MAMBA_SIZE_PRESET_OVERRIDES:-}"
FUSION_MAMBA_SIZE_PRESET_OVERRIDES="${FUSION_MAMBA_SIZE_PRESET_OVERRIDES:-}"
MAMBA_SIZE_ROUTER_TOP_K="${MAMBA_SIZE_ROUTER_TOP_K:-1}"
MAMBA_SIZE_ROUTER_WEIGHT_MODE="${MAMBA_SIZE_ROUTER_WEIGHT_MODE:-selection}"

USE_COMPRESSION_ROUTER="${USE_COMPRESSION_ROUTER:-true}"
COMPRESSION_ROUTER_STAGES="${COMPRESSION_ROUTER_STAGES:-16,8}"

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

STAGE_SCAN_TYPE_OVERRIDES="${STAGE_SCAN_TYPE_OVERRIDES:-}"
LOCAL_MAMBA_STAGE_RESOLUTIONS="${LOCAL_MAMBA_STAGE_RESOLUTIONS:-}"
LOCAL_MAMBA_WINDOW_SIZE_OVERRIDES="${LOCAL_MAMBA_WINDOW_SIZE_OVERRIDES:-}"
LOCAL_MAMBA_SHIFT_RESOLUTIONS="${LOCAL_MAMBA_SHIFT_RESOLUTIONS:-}"

validate_file() {
  local path="$1"
  local label="$2"
  [[ -f "${path}" ]] || die "Missing ${label}: ${path}"
}

[[ -d "${DATA_TAR_BASE}" ]] || die "DATA_TAR_BASE does not exist: ${DATA_TAR_BASE}"
validate_file "${DATA_TAR_BASE}/train-000000.tar" "first train shard"
validate_file "${DATA_TAR_BASE}/train-000082.tar" "last default train shard"
validate_file "${DATA_TAR_BASE}/val-000000.tar" "first validation shard"
validate_file "${DATA_TAR_BASE}/val-000040.tar" "last default validation shard"

if [[ -n "${CKPT_PATH}" ]]; then
  if [[ -f "${CKPT_PATH}" && "${CKPT_PATH}" == *.pt ]]; then
    CKPT_PATH="$(cd "$(dirname "${CKPT_PATH}")/.." && pwd)"
  elif [[ -d "${CKPT_PATH}" ]]; then
    CKPT_PATH="$(cd "${CKPT_PATH}" && pwd)"
  else
    die "Invalid --ckpt path: ${CKPT_PATH}"
  fi
  [[ -d "${CKPT_PATH}/checkpoints" ]] || die "Resume directory is missing checkpoints/: ${CKPT_PATH}"
fi

LAUNCH_ARGS=(
  --num_processes "${NUM_PROCESSES}"
  --num_machines 1
  --mixed_precision "${MIXED_PRECISION}"
  --main_process_ip 127.0.0.1
  --main_process_port "${MASTER_PORT}"
)
if (( NUM_PROCESSES > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu)
fi

HYDRA_ARGS=(
  data="${DATA_CONFIG}"
  model="${MODEL_CONFIG}"

  is_latent=true
  use_latent=true
  gradient_accumulation_steps="${GRAD_ACCUM_STEPS}"
  use_wandb="${USE_WANDB}"
  ckpt_every="${CKPT_EVERY}"
  log_every="${LOG_EVERY}"

  data.tar_base="${DATA_TAR_BASE}"
  data.image_size="${IMAGE_SIZE}"
  data.train.shards="\"${TRAIN_SHARDS}\""
  data.validation.shards="\"${VAL_SHARDS}\""
  data.batch_size="${BATCH_SIZE_PER_GPU}"
  data.val_batch_size="${VAL_BATCH_SIZE_PER_GPU}"
  data.num_workers="${NUM_WORKERS}"
  data.val_num_workers="${VAL_NUM_WORKERS}"
  data.train_steps="${TRAIN_STEPS}"
  data.sample_fid_n="${SAMPLE_FID_N}"
  data.sample_fid_every="${SAMPLE_FID_EVERY}"
  data.sample_fid_bs="${SAMPLE_FID_BS}"
  data.sample_vis_n="${SAMPLE_VIS_N}"

  model.params.img_dim="${MODEL_IMG_DIM}"
  model.params.in_channels="${IN_CHANNELS}"
  model.params.out_channels="${OUT_CHANNELS}"
  model.params.patch_size="${PATCH_SIZE}"
  model.params.embed_dim="${EMBED_DIM}"
  model.params.scan_type="${SCAN_TYPE}"
  model.params.use_pe="${USE_PE}"
  ++model.params.has_text=true
  ++model.params.d_context="${TEXT_CONTEXT_DIM}"
  model.params.hierarchy_window_size="${HIERARCHY_WINDOW_SIZE}"
  model.params.hierarchy_stride="${HIERARCHY_STRIDE}"
  model.params.first_layer_stride="${FIRST_LAYER_STRIDE}"
  model.params.hierarchy_stage_depth="${HIERARCHY_STAGE_DEPTH}"
  model.params.hierarchy_max_stages="${HIERARCHY_MAX_STAGES}"
  model.params.hierarchy_allow_partial="${HIERARCHY_ALLOW_PARTIAL}"
  model.params.context_compress_type="${CONTEXT_COMPRESS_TYPE}"
  model.params.share_stage_processor="${SHARE_STAGE_PROCESSOR}"
  model.params.hierarchical_output_mode="${HIERARCHICAL_OUTPUT_MODE}"
  model.params.highres_stage_depth="${HIGHRES_STAGE_DEPTH}"
  model.params.bottleneck_stage_depth="${BOTTLENECK_STAGE_DEPTH}"

  model.params.downsample_use_premix="${DOWNSAMPLE_USE_PREMIX}"
  model.params.downsample_premix_depth="${DOWNSAMPLE_PREMIX_DEPTH}"
  model.params.downsample_conv_type="${DOWNSAMPLE_CONV_TYPE}"
  model.params.highres_local_conv_depth="${HIGHRES_LOCAL_CONV_DEPTH}"
  model.params.highres_local_conv_type="${HIGHRES_LOCAL_CONV_TYPE}"
  model.params.prediction_head_conv_depth="${PREDICTION_HEAD_CONV_DEPTH}"
  model.params.prediction_head_conv_type="${PREDICTION_HEAD_CONV_TYPE}"

  model.params.use_multiscale_fusion_head="${USE_MULTISCALE_FUSION_HEAD}"
  model.params.fusion_mode="${FUSION_MODE}"
  model.params.fusion_selected_stages="\"${FUSION_SELECTED_STAGES}\""
  model.params.fusion_anchor_resolution="${FUSION_ANCHOR_RESOLUTION}"
  model.params.decoder_anchor_resolution="${DECODER_ANCHOR_RESOLUTION}"
  model.params.fusion_stage_dim="${FUSION_STAGE_DIM}"
  model.params.fusion_stage_dim_overrides="\"${FUSION_STAGE_DIM_OVERRIDES}\""
  model.params.fusion_gate_type="${FUSION_GATE_TYPE}"
  model.params.fusion_pos_embed_type="${FUSION_POS_EMBED_TYPE}"
  model.params.fusion_block_depth="${FUSION_BLOCK_DEPTH}"
  model.params.fusion_stage_depth_overrides="\"${FUSION_STAGE_DEPTH_OVERRIDES}\""
  model.params.fusion_channel_gate_stages="\"${FUSION_CHANNEL_GATE_STAGES}\""
  model.params.fusion_use_spatial_gate="${FUSION_USE_SPATIAL_GATE}"
  model.params.fusion_conv_depth="${FUSION_CONV_DEPTH}"
  model.params.fusion_pre_mamba_conv_depth="${FUSION_PRE_MAMBA_CONV_DEPTH}"
  model.params.fusion_post_mamba_conv_depth="${FUSION_POST_MAMBA_CONV_DEPTH}"
  model.params.anchor_builder_depth="${ANCHOR_BUILDER_DEPTH}"
  model.params.fusion_prediction_head_type="${FUSION_PREDICTION_HEAD_TYPE}"
  model.params.fusion_logging_verbose="${FUSION_LOGGING_VERBOSE}"

  model.params.final_skip_refiner_depth="${FINAL_SKIP_REFINER_DEPTH}"
  model.params.final_skip_refiner_conv_type="${FINAL_SKIP_REFINER_CONV_TYPE}"
  model.params.final_skip_refiner_use_channel_gate="${FINAL_SKIP_REFINER_USE_CHANNEL_GATE}"
  model.params.final_skip_refiner_use_spatial_gate="${FINAL_SKIP_REFINER_USE_SPATIAL_GATE}"

  ++model.params.use_factorized_top4_router="${USE_FACTORIZED_TOP4_ROUTER}"
  ++model.params.routed_stage_resolutions="\"${ROUTED_STAGE_RESOLUTIONS}\""
  ++model.params.routed_stage_count="${ROUTED_STAGE_COUNT}"
  ++model.params.include_anchor_in_stage_router="${INCLUDE_ANCHOR_IN_STAGE_ROUTER}"
  ++model.params.stage_router_top_k="${STAGE_ROUTER_TOP_K}"
  ++model.params.stage_router_weight_floor="${STAGE_ROUTER_WEIGHT_FLOOR}"
  ++model.params.stage_router_max_weight="${STAGE_ROUTER_MAX_WEIGHT}"
  ++model.params.stage_router_weight_mode="${STAGE_ROUTER_WEIGHT_MODE}"

  ++model.params.use_mamba_size_router="${USE_MAMBA_SIZE_ROUTER}"
  ++model.params.mamba_size_router_stages="\"${MAMBA_SIZE_ROUTER_STAGES}\""
  ++model.params.mamba_size_presets="\"${MAMBA_SIZE_PRESETS}\""
  ++model.params.encoder_mamba_size_presets="\"${ENCODER_MAMBA_SIZE_PRESETS}\""
  ++model.params.fusion_mamba_size_presets="\"${FUSION_MAMBA_SIZE_PRESETS}\""
  ++model.params.encoder_mamba_size_preset_overrides="\"${ENCODER_MAMBA_SIZE_PRESET_OVERRIDES}\""
  ++model.params.fusion_mamba_size_preset_overrides="\"${FUSION_MAMBA_SIZE_PRESET_OVERRIDES}\""
  ++model.params.mamba_size_router_top_k="${MAMBA_SIZE_ROUTER_TOP_K}"
  ++model.params.mamba_size_router_weight_mode="${MAMBA_SIZE_ROUTER_WEIGHT_MODE}"

  ++model.params.use_compression_router="${USE_COMPRESSION_ROUTER}"
  ++model.params.compression_router_stages="\"${COMPRESSION_ROUTER_STAGES}\""

  ++model.params.use_aux_4x4_context="${USE_AUX_4X4_CONTEXT}"
  ++model.params.aux_4x4_inject_target="${AUX_4X4_INJECT_TARGET}"
  ++model.params.aux_4x4_context_resolution="${AUX_4X4_CONTEXT_RESOLUTION}"
  ++model.params.aux_4x4_target_resolution="${AUX_4X4_TARGET_RESOLUTION}"
  ++model.params.aux_4x4_prepool_depth="${AUX_4X4_PREPOOL_DEPTH}"
  ++model.params.aux_4x4_context_depth="${AUX_4X4_CONTEXT_DEPTH}"
  ++model.params.aux_4x4_fusion_depth="${AUX_4X4_FUSION_DEPTH}"
  ++model.params.aux_4x4_conv_type="${AUX_4X4_CONV_TYPE}"
  ++model.params.aux_4x4_upsample_mode="${AUX_4X4_UPSAMPLE_MODE}"
  ++model.params.aux_4x4_use_timestep_gate="${AUX_4X4_USE_TIMESTEP_GATE}"
  ++model.params.aux_4x4_residual_scale_multiplier="${AUX_4X4_RESIDUAL_SCALE_MULTIPLIER}"

  ++model.params.stage_scan_type_overrides="\"${STAGE_SCAN_TYPE_OVERRIDES}\""
  ++model.params.local_mamba_stage_resolutions="\"${LOCAL_MAMBA_STAGE_RESOLUTIONS}\""
  ++model.params.local_mamba_window_size_overrides="\"${LOCAL_MAMBA_WINDOW_SIZE_OVERRIDES}\""
  ++model.params.local_mamba_shift_resolutions="\"${LOCAL_MAMBA_SHIFT_RESOLUTIONS}\""
  ++model.params.use_checkpoint="${USE_CHECKPOINT}"
)

[[ -z "${CKPT_PATH}" ]] || HYDRA_ARGS+=(ckpt="${CKPT_PATH}")
[[ -z "${OPTIM_LR}" ]] || HYDRA_ARGS+=(optim.lr="${OPTIM_LR}")
[[ -z "${EMA_RATE}" ]] || HYDRA_ARGS+=(ema_rate="${EMA_RATE}")
[[ -z "${NOTE}" ]] || HYDRA_ARGS+=(note="${NOTE}")
[[ -z "${TIMESTAMP}" ]] || HYDRA_ARGS+=(timestamp="${TIMESTAMP}")

cat <<EOF
Launching COCO 2014 text-latent training
  repo: ${REPO_DIR}
  data: ${DATA_TAR_BASE}
  train shards: ${TRAIN_SHARDS}
  val shards: ${VAL_SHARDS}
  model: ${MODEL_CONFIG}
  text conditioning: has_text=true, d_context=${TEXT_CONTEXT_DIM}
  latent input: ${IN_CHANNELS}x${MODEL_IMG_DIM}x${MODEL_IMG_DIM}
  cuda: ${CUDA_DEVICES} (${NUM_PROCESSES} process/es)
  batch/gpu: ${BATCH_SIZE_PER_GPU}, grad_accum: ${GRAD_ACCUM_STEPS}
  checkpointing: ${USE_CHECKPOINT}
  fusion: ${FUSION_MODE}, stages=${FUSION_SELECTED_STAGES}, anchor=${FUSION_ANCHOR_RESOLUTION}, dim=${FUSION_STAGE_DIM}, dim_overrides=${FUSION_STAGE_DIM_OVERRIDES:-none}
  compression router: ${USE_COMPRESSION_ROUTER}, stages=${COMPRESSION_ROUTER_STAGES}
  stage router: ${USE_FACTORIZED_TOP4_ROUTER}
  mamba-size router: ${USE_MAMBA_SIZE_ROUTER}
EOF

COMMAND=(
  python -m accelerate.commands.launch
  "${LAUNCH_ARGS[@]}"
  train_acc.py
  "${HYDRA_ARGS[@]}"
)

if [[ "${DRY_RUN}" == "true" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_DEVICES}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${COMMAND[@]}"
