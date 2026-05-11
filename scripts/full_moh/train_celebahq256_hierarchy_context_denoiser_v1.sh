#!/usr/bin/env bash
set -euo pipefail

# Full-MoH context-denoiser preset for CelebA-HQ 256px / 32x32 latents.
# This uses hierarchy_context_denoiser_v1:
# - integrated SSR is kept as the stage selector
# - CMR uses the same integrated controller scale prior
# - EDR is enabled from the model config
# - decoder fusion stages are replaced by stage-context projection + denoising

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

repo_abs_path() {
  local path="$1"
  if [[ -z "${path}" || "${path}" == /* ]]; then
    printf '%s\n' "${path}"
  elif [[ "${path}" == "~" ]]; then
    printf '%s\n' "${HOME}"
  elif [[ "${path}" == "~/"* ]]; then
    printf '%s\n' "${HOME}/${path#"~/"}"
  else
    local dir
    local base
    dir="$(dirname "${path}")"
    base="$(basename "${path}")"
    if [[ -d "${REPO_DIR}/${dir}" ]]; then
      printf '%s\n' "$(cd "${REPO_DIR}/${dir}" && pwd)/${base}"
    else
      printf '%s\n' "${REPO_DIR}/${path#./}"
    fi
  fi
}

find_conda_sh() {
  if [[ -n "${CONDA_SH:-}" ]]; then
    repo_abs_path "${CONDA_SH}"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "${conda_base}/etc/profile.d/conda.sh"
      return
    fi
  fi
  for candidate in "${HOME}/anaconda3/etc/profile.d/conda.sh" "${HOME}/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  printf '%s\n' ""
}

usage() {
  cat <<'EOF'
Usage:
  DATA_TAR_BASE=/path/to/celeba_hq_256_hf_shard \
  CUDA_DEVICES=4 \
  BATCH_SIZE_PER_GPU=1 \
  bash scripts/full_moh/train_celebahq256_hierarchy_context_denoiser_v1.sh [--ckpt PATH]

Options:
  --ckpt PATH    Resume from an experiment directory or a specific .pt checkpoint.
                 Relative paths are resolved from the repository root.
  -h, --help     Show this help message.
EOF
}

die() {
  echo "$*" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

export MODEL_CONFIG="${MODEL_CONFIG:-hierarchy_context_denoiser_v1}"
export SCAN_TYPE="${SCAN_TYPE:-v2}"
export CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export MASTER_PORT="${MASTER_PORT:-8879}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

export DATA_CONFIG="${DATA_CONFIG:-facehq_1024}"
export DATA_TAR_BASE="${DATA_TAR_BASE:-../../datasets/celeba_hq_256_hf_shard}"
if [[ -z "${TRAIN_SHARDS:-}" ]]; then
  export TRAIN_SHARDS='train-{000000..000029}.tar'
else
  export TRAIN_SHARDS
fi
if [[ -z "${VAL_SHARDS:-}" ]]; then
  export VAL_SHARDS="${TRAIN_SHARDS}"
else
  export VAL_SHARDS
fi
export LATENT_ONLY="${LATENT_ONLY:-true}"
export IMAGE_SIZE="${IMAGE_SIZE:-256}"
export TRAIN_STEPS="${TRAIN_STEPS:-600000}"
export MODEL_IMG_DIM="${MODEL_IMG_DIM:-32}"

export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
export VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-16}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-7}"
export VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
export SAMPLE_FID_EVERY="${SAMPLE_FID_EVERY:-10000}"
export SAMPLE_FID_START_STEP="${SAMPLE_FID_START_STEP:-}"
export SAMPLE_FID_BS="${SAMPLE_FID_BS:-4}"
export USE_WANDB="${USE_WANDB:-true}"
export USE_CHECKPOINT="${USE_CHECKPOINT:-false}"
export CKPT_EVERY="${CKPT_EVERY:-10000}"

export EMBED_DIM="${EMBED_DIM:-256}"
export HIERARCHY_WINDOW_SIZE="${HIERARCHY_WINDOW_SIZE:-2}"
export FIRST_LAYER_STRIDE="${FIRST_LAYER_STRIDE:-2}"
export HIERARCHY_STRIDE="${HIERARCHY_STRIDE:-2}"
export HIERARCHY_STAGE_DEPTH="${HIERARCHY_STAGE_DEPTH:-3}"
export HIGHRES_STAGE_DEPTH="${HIGHRES_STAGE_DEPTH:-4}"
export BOTTLENECK_STAGE_DEPTH="${BOTTLENECK_STAGE_DEPTH:-4}"
export CONTEXT_COMPRESS_TYPE="${CONTEXT_COMPRESS_TYPE:-mean}"
export SHARE_STAGE_PROCESSOR="${SHARE_STAGE_PROCESSOR:-false}"

export DOWNSAMPLE_USE_PREMIX="${DOWNSAMPLE_USE_PREMIX:-true}"
export DOWNSAMPLE_PREMIX_DEPTH="${DOWNSAMPLE_PREMIX_DEPTH:-1}"
export DOWNSAMPLE_CONV_TYPE="${DOWNSAMPLE_CONV_TYPE:-standard}"
export HIGHRES_LOCAL_CONV_DEPTH="${HIGHRES_LOCAL_CONV_DEPTH:-3}"
export HIGHRES_LOCAL_CONV_TYPE="${HIGHRES_LOCAL_CONV_TYPE:-standard}"

# Context denoiser path.
export CONTEXT_DIM="${CONTEXT_DIM:-768}"
export CONTEXT_RESOLUTION="${CONTEXT_RESOLUTION:-8}"
export CONTEXT_PROJECTOR_DEPTH="${CONTEXT_PROJECTOR_DEPTH:-1}"
export CONTEXT_PROJECTOR_CONV_TYPE="${CONTEXT_PROJECTOR_CONV_TYPE:-standard}"
export CONTEXT_AGGREGATION="${CONTEXT_AGGREGATION:-weighted_sum}"
export DENOISER_DIM="${DENOISER_DIM:-768}"
export DENOISER_DEPTH="${DENOISER_DEPTH:-4}"
export DENOISER_CONV_DEPTH="${DENOISER_CONV_DEPTH:-1}"
export DENOISER_SCAN_TYPE="${DENOISER_SCAN_TYPE:-${SCAN_TYPE}}"
export PREDICTION_HEAD_CONV_DEPTH="${PREDICTION_HEAD_CONV_DEPTH:-3}"
export PREDICTION_HEAD_CONV_TYPE="${PREDICTION_HEAD_CONV_TYPE:-standard}"

export ROUTED_STAGE_RESOLUTIONS="${ROUTED_STAGE_RESOLUTIONS:-32,16,8}"
export ROUTED_STAGE_COUNT="${ROUTED_STAGE_COUNT:-3}"
export INCLUDE_ANCHOR_IN_STAGE_ROUTER="${INCLUDE_ANCHOR_IN_STAGE_ROUTER:-true}"
export STAGE_ROUTER_TOP_K="${STAGE_ROUTER_TOP_K:-3}"
export STAGE_ROUTER_WEIGHT_FLOOR="${STAGE_ROUTER_WEIGHT_FLOOR:-0.0}"
export STAGE_ROUTER_MAX_WEIGHT="${STAGE_ROUTER_MAX_WEIGHT:-2.0}"
export STAGE_ROUTER_WEIGHT_MODE="${STAGE_ROUTER_WEIGHT_MODE:-equal_selection}"

export USE_COMPRESSION_ROUTER="${USE_COMPRESSION_ROUTER:-true}"
export COMPRESSION_ROUTER_STAGES="${COMPRESSION_ROUTER_STAGES:-16,8}"
export USE_ENCODER_MAMBA_DEPTH_ROUTER="${USE_ENCODER_MAMBA_DEPTH_ROUTER:-true}"
export ENCODER_MAMBA_DEPTH_ROUTER_STAGES="${ENCODER_MAMBA_DEPTH_ROUTER_STAGES:-auto}"
export ENCODER_MAMBA_DEPTH_ROUTER_TOP_K="${ENCODER_MAMBA_DEPTH_ROUTER_TOP_K:-1}"

export USE_INTEGRATED_ROUTER_CONTROLLER="${USE_INTEGRATED_ROUTER_CONTROLLER:-true}"
export INTEGRATED_CONTROLLER_STAGE_TOP_K="${INTEGRATED_CONTROLLER_STAGE_TOP_K:-3}"
export INTEGRATED_CONTROLLER_HIDDEN_DIM="${INTEGRATED_CONTROLLER_HIDDEN_DIM:-auto}"
export INTEGRATED_CONTROLLER_USE_CHANNEL_GATE="${INTEGRATED_CONTROLLER_USE_CHANNEL_GATE:-false}"
export INTEGRATED_CONTROLLER_CHANNEL_GATE_SCALE="${INTEGRATED_CONTROLLER_CHANNEL_GATE_SCALE:-0.1}"
export INTEGRATED_CONTROLLER_STAGE_SELECT_MODE="${INTEGRATED_CONTROLLER_STAGE_SELECT_MODE:-adaptive}"
export INTEGRATED_CONTROLLER_STAGE_SELECT_THRESHOLD="${INTEGRATED_CONTROLLER_STAGE_SELECT_THRESHOLD:-0.5}"
export INTEGRATED_CONTROLLER_STAGE_SELECT_THRESHOLD_MARGIN="${INTEGRATED_CONTROLLER_STAGE_SELECT_THRESHOLD_MARGIN:-0.0}"
export INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_STEPS="${INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_STEPS:-0}"
export INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_MIN_SELECTED="${INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_MIN_SELECTED:-0}"
export INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_THRESHOLD_MARGIN="${INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_THRESHOLD_MARGIN:-0.0}"
export INTEGRATED_CONTROLLER_STAGE_MIN_SELECTED="${INTEGRATED_CONTROLLER_STAGE_MIN_SELECTED:-2}"
export INTEGRATED_CONTROLLER_STAGE_BALANCE_MODE="${INTEGRATED_CONTROLLER_STAGE_BALANCE_MODE:-batch}"
export INTEGRATED_CONTROLLER_STAGE_USE_SCALE_PRIOR_CONTEXT="${INTEGRATED_CONTROLLER_STAGE_USE_SCALE_PRIOR_CONTEXT:-true}"

export NOTE="${NOTE:-hierarchy_context_denoiser_v1_celebahq256}"
export TIMESTAMP="${TIMESTAMP:-hierarchy_context_denoiser_v1_celebahq256}"
export WANDB_NAME="${WANDB_NAME:-${NOTE}}"
export WANDB_TAGS="${WANDB_TAGS:-full_moh,context_denoiser,integrated_controller,compression_router,encoder_depth_router,celebahq256,bf16}"

export CKPT_PATH="${CKPT:-}"
export OPTIM_LR="${OPTIM_LR:-}"
export EMA_RATE="${EMA_RATE:-}"
export TRAIN_SHUFFLE="${TRAIN_SHUFFLE:-}"

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

validate_shards() {
  local shard_spec="$1"
  local shard_name="$2"
  [[ "${shard_spec}" == *.tar ]] || die "${shard_name} must end with .tar: ${shard_spec}"
  if [[ "${shard_spec}" == *"{"* || "${shard_spec}" == *"}"* ]]; then
    [[ "${shard_spec}" == *"}.tar" ]] || die "Expected brace pattern like train-{000000..000029}.tar for ${shard_name}: ${shard_spec}"
  fi
}

validate_shards "${TRAIN_SHARDS}" "TRAIN_SHARDS"
validate_shards "${VAL_SHARDS}" "VAL_SHARDS"

DATA_TAR_BASE="$(repo_abs_path "${DATA_TAR_BASE}")"
export DATA_TAR_BASE

[[ -n "${DATA_TAR_BASE}" ]] || die "DATA_TAR_BASE is required."
[[ -d "${DATA_TAR_BASE}" ]] || die "DATA_TAR_BASE does not exist: ${DATA_TAR_BASE}"
[[ "${STAGE_ROUTER_WEIGHT_MODE}" == "equal_selection" ]] || die "STAGE_ROUTER_WEIGHT_MODE must be equal_selection."
[[ "${ENCODER_MAMBA_DEPTH_ROUTER_TOP_K}" == "1" ]] || die "ENCODER_MAMBA_DEPTH_ROUTER_TOP_K must be 1."
(( STAGE_ROUTER_TOP_K <= ROUTED_STAGE_COUNT )) || die "STAGE_ROUTER_TOP_K must be <= ROUTED_STAGE_COUNT."
(( INTEGRATED_CONTROLLER_STAGE_TOP_K <= ROUTED_STAGE_COUNT )) || die "INTEGRATED_CONTROLLER_STAGE_TOP_K must be <= ROUTED_STAGE_COUNT."

if [[ -n "${CKPT_PATH}" ]]; then
  CKPT_PATH="$(repo_abs_path "${CKPT_PATH}")"
  if [[ -f "${CKPT_PATH}" && "${CKPT_PATH}" == *.pt ]]; then
    CKPT_PATH="$(cd "$(dirname "${CKPT_PATH}")/.." && pwd)"
  elif [[ -d "${CKPT_PATH}" ]]; then
    CKPT_PATH="$(cd "${CKPT_PATH}" && pwd)"
  else
    die "Invalid --ckpt path: ${CKPT_PATH}"
  fi
  [[ -d "${CKPT_PATH}/checkpoints" ]] || die "Resume directory is missing checkpoints/: ${CKPT_PATH}"
fi

cd "${REPO_DIR}"

CONDA_SH="$(find_conda_sh)"
CONDA_ENV="${CONDA_ENV:-MoHmamba_128}"
[[ -f "${CONDA_SH}" ]] || die "Missing conda activation script. Set CONDA_SH=/path/to/conda.sh or make conda available on PATH."

export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  set +u
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  set -u
fi

export HF_HOME="${HF_HOME:-${REPO_DIR}/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_DIR}/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_DIR}/.cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

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
  gradient_accumulation_steps="${GRAD_ACCUM_STEPS}"
  use_wandb="${USE_WANDB}"
  is_latent=true
  use_latent=true
  data.tar_base="${DATA_TAR_BASE}"
  ++data.latent_only="${LATENT_ONLY}"
  data.image_size="${IMAGE_SIZE}"
  data.train_steps="${TRAIN_STEPS}"
  data.train.shards="\"${TRAIN_SHARDS}\""
  data.validation.shards="\"${VAL_SHARDS}\""
  data.batch_size="${BATCH_SIZE_PER_GPU}"
  data.val_batch_size="${VAL_BATCH_SIZE_PER_GPU}"
  data.num_workers="${NUM_WORKERS}"
  data.val_num_workers="${VAL_NUM_WORKERS}"
  data.sample_fid_bs="${SAMPLE_FID_BS}"

  model.params.img_dim="${MODEL_IMG_DIM}"
  model.params.in_channels=4
  model.params.out_channels=4
  model.params.patch_size=1
  model.params.embed_dim="${EMBED_DIM}"
  model.params.scan_type="${SCAN_TYPE}"
  model.params.use_pe=2
  model.params.hierarchy_window_size="${HIERARCHY_WINDOW_SIZE}"
  model.params.first_layer_stride="${FIRST_LAYER_STRIDE}"
  model.params.hierarchy_stride="${HIERARCHY_STRIDE}"
  model.params.hierarchy_stage_depth="${HIERARCHY_STAGE_DEPTH}"
  model.params.highres_stage_depth="${HIGHRES_STAGE_DEPTH}"
  model.params.bottleneck_stage_depth="${BOTTLENECK_STAGE_DEPTH}"
  model.params.context_compress_type="${CONTEXT_COMPRESS_TYPE}"
  model.params.share_stage_processor="${SHARE_STAGE_PROCESSOR}"

  model.params.downsample_use_premix="${DOWNSAMPLE_USE_PREMIX}"
  model.params.downsample_premix_depth="${DOWNSAMPLE_PREMIX_DEPTH}"
  model.params.downsample_conv_type="${DOWNSAMPLE_CONV_TYPE}"
  model.params.highres_local_conv_depth="${HIGHRES_LOCAL_CONV_DEPTH}"
  model.params.highres_local_conv_type="${HIGHRES_LOCAL_CONV_TYPE}"

  model.params.context_dim="${CONTEXT_DIM}"
  model.params.context_resolution="${CONTEXT_RESOLUTION}"
  model.params.context_projector_depth="${CONTEXT_PROJECTOR_DEPTH}"
  model.params.context_projector_conv_type="${CONTEXT_PROJECTOR_CONV_TYPE}"
  model.params.context_aggregation="${CONTEXT_AGGREGATION}"
  model.params.denoiser_dim="${DENOISER_DIM}"
  model.params.denoiser_depth="${DENOISER_DEPTH}"
  model.params.denoiser_conv_depth="${DENOISER_CONV_DEPTH}"
  model.params.denoiser_scan_type="${DENOISER_SCAN_TYPE}"
  model.params.prediction_head_conv_depth="${PREDICTION_HEAD_CONV_DEPTH}"
  model.params.prediction_head_conv_type="${PREDICTION_HEAD_CONV_TYPE}"

  model.params.routed_stage_resolutions="\"${ROUTED_STAGE_RESOLUTIONS}\""
  model.params.routed_stage_count="${ROUTED_STAGE_COUNT}"
  model.params.include_anchor_in_stage_router="${INCLUDE_ANCHOR_IN_STAGE_ROUTER}"
  model.params.stage_router_top_k="${STAGE_ROUTER_TOP_K}"
  model.params.stage_router_weight_floor="${STAGE_ROUTER_WEIGHT_FLOOR}"
  model.params.stage_router_max_weight="${STAGE_ROUTER_MAX_WEIGHT}"
  model.params.stage_router_weight_mode="${STAGE_ROUTER_WEIGHT_MODE}"

  model.params.use_integrated_router_controller="${USE_INTEGRATED_ROUTER_CONTROLLER}"
  model.params.integrated_controller_stage_top_k="${INTEGRATED_CONTROLLER_STAGE_TOP_K}"
  model.params.integrated_controller_hidden_dim="${INTEGRATED_CONTROLLER_HIDDEN_DIM}"
  model.params.integrated_controller_use_channel_gate="${INTEGRATED_CONTROLLER_USE_CHANNEL_GATE}"
  model.params.integrated_controller_channel_gate_scale="${INTEGRATED_CONTROLLER_CHANNEL_GATE_SCALE}"
  model.params.integrated_controller_stage_select_mode="${INTEGRATED_CONTROLLER_STAGE_SELECT_MODE}"
  model.params.integrated_controller_stage_select_threshold="${INTEGRATED_CONTROLLER_STAGE_SELECT_THRESHOLD}"
  model.params.integrated_controller_stage_select_threshold_margin="${INTEGRATED_CONTROLLER_STAGE_SELECT_THRESHOLD_MARGIN}"
  model.params.integrated_controller_stage_select_warmup_steps="${INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_STEPS}"
  model.params.integrated_controller_stage_select_warmup_min_selected="${INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_MIN_SELECTED}"
  model.params.integrated_controller_stage_select_warmup_threshold_margin="${INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_THRESHOLD_MARGIN}"
  model.params.integrated_controller_stage_min_selected="${INTEGRATED_CONTROLLER_STAGE_MIN_SELECTED}"
  model.params.integrated_controller_stage_balance_mode="${INTEGRATED_CONTROLLER_STAGE_BALANCE_MODE}"
  model.params.integrated_controller_stage_use_scale_prior_context="${INTEGRATED_CONTROLLER_STAGE_USE_SCALE_PRIOR_CONTEXT}"

  model.params.use_compression_router="${USE_COMPRESSION_ROUTER}"
  model.params.compression_router_stages="\"${COMPRESSION_ROUTER_STAGES}\""
  model.params.use_encoder_mamba_depth_router="${USE_ENCODER_MAMBA_DEPTH_ROUTER}"
  model.params.encoder_mamba_depth_router_stages="${ENCODER_MAMBA_DEPTH_ROUTER_STAGES}"
  model.params.encoder_mamba_depth_router_top_k="${ENCODER_MAMBA_DEPTH_ROUTER_TOP_K}"
  +model.params.use_checkpoint="${USE_CHECKPOINT}"
)

[[ -z "${CKPT_PATH}" ]] || HYDRA_ARGS+=(ckpt="${CKPT_PATH}")
[[ -z "${OPTIM_LR}" ]] || HYDRA_ARGS+=(optim.lr="${OPTIM_LR}")
[[ -z "${EMA_RATE}" ]] || HYDRA_ARGS+=(ema_rate="${EMA_RATE}")
[[ -z "${CKPT_EVERY}" ]] || HYDRA_ARGS+=(ckpt_every="${CKPT_EVERY}")
[[ -z "${SAMPLE_FID_EVERY}" ]] || HYDRA_ARGS+=(data.sample_fid_every="${SAMPLE_FID_EVERY}")
[[ -z "${SAMPLE_FID_START_STEP}" ]] || HYDRA_ARGS+=(++data.sample_fid_start_step="${SAMPLE_FID_START_STEP}")
[[ -z "${TRAIN_SHUFFLE}" ]] || HYDRA_ARGS+=(++data.train.shuffle="${TRAIN_SHUFFLE}")
[[ -z "${NOTE}" ]] || HYDRA_ARGS+=(note="${NOTE}")
[[ -z "${TIMESTAMP}" ]] || HYDRA_ARGS+=(timestamp="${TIMESTAMP}")

cat <<EOF
Launching hierarchy_context_denoiser_v1
  repo: ${REPO_DIR}
  data: ${DATA_TAR_BASE}
  shards: ${TRAIN_SHARDS}
  cuda: ${CUDA_DEVICES} (${NUM_PROCESSES} processes)
  mixed precision: ${MIXED_PRECISION}
  batch/gpu: ${BATCH_SIZE_PER_GPU}, grad_accum: ${GRAD_ACCUM_STEPS}
  checkpoint every: ${CKPT_EVERY:-default}
  context denoiser: context_dim=${CONTEXT_DIM}, context_res=${CONTEXT_RESOLUTION}, denoiser_dim=${DENOISER_DIM}, denoiser_depth=${DENOISER_DEPTH}
  SSR: ${USE_INTEGRATED_ROUTER_CONTROLLER}, mode=${INTEGRATED_CONTROLLER_STAGE_SELECT_MODE}, stages=${ROUTED_STAGE_RESOLUTIONS}, top_k=${INTEGRATED_CONTROLLER_STAGE_TOP_K}, min_selected=${INTEGRATED_CONTROLLER_STAGE_MIN_SELECTED}
  CMR: ${USE_COMPRESSION_ROUTER}, stages=${COMPRESSION_ROUTER_STAGES}
  EDR: ${USE_ENCODER_MAMBA_DEPTH_ROUTER}, stages=${ENCODER_MAMBA_DEPTH_ROUTER_STAGES}, top_k=${ENCODER_MAMBA_DEPTH_ROUTER_TOP_K}
EOF

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python -m accelerate.commands.launch \
  "${LAUNCH_ARGS[@]}" \
  train_acc.py \
  "${HYDRA_ARGS[@]}"
