#!/usr/bin/env bash
set -euo pipefail

# Fixed big native-skip preset for FaceHQ/CelebA-HQ 256px images / 32x32 latents.
#
# Design:
# - keep the existing native gated-sum fusion path
# - restore the FID-11-size encoder/fusion dimensions and depths directly
# - keep Mamba size routing off so depth/width stay fixed
# - keep compression deterministic: no compression router, one stride-1 premix
#   before the learned stride-2 downsample path

export MODEL_CONFIG="${MODEL_CONFIG:-hierarchy_hybrid_local_v2}"
export SCAN_TYPE="${SCAN_TYPE:-v2}"
export CUDA_DEVICES="${CUDA_DEVICES:-2,3}"
export MASTER_PORT="${MASTER_PORT:-8873}"

export DATA_TAR_BASE="${DATA_TAR_BASE:-/SSD4/yjjung/datasets/celeba_hq_256_hf_shard/}"
export IMAGE_SIZE="${IMAGE_SIZE:-256}"
export MODEL_IMG_DIM="${MODEL_IMG_DIM:-32}"
export FUSION_ANCHOR_RESOLUTION="${FUSION_ANCHOR_RESOLUTION:-8}"
export DECODER_ANCHOR_RESOLUTION="${DECODER_ANCHOR_RESOLUTION:-8}"

# Existing native fusion topology: 8x8 bottleneck, then fuse the 16x16 and
# 32x32 skips with gated_sum.
export FUSION_MODE="${FUSION_MODE:-gated_sum}"
export FUSION_SELECTED_STAGES="${FUSION_SELECTED_STAGES:-32,16}"
export FUSION_CHANNEL_GATE_STAGES="${FUSION_CHANNEL_GATE_STAGES:-16,32}"
export FUSION_USE_SPATIAL_GATE="${FUSION_USE_SPATIAL_GATE:-false}"
export FUSION_LOGGING_VERBOSE="${FUSION_LOGGING_VERBOSE:-true}"

# Stage router is off by default for this fixed-size restoration.
export USE_FACTORIZED_TOP4_ROUTER="${USE_FACTORIZED_TOP4_ROUTER:-false}"
export ROUTED_STAGE_RESOLUTIONS="${ROUTED_STAGE_RESOLUTIONS:-32,16,8}"
export ROUTED_STAGE_COUNT="${ROUTED_STAGE_COUNT:-3}"
export INCLUDE_ANCHOR_IN_STAGE_ROUTER="${INCLUDE_ANCHOR_IN_STAGE_ROUTER:-true}"
export STAGE_ROUTER_TOP_K="${STAGE_ROUTER_TOP_K:-3}"
export STAGE_ROUTER_WEIGHT_FLOOR="${STAGE_ROUTER_WEIGHT_FLOOR:-0.0}"
export STAGE_ROUTER_MAX_WEIGHT="${STAGE_ROUTER_MAX_WEIGHT:-2.0}"
export STAGE_ROUTER_WEIGHT_MODE="${STAGE_ROUTER_WEIGHT_MODE:-equal_selection}"

# Keep the FID-11-size model fixed; do not route Mamba depth/width.
export USE_MAMBA_SIZE_ROUTER="${USE_MAMBA_SIZE_ROUTER:-false}"
export MAMBA_SIZE_ROUTER_STAGES="${MAMBA_SIZE_ROUTER_STAGES:-32,16,8}"
export MAMBA_SIZE_PRESETS="${MAMBA_SIZE_PRESETS:-small:1:0.75,base:2:1.0,large:4:1.25}"
export MAMBA_SIZE_ROUTER_TOP_K="${MAMBA_SIZE_ROUTER_TOP_K:-1}"
export MAMBA_SIZE_ROUTER_WEIGHT_MODE="${MAMBA_SIZE_ROUTER_WEIGHT_MODE:-selection}"

# Compression is fixed, not routed. The stride-1 premix mirrors the big-model
# preset before the normal learned downsample.
export USE_COMPRESSION_ROUTER="${USE_COMPRESSION_ROUTER:-false}"
export COMPRESSION_ROUTER_STAGES="${COMPRESSION_ROUTER_STAGES:-}"
export DOWNSAMPLE_USE_PREMIX="${DOWNSAMPLE_USE_PREMIX:-true}"
export DOWNSAMPLE_PREMIX_DEPTH="${DOWNSAMPLE_PREMIX_DEPTH:-1}"
export DOWNSAMPLE_CONV_TYPE="${DOWNSAMPLE_CONV_TYPE:-standard}"

# Match the stronger native-skip capacity settings unless the caller overrides.
export EMBED_DIM="${EMBED_DIM:-256}"
export HIERARCHY_STAGE_DEPTH="${HIERARCHY_STAGE_DEPTH:-3}"
export HIGHRES_STAGE_DEPTH="${HIGHRES_STAGE_DEPTH:-4}"
export BOTTLENECK_STAGE_DEPTH="${BOTTLENECK_STAGE_DEPTH:-4}"
export HIGHRES_LOCAL_CONV_DEPTH="${HIGHRES_LOCAL_CONV_DEPTH:-3}"
export FUSION_STAGE_DIM="${FUSION_STAGE_DIM:-320}"
export FUSION_STAGE_DIM_OVERRIDES="${FUSION_STAGE_DIM_OVERRIDES:-16:448,32:512}"
export FUSION_BLOCK_DEPTH="${FUSION_BLOCK_DEPTH:-4}"
export FUSION_STAGE_DEPTH_OVERRIDES="${FUSION_STAGE_DEPTH_OVERRIDES:-16:4,32:6}"
export PREDICTION_HEAD_CONV_DEPTH="${PREDICTION_HEAD_CONV_DEPTH:-3}"
export FINAL_SKIP_REFINER_DEPTH="${FINAL_SKIP_REFINER_DEPTH:-3}"
export FINAL_SKIP_REFINER_USE_CHANNEL_GATE="${FINAL_SKIP_REFINER_USE_CHANNEL_GATE:-true}"
export FINAL_SKIP_REFINER_USE_SPATIAL_GATE="${FINAL_SKIP_REFINER_USE_SPATIAL_GATE:-false}"

export USE_CHECKPOINT="${USE_CHECKPOINT:-true}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-32}"
export VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_GPU}}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export SAMPLE_FID_BS="${SAMPLE_FID_BS:-4}"

export NOTE="${NOTE:-celeb256_fixed_big_native_skip_premix1}"
export TIMESTAMP="${TIMESTAMP:-fixed_big_native_skip_premix1}"
export WANDB_NAME="${WANDB_NAME:-${NOTE}}"
export WANDB_TAGS="${WANDB_TAGS:-fixed_big,celeb256,native_skip,gated_sum,premix1,no_mamba_size_router,no_compression_router}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_facehq1024_hierarchy_hybrid_local_v2_2gpu.sh" "$@"
