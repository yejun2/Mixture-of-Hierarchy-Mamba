#!/usr/bin/env bash
set -euo pipefail

# Big-parameter v2 native-skip preset.
# Keeps the best-performing native-skip topology, but scales both Mamba depth
# and channel dimensions for the first larger-model ablation.

export MODEL_CONFIG="${MODEL_CONFIG:-hierarchy_hybrid_local_v2}"
export SCAN_TYPE="${SCAN_TYPE:-v2}"
export CUDA_DEVICES="${CUDA_DEVICES:-2,3}"

# Keep effective global batch close to the native-skip baseline:
# baseline 32/GPU x 2 GPUs x accum 1 = 64, here 16/GPU x 2 GPUs x accum 2 = 64.
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-64}"
export VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_GPU}}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export SAMPLE_FID_BS="${SAMPLE_FID_BS:-4}"
export USE_CHECKPOINT="${USE_CHECKPOINT:-true}"

# Wider shared hidden dimension and deeper encoder/bottleneck Mamba stacks.
export EMBED_DIM="${EMBED_DIM:-256}"
export STAGE_DEPTH="${STAGE_DEPTH:-3}"
export HIGHRES_STAGE_DEPTH="${HIGHRES_STAGE_DEPTH:-4}"
export BOTTLENECK_STAGE_DEPTH="${BOTTLENECK_STAGE_DEPTH:-4}"

# A little more stride-1 processing before each learned compression step.
export DOWNSAMPLE_USE_PREMIX="${DOWNSAMPLE_USE_PREMIX:-true}"
export DOWNSAMPLE_PREMIX_DEPTH="${DOWNSAMPLE_PREMIX_DEPTH:-1}"
export DOWNSAMPLE_CONV_TYPE="${DOWNSAMPLE_CONV_TYPE:-standard}"

export HIGHRES_LOCAL_CONV_DEPTH="${HIGHRES_LOCAL_CONV_DEPTH:-3}"
export HIGHRES_LOCAL_CONV_TYPE="${HIGHRES_LOCAL_CONV_TYPE:-standard}"

# Native-skip fusion: 8x8 anchor, direct 16/32 skips, no anchor-builder path.
export FUSION_ANCHOR_RESOLUTION="${FUSION_ANCHOR_RESOLUTION:-8}"
export DECODER_ANCHOR_RESOLUTION="${DECODER_ANCHOR_RESOLUTION:-8}"
export FUSION_SELECTED_STAGES="${FUSION_SELECTED_STAGES:-32,16}"
export FUSION_MODE="${FUSION_MODE:-gated_sum}"

# Wider decoder/fusion stages and stronger decoder-side Mamba depth.
export FUSION_STAGE_DIM="${FUSION_STAGE_DIM:-320}"
export FUSION_STAGE_DIM_OVERRIDES="${FUSION_STAGE_DIM_OVERRIDES:-16:448,32:512}"
export FUSION_BLOCK_DEPTH="${FUSION_BLOCK_DEPTH:-4}"
export FUSION_STAGE_DEPTH_OVERRIDES="${FUSION_STAGE_DEPTH_OVERRIDES:-16:4,32:6}"
export FUSION_CHANNEL_GATE_STAGES="${FUSION_CHANNEL_GATE_STAGES:-16,32}"
export FUSION_USE_SPATIAL_GATE="${FUSION_USE_SPATIAL_GATE:-false}"
export FUSION_CONV_DEPTH="${FUSION_CONV_DEPTH:-1}"
export FUSION_PRE_MAMBA_CONV_DEPTH="${FUSION_PRE_MAMBA_CONV_DEPTH:-1}"
export FUSION_POST_MAMBA_CONV_DEPTH="${FUSION_POST_MAMBA_CONV_DEPTH:-1}"

export PREDICTION_HEAD_CONV_DEPTH="${PREDICTION_HEAD_CONV_DEPTH:-3}"
export PREDICTION_HEAD_CONV_TYPE="${PREDICTION_HEAD_CONV_TYPE:-standard}"

export FINAL_SKIP_REFINER_DEPTH="${FINAL_SKIP_REFINER_DEPTH:-3}"
export FINAL_SKIP_REFINER_CONV_TYPE="${FINAL_SKIP_REFINER_CONV_TYPE:-standard}"
export FINAL_SKIP_REFINER_USE_CHANNEL_GATE="${FINAL_SKIP_REFINER_USE_CHANNEL_GATE:-true}"
export FINAL_SKIP_REFINER_USE_SPATIAL_GATE="${FINAL_SKIP_REFINER_USE_SPATIAL_GATE:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_facehq1024_hierarchy_hybrid_local_v2_2gpu.sh" "$@"
