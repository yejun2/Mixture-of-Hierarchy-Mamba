#!/usr/bin/env bash
set -euo pipefail

# Adaptive-k dynamic integrated controller preset for CelebA-HQ 256px / 32x32 latents.
#
# Design intent:
#   - Route 32x32, 16x16, and 8x8 together so the controller can decide both
#     which stages are useful and how many stages k to keep.
#   - Use ReLU-gated sparse stage routing: a stage is selected when its current
#     sigmoid probability clears the probability threshold. This avoids ranking
#     stages against each other, so 32x32 can stay active whenever its own score
#     has real signal without being forced on every sample.
#   - Keep at least two stages as a permissive fallback, while still allowing all
#     three stages when their independent ReLU scores are high.
#   - Feed the selected stage prior into the integrated compression and encoder
#     depth routers so their branch choices are conditioned on the active SSR path.
#   - Keep SSR stage selection independent from the scale-prior router by
#     disabling scale-prior context for the stage scorer in this preset.
#   - Use dynamic bottleneck decoding. Among the selected stages, the lowest
#     selected resolution becomes the decoder start resolution.
#
# With this launcher, router/stage_selected_count_* in W&B is the selected k.
# dynamic_bottleneck/resolution_*_fraction shows which selected low-resolution
# stage actually became the bottleneck.

export MODEL_CONFIG="${MODEL_CONFIG:-hierarchy_hybrid_local_v2}"
export SCAN_TYPE="${SCAN_TYPE:-v2}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export MASTER_PORT="${MASTER_PORT:-0}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

export DATA_CONFIG="${DATA_CONFIG:-facehq_1024}"
export DATA_TAR_BASE="${DATA_TAR_BASE:-/SSD4/yjjung/datasets/celeba_hq_256_hf_shard}"
export IMAGE_SIZE="${IMAGE_SIZE:-256}"
export MODEL_IMG_DIM="${MODEL_IMG_DIM:-32}"

export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
export VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-16}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-7}"
export VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
export SAMPLE_FID_EVERY="${SAMPLE_FID_EVERY:-10000}"
export SAMPLE_FID_START_STEP="${SAMPLE_FID_START_STEP:-70000}"
export SAMPLE_FID_BS="${SAMPLE_FID_BS:-4}"
export USE_CHECKPOINT="${USE_CHECKPOINT:-false}"
export CKPT_EVERY="${CKPT_EVERY:-10000}"

export EMBED_DIM="${EMBED_DIM:-256}"
export HIERARCHY_STAGE_DEPTH="${HIERARCHY_STAGE_DEPTH:-3}"
export HIGHRES_STAGE_DEPTH="${HIGHRES_STAGE_DEPTH:-4}"
export BOTTLENECK_STAGE_DEPTH="${BOTTLENECK_STAGE_DEPTH:-4}"

export DOWNSAMPLE_USE_PREMIX="${DOWNSAMPLE_USE_PREMIX:-true}"
export DOWNSAMPLE_PREMIX_DEPTH="${DOWNSAMPLE_PREMIX_DEPTH:-1}"
export DOWNSAMPLE_CONV_TYPE="${DOWNSAMPLE_CONV_TYPE:-standard}"
export HIGHRES_LOCAL_CONV_DEPTH="${HIGHRES_LOCAL_CONV_DEPTH:-3}"
export HIGHRES_LOCAL_CONV_TYPE="${HIGHRES_LOCAL_CONV_TYPE:-standard}"

export FUSION_ANCHOR_RESOLUTION="${FUSION_ANCHOR_RESOLUTION:-8}"
export DECODER_ANCHOR_RESOLUTION="${DECODER_ANCHOR_RESOLUTION:-8}"
export FUSION_SELECTED_STAGES="${FUSION_SELECTED_STAGES:-32,16}"
export FUSION_MODE="${FUSION_MODE:-gated_sum}"
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

# Route every decoder candidate; k is selected adaptively instead of fixed.
export ROUTED_STAGE_RESOLUTIONS="${ROUTED_STAGE_RESOLUTIONS:-32,16,8}"
export ROUTED_STAGE_COUNT="${ROUTED_STAGE_COUNT:-3}"
export INCLUDE_ANCHOR_IN_STAGE_ROUTER="${INCLUDE_ANCHOR_IN_STAGE_ROUTER:-true}"
export STAGE_ROUTER_TOP_K="${STAGE_ROUTER_TOP_K:-3}"
export STAGE_ROUTER_WEIGHT_FLOOR="${STAGE_ROUTER_WEIGHT_FLOOR:-0.0}"
export STAGE_ROUTER_WEIGHT_MODE="${STAGE_ROUTER_WEIGHT_MODE:-equal_selection}"

export USE_FACTORIZED_TOP4_ROUTER="${USE_FACTORIZED_TOP4_ROUTER:-false}"
export USE_MAMBA_SIZE_ROUTER="${USE_MAMBA_SIZE_ROUTER:-false}"

export USE_COMPRESSION_ROUTER="${USE_COMPRESSION_ROUTER:-true}"
export COMPRESSION_ROUTER_STAGES="${COMPRESSION_ROUTER_STAGES:-16,8}"

export USE_INTEGRATED_ROUTER_CONTROLLER="${USE_INTEGRATED_ROUTER_CONTROLLER:-true}"
export INTEGRATED_CONTROLLER_STAGE_TOP_K="${INTEGRATED_CONTROLLER_STAGE_TOP_K:-3}"
export INTEGRATED_CONTROLLER_HIDDEN_DIM="${INTEGRATED_CONTROLLER_HIDDEN_DIM:-auto}"
export INTEGRATED_CONTROLLER_USE_CHANNEL_GATE="${INTEGRATED_CONTROLLER_USE_CHANNEL_GATE:-false}"
export INTEGRATED_CONTROLLER_CHANNEL_GATE_SCALE="${INTEGRATED_CONTROLLER_CHANNEL_GATE_SCALE:-0.1}"
export INTEGRATED_CONTROLLER_STAGE_SELECT_MODE="${INTEGRATED_CONTROLLER_STAGE_SELECT_MODE:-relu}"
export INTEGRATED_CONTROLLER_STAGE_SELECT_THRESHOLD="${INTEGRATED_CONTROLLER_STAGE_SELECT_THRESHOLD:-0.03}"
export INTEGRATED_CONTROLLER_STAGE_SELECT_THRESHOLD_MARGIN="${INTEGRATED_CONTROLLER_STAGE_SELECT_THRESHOLD_MARGIN:-0.0}"
export INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_STEPS="${INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_STEPS:-0}"
export INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_MIN_SELECTED="${INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_MIN_SELECTED:-0}"
export INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_THRESHOLD_MARGIN="${INTEGRATED_CONTROLLER_STAGE_SELECT_WARMUP_THRESHOLD_MARGIN:-0.0}"
export INTEGRATED_CONTROLLER_STAGE_MIN_SELECTED="${INTEGRATED_CONTROLLER_STAGE_MIN_SELECTED:-2}"
export INTEGRATED_CONTROLLER_STAGE_BALANCE_MODE="${INTEGRATED_CONTROLLER_STAGE_BALANCE_MODE:-none}"
export INTEGRATED_CONTROLLER_STAGE_USE_SCALE_PRIOR_CONTEXT="${INTEGRATED_CONTROLLER_STAGE_USE_SCALE_PRIOR_CONTEXT:-false}"

# Start from the lowest selected stage: 32 if only 32 is selected, 16 if 16 is
# the lowest selected stage, and 8 if 8 is selected.
export USE_DYNAMIC_BOTTLENECK="${USE_DYNAMIC_BOTTLENECK:-true}"
export DYNAMIC_BOTTLENECK_CANDIDATE_STAGES="${DYNAMIC_BOTTLENECK_CANDIDATE_STAGES:-32,16,8}"

export NOTE="${NOTE:-adaptive_k_dynamic_integrated_controller_hierarchy_celebahq256}"
export TIMESTAMP="${TIMESTAMP:-adaptive_k_dynamic_integrated_controller_hierarchy_celebahq256}"
export WANDB_NAME="${WANDB_NAME:-${NOTE}}"
export WANDB_TAGS="${WANDB_TAGS:-adaptive_k_dynamic_integrated_controller,dynamic_bottleneck,relu_stage_router,min2_stage_router,stage_selection_conditioned_depth_compress,independent_stage_router,compression_router,downsample_premix,no_mamba_size_router,celebahq256,bf16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../train_facehq1024_hierarchy_hybrid_local_v2_2gpu.sh" "$@"
