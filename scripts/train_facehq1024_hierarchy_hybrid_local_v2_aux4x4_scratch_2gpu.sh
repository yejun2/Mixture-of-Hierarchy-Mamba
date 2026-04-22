#!/usr/bin/env bash
set -euo pipefail

# Aux-4x4 from-scratch preset for the FID 13.46 native-skip configuration.
# This script never consumes a CKPT environment variable; pass --ckpt explicitly
# only when you intentionally want to resume an interrupted scratch run.

unset CKPT

export MASTER_PORT="${MASTER_PORT:-8875}"
export CUDA_DEVICES="${CUDA_DEVICES:-2,3}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-32}"
export VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_GPU}}"
export SAMPLE_FID_BS="${SAMPLE_FID_BS:-8}"
export TRAIN_STEPS="${TRAIN_STEPS:-600000}"
export OPTIM_LR="${OPTIM_LR:-}"

export DATA_CONFIG="${DATA_CONFIG:-facehq_1024}"
export DATA_TAR_BASE="${DATA_TAR_BASE:-/SSD4/vipnu/datasets/celeba_hq_256_hf_shard/}"
export IMAGE_SIZE="${IMAGE_SIZE:-256}"
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

export USE_AUX_4X4_CONTEXT="${USE_AUX_4X4_CONTEXT:-true}"
export AUX_4X4_INJECT_TARGET="${AUX_4X4_INJECT_TARGET:-final_map}"
export AUX_4X4_CONTEXT_RESOLUTION="${AUX_4X4_CONTEXT_RESOLUTION:-4}"
export AUX_4X4_TARGET_RESOLUTION="${AUX_4X4_TARGET_RESOLUTION:-auto}"
export AUX_4X4_PREPOOL_DEPTH="${AUX_4X4_PREPOOL_DEPTH:-1}"
export AUX_4X4_CONTEXT_DEPTH="${AUX_4X4_CONTEXT_DEPTH:-1}"
export AUX_4X4_FUSION_DEPTH="${AUX_4X4_FUSION_DEPTH:-2}"
export AUX_4X4_CONV_TYPE="${AUX_4X4_CONV_TYPE:-standard}"
export AUX_4X4_UPSAMPLE_MODE="${AUX_4X4_UPSAMPLE_MODE:-bilinear}"
export AUX_4X4_USE_TIMESTEP_GATE="${AUX_4X4_USE_TIMESTEP_GATE:-true}"
export AUX_4X4_RESIDUAL_SCALE_MULTIPLIER="${AUX_4X4_RESIDUAL_SCALE_MULTIPLIER:-0.5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_facehq1024_hierarchy_hybrid_local_v2_native_skip_2gpu.sh" "$@"
