#!/usr/bin/env bash
set -euo pipefail

# No-code Experiment 1:
# - stop compression at 8x8 instead of 4x4
# - remove anchor-builder by matching fusion/decoder anchors
# - decode with native 16/32 skips only
# - prefer gated residual fusion over concat
# - strengthen the final 32x32 detail reinjection path

export MODEL_CONFIG="${MODEL_CONFIG:-hierarchy_hybrid_local_v2}"
export SCAN_TYPE="${SCAN_TYPE:-v2}"

export FUSION_ANCHOR_RESOLUTION="${FUSION_ANCHOR_RESOLUTION:-8}"
export DECODER_ANCHOR_RESOLUTION="${DECODER_ANCHOR_RESOLUTION:-8}"
export FUSION_SELECTED_STAGES="${FUSION_SELECTED_STAGES:-32,16}"
export FUSION_STAGE_DIM="${FUSION_STAGE_DIM:-256}"
export FUSION_STAGE_DIM_OVERRIDES="${FUSION_STAGE_DIM_OVERRIDES:-16:320,32:384}"

export FUSION_MODE="${FUSION_MODE:-gated_sum}"
export FUSION_CHANNEL_GATE_STAGES="${FUSION_CHANNEL_GATE_STAGES:-16,32}"
export FUSION_USE_SPATIAL_GATE="${FUSION_USE_SPATIAL_GATE:-false}"

export FINAL_SKIP_REFINER_DEPTH="${FINAL_SKIP_REFINER_DEPTH:-2}"
export FINAL_SKIP_REFINER_CONV_TYPE="${FINAL_SKIP_REFINER_CONV_TYPE:-standard}"
export FINAL_SKIP_REFINER_USE_CHANNEL_GATE="${FINAL_SKIP_REFINER_USE_CHANNEL_GATE:-true}"
export FINAL_SKIP_REFINER_USE_SPATIAL_GATE="${FINAL_SKIP_REFINER_USE_SPATIAL_GATE:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_facehq1024_hierarchy_hybrid_local_v2_2gpu.sh" "$@"
