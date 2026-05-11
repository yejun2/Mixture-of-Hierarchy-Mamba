#!/usr/bin/env bash
set -euo pipefail

# Scratch-only dynamic bottleneck preset for 32x32 latents.
#
# A sparse stage router selects exactly two stages from 32, 16, 8.
# The effective per-sample bottleneck is the lowest-resolution selected stage:
#   selected={32,16} -> bottleneck=16
#   selected={32,8}  -> bottleneck=8
#   selected={16,8}  -> bottleneck=8

die() {
  echo "$*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/train_facehq1024_hierarchy_hybrid_local_v2_2gpu.sh"
[[ -f "${BASE_SCRIPT}" ]] || die "Missing base launcher: ${BASE_SCRIPT}"

if [[ -n "${CKPT:-}" ]]; then
  die "This dynamic bottleneck preset is scratch-only. Do not set CKPT."
fi
for arg in "$@"; do
  case "${arg}" in
    --ckpt|--ckpt=*)
      die "This dynamic bottleneck preset is scratch-only and does not accept --ckpt."
      ;;
  esac
done

export MODEL_CONFIG="${MODEL_CONFIG:-hierarchy_hybrid_dynamic_bottleneck_v2}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
export VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_GPU}}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
export IMAGE_SIZE="${IMAGE_SIZE:-256}"
export MODEL_IMG_DIM="${MODEL_IMG_DIM:-32}"
export DYNAMIC_STAGE_CANDIDATES="${DYNAMIC_STAGE_CANDIDATES:-32,16,8}"

[[ "${MODEL_IMG_DIM}" == "32" ]] || die "This execution script is for 32x32 latents; MODEL_IMG_DIM must be 32, got ${MODEL_IMG_DIM}."
[[ "${DYNAMIC_STAGE_CANDIDATES}" == "32,16,8" ]] || die "This execution script routes exactly stages 32,16,8; got ${DYNAMIC_STAGE_CANDIDATES}."

IFS=',' read -r -a _dynamic_candidates <<< "${DYNAMIC_STAGE_CANDIDATES}"
[[ "${#_dynamic_candidates[@]}" -eq 3 ]] || die "DYNAMIC_STAGE_CANDIDATES must contain exactly three comma-separated resolutions."

candidate_sum=0
candidate_min=
candidate_max=
for candidate in "${_dynamic_candidates[@]}"; do
  candidate="${candidate//[[:space:]]/}"
  [[ "${candidate}" =~ ^[0-9]+$ ]] || die "Stage candidate must be a positive integer: ${candidate}"
  (( candidate > 0 )) || die "Stage candidate must be positive: ${candidate}"
  candidate_sum=$((candidate_sum + candidate))
  if [[ -z "${candidate_min}" || "${candidate}" -lt "${candidate_min}" ]]; then
    candidate_min="${candidate}"
  fi
  if [[ -z "${candidate_max}" || "${candidate}" -gt "${candidate_max}" ]]; then
    candidate_max="${candidate}"
  fi
done
candidate_mid=$((candidate_sum - candidate_min - candidate_max))
[[ "${candidate_mid}" -gt "${candidate_min}" && "${candidate_mid}" -lt "${candidate_max}" ]] || \
  die "DYNAMIC_STAGE_CANDIDATES must contain three unique resolutions: ${DYNAMIC_STAGE_CANDIDATES}"

NORMALIZED_STAGE_CANDIDATES="${candidate_max},${candidate_mid},${candidate_min}"
export DYNAMIC_STAGE_CANDIDATES="${NORMALIZED_STAGE_CANDIDATES}"

export FUSION_ANCHOR_RESOLUTION="${FUSION_ANCHOR_RESOLUTION:-${candidate_min}}"
[[ "${FUSION_ANCHOR_RESOLUTION}" == "${candidate_min}" ]] || \
  die "FUSION_ANCHOR_RESOLUTION must equal min(DYNAMIC_STAGE_CANDIDATES)=${candidate_min}, got ${FUSION_ANCHOR_RESOLUTION}"

export DECODER_ANCHOR_RESOLUTION="${DECODER_ANCHOR_RESOLUTION:-${FUSION_ANCHOR_RESOLUTION}}"
[[ "${DECODER_ANCHOR_RESOLUTION}" == "${FUSION_ANCHOR_RESOLUTION}" ]] || \
  die "DECODER_ANCHOR_RESOLUTION must match FUSION_ANCHOR_RESOLUTION for dynamic bottleneck routing."

export FUSION_SELECTED_STAGES="${FUSION_SELECTED_STAGES:-${DYNAMIC_STAGE_CANDIDATES}}"
[[ "${FUSION_SELECTED_STAGES}" == "${DYNAMIC_STAGE_CANDIDATES}" ]] || \
  die "FUSION_SELECTED_STAGES must match DYNAMIC_STAGE_CANDIDATES for this preset."

export USE_FACTORIZED_TOP4_ROUTER="${USE_FACTORIZED_TOP4_ROUTER:-true}"
[[ "${USE_FACTORIZED_TOP4_ROUTER}" == "true" ]] || \
  die "USE_FACTORIZED_TOP4_ROUTER must be true for sparse stage selection."
export ROUTED_STAGE_RESOLUTIONS="${ROUTED_STAGE_RESOLUTIONS:-${DYNAMIC_STAGE_CANDIDATES}}"
[[ "${ROUTED_STAGE_RESOLUTIONS}" == "${DYNAMIC_STAGE_CANDIDATES}" ]] || \
  die "ROUTED_STAGE_RESOLUTIONS must match DYNAMIC_STAGE_CANDIDATES for exact two-of-three routing."
export ROUTED_STAGE_COUNT="${ROUTED_STAGE_COUNT:-3}"
[[ "${ROUTED_STAGE_COUNT}" == "3" ]] || die "ROUTED_STAGE_COUNT must be 3."
export INCLUDE_ANCHOR_IN_STAGE_ROUTER="${INCLUDE_ANCHOR_IN_STAGE_ROUTER:-true}"
[[ "${INCLUDE_ANCHOR_IN_STAGE_ROUTER}" == "true" ]] || \
  die "INCLUDE_ANCHOR_IN_STAGE_ROUTER must be true so the lowest candidate can be selected."
export STAGE_ROUTER_TOP_K="${STAGE_ROUTER_TOP_K:-2}"
[[ "${STAGE_ROUTER_TOP_K}" == "2" ]] || die "STAGE_ROUTER_TOP_K must be 2."
export STAGE_ROUTER_WEIGHT_MODE="${STAGE_ROUTER_WEIGHT_MODE:-equal_selection}"
[[ "${STAGE_ROUTER_WEIGHT_MODE}" == "equal_selection" ]] || \
  die "STAGE_ROUTER_WEIGHT_MODE must be equal_selection."
export STAGE_ROUTER_WEIGHT_FLOOR="${STAGE_ROUTER_WEIGHT_FLOOR:-0.0}"

export USE_AUX_4X4_CONTEXT="${USE_AUX_4X4_CONTEXT:-false}"
[[ "${USE_AUX_4X4_CONTEXT}" == "false" ]] || \
  die "USE_AUX_4X4_CONTEXT is not supported by this dynamic bottleneck preset."

export FUSION_STAGE_DIM_OVERRIDES="${FUSION_STAGE_DIM_OVERRIDES:-${candidate_mid}:448,${candidate_max}:512}"
export FUSION_STAGE_DEPTH_OVERRIDES="${FUSION_STAGE_DEPTH_OVERRIDES:-${candidate_mid}:4,${candidate_max}:6}"
export FUSION_CHANNEL_GATE_STAGES="${FUSION_CHANNEL_GATE_STAGES:-${candidate_mid},${candidate_max}}"

export USE_CHECKPOINT="${USE_CHECKPOINT:-true}"
export NOTE="${NOTE:-dynamic_bottleneck_sparse2_stage3_scratch}"
export TIMESTAMP="${TIMESTAMP:-dynamic_bottleneck_sparse2_stage3_scratch}"
export WANDB_TAGS="${WANDB_TAGS:-dynamic_bottleneck,sparse2,stage3,scratch}"

cat <<EOF
Launching dynamic bottleneck sparse2 scratch preset
  stages: ${DYNAMIC_STAGE_CANDIDATES}
  cuda: ${CUDA_DEVICES} (${NUM_PROCESSES} processes)
  batch/gpu: ${BATCH_SIZE_PER_GPU}
  workers/gpu: train=${NUM_WORKERS}, val=${VAL_NUM_WORKERS}
  router: select 2 / 3
  dynamic bottleneck: min(selected), minimum configured anchor=${FUSION_ANCHOR_RESOLUTION}
  model config: ${MODEL_CONFIG}
EOF

exec bash "${BASE_SCRIPT}" "$@"
