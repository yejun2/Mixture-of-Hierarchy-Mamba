#!/usr/bin/env bash
set -euo pipefail

cd /SSD4/vipnu/hierarchical_zigma_v1/zigma

set +u
source /SSD4/vipnu/anaconda3/etc/profile.d/conda.sh
conda activate zigma_server_cuda124
set -u

MASTER_PORT="${MASTER_PORT:-8868}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
VAL_BATCH_SIZE_PER_GPU="${VAL_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_GPU}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
SAMPLE_FID_BS="${SAMPLE_FID_BS:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

CUDA_VISIBLE_DEVICES="0,1" python -m accelerate.commands.launch \
  --num_processes 2 \
  --num_machines 1 \
  --multi_gpu \
  --mixed_precision fp16 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${MASTER_PORT}" \
  train_acc.py \
  data=facehq_1024 \
  model=hierarchical_sweep2_b1_pe2 \
  gradient_accumulation_steps="${GRAD_ACCUM_STEPS}" \
  is_latent=true \
  use_latent=true \
  data.tar_base=/SSD4/vipnu/datasets/ffhq_shard \
  data.train.shards="\"train-{000000..000060}.tar\"" \
  data.validation.shards="\"train-{000000..000060}.tar\"" \
  data.batch_size="${BATCH_SIZE_PER_GPU}" \
  data.val_batch_size="${VAL_BATCH_SIZE_PER_GPU}" \
  data.num_workers="${NUM_WORKERS}" \
  data.val_num_workers="${VAL_NUM_WORKERS}" \
  data.sample_fid_bs="${SAMPLE_FID_BS}" \
  model.params.img_dim=128 \
  model.params.in_channels=4 \
  model.params.patch_size=1 \
  model.params.use_pe=2 \
  model.params.hierarchical_output_mode=prediction \
  model.params.hierarchy_window_size=2 \
  model.params.hierarchy_stride=1 \
  model.params.context_compress_type=last \
  model.params.scan_type=zigzagN8 \
  model.params.hierarchy_stage_depth=4
