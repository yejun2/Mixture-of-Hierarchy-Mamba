#!/usr/bin/env bash
set -euo pipefail

cd /home/yejun/projects/hierarchical_zigma_v1/zigma

python train_acc.py \
  data=facehq_1024 \
  model=hierarchical_sweep2_b1_pe2 \
  is_latent=true \
  use_latent=true \
  data.tar_base=/home/yejun/projects/hierarchical_zigma_v1/datasets/ffhq_shard \
  data.train.shards='*.tar' \
  data.validation.shards='*.tar' \
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
