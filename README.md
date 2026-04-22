# Mixture-of-Hierarchy-Mamba (In preparation ICLR 2027)

Research code for hierarchical and multi-scale Mamba diffusion experiments built on
top of [CompVis/ZigMa](https://github.com/CompVis/zigma).

This fork keeps the original ZigMa training and sampling stack, then adds local
hierarchical context processors, hybrid decoder/fusion heads, and experiment
scripts for latent image diffusion.

## What Is Added

- `model_hierarchy_hybrid.py`: hierarchical Mamba encoder with decoder-side
  multi-scale fusion and optional local skip refinement.
- `model_hierarchy_only.py`: hierarchy-only local Mamba variant.
- `model_zigma.py`: extended ZigMa backbone with hierarchical context support.
- `config/model/hierarchy_*.yaml`: Hydra model configs for hierarchy-only,
  hierarchical-context, and hybrid local runs.
- `scripts/train_*hierarchy*.sh`: reproducible launch scripts for the main
  experiments and ablations.
- `scripts/diagnose_hierarchy_pipeline.py`: checkpoint inspection utility for
  stage/fusion tensor statistics.
- `scripts/analyze_fusion_ablation.py`: fixed-noise fusion ablation analysis.

## Repository Layout

```text
.
|-- config/                  # Hydra configs for data, model, optimizer, sampling
|-- datasets/                # WebDataset dataloaders
|-- dis_causal_conv1d/       # Local causal-conv1d dependency
|-- dis_mamba/               # Local Mamba/SSM dependency
|-- model_zigma.py           # Original ZigMa backbone plus hierarchy extensions
|-- model_hierarchy_hybrid.py
|-- model_hierarchy_only.py
|-- scripts/                 # Training, diagnosis, and analysis scripts
|-- train_acc.py             # Main Accelerate training entrypoint
`-- sample_acc.py            # Sampling/evaluation entrypoint
```

## Environment

The original project was developed around CUDA 11.8, Python 3.11, and PyTorch
2.2. Local experiments in this fork use an Accelerate + Hydra workflow.

```bash
conda create -n zigma python=3.11
conda activate zigma

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

pip install torchdiffeq matplotlib h5py timm diffusers accelerate loguru blobfile
pip install ml_collections wandb hydra-core opencv-python torch-fidelity webdataset
pip install einops pytorch_lightning torchmetrics moviepy imageio scikit-learn
pip install transformers==4.36.2 numpy-hilbert-curve av

cd dis_causal_conv1d && pip install -e . && cd ..
cd dis_mamba && pip install -e . && cd ..
```

If Mamba or causal-conv1d compilation fails, check the CUDA/PyTorch/GCC versions
first. These extensions are the most environment-sensitive part of the setup.

## Weights & Biases

Create `config/wandb/default.yaml` locally:

```yaml
key: YOUR_WANDB_KEY
entity: YOUR_WANDB_ENTITY
project: YOUR_PROJECT_NAME
```

Do not commit a real API key. The `.gitignore` ignores `config/wandb/` for new
local secrets, but a previously tracked placeholder file may still exist in the
repository history.

To disable W&B for a run:

```bash
python train_acc.py use_wandb=false ...
```

## Data

Training expects WebDataset shards. The latent image runs use shard samples with
keys such as:

- `image.png` or `image.jpg`
- `latent.npy` for FaceHQ-style latent shards
- `img_feature256.npy` and `caption_clip_feature.npy` for CelebA-MM-style
  latent/text shards

The training scripts expose dataset locations through environment variables:

```bash
DATA_TAR_BASE=/path/to/webdataset/shards
TRAIN_SHARDS='train-{000000..000029}.tar'
VAL_SHARDS='test-{000000..000004}.tar'
```

Outputs are written under `outputs/<hydra-job-name>/<timestamp>/`, with
checkpoints saved in the run's `checkpoints/` directory.

## Quick Start

Main hybrid local experiment with zigzag-8 scanning and native skip refinement:

```bash
DATA_TAR_BASE=/path/to/webdataset/shards \
CUDA_DEVICES=0,1 \
bash scripts/train_facehq1024_hierarchy_hybrid_local_z8_native_skip_2gpu.sh
```

Base hybrid local v2 run:

```bash
DATA_TAR_BASE=/path/to/webdataset/shards \
CUDA_DEVICES=0,1 \
bash scripts/train_facehq1024_hierarchy_hybrid_local_v2_2gpu.sh
```

Hierarchy-free ZigMa backbone baseline on the same latent shards:

```bash
DATA_TAR_BASE=/path/to/webdataset/shards \
CUDA_DEVICES=0,1 \
bash scripts/train_celebahq256_zigma_backbone_2gpu.sh
```

Resume from an experiment directory or a specific checkpoint:

```bash
bash scripts/train_facehq1024_hierarchy_hybrid_local_v2_2gpu.sh \
  --ckpt outputs/<run-name>/<timestamp>

bash scripts/train_facehq1024_hierarchy_hybrid_local_v2_2gpu.sh \
  --ckpt outputs/<run-name>/<timestamp>/checkpoints/0400000.pt
```

Common overrides:

```bash
CUDA_DEVICES=2,3
NUM_PROCESSES=2
BATCH_SIZE_PER_GPU=16
GRAD_ACCUM_STEPS=2
TRAIN_STEPS=600000
MASTER_PORT=8872
USE_CHECKPOINT=true
```

## Model Configs

| Config | Target | Notes |
| --- | --- | --- |
| `hierarchy_hybrid_local_v2` | `model_hierarchy_hybrid.HierarchicalMambaHybrid` | Main hybrid encoder/decoder with multi-scale fusion |
| `hierarchy_hybrid_local_v2_m` | `model_hierarchy_hybrid.HierarchicalMambaHybrid` | Medium-size hybrid variant |
| `hierarchy_only_local_v1` | `model_hierarchy_only.HierarchicalMambaLocal` | Local hierarchy-only model |
| `hierarchical_sweep2_b1_pe2` | `model_zigma.ZigMa` | ZigMa backbone with hierarchical context |
| `zigzag8_b1_pe2` | `model_zigma.ZigMa` | Original-style ZigMa baseline |

The main hybrid scripts train latent diffusion models with:

- 4 input/output channels
- `img_dim=32`
- patch size 1
- hierarchy window size 2
- multi-scale fusion stages such as `32,16,8`

## Sampling

Sampling uses `sample_acc.py` and a checkpoint path:

```bash
CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch \
  --num_processes 1 \
  sample_acc.py \
  model=hierarchy_hybrid_local_v2 \
  data=facehq_1024 \
  is_latent=true \
  use_latent=true \
  sample_mode=ODE \
  likelihood=false \
  ckpt=/path/to/checkpoint.pt \
  use_wandb=false
```

Generated samples are saved under `samples/` unless `sample_dir` is overridden.

## Analysis Utilities

Inspect hierarchy stage statistics from a checkpoint:

```bash
python scripts/diagnose_hierarchy_pipeline.py \
  --ckpt /path/to/checkpoint.pt \
  --output-json reports/diagnose.json
```

Run fixed-noise multi-scale fusion ablations:

```bash
python scripts/analyze_fusion_ablation.py \
  --ckpt /path/to/checkpoint.pt \
  --num-batches 4 \
  --grad-batches 1 \
  --output-json reports/fusion_ablation.json
```

## Notes For This Fork

- Several launch scripts contain local defaults such as `/SSD4/vipnu/...` and
  `zigma_server_cuda124`. Override environment variables or edit the script
  headers for another machine.
- `outputs/`, `samples/`, `checkpoints/`, logs, reports, and W&B runs are
  ignored by git.
- The upstream ZigMa repository is preserved as `upstream` in the recommended
  git remote layout.

## Upstream Attribution

This project is based on the official implementation of:

```bibtex
@InProceedings{hu2024zigma,
  title={ZigMa: A DiT-style Zigzag Mamba Diffusion Model},
  author={Vincent Tao Hu and Stefan Andreas Baumann and Ming Gui and Olga Grebenkova and Pingchuan Ma and Johannes Schusterbauer and Bjorn Ommer},
  booktitle={ECCV},
  year={2024}
}
```

Original project: <https://github.com/CompVis/zigma>

## License

This repository follows the upstream Apache-2.0 license. See
[`LICENSE.txt`](LICENSE.txt).
