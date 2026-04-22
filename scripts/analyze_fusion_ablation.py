#!/usr/bin/env python
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.wds_dataloader import WebDataModuleFromConfig
from transport import ModelType, WeightType, create_transport
import transport.path as transport_path
from transport.utils import mean_flat
from utils.train_utils import get_model


def has_text(args):
    return "celebamm" in args.data.name or "coco" in args.data.name


def is_video(args):
    if hasattr(args.model.params, "video_frames"):
        if args.model.params.video_frames > 0:
            return True
        if args.model.params.video_frames == 0:
            return False
        raise ValueError("video_frames must be >= 0")
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze multiscale fusion stage usage with fixed-noise ablations."
    )
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--weights",
        choices=("ema", "model"),
        default="ema",
        help="Which checkpoint weights to analyze",
    )
    parser.add_argument(
        "--split",
        choices=("auto", "train", "val"),
        default="auto",
        help="Data split to use for analysis",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=4,
        help="Number of batches used for loss and timestep-bin statistics",
    )
    parser.add_argument(
        "--grad-batches",
        type=int,
        default=1,
        help="Number of batches used for gradient norm probing",
    )
    parser.add_argument(
        "--t-bins",
        type=int,
        default=6,
        help="Number of timestep bins for baseline gate statistics",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for analysis, e.g. cuda or cuda:0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for batch-level transport sampling",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for analysis batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for dataloader workers",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the final JSON result",
    )
    return parser.parse_args()


def prepare_loaded_args(loaded_args, cli_args):
    if cli_args.batch_size is not None:
        loaded_args.data.batch_size = cli_args.batch_size
        if hasattr(loaded_args.data, "val_batch_size"):
            loaded_args.data.val_batch_size = cli_args.batch_size
    if cli_args.num_workers is not None:
        loaded_args.data.num_workers = cli_args.num_workers
        if hasattr(loaded_args.data, "val_num_workers"):
            loaded_args.data.val_num_workers = cli_args.num_workers
    return loaded_args


def choose_loader(datamod, loaded_args, split):
    include_images = not (loaded_args.use_latent and "facehq" in str(loaded_args.data.name))
    if split == "val" or (
        split == "auto" and getattr(loaded_args.data, "validation", None) is not None
    ):
        if getattr(loaded_args.data, "validation", None) is not None:
            return "val", datamod.val_dataloader(include_images=include_images)
    return "train", datamod.train_dataloader(include_images=include_images)


def extract_batch(data, loaded_args, device):
    if loaded_args.use_latent:
        if has_text(loaded_args):
            cap_feats = data["caption_feature"].to(device)
            x = data["img_feature"].to(device)
            y = cap_feats[:, 0]
        elif "facehq" in str(loaded_args.data.name):
            x = data["latent"].to(device)
            y = None
        elif "ucf101" in str(loaded_args.data.name):
            x = data["frame_feature256"].to(device)
            y = data["cls_id"].to(device)
        elif "celebav" in str(loaded_args.data.name):
            frame_count = loaded_args.model.params.video_frames
            x = data["frame_feature256"][:, :frame_count].to(device)
            y = None
        else:
            raise NotImplementedError(
                f"Unsupported latent dataset for analysis: {loaded_args.data.name}"
            )
    else:
        x = data["image"].to(device)
        y = None

    if loaded_args.is_latent:
        if loaded_args.use_latent:
            x = x * 0.18215
        else:
            raise NotImplementedError(
                "analyze_fusion_ablation.py currently expects use_latent=true when is_latent=true"
            )
    return x, y


def compute_loss_from_fixed_sample(transport, model_output, xt, x0, ut, t):
    if transport.model_type == ModelType.VELOCITY:
        return mean_flat((model_output - ut) ** 2)

    _, drift_var = transport.path_sampler.compute_drift(xt, t)
    sigma_t, _ = transport.path_sampler.compute_sigma_t(
        transport_path.expand_t_like_x(t, xt)
    )
    if transport.loss_type == WeightType.VELOCITY:
        weight = (drift_var / sigma_t) ** 2
    elif transport.loss_type == WeightType.LIKELIHOOD:
        weight = drift_var / (sigma_t**2)
    elif transport.loss_type == WeightType.NONE:
        weight = 1
    else:
        raise NotImplementedError(f"Unsupported loss type: {transport.loss_type}")

    if transport.model_type == ModelType.NOISE:
        return mean_flat(weight * ((model_output - x0) ** 2))
    if transport.model_type == ModelType.SCORE:
        return mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
    raise NotImplementedError(f"Unsupported model type: {transport.model_type}")


def sample_transport_inputs(transport, x1):
    t, x0, x1 = transport.sample(x1)
    t, xt, ut = transport.path_sampler.plan(t, x0, x1)
    return t, x0, xt, ut


def summarize_experiment_store(store):
    result = {}
    for name, stats in store.items():
        count = max(stats["count"], 1)
        result[name] = {
            "mean_loss": stats["loss_sum"] / count,
            "loss_delta_vs_baseline": None,
        }
        for key in ("stage_gate_means", "stage_raw_gate_means", "stage_feature_norms"):
            if stats[key]:
                result[name][key] = {
                    resolution: value / count for resolution, value in sorted(stats[key].items())
                }
    baseline_loss = result.get("baseline", {}).get("mean_loss")
    if baseline_loss is not None:
        for name in result:
            result[name]["loss_delta_vs_baseline"] = result[name]["mean_loss"] - baseline_loss
    return result


def summarize_t_bins(bin_store, stage_resolutions, t_bins):
    result = []
    for bin_idx in range(t_bins):
        bucket = bin_store[bin_idx]
        count = bucket["count"]
        entry = {
            "bin_idx": bin_idx,
            "t_start": bin_idx / t_bins,
            "t_end": (bin_idx + 1) / t_bins,
            "count": count,
        }
        if count > 0:
            for prefix in (
                "stage_gate_means",
                "stage_raw_gate_means",
                "stage_projected_norms",
                "stage_aligned_norms",
                "stage_feature_norms",
            ):
                entry[prefix] = {
                    resolution: bucket[prefix][resolution] / count
                    for resolution in stage_resolutions
                }
        result.append(entry)
    return result


def build_experiments(stage_resolutions):
    experiments = [{"name": "baseline", "overrides": {}}]
    for resolution in stage_resolutions:
        experiments.append(
            {
                "name": f"drop_stage_{resolution}",
                "overrides": {"disabled_stage_resolutions": [resolution]},
            }
        )
    for resolution in stage_resolutions:
        experiments.append(
            {
                "name": f"keep_only_stage_{resolution}",
                "overrides": {
                    "disabled_stage_resolutions": [
                        other for other in stage_resolutions if other != resolution
                    ]
                },
            }
        )
    for resolution in stage_resolutions:
        experiments.append(
            {
                "name": f"force_stage_{resolution}_open",
                "overrides": {"force_stage_gate_values": {resolution: 1.0}},
            }
        )
    experiments.append(
        {
            "name": "force_all_stages_open",
            "overrides": {
                "force_stage_gate_values": {
                    resolution: 1.0 for resolution in stage_resolutions
                }
            },
        }
    )
    return experiments


def stage_grad_norm(model, stage_resolutions):
    grad_stats = {}
    named_params = dict(model.named_parameters())
    is_hybrid = hasattr(model, "decoder_stage_resolutions")
    for resolution in stage_resolutions:
        projector_sq = 0.0
        gate_sq = 0.0
        if is_hybrid:
            if resolution == getattr(model, "decoder_anchor_resolution", None):
                stage_prefixes = ["anchor_builder."]
            else:
                decoder_resolutions = list(getattr(model, "decoder_stage_resolutions", []))
                if resolution in decoder_resolutions:
                    stage_idx = decoder_resolutions.index(resolution)
                    stage_prefixes = [f"decoder_stages.{stage_idx}."]
                else:
                    stage_prefixes = []
        else:
            projector_prefix = f"multiscale_fusion_head.projectors.{resolution}."
            gate_prefix = f"multiscale_fusion_head.gates.{resolution}."
        for name, param in named_params.items():
            if param.grad is None:
                continue
            grad_value = float(param.grad.detach().pow(2).sum().item())
            if is_hybrid:
                if any(name.startswith(prefix) for prefix in stage_prefixes):
                    if ".gate." in name or ".channel_gate." in name:
                        gate_sq += grad_value
                    elif ".skip_proj." in name or ".fuse_proj." in name:
                        projector_sq += grad_value
            else:
                if name.startswith(projector_prefix):
                    projector_sq += grad_value
                if name.startswith(gate_prefix):
                    gate_sq += grad_value
        grad_stats[resolution] = {
            "projector_grad_norm": projector_sq**0.5,
            "gate_grad_norm": gate_sq**0.5,
        }
    return grad_stats


def main():
    cli_args = parse_args()
    if not torch.cuda.is_available() and cli_args.device.startswith("cuda"):
        raise RuntimeError("CUDA is not available but a CUDA device was requested.")

    random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)

    checkpoint = torch.load(cli_args.ckpt, map_location="cpu")
    loaded_args = prepare_loaded_args(checkpoint["args"], cli_args)
    device = torch.device(cli_args.device)

    model, _, _ = get_model(loaded_args, device)
    state_dict = checkpoint[cli_args.weights]
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if not getattr(model, "use_multiscale_fusion_head", False):
        raise AssertionError("The loaded model does not have multiscale fusion enabled.")

    stage_resolutions = list(getattr(model, "fusion_selected_stages", []))
    transport = create_transport(
        path_type=loaded_args.train.path_type,
        prediction=loaded_args.train.prediction,
        loss_weight=loaded_args.train.loss_weight,
        train_eps=loaded_args.train.train_eps,
        sample_eps=loaded_args.train.sample_eps,
    )

    datamod = WebDataModuleFromConfig(**loaded_args.data)
    split_name, loader = choose_loader(datamod, loaded_args, cli_args.split)
    loader_iter = iter(loader)

    experiments = build_experiments(stage_resolutions)
    experiment_store = {
        experiment["name"]: {
            "loss_sum": 0.0,
            "count": 0,
            "stage_gate_means": defaultdict(float),
            "stage_raw_gate_means": defaultdict(float),
            "stage_feature_norms": defaultdict(float),
        }
        for experiment in experiments
    }
    t_bin_store = {
        idx: {
            "count": 0,
            "stage_gate_means": defaultdict(float),
            "stage_raw_gate_means": defaultdict(float),
            "stage_projected_norms": defaultdict(float),
            "stage_aligned_norms": defaultdict(float),
            "stage_feature_norms": defaultdict(float),
        }
        for idx in range(cli_args.t_bins)
    }
    has_per_sample_t_stats = False
    grad_store = {
        resolution: {
            "projector_grad_norm_sum": 0.0,
            "gate_grad_norm_sum": 0.0,
            "count": 0,
        }
        for resolution in stage_resolutions
    }

    with torch.no_grad():
        for _ in range(cli_args.num_batches):
            data = next(loader_iter)
            x, y = extract_batch(data, loaded_args, device)
            model_kwargs = {} if y is None else {"y": y}
            t, x0, xt, ut = sample_transport_inputs(transport, x)
            for experiment in experiments:
                model.clear_fusion_runtime_overrides()
                model.set_fusion_runtime_overrides(**experiment["overrides"])
                model_output = model.forward_transport(xt, t, **model_kwargs)
                batch_losses = compute_loss_from_fixed_sample(
                    transport, model_output, xt, x0, ut, t
                )
                batch_loss = float(batch_losses.mean().item())
                latest_stats = model.latest_fusion_stats

                store = experiment_store[experiment["name"]]
                store["loss_sum"] += batch_loss
                store["count"] += 1
                for resolution in stage_resolutions:
                    store["stage_gate_means"][resolution] += float(
                        latest_stats["stage_gate_means"][resolution]
                    )
                    store["stage_raw_gate_means"][resolution] += float(
                        latest_stats["stage_raw_gate_means"][resolution]
                    )
                    store["stage_feature_norms"][resolution] += float(
                        latest_stats["stage_feature_norms"][resolution]
                    )

                if experiment["name"] == "baseline":
                    gate_values = latest_stats.get("stage_gate_values")
                    raw_gate_values = latest_stats.get("stage_raw_gate_values")
                    projected_norm_values = latest_stats.get("stage_projected_norm_values")
                    aligned_norm_values = latest_stats.get("stage_aligned_norm_values")
                    feature_norm_values = latest_stats.get("stage_feature_norm_values")
                    if not all(
                        values is not None
                        for values in (
                            gate_values,
                            raw_gate_values,
                            projected_norm_values,
                            aligned_norm_values,
                            feature_norm_values,
                        )
                    ):
                        continue
                    has_per_sample_t_stats = True
                    for sample_idx, t_value in enumerate(t.detach().cpu().tolist()):
                        bin_idx = min(cli_args.t_bins - 1, int(t_value * cli_args.t_bins))
                        bucket = t_bin_store[bin_idx]
                        bucket["count"] += 1
                        for resolution in stage_resolutions:
                            bucket["stage_gate_means"][resolution] += float(
                                gate_values[resolution][sample_idx]
                            )
                            bucket["stage_raw_gate_means"][resolution] += float(
                                raw_gate_values[resolution][sample_idx]
                            )
                            bucket["stage_projected_norms"][resolution] += float(
                                projected_norm_values[resolution][sample_idx]
                            )
                            bucket["stage_aligned_norms"][resolution] += float(
                                aligned_norm_values[resolution][sample_idx]
                            )
                            bucket["stage_feature_norms"][resolution] += float(
                                feature_norm_values[resolution][sample_idx]
                            )
            model.clear_fusion_runtime_overrides()

    for _ in range(cli_args.grad_batches):
        data = next(loader_iter)
        x, y = extract_batch(data, loaded_args, device)
        model_kwargs = {} if y is None else {"y": y}
        t, x0, xt, ut = sample_transport_inputs(transport, x)

        model.zero_grad(set_to_none=True)
        model.clear_fusion_runtime_overrides()
        model_output = model.forward_transport(xt, t, **model_kwargs)
        loss = compute_loss_from_fixed_sample(transport, model_output, xt, x0, ut, t).mean()
        loss.backward()
        batch_grad_stats = stage_grad_norm(model, stage_resolutions)
        for resolution in stage_resolutions:
            grad_store[resolution]["projector_grad_norm_sum"] += batch_grad_stats[
                resolution
            ]["projector_grad_norm"]
            grad_store[resolution]["gate_grad_norm_sum"] += batch_grad_stats[resolution][
                "gate_grad_norm"
            ]
            grad_store[resolution]["count"] += 1

    grad_summary = {
        resolution: {
            "projector_grad_norm": stats["projector_grad_norm_sum"] / max(stats["count"], 1),
            "gate_grad_norm": stats["gate_grad_norm_sum"] / max(stats["count"], 1),
        }
        for resolution, stats in sorted(grad_store.items())
    }

    result = {
        "checkpoint": cli_args.ckpt,
        "weights": cli_args.weights,
        "split": split_name,
        "num_batches": cli_args.num_batches,
        "grad_batches": cli_args.grad_batches,
        "stage_resolutions": stage_resolutions,
        "experiments": summarize_experiment_store(experiment_store),
        "baseline_t_bins_available": has_per_sample_t_stats,
        "baseline_t_bins": (
            summarize_t_bins(t_bin_store, stage_resolutions, cli_args.t_bins)
            if has_per_sample_t_stats
            else []
        ),
        "baseline_grad_norms": grad_summary,
    }

    result_json = json.dumps(result, indent=2, sort_keys=True)
    print(result_json)
    if cli_args.output_json is not None:
        with open(cli_args.output_json, "w", encoding="ascii") as handle:
            handle.write(result_json)


if __name__ == "__main__":
    main()
