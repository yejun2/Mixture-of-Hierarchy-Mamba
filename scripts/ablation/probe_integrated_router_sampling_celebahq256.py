#!/usr/bin/env python
"""Sample from a checkpoint and summarize integrated-router decisions.

This script intentionally does not modify training or model code. It imports the
repo model, loads an exact checkpoint file, wraps router methods on that runtime
model instance, then writes samples plus CSV/JSON summaries.
"""

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = None
F = None
Image = None
Sampler = None
create_transport = None
get_model = None


DEFAULT_CKPT = (
    "/SSD4/yjjung/MoH/Mixture-of-Hierarchy-Mamba/outputs/"
    "hierarchy_hybrid_local_v2_facehq_1024_bs16/"
    "2026-05-04_03-45-00_integrated_controller_hierarchy_celebahq256/"
    "checkpoints/0130000.pt"
)


def has_text(args):
    return "celebamm" in str(args.data.name) or "coco" in str(args.data.name)


def is_video(args):
    if hasattr(args.model.params, "video_frames"):
        return int(args.model.params.video_frames) > 0
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate samples from an exact checkpoint and quantify integrated "
            "router logits/probabilities during ODE sampling."
        )
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Exact checkpoint .pt path.")
    parser.add_argument(
        "--weights",
        choices=("ema", "model"),
        default="ema",
        help="Checkpoint weights to sample with. train_acc.py samples with EMA.",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs/router_probe_integrated_controller_0130000"),
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Autocast precision for model sampling.",
    )
    parser.add_argument("--sampling-method", default="dopri5")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--t-bins", type=int, default=20)
    parser.add_argument(
        "--vae-id",
        default=None,
        help="Diffusers VAE id. Defaults to stabilityai/sd-vae-ft-${args.vae}.",
    )
    parser.add_argument(
        "--vae-local-files-only",
        action="store_true",
        help="Require the VAE to be present in the local diffusers cache.",
    )
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="Also save final latent tensors for each generated batch.",
    )
    return parser.parse_args()


def torch_load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_runtime_deps():
    global F, Image, Sampler, create_transport, get_model, torch

    import torch as torch_module
    import torch.nn.functional as functional
    from PIL import Image as image_module
    from transport import Sampler as sampler_class
    from transport import create_transport as create_transport_fn
    from utils.train_utils import get_model as get_model_fn

    torch = torch_module
    F = functional
    Image = image_module
    Sampler = sampler_class
    create_transport = create_transport_fn
    get_model = get_model_fn


def strip_module_prefix(state_dict):
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


class RunningStats:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = None
        self.max = None

    def update(self, values):
        if values is None:
            return
        tensor = torch.as_tensor(values).detach().float().flatten()
        if tensor.numel() == 0:
            return
        tensor = tensor.cpu()
        count = int(tensor.numel())
        self.count += count
        self.sum += float(tensor.sum().item())
        self.sumsq += float((tensor * tensor).sum().item())
        current_min = float(tensor.min().item())
        current_max = float(tensor.max().item())
        self.min = current_min if self.min is None else min(self.min, current_min)
        self.max = current_max if self.max is None else max(self.max, current_max)

    def update_scalar(self, value):
        self.update(torch.tensor([float(value)]))

    def to_dict(self):
        if self.count <= 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.sum / self.count
        variance = max(self.sumsq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.min,
            "max": self.max,
        }


class RouterProbe:
    def __init__(self, t_bins):
        self.t_bins = int(t_bins)
        self.global_stats = {}
        self.time_stats = {}
        self.per_sample_stats = {}
        self.stage_prob_trace = []
        self.forward_count = 0
        self.router_call_count = 0
        self.current_t = None
        self.current_sample_indices = None
        self.active = False

    def start_batch(self, sample_start, batch_size):
        self.current_sample_indices = list(range(sample_start, sample_start + batch_size))
        self.active = True

    def end_batch(self):
        self.current_sample_indices = None
        self.current_t = None
        self.active = False

    def begin_forward(self, t, batch_size):
        self.forward_count += 1
        if t is None:
            self.current_t = None
        else:
            self.current_t = float(torch.as_tensor(t).detach().float().mean().cpu().item())
        if (
            self.current_sample_indices is None
            or len(self.current_sample_indices) != int(batch_size)
        ):
            self.current_sample_indices = list(range(int(batch_size)))

    def _t_bin(self):
        if self.current_t is None:
            return -1
        idx = int(max(0.0, min(0.999999, self.current_t)) * self.t_bins)
        return max(0, min(self.t_bins - 1, idx))

    @staticmethod
    def _stats_key(router, context, choice, metric):
        return (str(router), str(context), str(choice), str(metric))

    def _update(self, router, context, choice, metric, values):
        key = self._stats_key(router, context, choice, metric)
        self.global_stats.setdefault(key, RunningStats()).update(values)

        time_key = (self._t_bin(), *key)
        self.time_stats.setdefault(time_key, RunningStats()).update(values)

        if values is None or self.current_sample_indices is None:
            return
        tensor = torch.as_tensor(values).detach().float()
        if tensor.dim() == 0:
            tensor = tensor.view(1).expand(len(self.current_sample_indices))
        tensor = tensor.flatten().cpu()
        for sample_idx, value in zip(self.current_sample_indices, tensor.tolist()):
            sample_key = (int(sample_idx), *key)
            self.per_sample_stats.setdefault(sample_key, RunningStats()).update_scalar(value)

    def record_matrix(
        self,
        router,
        context,
        choices,
        logits=None,
        probabilities=None,
        weights=None,
        mask=None,
        entropy_mode="categorical",
    ):
        if not self.active:
            return
        self.router_call_count += 1
        choices = [str(choice) for choice in choices]

        matrices = {
            "logit": logits,
            "prob": probabilities,
            "weight": weights,
            "selected": None if mask is None else mask.float(),
        }
        for metric, matrix in matrices.items():
            if matrix is None:
                continue
            matrix = torch.as_tensor(matrix).detach().float()
            for idx, choice in enumerate(choices):
                self._update(router, context, choice, metric, matrix[:, idx])

        if probabilities is None:
            return

        probs = torch.as_tensor(probabilities).detach().float().clamp(1e-8, 1.0)
        if str(router) == "stage" and str(context) == "skip":
            self._record_stage_prob_trace(
                choices=choices,
                probabilities=probs,
                logits=logits,
                weights=weights,
                mask=mask,
            )
        if entropy_mode == "bernoulli":
            inv = (1.0 - probs).clamp(1e-8, 1.0)
            entropy = -((probs * probs.log()) + (inv * inv.log())).sum(dim=-1)
        else:
            entropy = -(probs * probs.log()).sum(dim=-1)
        self._update(router, context, "all", "entropy", entropy)

        if probs.shape[-1] >= 2:
            top2 = torch.topk(probs, k=2, dim=-1).values
            self._update(router, context, "all", "top1_margin", top2[:, 0] - top2[:, 1])

        if mask is not None:
            selected_count = torch.as_tensor(mask).detach().float().sum(dim=-1)
            self._update(router, context, "all", "selected_count", selected_count)

    def _record_stage_prob_trace(
        self,
        choices,
        probabilities,
        logits=None,
        weights=None,
        mask=None,
    ):
        if self.current_sample_indices is None:
            return

        probabilities = torch.as_tensor(probabilities).detach().float().cpu()
        logits = (
            None
            if logits is None
            else torch.as_tensor(logits).detach().float().cpu()
        )
        weights = (
            None
            if weights is None
            else torch.as_tensor(weights).detach().float().cpu()
        )
        mask = (
            None
            if mask is None
            else torch.as_tensor(mask).detach().float().cpu()
        )

        for batch_offset, sample_idx in enumerate(self.current_sample_indices):
            row = {
                "sample_idx": int(sample_idx),
                "forward_idx": int(self.forward_count),
                "router_call_idx": int(self.router_call_count),
                "t": self.current_t,
                "bin_idx": self._t_bin(),
            }
            for choice_idx, choice in enumerate(choices):
                prefix = f"stage_{choice}"
                row[f"{prefix}_prob"] = float(probabilities[batch_offset, choice_idx])
                if logits is not None:
                    row[f"{prefix}_logit"] = float(logits[batch_offset, choice_idx])
                if weights is not None:
                    row[f"{prefix}_weight"] = float(weights[batch_offset, choice_idx])
                if mask is not None:
                    row[f"{prefix}_selected"] = float(mask[batch_offset, choice_idx])
            self.stage_prob_trace.append(row)

    def summary_rows(self):
        rows = []
        for key, stats in sorted(self.global_stats.items()):
            router, context, choice, metric = key
            row = {
                "router": router,
                "context": context,
                "choice": choice,
                "metric": metric,
                **stats.to_dict(),
            }
            rows.append(row)
        return rows

    def timeseries_rows(self):
        rows = []
        for key, stats in sorted(self.time_stats.items()):
            bin_idx, router, context, choice, metric = key
            if bin_idx < 0:
                t_start = None
                t_end = None
            else:
                t_start = bin_idx / self.t_bins
                t_end = (bin_idx + 1) / self.t_bins
            rows.append(
                {
                    "bin_idx": bin_idx,
                    "t_start": t_start,
                    "t_end": t_end,
                    "router": router,
                    "context": context,
                    "choice": choice,
                    "metric": metric,
                    **stats.to_dict(),
                }
            )
        return rows

    def stage_prob_trace_rows(self):
        return list(self.stage_prob_trace)

    def per_sample_rows(self):
        rows = []
        for key, stats in sorted(self.per_sample_stats.items()):
            sample_idx, router, context, choice, metric = key
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "router": router,
                    "context": context,
                    "choice": choice,
                    "metric": metric,
                    **stats.to_dict(),
                }
            )
        return rows


def entropy_mode_for_controller(controller):
    mode = str(getattr(controller, "stage_select_mode", "topk")).lower()
    return "bernoulli" if mode in {"adaptive", "adaptive_topk"} else "categorical"


def recompute_stage_logits(controller, condition, skip_maps, scale_prior, available):
    condition_summary = controller.condition_norm(condition)
    use_scale_prior_context = bool(
        getattr(controller, "stage_use_scale_prior_context", True)
    )
    prior_weights = scale_prior["weights"]
    prior_probabilities = scale_prior["probabilities"]
    if not use_scale_prior_context:
        prior_weights = torch.zeros_like(prior_weights)
        prior_probabilities = torch.zeros_like(prior_probabilities)
    stage_condition = torch.cat(
        [
            condition_summary,
            prior_weights,
            prior_probabilities,
        ],
        dim=-1,
    )
    stage_condition_hidden = controller.stage_condition_proj(stage_condition)
    raw_logits = []
    for resolution in available:
        feature_hidden = controller.stage_feature_proj(
            controller._summarize_map(skip_maps[resolution])
        )
        feature_logit = controller.stage_score_head(
            F.gelu(stage_condition_hidden + feature_hidden)
        )
        if use_scale_prior_context:
            idx = controller.resolution_to_index[int(resolution)]
            feature_logit = feature_logit + scale_prior["logits"][:, idx : idx + 1]
        raw_logits.append(feature_logit)
    return torch.cat(raw_logits, dim=-1)


def attach_integrated_controller_probe(model, probe):
    controller = getattr(model, "integrated_router_controller", None)
    if controller is None:
        return False

    resolutions = [int(resolution) for resolution in getattr(controller, "resolutions", [])]
    stage_entropy_mode = entropy_mode_for_controller(controller)

    original_scale_prior = controller.compute_scale_prior
    original_stage_weights = controller.compute_stage_weights
    original_compression = controller.compute_compression
    original_encoder_depth = controller.compute_encoder_depth

    def wrapped_scale_prior(condition):
        result = original_scale_prior(condition)
        probe.record_matrix(
            router="scale_prior",
            context="global",
            choices=resolutions,
            logits=result.get("logits"),
            probabilities=result.get("probabilities"),
            weights=result.get("weights"),
            mask=result.get("mask"),
            entropy_mode=stage_entropy_mode,
        )
        return result

    def wrapped_stage_weights(condition, skip_maps, scale_prior):
        result = original_stage_weights(condition, skip_maps, scale_prior)
        if scale_prior is not None:
            available = [resolution for resolution in resolutions if resolution in skip_maps]
            if available:
                raw_logits = recompute_stage_logits(
                    controller,
                    condition,
                    skip_maps,
                    scale_prior,
                    available,
                )
                selection = controller._select_stage_weights(raw_logits)
                probe.record_matrix(
                    router="stage",
                    context="skip",
                    choices=available,
                    logits=raw_logits,
                    probabilities=selection.get("probabilities"),
                    weights=selection.get("weights"),
                    mask=selection.get("mask"),
                    entropy_mode=stage_entropy_mode,
                )
        return result

    def wrapped_compression(
        condition,
        context_map,
        output_resolution,
        encoder_depth,
        scale_prior,
    ):
        result = original_compression(
            condition,
            context_map,
            output_resolution,
            encoder_depth,
            scale_prior,
        )
        logits = result.get("logits")
        if logits is not None:
            probabilities = torch.softmax(logits, dim=-1)
            selected = torch.argmax(logits, dim=-1)
            mask = F.one_hot(selected, num_classes=logits.shape[-1]).to(torch.bool)
            weights = mask.to(dtype=probabilities.dtype) + probabilities - probabilities.detach()
            probe.record_matrix(
                router="compression",
                context=f"out{int(output_resolution)}",
                choices=("stride0", "stride1", "stride2"),
                logits=logits,
                probabilities=probabilities,
                weights=weights,
                mask=mask,
                entropy_mode="categorical",
            )
        return result

    def wrapped_encoder_depth(
        condition,
        context_map,
        resolution,
        base_depth,
        scale_prior,
    ):
        result = original_encoder_depth(
            condition,
            context_map,
            resolution,
            base_depth,
            scale_prior,
        )
        logits = result.get("logits")
        if logits is not None:
            probabilities = torch.softmax(logits, dim=-1)
            selected = torch.argmax(logits, dim=-1)
            mask = F.one_hot(selected, num_classes=logits.shape[-1]).to(torch.bool)
            weights = mask.to(dtype=probabilities.dtype) + probabilities - probabilities.detach()
            probe.record_matrix(
                router="encoder_depth",
                context=f"res{int(resolution)}",
                choices=("shallow", "base", "deep"),
                logits=logits,
                probabilities=probabilities,
                weights=weights,
                mask=mask,
                entropy_mode="categorical",
            )
        return result

    controller.compute_scale_prior = wrapped_scale_prior
    controller.compute_stage_weights = wrapped_stage_weights
    controller.compute_compression = wrapped_compression
    controller.compute_encoder_depth = wrapped_encoder_depth
    return True


def autocast_context(device, mixed_precision):
    if device.type != "cuda" or mixed_precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_vae(loaded_args, cli_args, device):
    if not bool(getattr(loaded_args, "is_latent", False)):
        return None
    from diffusers.models import AutoencoderKL

    vae_id = cli_args.vae_id or f"stabilityai/sd-vae-ft-{loaded_args.vae}"
    vae = AutoencoderKL.from_pretrained(
        vae_id,
        local_files_only=bool(cli_args.vae_local_files_only),
    ).to(device)
    vae.eval()
    return vae


def decode_samples(latents, vae):
    if vae is None:
        return latents
    latents = latents.float() / 0.18215
    return vae.decode(latents).sample


def save_images(images, sample_start, samples_dir):
    samples_dir.mkdir(parents=True, exist_ok=True)
    images = torch.clamp(127.5 * images + 128.0, 0, 255).to(torch.uint8)
    for offset, image in enumerate(images):
        array = image.permute(1, 2, 0).contiguous().cpu().numpy()
        Image.fromarray(array).save(samples_dir / f"sample_{sample_start + offset:06d}.png")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stage_prob_rows(rows):
    return [
        row
        for row in rows
        if row.get("router") == "stage" and row.get("metric") == "prob"
    ]


def selected_model_metadata(model, loaded_args, checkpoint, cli_args):
    router = getattr(model, "integrated_router_controller", None)
    metadata = {
        "checkpoint": str(Path(cli_args.ckpt).resolve()),
        "weights": cli_args.weights,
        "train_steps": int(checkpoint.get("train_steps", -1)),
        "data_name": str(loaded_args.data.name),
        "model_name": str(loaded_args.model.name),
        "image_size": int(loaded_args.data.image_size),
        "is_latent": bool(loaded_args.is_latent),
        "use_latent": bool(loaded_args.use_latent),
        "num_samples": int(cli_args.num_samples),
        "batch_size": int(cli_args.batch_size),
        "sampling_method": str(cli_args.sampling_method),
        "num_steps": int(cli_args.num_steps),
        "atol": float(cli_args.atol),
        "rtol": float(cli_args.rtol),
        "mixed_precision": str(cli_args.mixed_precision),
        "seed": int(cli_args.seed),
        "integrated_controller_present": router is not None,
    }
    if router is not None:
        metadata.update(
            {
                "router_resolutions": [
                    int(resolution) for resolution in getattr(router, "resolutions", [])
                ],
                "stage_select_mode": str(getattr(router, "stage_select_mode", "unknown")),
                "stage_top_k": int(getattr(router, "top_k", -1)),
                "stage_select_threshold": float(
                    getattr(router, "stage_select_threshold", -1.0)
                ),
                "stage_select_threshold_mode": str(
                    getattr(router, "stage_select_threshold_mode", "fixed")
                ),
                "stage_select_threshold_margin": float(
                    getattr(router, "stage_select_threshold_margin", 0.0)
                ),
                "stage_select_warmup_steps": int(
                    getattr(router, "stage_select_warmup_steps", 0)
                ),
                "stage_select_warmup_min_selected": int(
                    getattr(router, "stage_select_warmup_min_selected", 0)
                ),
                "stage_select_warmup_threshold_margin": float(
                    getattr(router, "stage_select_warmup_threshold_margin", 0.0)
                ),
                "stage_min_selected": int(getattr(router, "stage_min_selected", -1)),
                "stage_balance_mode": str(
                    getattr(router, "stage_balance_mode", "unknown")
                ),
                "stage_use_scale_prior_context": bool(
                    getattr(router, "stage_use_scale_prior_context", True)
                ),
            }
        )
    return metadata


def main():
    cli_args = parse_args()
    load_runtime_deps()

    ckpt_path = Path(cli_args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Expected an exact checkpoint .pt file, got: {ckpt_path}"
        )
    if cli_args.num_samples <= 0 or cli_args.batch_size <= 0:
        raise ValueError("--num-samples and --batch-size must be positive.")
    if cli_args.t_bins <= 0:
        raise ValueError("--t-bins must be positive.")
    if cli_args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cli_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli_args.seed)

    output_dir = Path(cli_args.output_dir)
    samples_dir = output_dir / "samples"
    latents_dir = output_dir / "latents"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch_load_checkpoint(ckpt_path)
    loaded_args = checkpoint["args"]
    if has_text(loaded_args) or is_video(loaded_args):
        raise NotImplementedError(
            "This probe script currently targets unconditional image checkpoints."
        )

    device = torch.device(cli_args.device)
    model, in_channels, input_size = get_model(loaded_args, device)
    state_dict = strip_module_prefix(checkpoint[cli_args.weights])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    probe = RouterProbe(t_bins=cli_args.t_bins)
    if not attach_integrated_controller_probe(model, probe):
        raise RuntimeError("The loaded model has no integrated_router_controller.")

    transport = create_transport(
        path_type=loaded_args.train.path_type,
        prediction=loaded_args.train.prediction,
        loss_weight=loaded_args.train.loss_weight,
        train_eps=loaded_args.train.train_eps,
        sample_eps=loaded_args.train.sample_eps,
    )
    sample_fn = Sampler(transport).sample_ode(
        sampling_method=cli_args.sampling_method,
        num_steps=cli_args.num_steps,
        atol=cli_args.atol,
        rtol=cli_args.rtol,
    )
    model_forward = getattr(model, "forward_transport", model.forward)
    vae = load_vae(loaded_args, cli_args, device)

    metadata = selected_model_metadata(model, loaded_args, checkpoint, cli_args)
    with open(output_dir / "probe_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    generated = 0
    batch_idx = 0
    with torch.no_grad():
        while generated < cli_args.num_samples:
            batch_size = min(cli_args.batch_size, cli_args.num_samples - generated)
            z = torch.randn(
                batch_size,
                in_channels,
                input_size,
                input_size,
                device=device,
            )

            probe.start_batch(sample_start=generated, batch_size=batch_size)

            def probed_model_fn(x, t, **model_kwargs):
                probe.begin_forward(t, batch_size=x.shape[0])
                return model_forward(x, t, **model_kwargs)

            with autocast_context(device, cli_args.mixed_precision):
                latents = sample_fn(z, probed_model_fn)[-1]

            probe.end_batch()

            if cli_args.save_latents:
                latents_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    latents.detach().cpu(),
                    latents_dir / f"latents_batch_{batch_idx:04d}.pt",
                )

            images = decode_samples(latents, vae)
            save_images(images, generated, samples_dir)

            generated += batch_size
            batch_idx += 1
            print(
                f"[router-probe] generated {generated}/{cli_args.num_samples} samples",
                flush=True,
            )

    summary_rows = probe.summary_rows()
    timeseries_rows = probe.timeseries_rows()
    per_sample_rows = probe.per_sample_rows()
    stage_prob_trace_rows = probe.stage_prob_trace_rows()

    write_csv(output_dir / "router_summary.csv", summary_rows)
    write_csv(output_dir / "router_timeseries.csv", timeseries_rows)
    write_csv(output_dir / "router_per_sample.csv", per_sample_rows)
    write_csv(output_dir / "stage_prob_summary.csv", stage_prob_rows(summary_rows))
    write_csv(output_dir / "stage_prob_timeseries.csv", stage_prob_rows(timeseries_rows))
    write_csv(output_dir / "stage_prob_per_sample.csv", stage_prob_rows(per_sample_rows))
    write_csv(output_dir / "stage_prob_trace.csv", stage_prob_trace_rows)

    summary_json = {
        "metadata": {
            **metadata,
            "forward_count": probe.forward_count,
            "router_call_count": probe.router_call_count,
            "stage_prob_trace_rows": len(stage_prob_trace_rows),
        },
        "summary": summary_rows,
        "stage_prob_summary": stage_prob_rows(summary_rows),
    }
    with open(output_dir / "router_summary.json", "w") as handle:
        json.dump(summary_json, handle, indent=2, sort_keys=True)

    print(f"[router-probe] wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
