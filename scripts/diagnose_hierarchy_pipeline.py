#!/usr/bin/env python
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.wds_dataloader import WebDataModuleFromConfig
from model_zigma import map_to_tokens, tokens_to_map, layer_norm_fn, rms_norm_fn, RMSNorm
from transport import create_transport
from utils.train_utils import get_model


def has_text(args):
    return "celebamm" in args.data.name or "coco" in args.data.name


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
        else:
            raise NotImplementedError(
                f"Unsupported dataset for diagnosis: {loaded_args.data.name}"
            )
    else:
        x = data["image"].to(device)
        y = None

    if loaded_args.is_latent:
        if loaded_args.use_latent:
            x = x * 0.18215
        else:
            raise NotImplementedError(
                "diagnose_hierarchy_pipeline.py expects use_latent=true when is_latent=true"
            )
    return x, y


def choose_loader(datamod, loaded_args, split):
    include_images = not (loaded_args.use_latent and "facehq" in str(loaded_args.data.name))
    if split == "val" and getattr(loaded_args.data, "validation", None) is not None:
        return datamod.val_dataloader(include_images=include_images)
    return datamod.train_dataloader(include_images=include_images)


def summarize_tensor(tensor):
    data = tensor.detach()
    return {
        "shape": list(data.shape),
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "norm": float(data.norm().item()),
        "absmax": float(data.abs().max().item()),
    }


def run_processor_with_stats(processor, context_map, c, text=None):
    windows, out_hw = processor._extract_windows(context_map)
    residual = None
    hidden_states = windows
    repeat_factor = out_hw[0] * out_hw[1]
    repeated_c = c.repeat_interleave(repeat_factor, dim=0)
    repeated_text = (
        text.repeat_interleave(repeat_factor, dim=0) if text is not None else None
    )

    for block in processor.blocks:
        hidden_states, residual = block(
            hidden_states,
            residual=residual,
            c=repeated_c,
            text=repeated_text,
        )

    if not processor.fused_add_norm:
        residual = hidden_states if residual is None else residual + hidden_states
        stage_hidden = processor.stage_norm(
            residual.to(dtype=processor.stage_norm.weight.dtype)
        )
    else:
        fused_add_norm_fn = (
            rms_norm_fn
            if isinstance(processor.stage_norm, RMSNorm)
            else layer_norm_fn
        )
        stage_hidden = fused_add_norm_fn(
            hidden_states,
            processor.stage_norm.weight,
            processor.stage_norm.bias,
            eps=processor.stage_norm.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=processor.residual_in_fp32,
        )

    compressed_update = processor.context_compress_type == "mean"
    if compressed_update:
        update_tokens = stage_hidden.mean(dim=1)
        input_tokens = windows.mean(dim=1)
    else:
        update_tokens = stage_hidden[:, -1, :]
        input_tokens = windows[:, -1, :]
    residual_branch = processor.residual_proj(input_tokens)
    update_branch = processor.output_proj(update_tokens)
    stage_output = residual_branch + update_branch
    stage_output = stage_output.reshape(
        context_map.shape[0], out_hw[0], out_hw[1], processor.dim
    )
    stats = {
        "input_resolution": list(context_map.shape[1:3]),
        "output_resolution": list(out_hw),
        "windows": summarize_tensor(windows),
        "stage_hidden": summarize_tensor(stage_hidden),
        "pooled_input": summarize_tensor(input_tokens),
        "pooled_update": summarize_tensor(update_tokens),
        "residual_branch": summarize_tensor(residual_branch),
        "update_branch": summarize_tensor(update_branch),
        "stage_output": summarize_tensor(stage_output),
        "residual_to_update_norm_ratio": float(
            residual_branch.norm().item() / (update_branch.norm().item() + 1e-12)
        ),
    }
    return stage_output, stats


def run_manual_breakdown(model, xt, t, y=None):
    breakdown = {}
    hidden_states = model.x_embedder(xt)
    breakdown["patch_embed"] = summarize_tensor(hidden_states)

    c, y_embed, timestep_condition = model._get_condition(hidden_states, t, y)
    breakdown["timestep_condition"] = summarize_tensor(timestep_condition)
    breakdown["condition"] = summarize_tensor(c)

    if model.pos_embed is not None:
        hidden_states = hidden_states + model.pos_embed
        breakdown["top_pos_embed"] = summarize_tensor(model.pos_embed)
    initial_map = tokens_to_map(hidden_states, model.hierarchy_input_size)
    breakdown["initial_map"] = summarize_tensor(initial_map)

    current_map = initial_map
    selected_stage_maps = {}
    stage_details = []
    for stage_idx, layout in enumerate(model.hierarchy_stage_layout):
        if model.stage_processors is not None:
            processor = model.stage_processors[stage_idx]
        elif stage_idx == 0 and model.first_stage_processor is not None:
            processor = model.first_stage_processor
        else:
            processor = model.shared_processor
        current_map, stats = run_processor_with_stats(processor, current_map, c, y_embed)
        stats["stage_idx"] = stage_idx
        stats["expected_input_resolution"] = list(layout["input_resolution"])
        stats["expected_output_resolution"] = list(layout["output_resolution"])
        stage_details.append(stats)
        out_h, out_w = current_map.shape[1:3]
        if out_h == out_w and out_h in model.fusion_selected_stages:
            selected_stage_maps[int(out_h)] = current_map

    breakdown["hierarchy_stages"] = stage_details
    breakdown["final_map_pre_norm"] = summarize_tensor(current_map)

    final_tokens = map_to_tokens(current_map)
    if not model.fused_add_norm:
        final_tokens = model.norm_f(final_tokens.to(dtype=model.norm_f.weight.dtype))
    else:
        fused_add_norm_fn = (
            rms_norm_fn if isinstance(model.norm_f, RMSNorm) else layer_norm_fn
        )
        final_tokens = fused_add_norm_fn(
            final_tokens,
            model.norm_f.weight,
            model.norm_f.bias,
            eps=model.norm_f.eps,
            residual=None,
            prenorm=False,
            residual_in_fp32=False,
        )
    final_map = tokens_to_map(final_tokens, current_map.shape[1:3])
    breakdown["final_map_post_norm"] = summarize_tensor(final_map)
    if 1 in model.fusion_selected_stages and final_map.shape[1:3] == (1, 1):
        selected_stage_maps[1] = final_map

    fusion = model.multiscale_fusion_head
    fusion_details = {"per_stage": {}}
    aligned_features = []
    for resolution in fusion.selected_stage_resolutions:
        key = str(resolution)
        stage_map = selected_stage_maps[resolution]
        normalized_stage_map = fusion.input_norms[key](stage_map)
        projected = fusion.projectors[key](normalized_stage_map)
        aligned = fusion.aligner(projected)
        if fusion.pos_embed is not None:
            aligned = fusion.pos_embed(aligned)
        raw_gate = fusion.gates[key].compute_gate(timestep_condition, aligned.shape[0])
        gated = aligned * raw_gate
        aligned_features.append(gated)
        fusion_details["per_stage"][str(resolution)] = {
            "raw_stage_map": summarize_tensor(stage_map),
            "normalized_stage_map": summarize_tensor(normalized_stage_map),
            "projected": summarize_tensor(projected),
            "aligned": summarize_tensor(aligned),
            "raw_gate": summarize_tensor(raw_gate),
            "gated": summarize_tensor(gated),
        }

    if fusion.fusion_mode == "concat":
        fused = torch.cat(aligned_features, dim=-1)
    else:
        fused = torch.stack(aligned_features, dim=0).mean(dim=0)
    fusion_details["concat_or_sum"] = summarize_tensor(fused)
    fused = fusion.fusion_proj(fused)
    fusion_details["fusion_proj_out"] = summarize_tensor(fused)
    for block_idx, block in enumerate(fusion.fusion_blocks):
        fused = block(fused)
        fusion_details[f"fusion_block_{block_idx}"] = summarize_tensor(fused)
    predicted_tokens = fusion.prediction_head(fused)
    fusion_details["fused_anchor"] = summarize_tensor(fused)
    fusion_details["predicted_tokens"] = summarize_tensor(predicted_tokens)
    breakdown["fusion"] = fusion_details
    return breakdown, selected_stage_maps, timestep_condition, fused, predicted_tokens


def compare_manual_and_official(model, xt, t, y, predicted_tokens):
    with torch.no_grad():
        official = model.forward_transport(xt, t, y=y)
        manual = model.unpatchify(predicted_tokens)
    diff = (official - manual).detach()
    return {
        "official_output": summarize_tensor(official),
        "manual_output": summarize_tensor(manual),
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
    }


def temporarily_zero_parameter(parameter, fn):
    if parameter is None:
        return None
    with torch.no_grad():
        backup = parameter.detach().clone()
        parameter.zero_()
    try:
        return fn()
    finally:
        with torch.no_grad():
            parameter.copy_(backup)


def positional_embedding_effects(model, xt, t, y):
    with torch.no_grad():
        baseline = model.forward_transport(xt, t, y=y)
    effects = {"baseline": summarize_tensor(baseline)}

    if getattr(model, "pos_embed", None) is not None:
        zeroed = temporarily_zero_parameter(
            model.pos_embed, lambda: model.forward_transport(xt, t, y=y).detach()
        )
        delta = baseline - zeroed
        effects["top_pos_embed_zeroed"] = {
            "output_delta_norm": float(delta.norm().item()),
            "output_delta_mean_abs": float(delta.abs().mean().item()),
        }

    anchor_pos = getattr(getattr(model, "multiscale_fusion_head", None), "pos_embed", None)
    if anchor_pos is not None:
        zeroed = temporarily_zero_parameter(
            anchor_pos.pos_embed,
            lambda: model.forward_transport(xt, t, y=y).detach(),
        )
        delta = baseline - zeroed
        effects["anchor_pos_embed_zeroed"] = {
            "output_delta_norm": float(delta.norm().item()),
            "output_delta_mean_abs": float(delta.abs().mean().item()),
        }
    return effects


def anchor_usage_diagnostics(model, fused_anchor, target):
    prediction_head = model.multiscale_fusion_head.prediction_head
    diagnostics = {}

    with torch.no_grad():
        baseline_tokens = prediction_head(fused_anchor)
        baseline_output = model.unpatchify(baseline_tokens)
        baseline_norm = baseline_output.norm().item() + 1e-12
        per_position = {}
        for row in range(fused_anchor.shape[1]):
            for col in range(fused_anchor.shape[2]):
                modified = fused_anchor.clone()
                modified[:, row, col, :] = 0
                modified_output = model.unpatchify(prediction_head(modified))
                delta = (baseline_output - modified_output).norm().item() / baseline_norm
                per_position[f"{row},{col}"] = float(delta)
        diagnostics["zero_one_anchor_position_relative_output_delta"] = per_position

    fused_anchor_grad = fused_anchor.detach().clone().requires_grad_(True)
    pred = model.unpatchify(prediction_head(fused_anchor_grad))
    loss = F.mse_loss(pred, target)
    loss.backward()
    grad = fused_anchor_grad.grad.detach().norm(dim=-1)[0]
    diagnostics["anchor_input_grad_norms"] = [
        [float(v) for v in row] for row in grad.cpu().tolist()
    ]
    diagnostics["prediction_head_loss_on_fixed_anchor"] = float(loss.item())
    return diagnostics


def collect_gradient_summary(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()

    groups = {
        "x_embedder": ["x_embedder."],
        "top_pos_embed": ["pos_embed"],
        "t_embedder": ["t_embedder."],
        "stage_processors": ["stage_processors.", "first_stage_processor.", "shared_processor."],
        "norm_f": ["norm_f."],
        "fusion_projectors": ["multiscale_fusion_head.projectors."],
        "fusion_gates": ["multiscale_fusion_head.gates."],
        "fusion_anchor_pos_embed": ["multiscale_fusion_head.pos_embed."],
        "fusion_proj": ["multiscale_fusion_head.fusion_proj."],
        "fusion_blocks": ["multiscale_fusion_head.fusion_blocks."],
        "prediction_head": ["multiscale_fusion_head.prediction_head."],
    }

    summary = {}
    named_params = list(model.named_parameters())
    for group_name, prefixes in groups.items():
        params = [
            (name, param)
            for name, param in named_params
            if any(name.startswith(prefix) or name == prefix for prefix in prefixes)
        ]
        if not params:
            continue
        total = sum(param.numel() for _, param in params)
        with_grad = sum(param.numel() for _, param in params if param.grad is not None)
        nonzero_grad = 0
        grad_sq = 0.0
        max_abs = 0.0
        for _, param in params:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            nonzero_grad += int((grad != 0).sum().item())
            grad_sq += float(grad.pow(2).sum().item())
            max_abs = max(max_abs, float(grad.abs().max().item()))
        summary[group_name] = {
            "total_params": int(total),
            "params_with_grad": int(with_grad),
            "nonzero_grad_entries": int(nonzero_grad),
            "grad_norm": grad_sq**0.5,
            "max_abs_grad": max_abs,
        }
    return summary


def fixed_transport_sample(transport, x):
    t, x0, x1 = transport.sample(x)
    t, xt, ut = transport.path_sampler.plan(t, x0, x1)
    return t, x0, xt, ut


def output_loss(pred, target):
    return F.mse_loss(pred, target)


def run_head_only_probe(model, fused_anchor, target, steps, lr):
    head = copy.deepcopy(model.multiscale_fusion_head.prediction_head).to(fused_anchor.device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        pred = model.unpatchify(head(fused_anchor))
        loss = output_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "loss_ratio_final_over_initial": losses[-1] / (losses[0] + 1e-12),
    }


def run_fusion_head_probe(model, selected_stage_maps, timestep_condition, target, steps, lr):
    fusion_head = copy.deepcopy(model.multiscale_fusion_head).to(target.device)
    opt = torch.optim.Adam(fusion_head.parameters(), lr=lr)
    stage_maps = {k: v.detach() for k, v in selected_stage_maps.items()}
    cond = timestep_condition.detach()
    losses = []
    for _ in range(steps):
        pred_tokens, _ = fusion_head(stage_maps, cond)
        pred = model.unpatchify(pred_tokens)
        loss = output_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "loss_ratio_final_over_initial": losses[-1] / (losses[0] + 1e-12),
    }


def run_full_model_probe(model, xt, t, y, target, steps, lr):
    probe_model = copy.deepcopy(model).to(target.device)
    probe_model.train()
    opt = torch.optim.Adam(probe_model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        pred = probe_model.forward_transport(xt, t, y=y)
        loss = output_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "loss_ratio_final_over_initial": losses[-1] / (losses[0] + 1e-12),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose hierarchy/fusion learning path.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head-steps", type=int, default=60)
    parser.add_argument("--fusion-head-steps", type=int, default=60)
    parser.add_argument("--full-model-steps", type=int, default=40)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fusion-head-lr", type=float, default=5e-4)
    parser.add_argument("--full-model-lr", type=float, default=1e-4)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    loaded_args = checkpoint["args"]
    loaded_args.data.batch_size = args.batch_size
    loaded_args.data.val_batch_size = args.batch_size
    loaded_args.data.num_workers = args.num_workers
    loaded_args.data.val_num_workers = args.num_workers

    device = torch.device(args.device)
    model, _, _ = get_model(loaded_args, device)
    state_dict = checkpoint[args.weights]
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    transport = create_transport(
        path_type=loaded_args.train.path_type,
        prediction=loaded_args.train.prediction,
        loss_weight=loaded_args.train.loss_weight,
        train_eps=loaded_args.train.train_eps,
        sample_eps=loaded_args.train.sample_eps,
    )

    datamod = WebDataModuleFromConfig(**loaded_args.data)
    loader = choose_loader(datamod, loaded_args, args.split)
    batch = next(iter(loader))
    x, y = extract_batch(batch, loaded_args, device)

    t, _, xt, ut = fixed_transport_sample(transport, x)

    with torch.no_grad():
        breakdown, selected_stage_maps, timestep_condition, fused_anchor, predicted_tokens = (
            run_manual_breakdown(model, xt, t, y=y)
        )
        manual_compare = compare_manual_and_official(
            model, xt, t, y, predicted_tokens
        )
        official_pred = model.forward_transport(xt, t, y=y)
        baseline_loss = output_loss(official_pred, ut)

    grad_summary = collect_gradient_summary(model, output_loss(model.forward_transport(xt, t, y=y), ut))
    pos_effects = positional_embedding_effects(model, xt, t, y)
    anchor_usage = anchor_usage_diagnostics(model, fused_anchor.detach(), ut.detach())

    head_probe = run_head_only_probe(
        model,
        fused_anchor.detach(),
        ut.detach(),
        steps=args.head_steps,
        lr=args.head_lr,
    )
    torch.cuda.empty_cache()
    fusion_head_probe = run_fusion_head_probe(
        model,
        selected_stage_maps,
        timestep_condition,
        ut.detach(),
        steps=args.fusion_head_steps,
        lr=args.fusion_head_lr,
    )
    torch.cuda.empty_cache()
    full_model_probe = run_full_model_probe(
        model,
        xt.detach(),
        t.detach(),
        y.detach() if torch.is_tensor(y) else y,
        ut.detach(),
        steps=args.full_model_steps,
        lr=args.full_model_lr,
    )

    result = {
        "checkpoint": args.ckpt,
        "weights": args.weights,
        "data_name": loaded_args.data.name,
        "data_tar_base": loaded_args.data.tar_base,
        "image_size": loaded_args.data.image_size,
        "model_img_dim": loaded_args.model.params.img_dim,
        "baseline_output_loss": float(baseline_loss.item()),
        "manual_vs_official": manual_compare,
        "breakdown": breakdown,
        "gradient_summary": grad_summary,
        "positional_embedding_effects": pos_effects,
        "anchor_usage": anchor_usage,
        "overfit_probes": {
            "prediction_head_only": head_probe,
            "fusion_head_only": fusion_head_probe,
            "full_model": full_model_probe,
        },
    }

    result_json = json.dumps(result, indent=2, sort_keys=True)
    print(result_json)
    if args.output_json is not None:
        with open(args.output_json, "w", encoding="ascii") as handle:
            handle.write(result_json)


if __name__ == "__main__":
    main()
