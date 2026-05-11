import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from model_hierarchy_only import (
    AnchorPositionalEmbedding,
    ConvUpsampleBlock,
    StageTimestepGate,
    parse_stage_resolution_spec,
)
from model_zigma import (
    RMSNorm,
    LabelEmbedder,
    PatchEmbed_Video,
    TimestepEmbedder,
    build_scan_block_kwargs,
    compute_window_grid_size,
    create_block,
    get_2d_sincos_pos_embed,
    layer_norm_fn,
    map_to_tokens,
    rms_norm_fn,
    tokens_to_map,
)


def parse_stage_override_spec(stage_spec, value_parser=int):
    if stage_spec is None:
        return {}
    if isinstance(stage_spec, dict):
        items = list(stage_spec.items())
    elif isinstance(stage_spec, str):
        normalized = stage_spec.strip()
        if not normalized:
            return {}
        items = []
        for part in normalized.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(
                    "Expected stage override entries like '32:2,16:1', "
                    f"got {stage_spec!r}"
                )
            items.append(tuple(piece.strip() for piece in part.split(":", 1)))
    elif isinstance(stage_spec, (list, tuple)):
        items = []
        for part in stage_spec:
            if isinstance(part, str):
                if ":" not in part:
                    raise ValueError(
                        "Expected stage override entries like '32:2,16:1', "
                        f"got {stage_spec!r}"
                    )
                items.append(tuple(piece.strip() for piece in part.split(":", 1)))
            elif isinstance(part, (list, tuple)) and len(part) == 2:
                items.append((part[0], part[1]))
            else:
                raise ValueError(f"Unsupported stage override entry: {part!r}")
    else:
        raise ValueError(f"Unsupported stage override spec type: {type(stage_spec)}")

    parsed = {}
    for resolution, value in items:
        if isinstance(resolution, str):
            resolution = int(resolution.lower().replace("x", "").strip())
        else:
            resolution = int(resolution)
        parsed[resolution] = value_parser(value)
    return parsed


def parse_compute_preset_spec(preset_spec):
    if preset_spec is None:
        return [
            ("small", 1, 0.75),
            ("base", 2, 1.0),
            ("large", 4, 1.25),
        ]
    if isinstance(preset_spec, str):
        normalized = preset_spec.strip()
        if not normalized:
            return []
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
    elif isinstance(preset_spec, (list, tuple)):
        parts = list(preset_spec)
    else:
        raise ValueError(f"Unsupported compute preset spec type: {type(preset_spec)}")

    presets = []
    for part in parts:
        if isinstance(part, str):
            pieces = [piece.strip() for piece in part.split(":")]
            if len(pieces) != 3:
                raise ValueError(
                    "Expected compute preset entries like 'small:1:0.75', "
                    f"got {part!r}"
                )
            name, depth, dim_multiplier = pieces
        elif isinstance(part, (list, tuple)) and len(part) == 3:
            name, depth, dim_multiplier = part
        else:
            raise ValueError(f"Unsupported compute preset entry: {part!r}")
        depth = int(depth)
        dim_multiplier = float(dim_multiplier)
        if depth <= 0:
            raise ValueError(f"Compute preset depth must be positive, got {part!r}")
        if dim_multiplier <= 0:
            raise ValueError(
                f"Compute preset dim multiplier must be positive, got {part!r}"
            )
        presets.append((str(name), depth, dim_multiplier))
    return presets


def parse_stage_compute_preset_override_spec(override_spec):
    if override_spec is None:
        return {}
    if isinstance(override_spec, str):
        normalized = override_spec.strip()
        if not normalized:
            return {}
        parts = [part.strip() for part in normalized.split(";") if part.strip()]
    elif isinstance(override_spec, dict):
        return {
            int(resolution): parse_compute_preset_spec(presets)
            for resolution, presets in override_spec.items()
        }
    else:
        raise ValueError(
            "Unsupported stage compute preset override type: "
            f"{type(override_spec)}"
        )

    parsed = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(
                "Expected stage compute preset overrides like "
                "'16=small:3:0.75,base:4:1.0,large:4:1.25;32=...', "
                f"got {part!r}"
            )
        resolution, preset_spec = part.split("=", 1)
        parsed[int(resolution.strip().lower().replace("x", ""))] = (
            parse_compute_preset_spec(preset_spec)
        )
    return parsed


def round_dim_to_multiple(value, multiple=8):
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def is_auto_stage_spec(stage_spec):
    return stage_spec is None or (
        isinstance(stage_spec, str)
        and stage_spec.strip().lower() in {"", "auto", "default"}
    )


def resolve_stage_spec(stage_spec, auto_resolutions):
    if is_auto_stage_spec(stage_spec):
        return [int(resolution) for resolution in auto_resolutions]
    return parse_stage_resolution_spec(stage_spec)


class FactorizedRouterMLP(nn.Module):
    def __init__(self, condition_dim, feature_dim, num_outputs, device, dtype):
        super().__init__()
        hidden_dim = max(condition_dim, feature_dim, 128)
        self.condition_norm = nn.LayerNorm(
            condition_dim, eps=1e-6, device=device, dtype=dtype
        )
        self.summary_norm = nn.LayerNorm(
            feature_dim * 2, eps=1e-6, device=device, dtype=dtype
        )
        self.resolution_embed = nn.Parameter(
            torch.zeros(1, condition_dim, device=device, dtype=dtype)
        )
        self.input_proj = nn.Linear(
            condition_dim * 2 + feature_dim * 2,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        self.output_proj = nn.Linear(
            hidden_dim,
            num_outputs,
            device=device,
            dtype=dtype,
        )
        nn.init.normal_(self.resolution_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, condition, stage_map):
        summary_mean = stage_map.mean(dim=(1, 2))
        summary_std = stage_map.float().std(dim=(1, 2), unbiased=False).to(stage_map)
        summary = self.summary_norm(torch.cat([summary_mean, summary_std], dim=-1))
        condition_summary = self.condition_norm(condition)
        resolution_embed = self.resolution_embed.expand(condition.shape[0], -1)
        hidden = F.gelu(
            self.input_proj(
                torch.cat([condition_summary, summary, resolution_embed], dim=-1)
            )
        )
        return self.output_proj(hidden)


class StageContributionRouter(nn.Module):
    def __init__(
        self,
        resolutions,
        condition_dim,
        feature_dim,
        top_k,
        weight_floor,
        max_weight,
        weight_mode,
        device,
        dtype,
    ):
        super().__init__()
        self.resolutions = [int(resolution) for resolution in resolutions]
        self.top_k = int(top_k)
        self.weight_floor = float(weight_floor)
        self.max_weight = None if max_weight is None else float(max_weight)
        self.weight_mode = str(weight_mode).strip().lower()
        if self.weight_mode in {"prob", "probability", "weighted"}:
            self.weight_mode = "equal_selection"
        elif self.weight_mode in {"selection", "mask", "hard", "equal", "equal_selection"}:
            self.weight_mode = "equal_selection"
        else:
            raise ValueError(
                "stage router weight_mode must be 'equal_selection', "
                f"got {weight_mode!r}"
            )
        self.weight_floor = 0.0
        self.routers = nn.ModuleDict(
            {
                str(resolution): FactorizedRouterMLP(
                    condition_dim,
                    feature_dim,
                    1,
                    device,
                    dtype,
                )
                for resolution in self.resolutions
            }
        )

    def forward(self, condition, skip_maps):
        available = [
            resolution for resolution in self.resolutions if resolution in skip_maps
        ]
        if not available:
            return {}, {}
        raw_logits = torch.cat(
            [
                self.routers[str(resolution)](condition, skip_maps[resolution])
                for resolution in available
            ],
            dim=-1,
        )
        probabilities = torch.softmax(raw_logits, dim=-1)
        k = min(max(1, self.top_k), raw_logits.shape[-1])
        if k < raw_logits.shape[-1]:
            topk_indices = torch.topk(raw_logits, k=k, dim=-1).indices
            mask = torch.zeros_like(raw_logits, dtype=torch.bool)
            mask.scatter_(
                dim=-1,
                index=topk_indices,
                src=torch.ones_like(topk_indices, dtype=torch.bool),
            )
        else:
            mask = torch.ones_like(raw_logits, dtype=torch.bool)
        hard_weights = mask.to(dtype=probabilities.dtype) / float(k)
        # Sparse top-k routing for fusion: the router chooses which stages are
        # used, and all selected stages receive the same forward weight.
        weights = hard_weights + probabilities - probabilities.detach()
        weight_map = {
            resolution: weights[:, idx].view(-1, 1, 1, 1)
            for idx, resolution in enumerate(available)
        }
        stats = {
            "enabled": True,
            "top_k": int(k),
            "weight_mode": self.weight_mode,
            "resolutions": list(available),
            "logits": {
                resolution: raw_logits[:, idx].mean().detach()
                for idx, resolution in enumerate(available)
            },
            "probabilities": {
                resolution: probabilities[:, idx].mean().detach()
                for idx, resolution in enumerate(available)
            },
            "selected_fractions": {
                resolution: mask[:, idx].float().mean().detach()
                for idx, resolution in enumerate(available)
            },
            "weights": {
                resolution: weights[:, idx].mean().detach()
                for idx, resolution in enumerate(available)
            },
            "max_weight": weights.max().detach(),
            "min_weight": weights.min().detach(),
            "entropy": (
                -(probabilities * probabilities.clamp_min(1e-8).log())
                .sum(dim=-1)
                .mean()
                .detach()
            ),
        }
        return weight_map, stats


class IntegratedHierarchicalRoutingController(nn.Module):
    def __init__(
        self,
        resolutions,
        condition_dim,
        feature_dim,
        top_k,
        hidden_dim,
        use_channel_gate,
        channel_gate_scale,
        stage_select_mode,
        stage_select_threshold,
        stage_select_threshold_margin,
        stage_select_warmup_steps,
        stage_select_warmup_min_selected,
        stage_select_warmup_threshold_margin,
        stage_min_selected,
        stage_balance_mode,
        stage_use_scale_prior_context,
        device,
        dtype,
    ):
        super().__init__()
        self.resolutions = [int(resolution) for resolution in resolutions]
        self.resolution_to_index = {
            resolution: idx for idx, resolution in enumerate(self.resolutions)
        }
        self.top_k = int(top_k)
        self.feature_dim = int(feature_dim)
        self.use_channel_gate = bool(use_channel_gate)
        self.channel_gate_scale = float(channel_gate_scale)
        self.stage_select_mode = str(stage_select_mode).strip().lower()
        if self.stage_select_mode in {"topk", "top_k", "fixed_topk", "fixed"}:
            self.stage_select_mode = "topk"
        elif self.stage_select_mode in {
            "adaptive_topk",
            "adaptive_top_k",
            "variable_topk",
            "variable_top_k",
            "learned_k",
        }:
            self.stage_select_mode = "adaptive_topk"
        elif self.stage_select_mode in {
            "adaptive",
            "threshold",
            "sigmoid",
            "independent",
        }:
            self.stage_select_mode = "adaptive"
        elif self.stage_select_mode in {
            "relu",
            "relu_gate",
            "relu_routing",
            "relu_threshold",
            "ssr_relu",
            "remoe",
            "remoe_relu",
        }:
            self.stage_select_mode = "relu"
        else:
            raise ValueError(
                "integrated_controller_stage_select_mode must be 'topk', "
                f"'adaptive', 'adaptive_topk', or 'relu', got {stage_select_mode!r}"
            )
        threshold_spec = str(stage_select_threshold).strip().lower()
        self.stage_select_threshold_mode = "fixed"
        if threshold_spec in {
            "mean",
            "avg",
            "average",
            "prob_mean",
            "prob_avg",
            "probability_mean",
            "probability_avg",
            "stage_prob_mean",
            "stage_probability_mean",
            "running_prob_mean",
            "cumulative_prob_mean",
        }:
            self.stage_select_threshold = 0.0
            self.stage_select_threshold_mode = "cumulative_prob_mean"
        else:
            self.stage_select_threshold = float(stage_select_threshold)
        self.stage_select_threshold_margin = float(stage_select_threshold_margin)
        self.stage_select_warmup_steps = int(stage_select_warmup_steps)
        self.stage_select_warmup_min_selected = int(stage_select_warmup_min_selected)
        self.stage_select_warmup_threshold_margin = float(
            stage_select_warmup_threshold_margin
        )
        self.stage_min_selected = int(stage_min_selected)
        self.stage_balance_mode = str(stage_balance_mode).strip().lower()
        self.stage_use_scale_prior_context = bool(stage_use_scale_prior_context)
        if self.stage_balance_mode in {"", "none", "off", "false", "0"}:
            self.stage_balance_mode = "none"
        elif self.stage_balance_mode in {"batch", "balanced", "balance"}:
            self.stage_balance_mode = "batch"
        elif self.stage_balance_mode in {
            "batch_center",
            "batch_centered",
        }:
            self.stage_balance_mode = "batch_center"
        elif self.stage_balance_mode in {
            "center",
            "centered",
            "logit_center",
            "logit_centered",
        }:
            self.stage_balance_mode = "logit_center"
        else:
            raise ValueError(
                "integrated_controller_stage_balance_mode must be 'none', "
                f"'batch', 'batch_center', or 'logit_center', got {stage_balance_mode!r}"
            )
        self.max_resolution = max(self.resolutions) if self.resolutions else 1
        hidden_dim = (
            max(condition_dim, feature_dim, 128)
            if hidden_dim is None
            else int(hidden_dim)
        )
        if self.top_k <= 0:
            raise ValueError(
                f"integrated controller top_k must be positive, got {top_k}"
            )
        if (
            self.stage_select_mode != "relu"
            and self.stage_select_threshold_mode == "fixed"
            and not 0.0 < self.stage_select_threshold < 1.0
        ):
            raise ValueError(
                "integrated_controller_stage_select_threshold must be in (0, 1) "
                "or one of {'prob_mean', 'cumulative_prob_mean', 'mean'}, "
                f"got {stage_select_threshold}"
            )
        if not 0.0 <= self.stage_select_threshold_margin < 1.0:
            raise ValueError(
                "integrated_controller_stage_select_threshold_margin must be in "
                f"[0, 1), got {stage_select_threshold_margin}"
            )
        if self.stage_select_warmup_steps < 0:
            raise ValueError(
                "integrated_controller_stage_select_warmup_steps must be "
                f"non-negative, got {stage_select_warmup_steps}"
            )
        if self.stage_select_warmup_min_selected < 0:
            raise ValueError(
                "integrated_controller_stage_select_warmup_min_selected must be "
                f"non-negative, got {stage_select_warmup_min_selected}"
            )
        if not 0.0 <= self.stage_select_warmup_threshold_margin < 1.0:
            raise ValueError(
                "integrated_controller_stage_select_warmup_threshold_margin must "
                f"be in [0, 1), got {stage_select_warmup_threshold_margin}"
            )
        if self.stage_min_selected <= 0:
            raise ValueError(
                "integrated_controller_stage_min_selected must be positive, "
                f"got {stage_min_selected}"
            )
        self.register_buffer(
            "stage_probability_running_mean",
            torch.zeros(
                len(self.resolutions),
                device=device,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "stage_probability_running_count",
            torch.zeros((), device=device, dtype=torch.float32),
        )
        self.register_buffer(
            "stage_selection_warmup_updates",
            torch.zeros((), device=device, dtype=torch.float32),
        )

        self.condition_norm = nn.LayerNorm(
            condition_dim, eps=1e-6, device=device, dtype=dtype
        )
        self.summary_norm = nn.LayerNorm(
            feature_dim * 2, eps=1e-6, device=device, dtype=dtype
        )
        self.scale_prior_proj = nn.Linear(
            condition_dim,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        self.scale_prior_head = nn.Linear(
            hidden_dim,
            len(self.resolutions),
            device=device,
            dtype=dtype,
        )
        self.stage_feature_proj = nn.Linear(
            feature_dim * 2,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        self.stage_condition_proj = nn.Linear(
            condition_dim + len(self.resolutions) * 2,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        self.stage_score_head = nn.Linear(
            hidden_dim,
            1,
            device=device,
            dtype=dtype,
        )
        self.compression_proj = nn.Linear(
            condition_dim + feature_dim * 2 + len(self.resolutions) * 2 + 2,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        self.compression_stage_selection_proj = (
            nn.Linear(
                len(self.resolutions) * 2 + 1,
                hidden_dim,
                device=device,
                dtype=dtype,
            )
            if self.stage_select_mode == "relu"
            else None
        )
        self.compression_head = nn.Linear(
            hidden_dim,
            3,
            device=device,
            dtype=dtype,
        )
        self.encoder_depth_proj = nn.Linear(
            condition_dim + feature_dim * 2 + len(self.resolutions) * 2 + 2,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        self.encoder_depth_stage_selection_proj = (
            nn.Linear(
                len(self.resolutions) * 2 + 1,
                hidden_dim,
                device=device,
                dtype=dtype,
            )
            if self.stage_select_mode == "relu"
            else None
        )
        self.encoder_depth_head = nn.Linear(
            hidden_dim,
            3,
            device=device,
            dtype=dtype,
        )
        self.channel_gate_head = (
            nn.Linear(hidden_dim, feature_dim, device=device, dtype=dtype)
            if self.use_channel_gate
            else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scale_prior_proj.weight)
        nn.init.zeros_(self.scale_prior_proj.bias)
        nn.init.xavier_uniform_(self.stage_feature_proj.weight)
        nn.init.zeros_(self.stage_feature_proj.bias)
        nn.init.xavier_uniform_(self.stage_condition_proj.weight)
        nn.init.zeros_(self.stage_condition_proj.bias)
        nn.init.xavier_uniform_(self.compression_proj.weight)
        nn.init.zeros_(self.compression_proj.bias)
        if self.compression_stage_selection_proj is not None:
            nn.init.xavier_uniform_(self.compression_stage_selection_proj.weight)
            nn.init.zeros_(self.compression_stage_selection_proj.bias)
        nn.init.xavier_uniform_(self.encoder_depth_proj.weight)
        nn.init.zeros_(self.encoder_depth_proj.bias)
        if self.encoder_depth_stage_selection_proj is not None:
            nn.init.xavier_uniform_(self.encoder_depth_stage_selection_proj.weight)
            nn.init.zeros_(self.encoder_depth_stage_selection_proj.bias)
        nn.init.normal_(self.scale_prior_head.weight, std=1e-3)
        nn.init.zeros_(self.scale_prior_head.bias)
        nn.init.normal_(self.stage_score_head.weight, std=1e-3)
        nn.init.zeros_(self.stage_score_head.bias)
        nn.init.normal_(self.compression_head.weight, std=1e-3)
        nn.init.normal_(self.encoder_depth_head.weight, std=1e-3)
        with torch.no_grad():
            self.compression_head.bias.copy_(
                torch.tensor(
                    [-2.0, 2.0, -2.0],
                    device=self.compression_head.bias.device,
                    dtype=self.compression_head.bias.dtype,
                )
            )
            self.encoder_depth_head.bias.copy_(
                torch.tensor(
                    [-2.0, 2.0, -2.0],
                    device=self.encoder_depth_head.bias.device,
                    dtype=self.encoder_depth_head.bias.dtype,
                )
            )
        if self.channel_gate_head is not None:
            nn.init.normal_(self.channel_gate_head.weight, std=1e-3)
            nn.init.zeros_(self.channel_gate_head.bias)

    @staticmethod
    def _straight_through_topk(logits, top_k):
        probabilities = torch.softmax(logits, dim=-1)
        k = min(max(1, int(top_k)), logits.shape[-1])
        if k < logits.shape[-1]:
            topk_indices = torch.topk(logits, k=k, dim=-1).indices
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(
                dim=-1,
                index=topk_indices,
                src=torch.ones_like(topk_indices, dtype=torch.bool),
            )
        else:
            mask = torch.ones_like(logits, dtype=torch.bool)
        hard_weights = mask.to(dtype=probabilities.dtype) / float(k)
        weights = hard_weights + probabilities - probabilities.detach()
        return weights, probabilities, mask, k

    def _balance_selection_mask(self, logits, mask):
        if self.stage_balance_mode != "batch" or logits.shape[0] <= 1:
            return mask
        batch_size, stage_count = logits.shape
        selected_counts = mask.sum(dim=-1).clamp_min(1).to(torch.long)
        total_slots = int(selected_counts.sum().item())
        if total_slots <= 0 or stage_count <= 1:
            return mask

        detached_logits = logits.detach()
        mean_logits = detached_logits.mean(dim=0)
        stage_priority = torch.argsort(mean_logits, descending=True)
        quotas = torch.full(
            (stage_count,),
            total_slots // stage_count,
            dtype=torch.long,
            device=logits.device,
        )
        remainder = total_slots % stage_count
        if remainder > 0:
            quotas[stage_priority[:remainder]] += 1

        balanced = torch.zeros_like(mask)
        remaining_per_sample = selected_counts.clone()
        remaining_per_stage = quotas.clone()

        for stage_idx in stage_priority.tolist():
            if int(remaining_per_stage[stage_idx].item()) <= 0:
                continue
            sample_priority = torch.argsort(
                detached_logits[:, stage_idx], descending=True
            )
            for sample_idx in sample_priority.tolist():
                if int(remaining_per_stage[stage_idx].item()) <= 0:
                    break
                if int(remaining_per_sample[sample_idx].item()) <= 0:
                    continue
                if bool(balanced[sample_idx, stage_idx].item()):
                    continue
                balanced[sample_idx, stage_idx] = True
                remaining_per_sample[sample_idx] -= 1
                remaining_per_stage[stage_idx] -= 1

        for sample_idx in range(batch_size):
            if int(remaining_per_sample[sample_idx].item()) <= 0:
                continue
            stage_priority_for_sample = torch.argsort(
                detached_logits[sample_idx], descending=True
            )
            for stage_idx in stage_priority_for_sample.tolist():
                if int(remaining_per_sample[sample_idx].item()) <= 0:
                    break
                if bool(balanced[sample_idx, stage_idx].item()):
                    continue
                balanced[sample_idx, stage_idx] = True
                remaining_per_sample[sample_idx] -= 1

        return balanced

    def _selection_logits(self, logits):
        if self.stage_balance_mode == "batch_center" and logits.shape[0] > 1:
            stage_mean = logits.detach().mean(dim=0, keepdim=True)
            global_mean = stage_mean.mean(dim=-1, keepdim=True)
            return logits - stage_mean + global_mean
        if self.stage_balance_mode == "logit_center":
            return logits - logits.mean(dim=-1, keepdim=True)
        return logits

    def _running_probability_buffers(self):
        return (
            self.stage_probability_running_mean,
            self.stage_probability_running_count,
        )

    def _stage_warmup_progress(self):
        if self.stage_select_warmup_steps <= 0:
            return 1.0, 0.0
        progress = min(
            1.0,
            float(self.stage_selection_warmup_updates.item())
            / float(max(1, self.stage_select_warmup_steps)),
        )
        return progress, 1.0 - progress

    def _effective_threshold_margin(self):
        _, remaining = self._stage_warmup_progress()
        if remaining <= 0.0:
            return self.stage_select_threshold_margin
        return (
            self.stage_select_threshold_margin
            + remaining
            * (
                self.stage_select_warmup_threshold_margin
                - self.stage_select_threshold_margin
            )
        )

    def _effective_min_selected(self, min_selected):
        progress, _ = self._stage_warmup_progress()
        effective = int(min_selected)
        if progress < 1.0 and self.stage_select_warmup_min_selected > 0:
            effective = max(effective, self.stage_select_warmup_min_selected)
        return effective

    def _selection_threshold(self, probabilities, threshold_scope, update_running):
        if (
            self.stage_select_threshold_mode == "cumulative_prob_mean"
            and threshold_scope == "stage"
        ):
            batch_sum = probabilities.detach().float().sum(dim=0)
            batch_count = probabilities.new_tensor(
                float(probabilities.shape[0]),
                dtype=torch.float32,
            )
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                torch.distributed.all_reduce(
                    batch_sum,
                    op=torch.distributed.ReduceOp.SUM,
                )
                torch.distributed.all_reduce(
                    batch_count,
                    op=torch.distributed.ReduceOp.SUM,
                )
            batch_mean = batch_sum / batch_count.clamp_min(1.0)
            running_mean, running_count = self._running_probability_buffers()
            with torch.no_grad():
                if update_running:
                    total_count = running_count + batch_count.to(running_count)
                    updated_mean = (
                        running_mean * running_count
                        + batch_sum.to(running_mean)
                    ) / total_count.clamp_min(1.0)
                    running_mean.copy_(updated_mean)
                    running_count.copy_(total_count)
                    threshold_values = updated_mean
                elif float(running_count.item()) > 0.0:
                    threshold_values = running_mean
                else:
                    threshold_values = batch_mean.to(running_mean)
                threshold_margin = self._effective_threshold_margin()
                if threshold_margin > 0.0:
                    threshold_values = (
                        threshold_values - threshold_margin
                    ).clamp_min(0.0)
                if update_running and self.stage_select_warmup_steps > 0:
                    self.stage_selection_warmup_updates.add_(1.0)
            return threshold_values.to(dtype=probabilities.dtype).view(1, -1)
        fixed_threshold = (
            0.5
            if self.stage_select_threshold_mode == "cumulative_prob_mean"
            else float(self.stage_select_threshold)
        )
        return probabilities.new_full(
            (probabilities.shape[0], 1),
            fixed_threshold,
        )

    def _selection_scores(
        self,
        selection_logits,
        threshold,
        threshold_scope,
    ):
        if (
            self.stage_select_threshold_mode != "cumulative_prob_mean"
            or threshold_scope != "stage"
        ):
            return selection_logits
        threshold_logits = torch.logit(
            threshold.detach().float().clamp(1e-4, 1.0 - 1e-4)
        ).to(dtype=selection_logits.dtype)
        return selection_logits - threshold_logits

    def _relu_selection_scores(self, selection_logits, threshold, threshold_scope):
        if (
            self.stage_select_threshold_mode == "cumulative_prob_mean"
            and threshold_scope == "stage"
        ):
            return self._selection_scores(
                selection_logits,
                threshold,
                threshold_scope,
            )
        threshold_values = threshold.to(dtype=selection_logits.dtype)
        if threshold_values.shape[-1] == 1:
            threshold_values = threshold_values.expand_as(selection_logits)
        if self.stage_select_threshold_mode == "fixed":
            threshold_values = torch.where(
                (threshold_values > 0.0) & (threshold_values < 1.0),
                torch.logit(
                    threshold_values.float().clamp(1e-4, 1.0 - 1e-4)
                ).to(dtype=selection_logits.dtype),
                threshold_values,
            )
        return selection_logits - threshold_values

    def _straight_through_adaptive(self, logits, min_selected, threshold_scope):
        selection_logits = self._selection_logits(logits)
        raw_probabilities = torch.sigmoid(selection_logits)
        threshold = self._selection_threshold(
            raw_probabilities,
            threshold_scope,
            update_running=True,
        )
        probabilities = raw_probabilities
        selection_scores = self._selection_scores(
            selection_logits,
            threshold,
            threshold_scope,
        )
        mask = probabilities > threshold
        min_selected = self._effective_min_selected(min_selected)
        min_selected = min(max(1, int(min_selected)), logits.shape[-1])
        selected_counts = mask.sum(dim=-1, keepdim=True)
        if min_selected > 0:
            fallback_indices = torch.topk(
                selection_scores,
                k=min_selected,
                dim=-1,
            ).indices
            fallback_mask = torch.zeros_like(mask)
            fallback_mask.scatter_(
                dim=-1,
                index=fallback_indices,
                src=torch.ones_like(fallback_indices, dtype=torch.bool),
            )
            mask = torch.where(selected_counts < min_selected, fallback_mask, mask)
            selected_counts = mask.sum(dim=-1, keepdim=True)
        mask = self._balance_selection_mask(logits, mask)
        selected_counts = mask.sum(dim=-1, keepdim=True)
        selected_counts = selected_counts.clamp_min(1)
        hard_weights = mask.to(dtype=probabilities.dtype) / selected_counts.to(
            dtype=probabilities.dtype
        )
        soft_denominator = probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        soft_weights = probabilities / soft_denominator
        weights = hard_weights + soft_weights - soft_weights.detach()
        return (
            weights,
            probabilities,
            raw_probabilities,
            mask,
            selected_counts,
            threshold,
        )

    def _straight_through_adaptive_topk(
        self,
        logits,
        min_selected,
        max_selected,
        threshold_scope,
    ):
        selection_logits = self._selection_logits(logits)
        raw_probabilities = torch.sigmoid(selection_logits)
        stage_count = selection_logits.shape[-1]
        min_selected = self._effective_min_selected(min_selected)
        min_selected = min(max(1, int(min_selected)), stage_count)
        max_selected = min(max(min_selected, int(max_selected)), stage_count)
        threshold = self._selection_threshold(
            raw_probabilities,
            threshold_scope,
            update_running=True,
        )
        probabilities = raw_probabilities
        selection_scores = self._selection_scores(
            selection_logits,
            threshold,
            threshold_scope,
        )
        sorted_indices = torch.argsort(selection_scores, dim=-1, descending=True)
        ranks = torch.empty_like(sorted_indices)
        rank_values = torch.arange(
            stage_count,
            device=logits.device,
        ).view(1, -1).expand_as(sorted_indices)
        ranks.scatter_(dim=-1, index=sorted_indices, src=rank_values)

        count_mask = probabilities > threshold
        selected_counts = count_mask.sum(
            dim=-1,
            keepdim=True,
        )

        if (
            self.stage_select_threshold_mode == "cumulative_prob_mean"
            and threshold_scope == "stage"
        ):
            mask = count_mask
            if min_selected > 0:
                fallback_mask = ranks < min_selected
                mask = torch.where(
                    selected_counts < min_selected,
                    fallback_mask,
                    mask,
                )
                selected_counts = mask.sum(dim=-1, keepdim=True)
            if max_selected < stage_count:
                topk_mask = ranks < max_selected
                mask = torch.where(
                    selected_counts > max_selected,
                    topk_mask,
                    mask,
                )
            mask = self._balance_selection_mask(logits, mask)
            selected_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1)
            hard_weights = mask.to(dtype=probabilities.dtype) / selected_counts.to(
                dtype=probabilities.dtype
            )
            soft_denominator = probabilities.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)
            soft_weights = probabilities / soft_denominator
            weights = hard_weights + soft_weights - soft_weights.detach()
            return (
                weights,
                probabilities,
                raw_probabilities,
                mask,
                selected_counts,
                threshold,
            )

        selected_counts = selected_counts.clamp(
            min=min_selected,
            max=max_selected,
        )
        mask = ranks < selected_counts
        mask = self._balance_selection_mask(logits, mask)
        selected_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        hard_weights = mask.to(dtype=probabilities.dtype) / selected_counts.to(
            dtype=probabilities.dtype
        )
        soft_denominator = probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        soft_weights = probabilities / soft_denominator
        weights = hard_weights + soft_weights - soft_weights.detach()
        return (
            weights,
            probabilities,
            raw_probabilities,
            mask,
            selected_counts,
            threshold,
        )

    def _straight_through_relu(self, logits, min_selected, threshold_scope):
        selection_logits = self._selection_logits(logits)
        raw_probabilities = torch.sigmoid(selection_logits)
        threshold = self._selection_threshold(
            raw_probabilities,
            threshold_scope,
            update_running=True,
        )
        selection_scores = self._relu_selection_scores(
            selection_logits,
            threshold,
            threshold_scope,
        )
        relu_scores = F.relu(selection_scores)
        mask = relu_scores > 0
        stage_count = logits.shape[-1]
        min_selected = self._effective_min_selected(min_selected)
        min_selected = min(max(1, int(min_selected)), stage_count)
        selected_counts = mask.sum(dim=-1, keepdim=True)
        if min_selected > 0:
            fallback_indices = torch.topk(
                selection_scores,
                k=min_selected,
                dim=-1,
            ).indices
            fallback_mask = torch.zeros_like(mask)
            fallback_mask.scatter_(
                dim=-1,
                index=fallback_indices,
                src=torch.ones_like(fallback_indices, dtype=torch.bool),
            )
            mask = torch.where(selected_counts < min_selected, fallback_mask, mask)
        mask = self._balance_selection_mask(logits, mask)
        selected_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1)

        selected_relu_scores = torch.where(
            mask,
            relu_scores,
            torch.zeros_like(relu_scores),
        )
        relu_sum = selected_relu_scores.sum(dim=-1, keepdim=True)
        positive_selected = (selected_relu_scores > 0).sum(
            dim=-1,
            keepdim=True,
        )
        equal_weights = mask.to(dtype=relu_scores.dtype) / selected_counts.to(
            dtype=relu_scores.dtype
        )
        relu_weights = selected_relu_scores / relu_sum.clamp_min(1e-6)
        use_relu_weights = (relu_sum > 0) & (positive_selected == selected_counts)
        hard_weights = torch.where(use_relu_weights, relu_weights, equal_weights)

        soft_relu_sum = relu_scores.sum(dim=-1, keepdim=True)
        soft_relu_weights = relu_scores / soft_relu_sum.clamp_min(1e-6)
        softmax_weights = torch.softmax(selection_logits, dim=-1)
        soft_weights = torch.where(
            soft_relu_sum > 0,
            soft_relu_weights,
            softmax_weights,
        )
        weights = hard_weights + soft_weights - soft_weights.detach()
        probabilities = soft_weights
        return (
            weights,
            probabilities,
            raw_probabilities,
            mask,
            selected_counts,
            threshold,
            relu_scores,
        )

    def _select_stage_weights(self, logits, threshold_scope="stage"):
        if self.stage_select_mode in {"adaptive", "adaptive_topk", "relu"}:
            if self.stage_select_mode == "adaptive_topk":
                (
                    weights,
                    probabilities,
                    raw_probabilities,
                    mask,
                    selected_counts,
                    threshold,
                ) = (
                    self._straight_through_adaptive_topk(
                        logits,
                        self.stage_min_selected,
                        self.top_k,
                        threshold_scope,
                    )
                )
                top_k = min(max(self.stage_min_selected, self.top_k), logits.shape[-1])
                activations = None
            elif self.stage_select_mode == "relu":
                (
                    weights,
                    probabilities,
                    raw_probabilities,
                    mask,
                    selected_counts,
                    threshold,
                    activations,
                ) = (
                    self._straight_through_relu(
                        logits,
                        self.stage_min_selected,
                        threshold_scope,
                    )
                )
                top_k = 0
            else:
                (
                    weights,
                    probabilities,
                    raw_probabilities,
                    mask,
                    selected_counts,
                    threshold,
                ) = (
                    self._straight_through_adaptive(
                        logits,
                        self.stage_min_selected,
                        threshold_scope,
                    )
                )
                top_k = 0
                activations = None
            if self.stage_select_mode == "relu":
                entropy = (
                    -(probabilities * probabilities.clamp_min(1e-8).log())
                    .sum(dim=-1)
                    .mean()
                    .detach()
                )
            else:
                entropy = (
                    -(
                        probabilities * probabilities.clamp_min(1e-8).log()
                        + (1.0 - probabilities)
                        * (1.0 - probabilities).clamp_min(1e-8).log()
                    )
                    .sum(dim=-1)
                    .mean()
                    .detach()
                )
            threshold_mean = threshold.float().mean().detach()
            threshold_values = (
                threshold.expand_as(probabilities).float().mean(dim=0).detach()
            )
            selection = {
                "weights": weights,
                "probabilities": probabilities,
                "raw_probabilities": raw_probabilities,
                "mask": mask,
                "top_k": int(top_k),
                "threshold_mean": threshold_mean,
                "threshold_values": threshold_values,
                "selected_count_mean": selected_counts.float().mean().detach(),
                "selected_count_min": selected_counts.float().amin().detach(),
                "selected_count_max": selected_counts.float().amax().detach(),
                "entropy": entropy,
            }
            if activations is not None:
                selection["activations"] = activations
            return selection
        weights, probabilities, mask, k = self._straight_through_topk(
            logits,
            self.top_k,
        )
        mask = self._balance_selection_mask(logits, mask)
        selected_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        hard_weights = mask.to(dtype=probabilities.dtype) / selected_counts.to(
            dtype=probabilities.dtype
        )
        weights = hard_weights + probabilities - probabilities.detach()
        entropy = (
            -(probabilities * probabilities.clamp_min(1e-8).log())
            .sum(dim=-1)
            .mean()
            .detach()
        )
        return {
            "weights": weights,
            "probabilities": probabilities,
            "raw_probabilities": probabilities,
            "mask": mask,
            "top_k": int(k),
            "threshold_mean": logits.new_tensor(float(self.stage_select_threshold)),
            "threshold_values": logits.new_full(
                (logits.shape[-1],),
                float(self.stage_select_threshold),
            ),
            "selected_count_mean": selected_counts.float().mean().detach(),
            "selected_count_min": selected_counts.float().amin().detach(),
            "selected_count_max": selected_counts.float().amax().detach(),
            "entropy": entropy,
        }

    def _summarize_map(self, stage_map):
        summary_mean = stage_map.mean(dim=(1, 2))
        summary_std = stage_map.float().std(dim=(1, 2), unbiased=False).to(stage_map)
        return self.summary_norm(torch.cat([summary_mean, summary_std], dim=-1))

    def compute_scale_prior(self, condition):
        condition_summary = self.condition_norm(condition)
        hidden = F.gelu(self.scale_prior_proj(condition_summary))
        logits = self.scale_prior_head(hidden)
        selection = self._select_stage_weights(logits, threshold_scope="prior")
        weights = selection["weights"]
        probabilities = selection["probabilities"]
        mask = selection["mask"]
        activations = selection.get("activations")
        stats = {
            "top_k": int(selection["top_k"]),
            "stage_select_mode": self.stage_select_mode,
            "stage_balance_mode": self.stage_balance_mode,
            "stage_select_threshold": float(self.stage_select_threshold),
            "stage_select_threshold_mode": self.stage_select_threshold_mode,
            "stage_select_threshold_margin": float(
                self.stage_select_threshold_margin
            ),
            "stage_select_warmup_steps": int(self.stage_select_warmup_steps),
            "stage_select_warmup_min_selected": int(
                self.stage_select_warmup_min_selected
            ),
            "stage_select_warmup_threshold_margin": float(
                self.stage_select_warmup_threshold_margin
            ),
            "stage_select_warmup_updates": (
                self.stage_selection_warmup_updates.detach().clone()
            ),
            "stage_select_threshold_mean": selection["threshold_mean"],
            "stage_min_selected": int(self.stage_min_selected),
            "selected_count_mean": selection["selected_count_mean"],
            "selected_count_min": selection["selected_count_min"],
            "selected_count_max": selection["selected_count_max"],
            "resolutions": list(self.resolutions),
            "prior_logits": {
                resolution: logits[:, idx].mean().detach()
                for idx, resolution in enumerate(self.resolutions)
            },
            "prior_probabilities": {
                resolution: probabilities[:, idx].mean().detach()
                for idx, resolution in enumerate(self.resolutions)
            },
            "prior_thresholds": {
                resolution: selection["threshold_values"][idx]
                for idx, resolution in enumerate(self.resolutions)
            },
            "prior_selected_fractions": {
                resolution: mask[:, idx].float().mean().detach()
                for idx, resolution in enumerate(self.resolutions)
            },
            "prior_weights": {
                resolution: weights[:, idx].mean().detach()
                for idx, resolution in enumerate(self.resolutions)
            },
            "prior_entropy": selection["entropy"],
        }
        if activations is not None:
            stats["prior_activations"] = {
                resolution: activations[:, idx].mean().detach()
                for idx, resolution in enumerate(self.resolutions)
            }
        return {
            "logits": logits,
            "weights": weights,
            "probabilities": probabilities,
            "mask": mask,
            "activations": activations,
            "top_k": selection["top_k"],
            "stage_select_mode": self.stage_select_mode,
            "stats": stats,
        }

    def _stage_selection_context(self, scale_prior):
        selected_mask = scale_prior["mask"].to(dtype=scale_prior["weights"].dtype)
        selected_count = selected_mask.sum(dim=-1, keepdim=True) / float(
            max(1, len(self.resolutions))
        )
        return torch.cat(
            [
                scale_prior["weights"],
                selected_mask,
                selected_count,
            ],
            dim=-1,
        )

    def compute_compression(
        self,
        condition,
        context_map,
        output_resolution,
        encoder_depth,
        scale_prior,
    ):
        if scale_prior is None:
            scale_prior = self.compute_scale_prior(condition)
        condition_summary = self.condition_norm(condition)
        map_summary = self._summarize_map(context_map)
        batch_size = condition.shape[0]
        resolution_feature = condition.new_full(
            (batch_size, 1),
            float(output_resolution) / float(max(1, self.max_resolution)),
        )
        depth_feature = condition.new_full(
            (batch_size, 1),
            float(encoder_depth) / 16.0,
        )
        controller_input = torch.cat(
            [
                condition_summary,
                map_summary,
                scale_prior["weights"],
                scale_prior["probabilities"],
                resolution_feature,
                depth_feature,
            ],
            dim=-1,
        )
        hidden_pre_activation = self.compression_proj(controller_input)
        uses_stage_selection_context = self.stage_select_mode == "relu"
        if uses_stage_selection_context:
            stage_selection_context = self._stage_selection_context(scale_prior).to(
                dtype=hidden_pre_activation.dtype
            )
            hidden_pre_activation = (
                hidden_pre_activation
                + self.compression_stage_selection_proj(stage_selection_context)
            )
        hidden = F.gelu(hidden_pre_activation)
        logits = self.compression_head(hidden)
        channel_gate = None
        gate_stats = {}
        if self.channel_gate_head is not None:
            channel_delta = torch.tanh(self.channel_gate_head(hidden))
            channel_gate = 1.0 + self.channel_gate_scale * channel_delta
            gate_stats = {
                "integrated_channel_gate_mean": channel_gate.mean().detach(),
                "integrated_channel_gate_std": channel_gate.std(unbiased=False).detach(),
            }
        stats = {
            "uses_integrated_controller": 1.0,
            "integrated_output_resolution": float(output_resolution),
            "integrated_encoder_depth": float(encoder_depth),
            "integrated_scale_prior_entropy": scale_prior["stats"]["prior_entropy"],
            "integrated_uses_stage_selection_context": float(
                uses_stage_selection_context
            ),
            "integrated_scale_prior_selected_count": (
                scale_prior["mask"].float().sum(dim=-1).mean().detach()
            ),
        }
        stats.update(gate_stats)
        for resolution in self.resolutions:
            idx = self.resolution_to_index[resolution]
            stats[f"integrated_scale_{resolution}_prior_weight"] = (
                scale_prior["weights"][:, idx].mean().detach()
            )
            stats[f"integrated_scale_{resolution}_prior_prob"] = (
                scale_prior["probabilities"][:, idx].mean().detach()
            )
            stats[f"integrated_scale_{resolution}_prior_selected"] = (
                scale_prior["mask"][:, idx].float().mean().detach()
            )
        return {
            "logits": logits,
            "channel_gate": channel_gate,
            "stats": stats,
        }

    def compute_encoder_depth(
        self,
        condition,
        context_map,
        resolution,
        base_depth,
        scale_prior,
    ):
        if scale_prior is None:
            scale_prior = self.compute_scale_prior(condition)
        condition_summary = self.condition_norm(condition)
        map_summary = self._summarize_map(context_map)
        batch_size = condition.shape[0]
        resolution_feature = condition.new_full(
            (batch_size, 1),
            float(resolution) / float(max(1, self.max_resolution)),
        )
        depth_feature = condition.new_full(
            (batch_size, 1),
            float(base_depth) / 16.0,
        )
        controller_input = torch.cat(
            [
                condition_summary,
                map_summary,
                scale_prior["weights"],
                scale_prior["probabilities"],
                resolution_feature,
                depth_feature,
            ],
            dim=-1,
        )
        hidden_pre_activation = self.encoder_depth_proj(controller_input)
        uses_stage_selection_context = self.stage_select_mode == "relu"
        if uses_stage_selection_context:
            stage_selection_context = self._stage_selection_context(scale_prior).to(
                dtype=hidden_pre_activation.dtype
            )
            hidden_pre_activation = (
                hidden_pre_activation
                + self.encoder_depth_stage_selection_proj(stage_selection_context)
            )
        hidden = F.gelu(hidden_pre_activation)
        logits = self.encoder_depth_head(hidden)
        stats = {
            "uses_integrated_controller": 1.0,
            "integrated_depth_resolution": float(resolution),
            "integrated_depth_base_depth": float(base_depth),
            "integrated_depth_scale_prior_entropy": scale_prior["stats"]["prior_entropy"],
            "integrated_depth_uses_stage_selection_context": float(
                uses_stage_selection_context
            ),
            "integrated_depth_scale_prior_selected_count": (
                scale_prior["mask"].float().sum(dim=-1).mean().detach()
            ),
        }
        for stage_resolution in self.resolutions:
            idx = self.resolution_to_index[stage_resolution]
            stats[f"integrated_depth_scale_{stage_resolution}_prior_weight"] = (
                scale_prior["weights"][:, idx].mean().detach()
            )
            stats[f"integrated_depth_scale_{stage_resolution}_prior_prob"] = (
                scale_prior["probabilities"][:, idx].mean().detach()
            )
            stats[f"integrated_depth_scale_{stage_resolution}_prior_selected"] = (
                scale_prior["mask"][:, idx].float().mean().detach()
            )
        return {"logits": logits, "stats": stats}

    def compute_stage_weights(self, condition, skip_maps, scale_prior):
        available = [
            resolution for resolution in self.resolutions if resolution in skip_maps
        ]
        if not available:
            return {}, {}
        if scale_prior is None:
            scale_prior = self.compute_scale_prior(condition)
        condition_summary = self.condition_norm(condition)
        prior_weights = scale_prior["weights"]
        prior_probabilities = scale_prior["probabilities"]
        if not self.stage_use_scale_prior_context:
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
        stage_condition_hidden = self.stage_condition_proj(stage_condition)
        raw_logits = []
        for resolution in available:
            feature_hidden = self.stage_feature_proj(
                self._summarize_map(skip_maps[resolution])
            )
            feature_logit = self.stage_score_head(
                F.gelu(stage_condition_hidden + feature_hidden)
            )
            if self.stage_use_scale_prior_context:
                idx = self.resolution_to_index[resolution]
                feature_logit = feature_logit + scale_prior["logits"][:, idx : idx + 1]
            raw_logits.append(feature_logit)
        raw_logits = torch.cat(raw_logits, dim=-1)
        selection = self._select_stage_weights(raw_logits, threshold_scope="stage")
        weights = selection["weights"]
        probabilities = selection["probabilities"]
        raw_probabilities = selection["raw_probabilities"]
        activations = selection.get("activations")
        mask = selection["mask"]
        weight_map = {
            resolution: weights[:, idx].view(-1, 1, 1, 1)
            for idx, resolution in enumerate(available)
        }
        stats = {
            "enabled": True,
            "uses_integrated_controller": 1.0,
            "top_k": int(selection["top_k"]),
            "stage_select_mode": self.stage_select_mode,
            "stage_balance_mode": self.stage_balance_mode,
            "stage_use_scale_prior_context": self.stage_use_scale_prior_context,
            "stage_select_threshold": float(self.stage_select_threshold),
            "stage_select_threshold_mode": self.stage_select_threshold_mode,
            "stage_select_threshold_margin": float(
                self.stage_select_threshold_margin
            ),
            "stage_select_threshold_mean": selection["threshold_mean"],
            "stage_min_selected": int(self.stage_min_selected),
            "selected_count_mean": selection["selected_count_mean"],
            "selected_count_min": selection["selected_count_min"],
            "selected_count_max": selection["selected_count_max"],
            "weight_mode": "equal_selection",
            "resolutions": list(available),
            "logits": {
                resolution: raw_logits[:, idx].mean().detach()
                for idx, resolution in enumerate(available)
            },
            "probabilities": {
                resolution: probabilities[:, idx].mean().detach()
                for idx, resolution in enumerate(available)
            },
            "raw_probabilities": {
                resolution: raw_probabilities[:, idx].mean().detach()
                for idx, resolution in enumerate(available)
            },
            "thresholds": {
                resolution: selection["threshold_values"][idx]
                for idx, resolution in enumerate(available)
            },
            "selected_fractions": {
                resolution: mask[:, idx].float().mean().detach()
                for idx, resolution in enumerate(available)
            },
            "weights": {
                resolution: weights[:, idx].mean().detach()
                for idx, resolution in enumerate(available)
            },
            "max_weight": weights.max().detach(),
            "min_weight": weights.min().detach(),
            "entropy": selection["entropy"],
            "prior_entropy": scale_prior["stats"]["prior_entropy"],
            "prior_probabilities": scale_prior["stats"]["prior_probabilities"],
            "prior_selected_fractions": scale_prior["stats"][
                "prior_selected_fractions"
            ],
            "prior_weights": scale_prior["stats"]["prior_weights"],
        }
        if activations is not None:
            stats["activations"] = {
                resolution: activations[:, idx].mean().detach()
                for idx, resolution in enumerate(available)
            }
        return weight_map, stats


class RoutedMambaSizeProcessor(nn.Module):
    def __init__(
        self,
        dim,
        resolution,
        preset_specs,
        has_text,
        ssm_cfg,
        norm_epsilon,
        rms_norm,
        residual_in_fp32,
        fused_add_norm,
        drop_path_rate,
        scan_type,
        processor_scan_type,
        processor_window_size,
        processor_shift_size,
        router_top_k,
        router_weight_mode,
        use_jit,
        use_checkpoint,
        device,
        dtype,
    ):
        super().__init__()
        self.dim = int(dim)
        self.resolution = int(resolution)
        self.preset_names = []
        self.router_top_k = int(router_top_k)
        self.router_weight_mode = str(router_weight_mode).strip().lower()
        if self.router_weight_mode in {"prob", "probability", "weighted"}:
            self.router_weight_mode = "selection"
        elif self.router_weight_mode in {"selection", "mask", "hard"}:
            self.router_weight_mode = "selection"
        else:
            raise ValueError(
                "mamba size router weight_mode must be 'selection', "
                f"got {router_weight_mode!r}"
            )
        self.routers = FactorizedRouterMLP(
            dim,
            dim,
            len(preset_specs),
            device,
            dtype,
        )
        self.experts = nn.ModuleList()
        self.input_projs = nn.ModuleList()
        self.output_projs = nn.ModuleList()
        self.condition_projs = nn.ModuleList()
        self.text_projs = nn.ModuleList()
        processor_cls = (
            MapWindowMambaResidualProcessor
            if int(processor_window_size) > 0
            else MapMambaResidualProcessor
        )
        for name, depth, dim_multiplier in preset_specs:
            internal_dim = round_dim_to_multiple(dim * dim_multiplier)
            self.preset_names.append(str(name))
            self.input_projs.append(
                nn.Identity()
                if internal_dim == dim
                else nn.Linear(dim, internal_dim, device=device, dtype=dtype)
            )
            self.condition_projs.append(
                nn.Identity()
                if internal_dim == dim
                else nn.Linear(dim, internal_dim, device=device, dtype=dtype)
            )
            self.text_projs.append(
                nn.Identity()
                if internal_dim == dim
                else nn.Linear(dim, internal_dim, device=device, dtype=dtype)
            )
            processor_kwargs = dict(
                dim=internal_dim,
                depth=depth,
                resolution=resolution,
                has_text=has_text,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                drop_path_values=torch.linspace(0, drop_path_rate, depth).tolist(),
                scan_type=processor_scan_type if processor_scan_type else scan_type,
                use_jit=use_jit,
                use_checkpoint=use_checkpoint,
                device=device,
                dtype=dtype,
            )
            if processor_cls is MapWindowMambaResidualProcessor:
                processor_kwargs.update(
                    window_size=int(processor_window_size),
                    shift_size=int(processor_shift_size),
                )
            self.experts.append(processor_cls(**processor_kwargs))
            self.output_projs.append(
                nn.Identity()
                if internal_dim == dim
                else nn.Linear(internal_dim, dim, device=device, dtype=dtype)
            )
        if "base" in self.preset_names:
            base_idx = self.preset_names.index("base")
            with torch.no_grad():
                self.routers.output_proj.bias.fill_(-2.0)
                self.routers.output_proj.bias[base_idx] = 2.0

    def forward(self, context_map, c, text=None):
        logits = self.routers(c, context_map)
        probabilities = torch.softmax(logits, dim=-1)
        k = 1
        selected_indices = torch.argmax(logits, dim=-1)
        mask = F.one_hot(selected_indices, num_classes=probabilities.shape[-1]).to(
            dtype=torch.bool
        )
        hard_weights = mask.to(dtype=probabilities.dtype)
        # Sparse top-1 routing: only the selected preset expert is executed for
        # each sample. The straight-through weight keeps router gradients while
        # the forward pass remains a hard one-preset selection.
        router_weights = hard_weights + probabilities - probabilities.detach()
        output = context_map.new_zeros(context_map.shape)
        expert_stats = {}
        for idx, (name, input_proj, condition_proj, text_proj, expert, output_proj) in enumerate(
            zip(
                self.preset_names,
                self.input_projs,
                self.condition_projs,
                self.text_projs,
                self.experts,
                self.output_projs,
            )
        ):
            selected = selected_indices == idx
            if not selected.any():
                expert_stats[name] = {
                    "weight": router_weights[:, idx].mean().detach(),
                    "probability": probabilities[:, idx].mean().detach(),
                    "selected_fraction": selected.float().mean().detach(),
                    "output_norm": context_map.new_tensor(0.0).detach(),
                    "depth": float(len(expert.blocks)),
                    "dim": float(expert.dim),
                }
                continue
            expert_input = input_proj(context_map[selected])
            expert_condition = condition_proj(c[selected])
            expert_text = text_proj(text[selected]) if text is not None else None
            expert_output, stats = expert(
                expert_input,
                expert_condition,
                text=expert_text,
            )
            expert_output = output_proj(expert_output)
            weight = router_weights[selected, idx].view(-1, 1, 1, 1)
            output[selected] = expert_output * weight
            expert_stats[name] = {
                "weight": router_weights[:, idx].mean().detach(),
                "probability": probabilities[:, idx].mean().detach(),
                "selected_fraction": selected.float().mean().detach(),
                "output_norm": expert_output.norm(dim=-1).mean().detach(),
                "depth": float(len(expert.blocks)),
                "dim": float(expert.dim),
            }
        stats = {
            "resolution": int(self.resolution),
            "uses_mamba_size_router": 1.0,
            "preset_top_k": float(k),
            "preset_weight_mode_selection": float(self.router_weight_mode == "selection"),
            "preset_weights": {
                name: values["weight"] for name, values in expert_stats.items()
            },
            "preset_probabilities": {
                name: values["probability"] for name, values in expert_stats.items()
            },
            "preset_selected_fractions": {
                name: values["selected_fraction"] for name, values in expert_stats.items()
            },
            "preset_output_norms": {
                name: values["output_norm"] for name, values in expert_stats.items()
            },
            "preset_depths": {
                name: values["depth"] for name, values in expert_stats.items()
            },
            "preset_dims": {
                name: values["dim"] for name, values in expert_stats.items()
            },
            "output_norm": output.norm(dim=-1).mean().detach(),
        }
        return output, stats


class RoutedEncoderMambaDepthProcessor(nn.Module):
    def __init__(
        self,
        dim,
        resolution,
        base_depth,
        has_text,
        ssm_cfg,
        norm_epsilon,
        rms_norm,
        residual_in_fp32,
        fused_add_norm,
        drop_path_rate,
        scan_type,
        processor_scan_type,
        processor_window_size,
        processor_shift_size,
        router_top_k,
        use_jit,
        use_checkpoint,
        device,
        dtype,
    ):
        super().__init__()
        self.dim = int(dim)
        self.resolution = int(resolution)
        self.base_depth = int(base_depth)
        self.router_top_k = int(router_top_k)
        if self.router_top_k != 1:
            raise ValueError(
                "encoder_mamba_depth_router_top_k must be 1 for sparse depth routing, "
                f"got {router_top_k}"
            )
        if int(processor_window_size) > 0:
            raise ValueError(
                "shared-prefix encoder_mamba_depth_router currently supports full-map "
                "Mamba stages only; disable local_mamba_stage_resolutions for routed "
                f"encoder resolution {resolution}"
            )
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.processor_scan_type = processor_scan_type if processor_scan_type else scan_type
        self.branch_names = ["shallow", "base", "deep"]
        self.branch_depths = [
            max(1, self.base_depth - 1),
            max(1, self.base_depth),
            max(1, self.base_depth + 1),
        ]
        self.base_branch_idx = 1
        self.max_depth = max(self.branch_depths)
        factory_kwargs = {"device": device, "dtype": dtype}
        block_kwargs = {"use_jit": use_jit}
        if self.processor_scan_type != "v2":
            block_kwargs.update(
                build_scan_block_kwargs(
                    scan_type=self.processor_scan_type,
                    patch_side_len=resolution,
                    depth=self.max_depth,
                    device=device,
                    extras=0,
                )
            )
        drop_path_values = torch.linspace(0, drop_path_rate, self.max_depth).tolist()
        self.blocks = nn.ModuleList(
            [
                create_block(
                    dim,
                    has_text=has_text,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=layer_idx,
                    scan_type=self.processor_scan_type,
                    drop_path=drop_path_values[layer_idx],
                    **block_kwargs,
                    **factory_kwargs,
                )
                for layer_idx in range(self.max_depth)
            ]
        )
        self.stage_norm = (nn.LayerNorm if not rms_norm else RMSNorm)(
            dim, eps=norm_epsilon, **factory_kwargs
        )
        self.output_proj = nn.Linear(dim, dim, **factory_kwargs)

    def _run_depth(self, context_map, c, text, depth):
        input_tokens = map_to_tokens(context_map)
        residual = None
        hidden_states = input_tokens
        for block in self.blocks[: int(depth)]:
            if self.use_checkpoint:
                hidden_states, residual = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    residual,
                    c,
                    text,
                    use_reentrant=False,
                )
            else:
                hidden_states, residual = block(
                    hidden_states,
                    residual=residual,
                    c=c,
                    text=text,
                )
        if not self.fused_add_norm:
            residual = hidden_states if residual is None else residual + hidden_states
            stage_hidden = self.stage_norm(
                residual.to(dtype=self.stage_norm.weight.dtype)
            )
        else:
            fused_add_norm_fn = (
                rms_norm_fn
                if isinstance(self.stage_norm, RMSNorm)
                else layer_norm_fn
            )
            stage_hidden = fused_add_norm_fn(
                hidden_states,
                self.stage_norm.weight,
                self.stage_norm.bias,
                eps=self.stage_norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        output_tokens = input_tokens + self.output_proj(stage_hidden)
        output_map = tokens_to_map(output_tokens, context_map.shape[1:3])
        return output_map, stage_hidden, output_tokens

    def forward(
        self,
        context_map,
        c,
        text=None,
        router_logits=None,
        controller_stats=None,
    ):
        if router_logits is None:
            router_logits = context_map.new_full(
                (context_map.shape[0], len(self.branch_names)),
                -2.0,
            )
            router_logits[:, self.base_branch_idx] = 2.0
        probabilities = torch.softmax(router_logits, dim=-1)
        selected_indices = torch.argmax(router_logits, dim=-1)
        mask = F.one_hot(
            selected_indices,
            num_classes=router_logits.shape[-1],
        ).to(dtype=torch.bool)
        hard_weights = mask.to(dtype=probabilities.dtype)
        router_weights = hard_weights + probabilities - probabilities.detach()
        output = context_map.new_zeros(context_map.shape)
        branch_stats = {}
        for branch_idx, name in enumerate(self.branch_names):
            selected = selected_indices == branch_idx
            depth = self.branch_depths[branch_idx]
            if not selected.any():
                branch_stats[name] = {
                    "weight": router_weights[:, branch_idx].mean().detach(),
                    "probability": probabilities[:, branch_idx].mean().detach(),
                    "selected_fraction": selected.float().mean().detach(),
                    "output_norm": context_map.new_tensor(0.0).detach(),
                    "depth": float(depth),
                }
                continue
            selected_text = text[selected] if text is not None else None
            branch_output, _, _ = self._run_depth(
                context_map[selected],
                c[selected],
                selected_text,
                depth,
            )
            output[selected] = branch_output * router_weights[
                selected,
                branch_idx,
            ].view(-1, 1, 1, 1)
            branch_stats[name] = {
                "weight": router_weights[:, branch_idx].mean().detach(),
                "probability": probabilities[:, branch_idx].mean().detach(),
                "selected_fraction": selected.float().mean().detach(),
                "output_norm": branch_output.norm(dim=-1).mean().detach(),
                "depth": float(depth),
            }
        selected_depth_mean = sum(
            branch_stats[name]["selected_fraction"] * self.branch_depths[idx]
            for idx, name in enumerate(self.branch_names)
        )
        stats = {
            "resolution": int(self.resolution),
            "uses_encoder_mamba_depth_router": 1.0,
            "uses_shared_prefix_depth_router": 1.0,
            "depth_router_top_k": float(self.router_top_k),
            "base_depth": float(self.base_depth),
            "max_depth": float(self.max_depth),
            "selected_depth_mean": selected_depth_mean.detach(),
            "depth_weights": {
                name: values["weight"] for name, values in branch_stats.items()
            },
            "depth_probabilities": {
                name: values["probability"] for name, values in branch_stats.items()
            },
            "depth_selected_fractions": {
                name: values["selected_fraction"] for name, values in branch_stats.items()
            },
            "depth_output_norms": {
                name: values["output_norm"] for name, values in branch_stats.items()
            },
            "depth_branch_depths": {
                name: values["depth"] for name, values in branch_stats.items()
            },
            "output_norm": output.norm(dim=-1).mean().detach(),
        }
        if controller_stats:
            stats.update(controller_stats)
        return output, stats


class RoutedCompressionDownsample2d(nn.Module):
    def __init__(
        self,
        dim,
        stride,
        use_premix,
        premix_depth,
        conv_type,
        condition_dim,
        device,
        dtype,
    ):
        super().__init__()
        self.stride = int(stride)
        self.base_premix_depth = int(premix_depth)
        self.branch_premix_depths = [
            max(0, self.base_premix_depth - 1),
            self.base_premix_depth,
            self.base_premix_depth + 1,
        ]
        self.router = FactorizedRouterMLP(
            condition_dim,
            dim,
            3,
            device,
            dtype,
        )
        self.premix_branches = nn.ModuleList(
            [
                self._make_premix_branch(
                    dim,
                    use_premix,
                    branch_premix_depth,
                    conv_type,
                    device,
                    dtype,
                )
                for branch_premix_depth in self.branch_premix_depths
            ]
        )
        self.downsample_norm = nn.GroupNorm(1, dim, eps=1e-6).to(device).to(dtype)
        self.conv = make_spatial_conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            conv_type=conv_type,
            device=device,
            dtype=dtype,
        )
        self.skip = nn.Identity()
        with torch.no_grad():
            # Branch 1 mirrors the non-routed LearnedDownsample2d configuration;
            # the other branches only vary the number of stride-1 premix blocks.
            self.router.output_proj.bias.copy_(
                torch.tensor([-2.0, 2.0, -2.0], device=device, dtype=dtype)
            )

    @staticmethod
    def _make_premix_branch(dim, use_premix, premix_depth, conv_type, device, dtype):
        if not use_premix:
            return nn.Identity()
        if int(premix_depth) == 0:
            return nn.Sequential(
                nn.GroupNorm(1, dim, eps=1e-6).to(device).to(dtype),
                nn.GELU(),
                make_spatial_conv2d(
                    dim,
                    dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_type=conv_type,
                    device=device,
                    dtype=dtype,
                ),
            )
        return nn.Sequential(
            *[
                ConvResidualBlock2d(
                    dim,
                    dim,
                    conv_type=conv_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(int(premix_depth))
            ]
        )

    def forward(
        self,
        context_map,
        condition,
        external_logits=None,
        channel_gate=None,
    ):
        logits = self.router(condition, context_map) if external_logits is None else external_logits
        probabilities = torch.softmax(logits, dim=-1)
        selected_indices = torch.argmax(logits, dim=-1)
        mask = F.one_hot(selected_indices, num_classes=logits.shape[-1]).to(
            dtype=torch.bool
        )
        if not torch.all(mask.sum(dim=-1) == 1):
            raise RuntimeError("compression router must select exactly one branch")
        hard_weights = mask.to(dtype=probabilities.dtype)
        weights = hard_weights + probabilities - probabilities.detach()

        x = context_map.permute(0, 3, 1, 2).contiguous()
        if channel_gate is not None:
            if channel_gate.dim() != 2 or channel_gate.shape[-1] != x.shape[1]:
                raise ValueError(
                    "compression channel_gate must have shape [batch, channels], "
                    f"got {tuple(channel_gate.shape)} for channels={x.shape[1]}"
                )
            x = x * channel_gate.to(dtype=x.dtype).view(x.shape[0], x.shape[1], 1, 1)
        residual = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        residual = self.skip(residual)
        output = None
        for branch_idx, premix_branch in enumerate(self.premix_branches):
            selected = selected_indices == branch_idx
            if not selected.any():
                continue
            branch_input = premix_branch(x[selected])
            branch_output = self.conv(F.gelu(self.downsample_norm(branch_input)))
            branch_residual = residual[selected]
            if branch_output.shape[-2:] != branch_residual.shape[-2:]:
                branch_residual = F.interpolate(
                    branch_residual, size=branch_output.shape[-2:], mode="nearest"
                )
            branch_output = branch_output + branch_residual
            if output is None:
                output = context_map.new_zeros(
                    (context_map.shape[0], *branch_output.shape[1:])
                )
            output[selected] = branch_output * weights[selected, branch_idx].view(
                -1, 1, 1, 1
            )
        if output is None:
            raise RuntimeError("compression router selected no branches")
        output_map = output.permute(0, 2, 3, 1).contiguous()
        stats = {
            "uses_compression_router": 1.0,
            "uses_external_compression_logits": float(external_logits is not None),
            "uses_compression_channel_gate": float(channel_gate is not None),
            "stride0_weight": weights[:, 0].mean().detach(),
            "stride1_weight": weights[:, 1].mean().detach(),
            "stride2_weight": weights[:, 2].mean().detach(),
            "stride0_probability": probabilities[:, 0].mean().detach(),
            "stride1_probability": probabilities[:, 1].mean().detach(),
            "stride2_probability": probabilities[:, 2].mean().detach(),
            "stride0_selected": mask[:, 0].float().mean().detach(),
            "stride1_selected": mask[:, 1].float().mean().detach(),
            "stride2_selected": mask[:, 2].float().mean().detach(),
            "output_norm": output_map.norm(dim=-1).mean().detach(),
        }
        return output_map, stats


class MapMambaResidualProcessor(nn.Module):
    """Run a stack of Mamba blocks at a fixed spatial resolution."""

    def __init__(
        self,
        dim,
        depth,
        resolution,
        has_text,
        ssm_cfg,
        norm_epsilon,
        rms_norm,
        residual_in_fp32,
        fused_add_norm,
        drop_path_values,
        scan_type,
        use_jit,
        use_checkpoint,
        device,
        dtype,
    ):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        factory_kwargs = {"device": device, "dtype": dtype}

        block_kwargs = {"use_jit": use_jit}
        if scan_type != "v2":
            block_kwargs.update(
                build_scan_block_kwargs(
                    scan_type=scan_type,
                    patch_side_len=resolution,
                    depth=depth,
                    device=device,
                    extras=0,
                )
            )

        self.blocks = nn.ModuleList(
            [
                create_block(
                    dim,
                    has_text=has_text,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=layer_idx,
                    scan_type=scan_type,
                    drop_path=drop_path_values[layer_idx],
                    **block_kwargs,
                    **factory_kwargs,
                )
                for layer_idx in range(depth)
            ]
        )
        self.stage_norm = (nn.LayerNorm if not rms_norm else RMSNorm)(
            dim, eps=norm_epsilon, **factory_kwargs
        )
        self.output_proj = nn.Linear(dim, dim, **factory_kwargs)

    def forward(self, context_map, c, text=None):
        input_tokens = map_to_tokens(context_map)
        residual = None
        hidden_states = input_tokens

        for block in self.blocks:
            if self.use_checkpoint:
                hidden_states, residual = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    residual,
                    c,
                    text,
                    use_reentrant=False,
                )
            else:
                hidden_states, residual = block(
                    hidden_states,
                    residual=residual,
                    c=c,
                    text=text,
                )

        if not self.fused_add_norm:
            residual = hidden_states if residual is None else residual + hidden_states
            stage_hidden = self.stage_norm(
                residual.to(dtype=self.stage_norm.weight.dtype)
            )
        else:
            fused_add_norm_fn = (
                rms_norm_fn
                if isinstance(self.stage_norm, RMSNorm)
                else layer_norm_fn
            )
            stage_hidden = fused_add_norm_fn(
                hidden_states,
                self.stage_norm.weight,
                self.stage_norm.bias,
                eps=self.stage_norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        output_tokens = input_tokens + self.output_proj(stage_hidden)
        output_map = tokens_to_map(output_tokens, context_map.shape[1:3])
        stats = {
            "resolution": int(self.resolution),
            "input_norm": input_tokens.norm(dim=-1).mean().detach(),
            "stage_hidden_norm": stage_hidden.norm(dim=-1).mean().detach(),
            "output_norm": output_tokens.norm(dim=-1).mean().detach(),
        }
        return output_map, stats


class MapWindowMambaResidualProcessor(nn.Module):
    """Run a stack of Mamba blocks inside local windows and stitch the map back."""

    def __init__(
        self,
        dim,
        depth,
        resolution,
        window_size,
        shift_size,
        has_text,
        ssm_cfg,
        norm_epsilon,
        rms_norm,
        residual_in_fp32,
        fused_add_norm,
        drop_path_values,
        scan_type,
        use_jit,
        use_checkpoint,
        device,
        dtype,
    ):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if self.window_size > resolution:
            raise ValueError(
                "window_size must not exceed resolution, "
                f"got {self.window_size} > {resolution}"
            )
        if self.shift_size < 0:
            raise ValueError(f"shift_size must be non-negative, got {shift_size}")
        if self.shift_size >= self.window_size:
            raise ValueError(
                "shift_size must be smaller than window_size, "
                f"got {self.shift_size} >= {self.window_size}"
            )
        self.grid_hw = (
            compute_window_grid_size(resolution, self.window_size, self.window_size)[0],
            compute_window_grid_size(resolution, self.window_size, self.window_size)[0],
        )
        self.pad_hw = (
            compute_window_grid_size(resolution, self.window_size, self.window_size)[1],
            compute_window_grid_size(resolution, self.window_size, self.window_size)[1],
        )
        factory_kwargs = {"device": device, "dtype": dtype}

        block_kwargs = {"use_jit": use_jit}
        if scan_type != "v2":
            block_kwargs.update(
                build_scan_block_kwargs(
                    scan_type=scan_type,
                    patch_side_len=self.window_size,
                    depth=depth,
                    device=device,
                    extras=0,
                )
            )

        self.blocks = nn.ModuleList(
            [
                create_block(
                    dim,
                    has_text=has_text,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=layer_idx,
                    scan_type=scan_type,
                    drop_path=drop_path_values[layer_idx],
                    **block_kwargs,
                    **factory_kwargs,
                )
                for layer_idx in range(depth)
            ]
        )
        self.stage_norm = (nn.LayerNorm if not rms_norm else RMSNorm)(
            dim, eps=norm_epsilon, **factory_kwargs
        )
        self.output_proj = nn.Linear(dim, dim, **factory_kwargs)

    def _extract_windows(self, context_map):
        b, h, w, c = context_map.shape
        x = context_map
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x = x.permute(0, 3, 1, 2)
        pad_h, pad_w = self.pad_hw
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = x.unfold(2, self.window_size, self.window_size).unfold(
            3, self.window_size, self.window_size
        )
        x = x.permute(0, 2, 3, 4, 5, 1).contiguous()
        out_h, out_w = x.shape[1], x.shape[2]
        windows = x.reshape(b * out_h * out_w, self.window_size * self.window_size, c)
        return windows, (out_h, out_w), (h, w)

    def _merge_windows(self, output_tokens, batch_size, out_hw, original_hw):
        out_h, out_w = out_hw
        merged = output_tokens.reshape(
            batch_size,
            out_h,
            out_w,
            self.window_size,
            self.window_size,
            self.dim,
        )
        merged = merged.permute(0, 5, 1, 3, 2, 4).contiguous()
        merged = merged.reshape(
            batch_size,
            self.dim,
            out_h * self.window_size,
            out_w * self.window_size,
        )
        merged = merged[:, :, : original_hw[0], : original_hw[1]]
        merged = merged.permute(0, 2, 3, 1).contiguous()
        if self.shift_size > 0:
            merged = torch.roll(
                merged, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        return merged

    def forward(self, context_map, c, text=None):
        input_windows, out_hw, original_hw = self._extract_windows(context_map)
        residual = None
        hidden_states = input_windows
        repeated_c = c.repeat_interleave(out_hw[0] * out_hw[1], dim=0)
        repeated_text = (
            text.repeat_interleave(out_hw[0] * out_hw[1], dim=0)
            if text is not None
            else None
        )

        for block in self.blocks:
            if self.use_checkpoint:
                hidden_states, residual = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    residual,
                    repeated_c,
                    repeated_text,
                    use_reentrant=False,
                )
            else:
                hidden_states, residual = block(
                    hidden_states,
                    residual=residual,
                    c=repeated_c,
                    text=repeated_text,
                )

        if not self.fused_add_norm:
            residual = hidden_states if residual is None else residual + hidden_states
            stage_hidden = self.stage_norm(
                residual.to(dtype=self.stage_norm.weight.dtype)
            )
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.stage_norm, RMSNorm) else layer_norm_fn
            )
            stage_hidden = fused_add_norm_fn(
                hidden_states,
                self.stage_norm.weight,
                self.stage_norm.bias,
                eps=self.stage_norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        output_tokens = input_windows + self.output_proj(stage_hidden)
        output_map = self._merge_windows(
            output_tokens, context_map.shape[0], out_hw, original_hw
        )
        stats = {
            "resolution": int(self.resolution),
            "window_size": int(self.window_size),
            "shift_size": int(self.shift_size),
            "uses_local_window_mamba": 1.0,
            "input_norm": input_windows.norm(dim=-1).mean().detach(),
            "stage_hidden_norm": stage_hidden.norm(dim=-1).mean().detach(),
            "output_norm": output_tokens.norm(dim=-1).mean().detach(),
        }
        return output_map, stats


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            device=device,
            dtype=dtype,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def make_spatial_conv2d(
    in_channels,
    out_channels,
    *,
    kernel_size=3,
    stride=1,
    padding=1,
    conv_type="standard",
    device=None,
    dtype=None,
):
    if conv_type == "standard":
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            device=device,
            dtype=dtype,
        )
    if conv_type == "separable":
        return DepthwiseSeparableConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unsupported conv_type: {conv_type}")


class LearnedDownsample2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        use_premix,
        premix_depth,
        conv_type,
        device,
        dtype,
    ):
        super().__init__()
        self.stride = stride
        self.use_premix = use_premix
        self.premix_depth = int(premix_depth)
        self.conv_type = conv_type
        if self.use_premix and self.premix_depth == 0:
            self.premix_norm = nn.GroupNorm(1, in_channels, eps=1e-6).to(device).to(dtype)
            self.premix_conv = make_spatial_conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_type=conv_type,
                device=device,
                dtype=dtype,
            )
        else:
            self.premix_norm = None
            self.premix_conv = None
        self.premix_blocks = nn.ModuleList(
            [
                ConvResidualBlock2d(
                    in_channels,
                    in_channels,
                    conv_type=conv_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(self.premix_depth if self.use_premix else 0)
            ]
        )
        self.downsample_norm = nn.GroupNorm(1, in_channels, eps=1e-6).to(device).to(dtype)
        self.conv = make_spatial_conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            conv_type=conv_type,
            device=device,
            dtype=dtype,
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, context_map):
        x = context_map.permute(0, 3, 1, 2).contiguous()
        residual = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        residual = self.skip(residual)
        if self.premix_conv is not None:
            x = self.premix_conv(F.gelu(self.premix_norm(x)))
        for block in self.premix_blocks:
            x = block(x)
        x = self.conv(F.gelu(self.downsample_norm(x)))
        if x.shape[-2:] != residual.shape[-2:]:
            residual = F.interpolate(residual, size=x.shape[-2:], mode="nearest")
        x = x + residual
        return x.permute(0, 2, 3, 1).contiguous()


class ConvResidualBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_type="standard",
        device=None,
        dtype=None,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = nn.GroupNorm(1, in_channels, eps=1e-6).to(device).to(dtype)
        self.conv1 = make_spatial_conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            conv_type=conv_type,
            device=device,
            dtype=dtype,
        )
        self.norm2 = nn.GroupNorm(1, out_channels, eps=1e-6).to(device).to(dtype)
        self.conv2 = make_spatial_conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            conv_type=conv_type,
            device=device,
            dtype=dtype,
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.conv2(F.gelu(self.norm2(x)))
        return x + residual


class MapConvResidualStack(nn.Module):
    def __init__(self, dim, depth, device, dtype, conv_type="standard"):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ConvResidualBlock2d(
                    dim,
                    dim,
                    conv_type=conv_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, stage_map):
        if len(self.blocks) == 0:
            return stage_map
        x = stage_map.permute(0, 3, 1, 2).contiguous()
        for block in self.blocks:
            x = block(x)
        return x.permute(0, 2, 3, 1).contiguous()


class SpatialFusionBlock(nn.Module):
    def __init__(
        self,
        low_dim,
        skip_dim,
        out_dim,
        refinement_depth,
        device,
        dtype,
    ):
        super().__init__()
        self.input_norm = nn.GroupNorm(1, low_dim + skip_dim, eps=1e-6).to(device).to(dtype)
        self.input_conv = nn.Conv2d(
            low_dim + skip_dim,
            out_dim,
            kernel_size=3,
            padding=1,
            device=device,
            dtype=dtype,
        )
        self.skip = (
            nn.Identity()
            if low_dim == out_dim
            else nn.Conv2d(
                low_dim,
                out_dim,
                kernel_size=1,
                device=device,
                dtype=dtype,
            )
        )
        self.refinement = nn.ModuleList(
            [
                ConvResidualBlock2d(
                    out_dim,
                    out_dim,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(refinement_depth)
            ]
        )

    def forward(self, low_map, skip_map):
        low = low_map.permute(0, 3, 1, 2).contiguous()
        skip = skip_map.permute(0, 3, 1, 2).contiguous()
        fused = self.input_conv(
            F.gelu(self.input_norm(torch.cat([low, skip], dim=1)))
        )
        fused = fused + self.skip(low)
        for block in self.refinement:
            fused = block(fused)
        return fused.permute(0, 2, 3, 1).contiguous()


class MapPredictionHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
        prediction_head_type,
        conv_block_depth,
        conv_type,
        device,
        dtype,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.patch_dim = patch_size * patch_size * out_channels
        self.prediction_head_type = prediction_head_type
        self.norm = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
            device=device,
            dtype=dtype,
        )
        self.use_conv_head = prediction_head_type == "anchor_conv_upsample"
        if self.use_conv_head:
            self.stem = make_spatial_conv2d(
                hidden_size,
                hidden_size,
                kernel_size=3,
                padding=1,
                conv_type=conv_type,
                device=device,
                dtype=dtype,
            )
            self.blocks = nn.ModuleList(
                [
                    ConvResidualBlock2d(
                        hidden_size,
                        hidden_size,
                        conv_type=conv_type,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in range(conv_block_depth)
                ]
            )
            self.out_norm = nn.GroupNorm(1, hidden_size, eps=1e-6).to(device).to(dtype)
            self.out_conv = nn.Conv2d(
                hidden_size,
                self.patch_dim,
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            )
        else:
            self.linear = nn.Linear(
                hidden_size,
                self.patch_dim,
                device=device,
                dtype=dtype,
            )

    def forward(self, context_map):
        normalized = self.norm(context_map)
        if not self.use_conv_head:
            tokens = map_to_tokens(normalized)
            return self.linear(tokens)

        x = normalized.permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_conv(F.gelu(self.out_norm(x)))
        return map_to_tokens(x.permute(0, 2, 3, 1).contiguous())


class ChildOffsetPositionalEmbedding(nn.Module):
    """Repeat a learned 2x2 offset code after each x2 upsample."""

    def __init__(self, dim, device, dtype):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 2, 2, dim, device=device, dtype=dtype)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, stage_map):
        _, h, w, _ = stage_map.shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(
                "ChildOffsetPositionalEmbedding requires even spatial dimensions, "
                f"got {(h, w)}"
            )
        return stage_map + self.pos_embed.repeat(1, h // 2, w // 2, 1)


class StageContentAwareChannelGate(nn.Module):
    """Predict a per-channel gate residual from the decoder context and skip summary."""

    def __init__(self, decoder_dim, condition_dim, device, dtype):
        super().__init__()
        hidden_dim = max(decoder_dim, condition_dim)
        self.condition_norm = nn.LayerNorm(condition_dim, eps=1e-6, device=device, dtype=dtype)
        self.low_norm = nn.LayerNorm(decoder_dim, eps=1e-6, device=device, dtype=dtype)
        self.skip_norm = nn.LayerNorm(decoder_dim, eps=1e-6, device=device, dtype=dtype)
        self.input_proj = nn.Linear(
            condition_dim + decoder_dim * 2,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        self.output_proj = nn.Linear(
            hidden_dim,
            decoder_dim,
            device=device,
            dtype=dtype,
        )
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, condition, low_map, skip_map):
        low_summary = self.low_norm(low_map.mean(dim=(1, 2)))
        skip_summary = self.skip_norm(skip_map.mean(dim=(1, 2)))
        condition_summary = self.condition_norm(condition)
        hidden = F.gelu(
            self.input_proj(
                torch.cat([condition_summary, low_summary, skip_summary], dim=-1)
            )
        )
        return self.output_proj(hidden).view(condition.shape[0], 1, 1, -1)


class StageContentAwareSpatialGate(nn.Module):
    """Predict a per-location gate residual from local decoder/skip features."""

    def __init__(self, decoder_dim, condition_dim, device, dtype):
        super().__init__()
        self.condition_norm = nn.LayerNorm(condition_dim, eps=1e-6, device=device, dtype=dtype)
        self.low_norm = nn.LayerNorm(decoder_dim, eps=1e-6, device=device, dtype=dtype)
        self.skip_norm = nn.LayerNorm(decoder_dim, eps=1e-6, device=device, dtype=dtype)
        self.condition_proj = nn.Linear(
            condition_dim,
            decoder_dim,
            device=device,
            dtype=dtype,
        )
        self.input_norm = nn.GroupNorm(1, decoder_dim * 3, eps=1e-6).to(device).to(dtype)
        self.input_conv = nn.Conv2d(
            decoder_dim * 3,
            decoder_dim,
            kernel_size=3,
            padding=1,
            device=device,
            dtype=dtype,
        )
        self.hidden_norm = nn.GroupNorm(1, decoder_dim, eps=1e-6).to(device).to(dtype)
        self.hidden_conv = nn.Conv2d(
            decoder_dim,
            decoder_dim,
            kernel_size=3,
            padding=1,
            device=device,
            dtype=dtype,
        )
        self.output_conv = nn.Conv2d(
            decoder_dim,
            1,
            kernel_size=1,
            device=device,
            dtype=dtype,
        )
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, condition, low_map, skip_map):
        batch_size, height, width, _ = low_map.shape
        low = self.low_norm(low_map).permute(0, 3, 1, 2).contiguous()
        skip = self.skip_norm(skip_map).permute(0, 3, 1, 2).contiguous()
        condition_map = self.condition_proj(self.condition_norm(condition)).view(
            batch_size, -1, 1, 1
        )
        condition_map = condition_map.expand(-1, -1, height, width)
        hidden = torch.cat([low, skip, condition_map], dim=1)
        hidden = self.input_conv(F.gelu(self.input_norm(hidden)))
        hidden = self.hidden_conv(F.gelu(self.hidden_norm(hidden)))
        return self.output_conv(hidden).permute(0, 2, 3, 1).contiguous()


class FinalSkipRefiner(nn.Module):
    """Re-inject the highest-resolution encoder skip before the prediction head."""

    def __init__(
        self,
        resolution,
        decoder_dim,
        skip_dim,
        condition_dim,
        refinement_depth,
        refinement_conv_type,
        use_channel_gate,
        use_spatial_gate,
        device,
        dtype,
    ):
        super().__init__()
        self.resolution = resolution
        self.decoder_dim = decoder_dim
        self.skip_proj = (
            nn.Identity()
            if skip_dim == decoder_dim
            else nn.Linear(skip_dim, decoder_dim, device=device, dtype=dtype)
        )
        self.gate = StageTimestepGate(condition_dim, device, dtype)
        self.channel_gate = (
            StageContentAwareChannelGate(decoder_dim, condition_dim, device, dtype)
            if use_channel_gate
            else None
        )
        self.spatial_gate = (
            StageContentAwareSpatialGate(decoder_dim, condition_dim, device, dtype)
            if use_spatial_gate
            else None
        )
        self.local_fusion = SpatialFusionBlock(
            low_dim=decoder_dim,
            skip_dim=decoder_dim,
            out_dim=decoder_dim,
            refinement_depth=0,
            device=device,
            dtype=dtype,
        )
        self.refinement = MapConvResidualStack(
            decoder_dim,
            depth=refinement_depth,
            device=device,
            dtype=dtype,
            conv_type=refinement_conv_type,
        )

    def forward(
        self,
        final_map,
        skip_map,
        condition,
        disable_skip=False,
        force_skip_gate_value=None,
        gate_floor=None,
        stage_router_weight=None,
    ):
        stage_stats = {
            "resolution": int(self.resolution),
            "decoder_dim": int(self.decoder_dim),
            "skip_used": float(skip_map is not None),
            "uses_content_channel_gate": float(self.channel_gate is not None),
            "uses_spatial_gate": float(self.spatial_gate is not None),
        }
        if skip_map is None:
            refined = self.refinement(final_map)
            stage_stats["output_norm"] = refined.norm(dim=-1).mean().detach()
            return refined, stage_stats

        skip = self.skip_proj(skip_map)
        raw_gate = self.gate.compute_gate(condition, skip.shape[0])
        gate_logits = torch.logit(raw_gate.clamp(1e-6, 1.0 - 1e-6))
        gate = raw_gate
        if self.spatial_gate is not None:
            spatial_gate_delta = self.spatial_gate(condition, final_map, skip)
            gate_logits = gate_logits + spatial_gate_delta
            stage_stats["spatial_gate_delta_norm"] = spatial_gate_delta.norm(dim=-1).mean().detach()
        if self.channel_gate is not None:
            channel_gate_delta = self.channel_gate(condition, final_map, skip)
            gate_logits = gate_logits + channel_gate_delta
            stage_stats["channel_gate_delta_norm"] = channel_gate_delta.norm(dim=-1).mean().detach()
        if self.spatial_gate is not None or self.channel_gate is not None:
            gate = torch.sigmoid(gate_logits)
        if disable_skip:
            gate = torch.zeros_like(gate)
        if force_skip_gate_value is not None:
            gate = torch.full_like(gate, float(force_skip_gate_value))
        if gate_floor is not None:
            gate = gate_floor + (1.0 - gate_floor) * gate
        if stage_router_weight is not None:
            gate = gate * stage_router_weight
            stage_stats["stage_router_weight"] = stage_router_weight.mean().detach()

        gated_skip = skip * gate
        refined = self.local_fusion(final_map, gated_skip)
        refined = self.refinement(refined)
        stage_stats["raw_gate_mean"] = raw_gate.mean().detach()
        stage_stats["raw_gate_std"] = raw_gate.std(unbiased=False).detach()
        stage_stats["gate_mean"] = gate.mean().detach()
        stage_stats["gate_std"] = gate.std(unbiased=False).detach()
        stage_stats["skip_norm"] = gated_skip.norm(dim=-1).mean().detach()
        stage_stats["output_norm"] = refined.norm(dim=-1).mean().detach()
        return refined, stage_stats


class Aux4x4DecoderContextAdapter(nn.Module):
    """Turn a coarse 4x4 bottleneck summary into a gated decoder-anchor residual."""

    def __init__(
        self,
        context_dim,
        target_dim,
        context_resolution,
        target_resolution,
        condition_dim,
        prepool_depth,
        context_depth,
        fusion_depth,
        conv_type,
        upsample_mode,
        use_timestep_gate,
        residual_scale_multiplier,
        device,
        dtype,
    ):
        super().__init__()
        self.context_resolution = int(context_resolution)
        self.target_resolution = int(target_resolution)
        self.target_dim = int(target_dim)
        self.upsample_mode = upsample_mode
        self.use_timestep_gate = bool(use_timestep_gate)
        self.residual_scale_multiplier = float(residual_scale_multiplier)

        self.prepool_refine = MapConvResidualStack(
            context_dim,
            depth=prepool_depth,
            device=device,
            dtype=dtype,
            conv_type=conv_type,
        )
        self.context_norm = nn.LayerNorm(
            context_dim, eps=1e-6, device=device, dtype=dtype
        )
        self.context_proj = nn.Linear(
            context_dim, target_dim, device=device, dtype=dtype
        )
        self.context_pos_embed = AnchorPositionalEmbedding(
            self.context_resolution, target_dim, device, dtype
        )
        self.context_refine = MapConvResidualStack(
            target_dim,
            depth=context_depth,
            device=device,
            dtype=dtype,
            conv_type=conv_type,
        )
        self.target_norm = nn.LayerNorm(
            target_dim, eps=1e-6, device=device, dtype=dtype
        )
        self.fusion_norm = nn.GroupNorm(1, target_dim * 2, eps=1e-6).to(device).to(dtype)
        self.fusion_conv = make_spatial_conv2d(
            target_dim * 2,
            target_dim,
            kernel_size=3,
            padding=1,
            conv_type=conv_type,
            device=device,
            dtype=dtype,
        )
        self.fusion_refine = MapConvResidualStack(
            target_dim,
            depth=fusion_depth,
            device=device,
            dtype=dtype,
            conv_type=conv_type,
        )
        self.out_norm = nn.GroupNorm(1, target_dim, eps=1e-6).to(device).to(dtype)
        self.out_conv = make_spatial_conv2d(
            target_dim,
            target_dim,
            kernel_size=1,
            padding=0,
            conv_type=conv_type,
            device=device,
            dtype=dtype,
        )
        self.timestep_gate = (
            StageTimestepGate(condition_dim, device, dtype)
            if self.use_timestep_gate
            else None
        )
        self.residual_scale = nn.Parameter(
            torch.zeros(1, 1, 1, target_dim, device=device, dtype=dtype)
        )

    def _resize_cf(self, x, size):
        if x.shape[-2:] == size:
            return x
        kwargs = {}
        if self.upsample_mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        return F.interpolate(x, size=size, mode=self.upsample_mode, **kwargs)

    def _make_context_4x4(self, context_map):
        context_map = self.prepool_refine(context_map)
        context_cf = context_map.permute(0, 3, 1, 2).contiguous()
        if context_cf.shape[-2:] != (
            self.context_resolution,
            self.context_resolution,
        ):
            context_cf = F.adaptive_avg_pool2d(
                context_cf,
                (self.context_resolution, self.context_resolution),
            )
        return context_cf.permute(0, 2, 3, 1).contiguous()

    def forward(self, context_map, target_map, condition):
        context_4x4 = self._make_context_4x4(context_map)
        context = self.context_proj(self.context_norm(context_4x4))
        context = self.context_pos_embed(context)
        context = self.context_refine(context)

        context_cf = context.permute(0, 3, 1, 2).contiguous()
        context_cf = self._resize_cf(context_cf, target_map.shape[1:3])
        target_cf = self.target_norm(target_map).permute(0, 3, 1, 2).contiguous()
        fused = torch.cat([target_cf, context_cf], dim=1)
        delta_cf = self.fusion_conv(F.gelu(self.fusion_norm(fused)))
        delta = delta_cf.permute(0, 2, 3, 1).contiguous()
        delta = self.fusion_refine(delta)
        delta_cf = delta.permute(0, 3, 1, 2).contiguous()
        delta_cf = self.out_conv(F.gelu(self.out_norm(delta_cf)))
        delta = delta_cf.permute(0, 2, 3, 1).contiguous()

        raw_gate = None
        gate = self.residual_scale
        if self.timestep_gate is not None:
            raw_gate = self.timestep_gate.compute_gate(condition, target_map.shape[0])
            gate = gate * raw_gate
        effective_gate = self.residual_scale_multiplier * gate
        output = target_map + effective_gate * delta
        stats = {
            "context_norm": context_4x4.norm(dim=-1).mean().detach(),
            "projected_context_norm": context.norm(dim=-1).mean().detach(),
            "upsampled_context_norm": context_cf.norm(dim=1).mean().detach(),
            "delta_norm": delta.norm(dim=-1).mean().detach(),
            "residual_scale_abs_mean": self.residual_scale.abs().mean().detach(),
            "residual_scale_abs_max": self.residual_scale.abs().max().detach(),
            "effective_gate_abs_mean": effective_gate.abs().mean().detach(),
            "residual_scale_multiplier": float(self.residual_scale_multiplier),
            "output_norm": output.norm(dim=-1).mean().detach(),
        }
        if raw_gate is not None:
            stats["timestep_gate_mean"] = raw_gate.mean().detach()
            stats["timestep_gate_std"] = raw_gate.std(unbiased=False).detach()
        return output, stats


class DecoderFusionStage(nn.Module):
    def __init__(
        self,
        resolution,
        input_dim,
        output_dim,
        skip_dim,
        stage_depth,
        fusion_conv_depth,
        pre_conv_depth,
        post_conv_depth,
        fusion_mode,
        gate_type,
        use_channel_gate,
        use_spatial_gate,
        pos_embed_type,
        condition_dim,
        text_dim,
        has_text,
        ssm_cfg,
        norm_epsilon,
        rms_norm,
        residual_in_fp32,
        fused_add_norm,
        scan_type,
        processor_scan_type,
        processor_window_size,
        processor_shift_size,
        use_mamba_size_router,
        mamba_size_presets,
        mamba_size_router_top_k,
        mamba_size_router_weight_mode,
        use_jit,
        use_checkpoint,
        device,
        dtype,
    ):
        super().__init__()
        self.resolution = resolution
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fusion_mode = fusion_mode
        self.gate_type = gate_type
        self.pos_embed_type = pos_embed_type
        self.use_channel_gate = use_channel_gate
        self.use_spatial_gate = use_spatial_gate
        self.text_dim = int(condition_dim if text_dim is None else text_dim)
        self.processor_scan_type = processor_scan_type
        self.processor_window_size = int(processor_window_size)
        self.processor_shift_size = int(processor_shift_size)
        self.upsample = ConvUpsampleBlock(
            input_dim,
            output_dim,
            upsample=True,
            device=device,
            dtype=dtype,
        )
        self.skip_proj = (
            nn.Identity()
            if skip_dim == output_dim
            else nn.Linear(skip_dim, output_dim, device=device, dtype=dtype)
        )
        if fusion_mode == "concat":
            self.local_fusion = SpatialFusionBlock(
                low_dim=output_dim,
                skip_dim=output_dim,
                out_dim=output_dim,
                refinement_depth=fusion_conv_depth,
                device=device,
                dtype=dtype,
            )
            self.gated_sum_refinement = None
        elif fusion_mode == "gated_sum":
            self.local_fusion = None
            self.gated_sum_refinement = MapConvResidualStack(
                output_dim,
                depth=max(1, fusion_conv_depth),
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        if gate_type != "stage_timestep":
            raise ValueError(f"Unsupported fusion_gate_type: {gate_type}")
        if pos_embed_type not in {"anchor_shared", "none"}:
            raise ValueError(f"Unsupported fusion_pos_embed_type: {pos_embed_type}")
        self.gate = StageTimestepGate(condition_dim, device, dtype)
        self.channel_gate = (
            StageContentAwareChannelGate(output_dim, condition_dim, device, dtype)
            if use_channel_gate
            else None
        )
        self.spatial_gate = (
            StageContentAwareSpatialGate(output_dim, condition_dim, device, dtype)
            if use_spatial_gate
            else None
        )
        self.condition_proj = (
            nn.Identity()
            if condition_dim == output_dim
            else nn.Linear(condition_dim, output_dim, device=device, dtype=dtype)
        )
        self.uses_decoder_text_proj = bool(has_text and self.text_dim != output_dim)
        self.text_proj = (
            None
            if not has_text
            else (
                nn.Linear(self.text_dim, output_dim, device=device, dtype=dtype)
                if self.uses_decoder_text_proj
                else nn.Identity()
            )
        )
        self.absolute_pos_embed = (
            AnchorPositionalEmbedding(resolution, output_dim, device, dtype)
            if pos_embed_type == "anchor_shared"
            else None
        )
        self.child_pos_embed = (
            ChildOffsetPositionalEmbedding(output_dim, device, dtype)
            if pos_embed_type == "anchor_shared"
            else None
        )
        self.low_only_refinement = MapConvResidualStack(
            output_dim,
            depth=max(1, fusion_conv_depth),
            device=device,
            dtype=dtype,
        )
        self.pre_mamba_blocks = MapConvResidualStack(
            output_dim,
            depth=pre_conv_depth,
            device=device,
            dtype=dtype,
        )
        if use_mamba_size_router:
            self.processor = RoutedMambaSizeProcessor(
                dim=output_dim,
                resolution=resolution,
                preset_specs=mamba_size_presets,
                has_text=has_text,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                drop_path_rate=0.0,
                scan_type=scan_type,
                processor_scan_type=self.processor_scan_type,
                processor_window_size=self.processor_window_size,
                processor_shift_size=self.processor_shift_size,
                router_top_k=mamba_size_router_top_k,
                router_weight_mode=mamba_size_router_weight_mode,
                use_jit=use_jit,
                use_checkpoint=use_checkpoint,
                device=device,
                dtype=dtype,
            )
        else:
            dpr = torch.linspace(0, 0.0, stage_depth).tolist()
            processor_cls = (
                MapWindowMambaResidualProcessor
                if self.processor_window_size > 0
                else MapMambaResidualProcessor
            )
            processor_kwargs = dict(
                dim=output_dim,
                depth=stage_depth,
                resolution=resolution,
                has_text=has_text,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                drop_path_values=dpr,
                scan_type=self.processor_scan_type,
                use_jit=use_jit,
                use_checkpoint=use_checkpoint,
                device=device,
                dtype=dtype,
            )
            if processor_cls is MapWindowMambaResidualProcessor:
                processor_kwargs.update(
                    window_size=self.processor_window_size,
                    shift_size=self.processor_shift_size,
                )
            self.processor = processor_cls(**processor_kwargs)
        self.post_mamba_blocks = MapConvResidualStack(
            output_dim,
            depth=post_conv_depth,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        low_map,
        skip_map,
        condition,
        text=None,
        disable_stage=False,
        force_stage_gate_value=None,
        gate_floor=None,
        stage_router_weight=None,
    ):
        low = low_map.permute(0, 3, 1, 2).contiguous()
        low = self.upsample(low).permute(0, 2, 3, 1).contiguous()
        if low.shape[1:3] != (self.resolution, self.resolution):
            low_cf = low.permute(0, 3, 1, 2).contiguous()
            low_cf = F.interpolate(
                low_cf,
                size=(self.resolution, self.resolution),
                mode="nearest",
            )
            low = low_cf.permute(0, 2, 3, 1).contiguous()
        if self.child_pos_embed is not None:
            low = self.child_pos_embed(low)

        stage_stats = {
            "resolution": int(self.resolution),
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "skip_used": float(skip_map is not None),
            "uses_stage_pos_embed": float(self.absolute_pos_embed is not None),
            "uses_child_pos_embed": float(self.child_pos_embed is not None),
            "uses_content_channel_gate": float(self.channel_gate is not None),
            "uses_spatial_gate": float(self.spatial_gate is not None),
            "uses_local_window_mamba": float(self.processor_window_size > 0),
            "uses_decoder_text_proj": float(self.uses_decoder_text_proj),
        }
        processor_condition = self.condition_proj(condition)
        if text is not None:
            stage_stats["decoder_text_input_dim"] = int(text.shape[-1])
        processor_text = (
            self.text_proj(text)
            if text is not None and self.text_proj is not None
            else text
        )
        if processor_text is not None:
            stage_stats["decoder_text_dim"] = int(processor_text.shape[-1])
        if skip_map is None:
            if self.absolute_pos_embed is not None:
                low = self.absolute_pos_embed(low)
            low = self.low_only_refinement(low)
            low = self.pre_mamba_blocks(low)
            refined, block_stats = self.processor(
                low, processor_condition, text=processor_text
            )
            refined = self.post_mamba_blocks(refined)
            stage_stats.update(block_stats)
            return refined, stage_stats

        skip = self.skip_proj(skip_map)
        raw_gate = self.gate.compute_gate(condition, skip.shape[0])
        gate_logits = torch.logit(raw_gate.clamp(1e-6, 1.0 - 1e-6))
        gate = raw_gate
        if self.spatial_gate is not None:
            spatial_gate_delta = self.spatial_gate(condition, low, skip)
            gate_logits = gate_logits + spatial_gate_delta
            stage_stats["spatial_gate_delta_norm"] = spatial_gate_delta.norm(dim=-1).mean().detach()
        if self.channel_gate is not None:
            channel_gate_delta = self.channel_gate(condition, low, skip)
            gate_logits = gate_logits + channel_gate_delta
            stage_stats["channel_gate_delta_norm"] = channel_gate_delta.norm(dim=-1).mean().detach()
        if self.spatial_gate is not None or self.channel_gate is not None:
            gate = torch.sigmoid(gate_logits)
        if disable_stage:
            gate = torch.zeros_like(gate)
        if force_stage_gate_value is not None:
            gate = torch.full_like(gate, float(force_stage_gate_value))
        if gate_floor is not None:
            gate = gate_floor + (1.0 - gate_floor) * gate
        if stage_router_weight is not None:
            gate = gate * stage_router_weight
            stage_stats["stage_router_weight"] = stage_router_weight.mean().detach()

        gated_skip = skip * gate
        if self.fusion_mode == "concat":
            fused = self.local_fusion(low, gated_skip)
        else:
            fused = low + gated_skip
            fused = self.gated_sum_refinement(fused)
        if self.absolute_pos_embed is not None:
            fused = self.absolute_pos_embed(fused)
        fused = self.pre_mamba_blocks(fused)

        refined, block_stats = self.processor(
            fused, processor_condition, text=processor_text
        )
        refined = self.post_mamba_blocks(refined)
        stage_stats.update(block_stats)
        stage_stats["raw_gate_mean"] = raw_gate.mean().detach()
        stage_stats["raw_gate_std"] = raw_gate.std(unbiased=False).detach()
        stage_stats["gate_mean"] = gate.mean().detach()
        stage_stats["gate_std"] = gate.std(unbiased=False).detach()
        stage_stats["skip_norm"] = gated_skip.norm(dim=-1).mean().detach()
        return refined, stage_stats

    def forward_from_skip(
        self,
        skip_map,
        condition,
        text=None,
        disable_stage=False,
        force_stage_gate_value=None,
        gate_floor=None,
        stage_router_weight=None,
    ):
        stage_stats = {
            "resolution": int(self.resolution),
            "input_dim": int(self.output_dim),
            "output_dim": int(self.output_dim),
            "skip_used": 1.0,
            "started_from_bottleneck_skip": 1.0,
            "uses_stage_pos_embed": float(self.absolute_pos_embed is not None),
            "uses_child_pos_embed": 0.0,
            "uses_content_channel_gate": float(self.channel_gate is not None),
            "uses_spatial_gate": float(self.spatial_gate is not None),
            "uses_local_window_mamba": float(self.processor_window_size > 0),
            "uses_decoder_text_proj": float(self.uses_decoder_text_proj),
        }
        processor_condition = self.condition_proj(condition)
        if text is not None:
            stage_stats["decoder_text_input_dim"] = int(text.shape[-1])
        processor_text = (
            self.text_proj(text)
            if text is not None and self.text_proj is not None
            else text
        )
        if processor_text is not None:
            stage_stats["decoder_text_dim"] = int(processor_text.shape[-1])

        skip = self.skip_proj(skip_map)
        raw_gate = self.gate.compute_gate(condition, skip.shape[0])
        gate_logits = torch.logit(raw_gate.clamp(1e-6, 1.0 - 1e-6))
        gate = raw_gate
        if self.spatial_gate is not None:
            spatial_gate_delta = self.spatial_gate(condition, skip, skip)
            gate_logits = gate_logits + spatial_gate_delta
            stage_stats["spatial_gate_delta_norm"] = (
                spatial_gate_delta.norm(dim=-1).mean().detach()
            )
        if self.channel_gate is not None:
            channel_gate_delta = self.channel_gate(condition, skip, skip)
            gate_logits = gate_logits + channel_gate_delta
            stage_stats["channel_gate_delta_norm"] = (
                channel_gate_delta.norm(dim=-1).mean().detach()
            )
        if self.spatial_gate is not None or self.channel_gate is not None:
            gate = torch.sigmoid(gate_logits)
        if disable_stage:
            gate = torch.zeros_like(gate)
        if force_stage_gate_value is not None:
            gate = torch.full_like(gate, float(force_stage_gate_value))
        if gate_floor is not None:
            gate = gate_floor + (1.0 - gate_floor) * gate
        if stage_router_weight is not None:
            gate = gate * stage_router_weight
            stage_stats["stage_router_weight"] = stage_router_weight.mean().detach()

        started = skip * gate
        if self.absolute_pos_embed is not None:
            started = self.absolute_pos_embed(started)
        started = self.pre_mamba_blocks(started)
        refined, block_stats = self.processor(
            started, processor_condition, text=processor_text
        )
        refined = self.post_mamba_blocks(refined)
        stage_stats.update(block_stats)
        stage_stats["raw_gate_mean"] = raw_gate.mean().detach()
        stage_stats["raw_gate_std"] = raw_gate.std(unbiased=False).detach()
        stage_stats["gate_mean"] = gate.mean().detach()
        stage_stats["gate_std"] = gate.std(unbiased=False).detach()
        stage_stats["skip_norm"] = started.norm(dim=-1).mean().detach()
        return refined, stage_stats


class HierarchicalMambaHybrid(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_dim,
        img_dim,
        patch_size=1,
        out_channels=None,
        has_text=False,
        num_classes=-1,
        d_context=0,
        drop_path_rate=0.0,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=True,
        fused_add_norm=True,
        residual_in_fp32=True,
        scan_type="v2",
        video_frames=0,
        tpe=False,
        device="cuda",
        use_pe=0,
        use_jit=True,
        use_checkpoint=False,
        dtype=torch.float32,
        hierarchy_window_size=2,
        hierarchy_stride=2,
        first_layer_stride=None,
        context_compress_type="mean",
        hierarchy_max_stages=None,
        hierarchy_stage_depth=2,
        hierarchy_allow_partial=True,
        share_stage_processor=False,
        hierarchical_output_mode="prediction",
        use_multiscale_fusion_head=True,
        fusion_mode="concat",
        fusion_selected_stages="32,16,8",
        fusion_anchor_resolution=4,
        decoder_anchor_resolution=None,
        fusion_stage_dim=256,
        fusion_stage_dim_overrides=None,
        fusion_gate_type="stage_timestep",
        fusion_pos_embed_type="anchor_shared",
        fusion_block_depth=1,
        fusion_stage_depth_overrides=None,
        fusion_channel_gate_stages=None,
        fusion_use_spatial_gate=False,
        fusion_conv_depth=0,
        fusion_pre_mamba_conv_depth=0,
        fusion_post_mamba_conv_depth=0,
        anchor_builder_depth=None,
        stage_scan_type_overrides=None,
        local_mamba_stage_resolutions="",
        local_mamba_window_size_overrides=None,
        local_mamba_shift_resolutions="",
        fusion_prediction_head_type="anchor_conv_upsample",
        fusion_logging_verbose=True,
        highres_stage_depth=2,
        bottleneck_stage_depth=2,
        downsample_use_premix=False,
        downsample_premix_depth=0,
        downsample_conv_type="standard",
        highres_local_conv_depth=0,
        highres_local_conv_type="standard",
        prediction_head_conv_depth=0,
        prediction_head_conv_type="standard",
        final_skip_refiner_depth=0,
        final_skip_refiner_conv_type="standard",
        final_skip_refiner_use_channel_gate=True,
        final_skip_refiner_use_spatial_gate=True,
        use_aux_4x4_context=False,
        aux_4x4_inject_target="decoder_anchor",
        aux_4x4_context_resolution=4,
        aux_4x4_target_resolution=None,
        aux_4x4_prepool_depth=1,
        aux_4x4_context_depth=1,
        aux_4x4_fusion_depth=1,
        aux_4x4_conv_type="standard",
        aux_4x4_upsample_mode="bilinear",
        aux_4x4_use_timestep_gate=True,
        aux_4x4_residual_scale_multiplier=1.0,
        use_factorized_top4_router=False,
        routed_stage_resolutions="auto",
        routed_stage_count=4,
        include_anchor_in_stage_router=True,
        stage_router_top_k=4,
        stage_router_weight_floor=0.0,
        stage_router_max_weight=2.0,
        stage_router_weight_mode="selection",
        use_mamba_size_router=False,
        mamba_size_router_stages="auto",
        mamba_size_presets="small:1:0.75,base:2:1.0,large:4:1.25",
        encoder_mamba_size_presets=None,
        fusion_mamba_size_presets=None,
        encoder_mamba_size_preset_overrides=None,
        fusion_mamba_size_preset_overrides=None,
        mamba_size_router_top_k=1,
        mamba_size_router_weight_mode="selection",
        use_compression_router=False,
        compression_router_stages="auto",
        use_integrated_router_controller=False,
        integrated_controller_stage_top_k=None,
        integrated_controller_hidden_dim=None,
        integrated_controller_use_channel_gate=False,
        integrated_controller_channel_gate_scale=0.1,
        integrated_controller_stage_select_mode="topk",
        integrated_controller_stage_select_threshold=0.5,
        integrated_controller_stage_select_threshold_margin=0.0,
        integrated_controller_stage_select_warmup_steps=0,
        integrated_controller_stage_select_warmup_min_selected=0,
        integrated_controller_stage_select_warmup_threshold_margin=0.0,
        integrated_controller_stage_min_selected=1,
        integrated_controller_stage_balance_mode="none",
        integrated_controller_stage_use_scale_prior_context=True,
        use_dynamic_bottleneck=False,
        dynamic_bottleneck_candidate_stages="auto",
        use_encoder_mamba_depth_router=None,
        encoder_mamba_depth_router_stages="auto",
        encoder_mamba_depth_router_top_k=1,
        ):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.has_text = has_text
        self.ssm_cfg = ssm_cfg
        self.norm_epsilon = norm_epsilon
        self.rms_norm = rms_norm
        self.use_jit = use_jit
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.drop_path_rate = drop_path_rate
        self.scan_type = scan_type
        self.use_pe = use_pe
        self.video_frames = video_frames
        self.tpe = tpe
        self.use_checkpoint = use_checkpoint
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.hierarchical_context = True
        self.hierarchy_window_size = hierarchy_window_size
        self.hierarchy_stride = hierarchy_stride
        self.first_layer_stride = (
            hierarchy_stride if first_layer_stride is None else first_layer_stride
        )
        self.context_compress_type = context_compress_type
        self.hierarchy_max_stages = hierarchy_max_stages
        self.hierarchy_stage_depth = hierarchy_stage_depth
        self.hierarchy_allow_partial = hierarchy_allow_partial
        self.hierarchical_output_mode = hierarchical_output_mode
        self.share_stage_processor = share_stage_processor
        self.use_multiscale_fusion_head = use_multiscale_fusion_head
        self.fusion_mode = fusion_mode
        self.fusion_selected_stages = parse_stage_resolution_spec(fusion_selected_stages)
        self.fusion_anchor_resolution = fusion_anchor_resolution
        self.bottleneck_resolution = fusion_anchor_resolution
        self.decoder_anchor_resolution = (
            fusion_anchor_resolution
            if decoder_anchor_resolution is None
            else int(decoder_anchor_resolution)
        )
        self.fusion_stage_dim = fusion_stage_dim
        self.fusion_stage_dim_overrides = parse_stage_override_spec(
            fusion_stage_dim_overrides, int
        )
        self.fusion_gate_type = fusion_gate_type
        self.fusion_pos_embed_type = fusion_pos_embed_type
        self.fusion_block_depth = fusion_block_depth
        self.fusion_stage_depth_overrides = parse_stage_override_spec(
            fusion_stage_depth_overrides, int
        )
        self.fusion_channel_gate_stages = {
            int(resolution)
            for resolution in parse_stage_resolution_spec(fusion_channel_gate_stages)
        }
        self.fusion_use_spatial_gate = bool(fusion_use_spatial_gate)
        self.fusion_conv_depth = int(fusion_conv_depth)
        self.fusion_pre_mamba_conv_depth = int(fusion_pre_mamba_conv_depth)
        self.fusion_post_mamba_conv_depth = int(fusion_post_mamba_conv_depth)
        self.anchor_builder_depth = (
            fusion_block_depth
            if anchor_builder_depth is None
            else int(anchor_builder_depth)
        )
        self.stage_scan_type_overrides = parse_stage_override_spec(
            stage_scan_type_overrides,
            lambda value: str(value).strip(),
        )
        self.local_mamba_stage_resolutions = set(
            parse_stage_resolution_spec(local_mamba_stage_resolutions)
        )
        self.local_mamba_window_size_overrides = parse_stage_override_spec(
            local_mamba_window_size_overrides, int
        )
        self.local_mamba_shift_resolutions = set(
            parse_stage_resolution_spec(local_mamba_shift_resolutions)
        )
        self.fusion_prediction_head_type = fusion_prediction_head_type
        self.fusion_logging_verbose = fusion_logging_verbose
        self.highres_stage_depth = highres_stage_depth
        self.bottleneck_stage_depth = bottleneck_stage_depth
        self.downsample_use_premix = bool(downsample_use_premix)
        self.downsample_premix_depth = int(downsample_premix_depth)
        self.downsample_conv_type = downsample_conv_type
        self.highres_local_conv_depth = int(highres_local_conv_depth)
        self.highres_local_conv_type = highres_local_conv_type
        self.prediction_head_conv_depth = int(prediction_head_conv_depth)
        self.prediction_head_conv_type = prediction_head_conv_type
        self.final_skip_refiner_depth = int(final_skip_refiner_depth)
        self.final_skip_refiner_conv_type = final_skip_refiner_conv_type
        self.final_skip_refiner_use_channel_gate = bool(final_skip_refiner_use_channel_gate)
        self.final_skip_refiner_use_spatial_gate = bool(final_skip_refiner_use_spatial_gate)
        self.use_aux_4x4_context = bool(use_aux_4x4_context)
        aux_4x4_inject_target = str(aux_4x4_inject_target).strip().lower()
        if aux_4x4_inject_target in {"anchor", "decoder", "decoder_anchor"}:
            aux_4x4_inject_target = "decoder_anchor"
        elif aux_4x4_inject_target in {"final", "final_map", "head", "prediction_head"}:
            aux_4x4_inject_target = "final_map"
        else:
            raise ValueError(
                "aux_4x4_inject_target must be decoder_anchor or final_map, "
                f"got {aux_4x4_inject_target!r}"
            )
        self.aux_4x4_inject_target = aux_4x4_inject_target
        self.aux_4x4_context_resolution = int(aux_4x4_context_resolution)
        self.aux_4x4_target_resolution_spec = aux_4x4_target_resolution
        self.aux_4x4_target_resolution = None
        self.aux_4x4_prepool_depth = int(aux_4x4_prepool_depth)
        self.aux_4x4_context_depth = int(aux_4x4_context_depth)
        self.aux_4x4_fusion_depth = int(aux_4x4_fusion_depth)
        self.aux_4x4_conv_type = aux_4x4_conv_type
        self.aux_4x4_upsample_mode = aux_4x4_upsample_mode
        self.aux_4x4_use_timestep_gate = bool(aux_4x4_use_timestep_gate)
        self.aux_4x4_residual_scale_multiplier = float(
            aux_4x4_residual_scale_multiplier
        )
        self.use_factorized_top4_router = bool(use_factorized_top4_router)
        self.routed_stage_resolution_spec = routed_stage_resolutions
        self.routed_stage_count = int(routed_stage_count)
        self.include_anchor_in_stage_router = bool(include_anchor_in_stage_router)
        self.routed_stage_resolutions = []
        self.stage_router_top_k = int(stage_router_top_k)
        self.stage_router_weight_floor = float(stage_router_weight_floor)
        self.stage_router_max_weight = (
            None if stage_router_max_weight is None else float(stage_router_max_weight)
        )
        self.stage_router_weight_mode = str(stage_router_weight_mode).strip().lower()
        if self.stage_router_weight_mode in {"prob", "probability", "weighted"}:
            self.stage_router_weight_mode = "equal_selection"
        elif self.stage_router_weight_mode in {
            "selection",
            "mask",
            "hard",
            "equal",
            "equal_selection",
        }:
            self.stage_router_weight_mode = "equal_selection"
        else:
            raise ValueError(
                "stage_router_weight_mode must be 'equal_selection', "
                f"got {stage_router_weight_mode!r}"
            )
        self.stage_router_weight_floor = 0.0
        self.use_mamba_size_router = bool(use_mamba_size_router)
        self.mamba_size_router_stage_spec = mamba_size_router_stages
        self.mamba_size_router_stages = set()
        self.encoder_mamba_size_presets = parse_compute_preset_spec(
            encoder_mamba_size_presets
            if encoder_mamba_size_presets is not None
            else mamba_size_presets
        )
        self.fusion_mamba_size_presets = parse_compute_preset_spec(
            fusion_mamba_size_presets
            if fusion_mamba_size_presets is not None
            else mamba_size_presets
        )
        self.encoder_mamba_size_preset_overrides = (
            parse_stage_compute_preset_override_spec(
                encoder_mamba_size_preset_overrides
            )
        )
        self.fusion_mamba_size_preset_overrides = (
            parse_stage_compute_preset_override_spec(
                fusion_mamba_size_preset_overrides
            )
        )
        self.mamba_size_presets = self.encoder_mamba_size_presets
        self.mamba_size_router_top_k = int(mamba_size_router_top_k)
        self.mamba_size_router_weight_mode = str(mamba_size_router_weight_mode).strip().lower()
        if self.mamba_size_router_weight_mode in {"prob", "probability", "weighted"}:
            self.mamba_size_router_weight_mode = "selection"
        elif self.mamba_size_router_weight_mode in {"selection", "mask", "hard"}:
            self.mamba_size_router_weight_mode = "selection"
        else:
            raise ValueError(
                "mamba_size_router_weight_mode must be 'selection', "
                f"got {mamba_size_router_weight_mode!r}"
            )
        self.use_compression_router = bool(use_compression_router)
        self.compression_router_stage_spec = compression_router_stages
        self.compression_router_stages = set()
        self.use_integrated_router_controller = bool(use_integrated_router_controller)
        self.integrated_controller_stage_top_k = (
            self.stage_router_top_k
            if integrated_controller_stage_top_k is None
            else int(integrated_controller_stage_top_k)
        )
        integrated_controller_hidden_dim_spec = integrated_controller_hidden_dim
        self.integrated_controller_hidden_dim = (
            None
            if integrated_controller_hidden_dim_spec is None
            or (
                isinstance(integrated_controller_hidden_dim_spec, str)
                and integrated_controller_hidden_dim_spec.strip().lower()
                in {"", "auto", "none", "null"}
            )
            else int(integrated_controller_hidden_dim_spec)
        )
        self.integrated_controller_use_channel_gate = bool(
            integrated_controller_use_channel_gate
        )
        self.integrated_controller_channel_gate_scale = float(
            integrated_controller_channel_gate_scale
        )
        self.integrated_controller_stage_select_mode = str(
            integrated_controller_stage_select_mode
        ).strip().lower()
        if self.integrated_controller_stage_select_mode in {
            "topk",
            "top_k",
            "fixed_topk",
            "fixed",
        }:
            self.integrated_controller_stage_select_mode = "topk"
        elif self.integrated_controller_stage_select_mode in {
            "adaptive_topk",
            "adaptive_top_k",
            "variable_topk",
            "variable_top_k",
            "learned_k",
        }:
            self.integrated_controller_stage_select_mode = "adaptive_topk"
        elif self.integrated_controller_stage_select_mode in {
            "adaptive",
            "threshold",
            "sigmoid",
            "independent",
        }:
            self.integrated_controller_stage_select_mode = "adaptive"
        elif self.integrated_controller_stage_select_mode in {
            "relu",
            "relu_gate",
            "relu_routing",
            "relu_threshold",
            "ssr_relu",
            "remoe",
            "remoe_relu",
        }:
            self.integrated_controller_stage_select_mode = "relu"
        threshold_spec = (
            str(integrated_controller_stage_select_threshold).strip().lower()
        )
        self.integrated_controller_stage_select_threshold_mode = "fixed"
        if threshold_spec in {
            "mean",
            "avg",
            "average",
            "prob_mean",
            "prob_avg",
            "probability_mean",
            "probability_avg",
            "stage_prob_mean",
            "stage_probability_mean",
            "running_prob_mean",
            "cumulative_prob_mean",
        }:
            self.integrated_controller_stage_select_threshold = threshold_spec
            self.integrated_controller_stage_select_threshold_value = 0.0
            self.integrated_controller_stage_select_threshold_mode = (
                "cumulative_prob_mean"
            )
        else:
            self.integrated_controller_stage_select_threshold = float(
                integrated_controller_stage_select_threshold
            )
            self.integrated_controller_stage_select_threshold_value = float(
                self.integrated_controller_stage_select_threshold
            )
        self.integrated_controller_stage_select_threshold_margin = float(
            integrated_controller_stage_select_threshold_margin
        )
        self.integrated_controller_stage_select_warmup_steps = int(
            integrated_controller_stage_select_warmup_steps
        )
        self.integrated_controller_stage_select_warmup_min_selected = int(
            integrated_controller_stage_select_warmup_min_selected
        )
        self.integrated_controller_stage_select_warmup_threshold_margin = float(
            integrated_controller_stage_select_warmup_threshold_margin
        )
        self.integrated_controller_stage_min_selected = int(
            integrated_controller_stage_min_selected
        )
        self.integrated_controller_stage_balance_mode = str(
            integrated_controller_stage_balance_mode
        ).strip().lower()
        self.integrated_controller_stage_use_scale_prior_context = bool(
            integrated_controller_stage_use_scale_prior_context
        )
        if self.integrated_controller_stage_balance_mode in {
            "",
            "none",
            "off",
            "false",
            "0",
        }:
            self.integrated_controller_stage_balance_mode = "none"
        elif self.integrated_controller_stage_balance_mode in {
            "batch",
            "balanced",
            "balance",
        }:
            self.integrated_controller_stage_balance_mode = "batch"
        elif self.integrated_controller_stage_balance_mode in {
            "batch_center",
            "batch_centered",
        }:
            self.integrated_controller_stage_balance_mode = "batch_center"
        elif self.integrated_controller_stage_balance_mode in {
            "center",
            "centered",
            "logit_center",
            "logit_centered",
        }:
            self.integrated_controller_stage_balance_mode = "logit_center"
        self.use_dynamic_bottleneck = bool(use_dynamic_bottleneck)
        self.dynamic_bottleneck_candidate_spec = dynamic_bottleneck_candidate_stages
        self.dynamic_bottleneck_candidates = []
        self.dynamic_bottleneck_candidates_ascending = []
        self.use_encoder_mamba_depth_router = (
            self.use_integrated_router_controller
            if use_encoder_mamba_depth_router is None
            else bool(use_encoder_mamba_depth_router)
        )
        self.encoder_mamba_depth_router_stage_spec = encoder_mamba_depth_router_stages
        self.encoder_mamba_depth_router_stages = set()
        self.encoder_mamba_depth_router_top_k = int(encoder_mamba_depth_router_top_k)
        self.latest_hierarchy_stats = {}
        self.latest_fusion_stats = {}
        self.latest_backbone_stats = {}
        self.latest_factorized_router_stats = {}
        self.disabled_stage_resolutions = set()
        self.force_stage_gate_values = {}
        self.gate_floor = None
        if str(device) == "cpu":
            fused_add_norm = False
            rms_norm = False

        if video_frames != 0:
            raise NotImplementedError("HierarchicalMambaHybrid currently supports images only")
        if hierarchical_output_mode not in {"prediction", "context"}:
            raise ValueError(
                f"Unsupported hierarchical_output_mode: {hierarchical_output_mode}"
            )
        if fusion_anchor_resolution <= 0:
            raise ValueError(
                f"fusion_anchor_resolution must be positive, got {fusion_anchor_resolution}"
            )
        if self.decoder_anchor_resolution <= 0:
            raise ValueError(
                "decoder_anchor_resolution must be positive, "
                f"got {self.decoder_anchor_resolution}"
            )
        if fusion_stage_dim <= 0:
            raise ValueError(f"fusion_stage_dim must be positive, got {fusion_stage_dim}")
        if hierarchy_stage_depth <= 0:
            raise ValueError(
                f"hierarchy_stage_depth must be positive, got {hierarchy_stage_depth}"
            )
        if highres_stage_depth <= 0:
            raise ValueError(
                f"highres_stage_depth must be positive, got {highres_stage_depth}"
            )
        if bottleneck_stage_depth <= 0:
            raise ValueError(
                f"bottleneck_stage_depth must be positive, got {bottleneck_stage_depth}"
            )
        if fusion_block_depth <= 0:
            raise ValueError(
                f"fusion_block_depth must be positive, got {fusion_block_depth}"
            )
        if self.fusion_conv_depth < 0:
            raise ValueError(
                f"fusion_conv_depth must be non-negative, got {fusion_conv_depth}"
            )
        if self.fusion_pre_mamba_conv_depth < 0:
            raise ValueError(
                "fusion_pre_mamba_conv_depth must be non-negative, "
                f"got {fusion_pre_mamba_conv_depth}"
            )
        if self.fusion_post_mamba_conv_depth < 0:
            raise ValueError(
                "fusion_post_mamba_conv_depth must be non-negative, "
                f"got {fusion_post_mamba_conv_depth}"
            )
        if self.anchor_builder_depth <= 0:
            raise ValueError(
                f"anchor_builder_depth must be positive, got {self.anchor_builder_depth}"
            )
        for resolution, window_size in self.local_mamba_window_size_overrides.items():
            if window_size <= 0:
                raise ValueError(
                    "local_mamba_window_size_overrides values must be positive, "
                    f"got {resolution}:{window_size}"
                )
        if self.downsample_premix_depth < 0:
            raise ValueError(
                "downsample_premix_depth must be non-negative, "
                f"got {downsample_premix_depth}"
            )
        for name, conv_type in {
            "downsample_conv_type": self.downsample_conv_type,
            "highres_local_conv_type": self.highres_local_conv_type,
            "prediction_head_conv_type": self.prediction_head_conv_type,
            "final_skip_refiner_conv_type": self.final_skip_refiner_conv_type,
            "aux_4x4_conv_type": self.aux_4x4_conv_type,
        }.items():
            if conv_type not in {"standard", "separable"}:
                raise ValueError(
                    f"{name} must be either 'standard' or 'separable', got {conv_type!r}"
                )
        if self.highres_local_conv_depth < 0:
            raise ValueError(
                f"highres_local_conv_depth must be non-negative, got {highres_local_conv_depth}"
            )
        if self.prediction_head_conv_depth < 0:
            raise ValueError(
                "prediction_head_conv_depth must be non-negative, "
                f"got {prediction_head_conv_depth}"
            )
        if self.final_skip_refiner_depth < 0:
            raise ValueError(
                "final_skip_refiner_depth must be non-negative, "
                f"got {final_skip_refiner_depth}"
            )
        if self.aux_4x4_context_resolution <= 0:
            raise ValueError(
                "aux_4x4_context_resolution must be positive, "
                f"got {aux_4x4_context_resolution}"
            )
        if self.aux_4x4_prepool_depth < 0:
            raise ValueError(
                "aux_4x4_prepool_depth must be non-negative, "
                f"got {aux_4x4_prepool_depth}"
            )
        if self.aux_4x4_context_depth < 0:
            raise ValueError(
                "aux_4x4_context_depth must be non-negative, "
                f"got {aux_4x4_context_depth}"
            )
        if self.aux_4x4_fusion_depth < 0:
            raise ValueError(
                "aux_4x4_fusion_depth must be non-negative, "
                f"got {aux_4x4_fusion_depth}"
            )
        if self.aux_4x4_upsample_mode not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError(
                "aux_4x4_upsample_mode must be nearest, bilinear, or bicubic, "
                f"got {aux_4x4_upsample_mode!r}"
            )
        if self.stage_router_top_k <= 0:
            raise ValueError(
                f"stage_router_top_k must be positive, got {stage_router_top_k}"
            )
        if self.use_factorized_top4_router and self.stage_router_top_k < 2:
            raise ValueError(
                "stage_router_top_k=1 disables multi-stage routing and is not "
                "supported for this factorized-router preset."
            )
        if self.routed_stage_count <= 0:
            raise ValueError(
                f"routed_stage_count must be positive, got {routed_stage_count}"
            )
        if not 0.0 <= self.stage_router_weight_floor < 1.0:
            raise ValueError(
                "stage_router_weight_floor must be in [0, 1), "
                f"got {stage_router_weight_floor}"
            )
        if self.stage_router_max_weight is not None and self.stage_router_max_weight <= 0:
            raise ValueError(
                "stage_router_max_weight must be positive or null, "
                f"got {stage_router_max_weight}"
            )
        if (
            self.stage_router_weight_mode == "equal_selection"
            and self.stage_router_weight_floor > 0
        ):
            raise ValueError(
                "stage_router_weight_floor must be 0 when "
                "stage_router_weight_mode='equal_selection'"
            )
        if self.use_mamba_size_router and not self.encoder_mamba_size_presets:
            raise ValueError("encoder_mamba_size_presets must not be empty when enabled")
        if self.use_mamba_size_router and not self.fusion_mamba_size_presets:
            raise ValueError("fusion_mamba_size_presets must not be empty when enabled")
        if self.mamba_size_router_top_k <= 0:
            raise ValueError(
                "mamba_size_router_top_k must be positive, "
                f"got {mamba_size_router_top_k}"
            )
        if self.use_integrated_router_controller and self.use_mamba_size_router:
            raise ValueError(
                "integrated router controller does not include mamba-size routing; "
                "set use_mamba_size_router=false"
            )
        if self.use_encoder_mamba_depth_router and not self.use_integrated_router_controller:
            raise ValueError(
                "encoder_mamba_depth_router requires use_integrated_router_controller=true"
            )
        if self.encoder_mamba_depth_router_top_k != 1:
            raise ValueError(
                "encoder_mamba_depth_router_top_k must be 1 for sparse depth routing, "
                f"got {self.encoder_mamba_depth_router_top_k}"
            )
        if self.integrated_controller_stage_top_k <= 0:
            raise ValueError(
                "integrated_controller_stage_top_k must be positive, "
                f"got {self.integrated_controller_stage_top_k}"
            )
        for resolution, depth in self.fusion_stage_depth_overrides.items():
            if depth <= 0:
                raise ValueError(
                    f"fusion_stage_depth_overrides[{resolution}] must be positive, got {depth}"
                )
        for resolution, dim in self.fusion_stage_dim_overrides.items():
            if dim <= 0:
                raise ValueError(
                    f"fusion_stage_dim_overrides[{resolution}] must be positive, got {dim}"
                )

        num_patches = (img_dim // patch_size) ** 2
        self.patch_side_len = int(math.sqrt(num_patches))
        self.hierarchy_input_size = (self.patch_side_len, self.patch_side_len)
        if fusion_anchor_resolution > self.patch_side_len:
            raise ValueError(
                "fusion_anchor_resolution must not exceed the token resolution, "
                f"got {fusion_anchor_resolution} > {self.patch_side_len}"
            )
        if self.decoder_anchor_resolution > self.patch_side_len:
            raise ValueError(
                "decoder_anchor_resolution must not exceed the token resolution, "
                f"got {self.decoder_anchor_resolution} > {self.patch_side_len}"
            )
        if self.decoder_anchor_resolution < fusion_anchor_resolution:
            raise ValueError(
                "decoder_anchor_resolution must be >= bottleneck resolution, "
                f"got {self.decoder_anchor_resolution} < {fusion_anchor_resolution}"
            )
        target_resolution_spec = self.aux_4x4_target_resolution_spec
        if target_resolution_spec in {None, "", "auto", 0, "0"}:
            self.aux_4x4_target_resolution = (
                self.decoder_anchor_resolution
                if self.aux_4x4_inject_target == "decoder_anchor"
                else self.patch_side_len
            )
        elif target_resolution_spec in {"decoder_anchor", "anchor", "decoder"}:
            self.aux_4x4_target_resolution = self.decoder_anchor_resolution
        elif target_resolution_spec in {"final", "final_map", "head", "prediction_head"}:
            self.aux_4x4_target_resolution = self.patch_side_len
        else:
            self.aux_4x4_target_resolution = int(target_resolution_spec)
        if self.use_aux_4x4_context:
            expected_aux_target_resolution = (
                self.decoder_anchor_resolution
                if self.aux_4x4_inject_target == "decoder_anchor"
                else self.patch_side_len
            )
            if self.aux_4x4_target_resolution != expected_aux_target_resolution:
                raise ValueError(
                    "aux_4x4_target_resolution does not match "
                    f"{self.aux_4x4_inject_target}: got "
                    f"{self.aux_4x4_target_resolution}, expected "
                    f"{expected_aux_target_resolution}"
                )

        if video_frames == 0:
            self.x_embedder = (
                PatchEmbed(img_dim, patch_size, in_channels, embed_dim, bias=True)
                .to(device)
                .to(dtype)
            )
        else:
            self.x_embedder = (
                PatchEmbed_Video(img_dim, patch_size, in_channels, embed_dim, bias=True)
                .to(device)
                .to(dtype)
            )
        self.t_embedder = (
            TimestepEmbedder(self.embed_dim, dtype=dtype).to(device).to(dtype)
        )

        if has_text:
            self.y_embedder = nn.Linear(d_context, embed_dim).to(device).to(dtype)
        elif num_classes > 0:
            self.y_embedder = (
                LabelEmbedder(num_classes, hidden_size=embed_dim, dropout_prob=0.0)
                .to(device)
                .to(dtype)
            )
        else:
            self.y_embedder = None
        self.has_text = has_text
        self.num_classes = num_classes

        if self.use_pe == 1:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim, device=device, dtype=dtype),
                requires_grad=False,
            )
        elif self.use_pe == 2:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim, device=device, dtype=dtype)
            )
        elif self.use_pe == 0:
            self.pos_embed = None
        else:
            raise ValueError("use_pe should be 0, 1, or 2")

        self.hierarchy_stage_layout = []
        current_resolution = self.patch_side_len
        current_stride = self.first_layer_stride
        while current_resolution > fusion_anchor_resolution:
            if current_resolution % current_stride != 0:
                raise ValueError(
                    "Hybrid hierarchy requires exact divisibility at each transition. "
                    f"resolution={current_resolution}, stride={current_stride}"
                )
            next_resolution = current_resolution // current_stride
            self.hierarchy_stage_layout.append(
                {
                    "layer_idx": len(self.hierarchy_stage_layout),
                    "input_resolution": (current_resolution, current_resolution),
                    "output_resolution": (next_resolution, next_resolution),
                    "stride": current_stride,
                }
            )
            current_resolution = next_resolution
            current_stride = self.hierarchy_stride
            if hierarchy_max_stages is not None and len(self.hierarchy_stage_layout) >= hierarchy_max_stages:
                break

        if current_resolution != fusion_anchor_resolution:
            raise ValueError(
                "Hybrid hierarchy must reach the configured bottleneck resolution. "
                f"Reached {current_resolution}, expected {fusion_anchor_resolution}."
            )

        self.hierarchy_final_resolution = (fusion_anchor_resolution, fusion_anchor_resolution)
        self.hierarchy_reaches_global_context = self.hierarchy_final_resolution == (1, 1)
        available_routed_stage_resolutions = [self.patch_side_len]
        available_routed_stage_resolutions.extend(
            layout["output_resolution"][0]
            for layout in self.hierarchy_stage_layout
            if (
                layout["output_resolution"][0] > self.fusion_anchor_resolution
                or self.include_anchor_in_stage_router
            )
        )
        auto_routed_stage_resolutions = list(available_routed_stage_resolutions)
        auto_routed_stage_resolutions = auto_routed_stage_resolutions[
            : self.routed_stage_count
        ]
        auto_compression_stage_resolutions = [
            layout["output_resolution"][0]
            for layout in self.hierarchy_stage_layout
        ]
        auto_encoder_depth_router_resolutions = [self.patch_side_len]
        auto_encoder_depth_router_resolutions.extend(auto_compression_stage_resolutions)
        self.routed_stage_resolutions = resolve_stage_spec(
            self.routed_stage_resolution_spec,
            auto_routed_stage_resolutions,
        )
        self.mamba_size_router_stages = set(
            resolve_stage_spec(
                self.mamba_size_router_stage_spec,
                auto_routed_stage_resolutions,
            )
        )
        self.compression_router_stages = set(
            resolve_stage_spec(
                self.compression_router_stage_spec,
                auto_compression_stage_resolutions,
            )
        )
        self.encoder_mamba_depth_router_stages = set(
            resolve_stage_spec(
                self.encoder_mamba_depth_router_stage_spec,
                auto_encoder_depth_router_resolutions,
            )
        )
        valid_router_resolutions = set(available_routed_stage_resolutions)
        invalid_routed_resolutions = sorted(
            set(self.routed_stage_resolutions) - valid_router_resolutions
        )
        invalid_mamba_router_resolutions = sorted(
            self.mamba_size_router_stages - valid_router_resolutions
        )
        invalid_compression_router_resolutions = sorted(
            self.compression_router_stages - set(auto_compression_stage_resolutions)
        )
        invalid_depth_router_resolutions = sorted(
            self.encoder_mamba_depth_router_stages
            - set(auto_encoder_depth_router_resolutions)
        )
        if invalid_routed_resolutions:
            raise ValueError(
                "routed_stage_resolutions includes resolutions outside the current "
                f"pyramid: {invalid_routed_resolutions}; "
                f"valid={sorted(valid_router_resolutions, reverse=True)}"
            )
        if invalid_mamba_router_resolutions:
            raise ValueError(
                "mamba_size_router_stages includes resolutions outside the current "
                f"pyramid: {invalid_mamba_router_resolutions}; "
                f"valid={sorted(valid_router_resolutions, reverse=True)}"
            )
        if invalid_compression_router_resolutions:
            raise ValueError(
                "compression_router_stages includes transition outputs outside the "
                f"current pyramid: {invalid_compression_router_resolutions}; "
                f"valid={sorted(auto_compression_stage_resolutions, reverse=True)}"
            )
        if invalid_depth_router_resolutions:
            raise ValueError(
                "encoder_mamba_depth_router_stages includes resolutions outside the "
                f"current encoder pyramid: {invalid_depth_router_resolutions}; "
                f"valid={sorted(auto_encoder_depth_router_resolutions, reverse=True)}"
            )

        if is_auto_stage_spec(self.dynamic_bottleneck_candidate_spec):
            dynamic_bottleneck_candidates = list(self.routed_stage_resolutions)
        else:
            dynamic_bottleneck_candidates = parse_stage_resolution_spec(
                self.dynamic_bottleneck_candidate_spec
            )
        dynamic_bottleneck_candidates = [
            int(resolution) for resolution in dynamic_bottleneck_candidates
        ]
        invalid_dynamic_bottleneck_resolutions = sorted(
            set(dynamic_bottleneck_candidates) - valid_router_resolutions
        )
        if invalid_dynamic_bottleneck_resolutions:
            raise ValueError(
                "dynamic_bottleneck_candidate_stages includes resolutions outside "
                "the routed pyramid: "
                f"{invalid_dynamic_bottleneck_resolutions}; "
                f"valid={sorted(valid_router_resolutions, reverse=True)}"
            )
        if self.use_dynamic_bottleneck:
            if not dynamic_bottleneck_candidates:
                raise ValueError("dynamic bottleneck requires at least one candidate stage")
            if not (
                self.use_integrated_router_controller or self.use_factorized_top4_router
            ):
                raise ValueError(
                    "dynamic bottleneck requires a stage router; enable "
                    "use_integrated_router_controller or use_factorized_top4_router"
                )
            if self.decoder_anchor_resolution != self.fusion_anchor_resolution:
                raise ValueError(
                    "dynamic bottleneck requires decoder_anchor_resolution == "
                    "fusion_anchor_resolution so any selected stage can start the "
                    "same decoder chain; got "
                    f"{self.decoder_anchor_resolution} != {self.fusion_anchor_resolution}"
                )
            if min(dynamic_bottleneck_candidates) < self.fusion_anchor_resolution:
                raise ValueError(
                    "dynamic bottleneck candidates must be at or above "
                    "fusion_anchor_resolution; got "
                    f"{dynamic_bottleneck_candidates} with "
                    f"fusion_anchor_resolution={self.fusion_anchor_resolution}"
                )
        self.dynamic_bottleneck_candidates = sorted(
            set(dynamic_bottleneck_candidates), reverse=True
        )
        self.dynamic_bottleneck_candidates_ascending = sorted(
            self.dynamic_bottleneck_candidates
        )

        self.highres_local_stem = MapConvResidualStack(
            embed_dim,
            depth=self.highres_local_conv_depth,
            device=device,
            dtype=dtype,
            conv_type=self.highres_local_conv_type,
        )
        self.highres_processor = self._build_stage_processor(
            self.patch_side_len, highres_stage_depth
        )

        self.downsamplers = nn.ModuleList()
        self.encoder_processors = nn.ModuleDict()
        for layout in self.hierarchy_stage_layout:
            out_resolution = layout["output_resolution"][0]
            if self.use_compression_router and out_resolution in self.compression_router_stages:
                self.downsamplers.append(
                    RoutedCompressionDownsample2d(
                        embed_dim,
                        stride=layout["stride"],
                        use_premix=self.downsample_use_premix,
                        premix_depth=self.downsample_premix_depth,
                        conv_type=self.downsample_conv_type,
                        condition_dim=self.embed_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
            else:
                self.downsamplers.append(
                    LearnedDownsample2d(
                        embed_dim,
                        embed_dim,
                        stride=layout["stride"],
                        use_premix=self.downsample_use_premix,
                        premix_depth=self.downsample_premix_depth,
                        conv_type=self.downsample_conv_type,
                        device=device,
                        dtype=dtype,
                    )
                )
            if out_resolution > self.fusion_anchor_resolution:
                self.encoder_processors[str(out_resolution)] = self._build_stage_processor(
                    out_resolution, hierarchy_stage_depth
                )

        self.integrated_router_controller = None
        self.stage_contribution_router = None
        if self.use_integrated_router_controller:
            self.integrated_router_controller = IntegratedHierarchicalRoutingController(
                self.routed_stage_resolutions,
                condition_dim=self.embed_dim,
                feature_dim=embed_dim,
                top_k=self.integrated_controller_stage_top_k,
                hidden_dim=self.integrated_controller_hidden_dim,
                use_channel_gate=self.integrated_controller_use_channel_gate,
                channel_gate_scale=self.integrated_controller_channel_gate_scale,
                stage_select_mode=self.integrated_controller_stage_select_mode,
                stage_select_threshold=(
                    self.integrated_controller_stage_select_threshold
                ),
                stage_select_threshold_margin=(
                    self.integrated_controller_stage_select_threshold_margin
                ),
                stage_select_warmup_steps=(
                    self.integrated_controller_stage_select_warmup_steps
                ),
                stage_select_warmup_min_selected=(
                    self.integrated_controller_stage_select_warmup_min_selected
                ),
                stage_select_warmup_threshold_margin=(
                    self.integrated_controller_stage_select_warmup_threshold_margin
                ),
                stage_min_selected=self.integrated_controller_stage_min_selected,
                stage_balance_mode=self.integrated_controller_stage_balance_mode,
                stage_use_scale_prior_context=(
                    self.integrated_controller_stage_use_scale_prior_context
                ),
                device=device,
                dtype=dtype,
            )
        elif self.use_factorized_top4_router:
            self.stage_contribution_router = StageContributionRouter(
                self.routed_stage_resolutions,
                condition_dim=self.embed_dim,
                feature_dim=embed_dim,
                top_k=self.stage_router_top_k,
                weight_floor=self.stage_router_weight_floor,
                max_weight=self.stage_router_max_weight,
                weight_mode=self.stage_router_weight_mode,
                device=device,
                dtype=dtype,
            )

        self.bottleneck_processor = self._build_stage_processor(
            self.fusion_anchor_resolution, bottleneck_stage_depth
        )

        selected_skip_resolutions = set(self.fusion_selected_stages) | {self.patch_side_len}
        if self.use_dynamic_bottleneck:
            selected_skip_resolutions.update(self.dynamic_bottleneck_candidates)
        required_skip_resolutions = set(selected_skip_resolutions)
        if self.decoder_anchor_resolution > self.fusion_anchor_resolution:
            required_skip_resolutions.add(self.decoder_anchor_resolution)
        valid_skip_resolutions = {self.patch_side_len}
        valid_skip_resolutions.update(
            layout["output_resolution"][0]
            for layout in self.hierarchy_stage_layout
            if layout["output_resolution"][0] >= self.fusion_anchor_resolution
        )
        invalid_skip_resolutions = sorted(required_skip_resolutions - valid_skip_resolutions)
        if invalid_skip_resolutions:
            raise ValueError(
                "fusion_selected_stages includes resolutions that do not exist in the "
                f"hybrid pyramid: {invalid_skip_resolutions}; valid={sorted(valid_skip_resolutions)}"
            )
        invalid_stage_depth_resolutions = sorted(
            set(self.fusion_stage_depth_overrides) - required_skip_resolutions
        )
        if invalid_stage_depth_resolutions:
            raise ValueError(
                "fusion_stage_depth_overrides includes resolutions that are not used by the "
                f"decoder: {invalid_stage_depth_resolutions}; "
                f"valid={sorted(required_skip_resolutions)}"
            )
        invalid_stage_dim_resolutions = sorted(
            set(self.fusion_stage_dim_overrides) - required_skip_resolutions
        )
        if invalid_stage_dim_resolutions:
            raise ValueError(
                "fusion_stage_dim_overrides includes resolutions that are not used by the "
                f"decoder: {invalid_stage_dim_resolutions}; "
                f"valid={sorted(required_skip_resolutions)}"
            )
        invalid_channel_gate_stages = sorted(
            self.fusion_channel_gate_stages - required_skip_resolutions
        )
        if invalid_channel_gate_stages:
            raise ValueError(
                "fusion_channel_gate_stages includes resolutions that are not used by the "
                f"decoder: {invalid_channel_gate_stages}; "
                f"valid={sorted(required_skip_resolutions)}"
            )
        self.fusion_selected_stages = sorted(selected_skip_resolutions, reverse=True)
        self.required_skip_resolutions = sorted(required_skip_resolutions, reverse=True)

        self.decoder_stage_resolutions = sorted(
            [
                resolution
                for resolution in valid_skip_resolutions
                if resolution > self.decoder_anchor_resolution
            ]
        )
        self.decoder_stage_depths = {
            resolution: self.fusion_stage_depth_overrides.get(
                resolution, self.fusion_block_depth
            )
            for resolution in self.decoder_stage_resolutions
        }
        self.decoder_stage_dims = {
            self.decoder_anchor_resolution: self.fusion_stage_dim_overrides.get(
                self.decoder_anchor_resolution, fusion_stage_dim
            )
        }
        self.decoder_stage_input_dims = {}
        previous_dim = self.decoder_stage_dims[self.decoder_anchor_resolution]
        for resolution in self.decoder_stage_resolutions:
            self.decoder_stage_input_dims[resolution] = previous_dim
            stage_dim = self.fusion_stage_dim_overrides.get(resolution, fusion_stage_dim)
            self.decoder_stage_dims[resolution] = stage_dim
            previous_dim = stage_dim
        self.final_decoder_dim = previous_dim
        self.aux_4x4_context_adapter = None
        if self.use_aux_4x4_context:
            aux_4x4_target_dim = (
                self.decoder_stage_dims[self.decoder_anchor_resolution]
                if self.aux_4x4_inject_target == "decoder_anchor"
                else self.final_decoder_dim
            )
            self.aux_4x4_context_adapter = Aux4x4DecoderContextAdapter(
                context_dim=embed_dim,
                target_dim=aux_4x4_target_dim,
                context_resolution=self.aux_4x4_context_resolution,
                target_resolution=self.aux_4x4_target_resolution,
                condition_dim=self.embed_dim,
                prepool_depth=self.aux_4x4_prepool_depth,
                context_depth=self.aux_4x4_context_depth,
                fusion_depth=self.aux_4x4_fusion_depth,
                conv_type=self.aux_4x4_conv_type,
                upsample_mode=self.aux_4x4_upsample_mode,
                use_timestep_gate=self.aux_4x4_use_timestep_gate,
                residual_scale_multiplier=self.aux_4x4_residual_scale_multiplier,
                device=device,
                dtype=dtype,
            )

        self.anchor_builder_stage_depth = self.fusion_stage_depth_overrides.get(
            self.decoder_anchor_resolution, self.anchor_builder_depth
        )
        self.bottleneck_proj = (
            nn.Identity()
            if self.decoder_stage_dims[self.decoder_anchor_resolution] == embed_dim
            else nn.Linear(
                embed_dim,
                self.decoder_stage_dims[self.decoder_anchor_resolution],
                device=device,
                dtype=dtype,
            )
        )
        self.decoder_anchor_condition_proj = None
        self.decoder_anchor_text_proj = None
        self.decoder_anchor_pre_mamba_blocks = None
        self.decoder_anchor_processor = None
        self.decoder_anchor_post_mamba_blocks = None
        if has_text and self.decoder_anchor_resolution == self.fusion_anchor_resolution:
            decoder_anchor_dim = self.decoder_stage_dims[self.decoder_anchor_resolution]
            self.decoder_anchor_condition_proj = (
                nn.Identity()
                if self.embed_dim == decoder_anchor_dim
                else nn.Linear(
                    self.embed_dim, decoder_anchor_dim, device=device, dtype=dtype
                )
            )
            self.decoder_anchor_text_proj = (
                nn.Identity()
                if d_context == decoder_anchor_dim
                else nn.Linear(d_context, decoder_anchor_dim, device=device, dtype=dtype)
            )
            self.decoder_anchor_pre_mamba_blocks = MapConvResidualStack(
                decoder_anchor_dim,
                depth=self.fusion_pre_mamba_conv_depth,
                device=device,
                dtype=dtype,
            )
            if self.use_mamba_size_router and (
                self.decoder_anchor_resolution in self.mamba_size_router_stages
            ):
                self.decoder_anchor_processor = RoutedMambaSizeProcessor(
                    dim=decoder_anchor_dim,
                    resolution=self.decoder_anchor_resolution,
                    preset_specs=self._resolve_fusion_mamba_size_presets(
                        self.decoder_anchor_resolution
                    ),
                    has_text=has_text,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    drop_path_rate=0.0,
                    scan_type=scan_type,
                    processor_scan_type=self._resolve_stage_scan_type(
                        self.decoder_anchor_resolution
                    ),
                    processor_window_size=self._resolve_local_window_size(
                        self.decoder_anchor_resolution
                    ),
                    processor_shift_size=self._resolve_local_shift_size(
                        self.decoder_anchor_resolution
                    ),
                    router_top_k=self.mamba_size_router_top_k,
                    router_weight_mode=self.mamba_size_router_weight_mode,
                    use_jit=use_jit,
                    use_checkpoint=use_checkpoint,
                    device=device,
                    dtype=dtype,
                )
            else:
                decoder_anchor_processor_cls = (
                    MapWindowMambaResidualProcessor
                    if self._resolve_local_window_size(self.decoder_anchor_resolution) > 0
                    else MapMambaResidualProcessor
                )
                decoder_anchor_processor_kwargs = dict(
                    dim=decoder_anchor_dim,
                    depth=self.anchor_builder_stage_depth,
                    resolution=self.decoder_anchor_resolution,
                    has_text=has_text,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    drop_path_values=torch.linspace(
                        0, 0.0, self.anchor_builder_stage_depth
                    ).tolist(),
                    scan_type=self._resolve_stage_scan_type(
                        self.decoder_anchor_resolution
                    ),
                    use_jit=use_jit,
                    use_checkpoint=use_checkpoint,
                    device=device,
                    dtype=dtype,
                )
                if decoder_anchor_processor_cls is MapWindowMambaResidualProcessor:
                    decoder_anchor_processor_kwargs.update(
                        window_size=self._resolve_local_window_size(
                            self.decoder_anchor_resolution
                        ),
                        shift_size=self._resolve_local_shift_size(
                            self.decoder_anchor_resolution
                        ),
                    )
                self.decoder_anchor_processor = decoder_anchor_processor_cls(
                    **decoder_anchor_processor_kwargs
                )
            self.decoder_anchor_post_mamba_blocks = MapConvResidualStack(
                decoder_anchor_dim,
                depth=self.fusion_post_mamba_conv_depth,
                device=device,
                dtype=dtype,
            )
        self.anchor_builder = None
        if self.decoder_anchor_resolution > self.fusion_anchor_resolution:
            self.anchor_builder = DecoderFusionStage(
                resolution=self.decoder_anchor_resolution,
                input_dim=self.decoder_stage_dims[self.decoder_anchor_resolution],
                output_dim=self.decoder_stage_dims[self.decoder_anchor_resolution],
                skip_dim=embed_dim,
                stage_depth=self.anchor_builder_stage_depth,
                fusion_conv_depth=self.fusion_conv_depth,
                pre_conv_depth=self.fusion_pre_mamba_conv_depth,
                post_conv_depth=self.fusion_post_mamba_conv_depth,
                fusion_mode=fusion_mode,
                gate_type=fusion_gate_type,
                use_channel_gate=self.decoder_anchor_resolution in self.fusion_channel_gate_stages,
                use_spatial_gate=self.fusion_use_spatial_gate,
                pos_embed_type=fusion_pos_embed_type,
                condition_dim=self.embed_dim,
                text_dim=d_context if has_text else None,
                has_text=has_text,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                scan_type=scan_type,
                processor_scan_type=self._resolve_stage_scan_type(
                    self.decoder_anchor_resolution
                ),
                processor_window_size=self._resolve_local_window_size(
                    self.decoder_anchor_resolution
                ),
                processor_shift_size=self._resolve_local_shift_size(
                    self.decoder_anchor_resolution
                ),
                use_mamba_size_router=(
                    self.use_mamba_size_router
                    and self.decoder_anchor_resolution in self.mamba_size_router_stages
                ),
                mamba_size_presets=self._resolve_fusion_mamba_size_presets(
                    self.decoder_anchor_resolution
                ),
                mamba_size_router_top_k=self.mamba_size_router_top_k,
                mamba_size_router_weight_mode=self.mamba_size_router_weight_mode,
                use_jit=use_jit,
                use_checkpoint=use_checkpoint,
                device=device,
                dtype=dtype,
            )
        self.decoder_stages = nn.ModuleList(
            [
                DecoderFusionStage(
                    resolution=resolution,
                    input_dim=self.decoder_stage_input_dims[resolution],
                    output_dim=self.decoder_stage_dims[resolution],
                    skip_dim=embed_dim,
                    stage_depth=self.decoder_stage_depths[resolution],
                    fusion_conv_depth=self.fusion_conv_depth,
                    pre_conv_depth=self.fusion_pre_mamba_conv_depth,
                    post_conv_depth=self.fusion_post_mamba_conv_depth,
                    fusion_mode=fusion_mode,
                    gate_type=fusion_gate_type,
                    use_channel_gate=resolution in self.fusion_channel_gate_stages,
                    use_spatial_gate=self.fusion_use_spatial_gate,
                    pos_embed_type=fusion_pos_embed_type,
                    condition_dim=self.embed_dim,
                    text_dim=d_context if has_text else None,
                    has_text=has_text,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    scan_type=scan_type,
                    processor_scan_type=self._resolve_stage_scan_type(resolution),
                    processor_window_size=self._resolve_local_window_size(resolution),
                    processor_shift_size=self._resolve_local_shift_size(resolution),
                    use_mamba_size_router=(
                        self.use_mamba_size_router
                        and resolution in self.mamba_size_router_stages
                    ),
                    mamba_size_presets=self._resolve_fusion_mamba_size_presets(
                        resolution
                    ),
                    mamba_size_router_top_k=self.mamba_size_router_top_k,
                    mamba_size_router_weight_mode=self.mamba_size_router_weight_mode,
                    use_jit=use_jit,
                    use_checkpoint=use_checkpoint,
                    device=device,
                    dtype=dtype,
                )
                for resolution in self.decoder_stage_resolutions
            ]
        )
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            self.final_decoder_dim, eps=norm_epsilon, **self.factory_kwargs
        )
        self.prediction_head = MapPredictionHead(
            hidden_size=self.final_decoder_dim,
            patch_size=patch_size,
            out_channels=self.out_channels,
            prediction_head_type=fusion_prediction_head_type,
            conv_block_depth=self.prediction_head_conv_depth,
            conv_type=self.prediction_head_conv_type,
            device=device,
            dtype=dtype,
        )
        self.final_skip_refiner = None
        if self.final_skip_refiner_depth > 0:
            self.final_skip_refiner = FinalSkipRefiner(
                resolution=self.patch_side_len,
                decoder_dim=self.final_decoder_dim,
                skip_dim=embed_dim,
                condition_dim=self.embed_dim,
                refinement_depth=self.final_skip_refiner_depth,
                refinement_conv_type=self.final_skip_refiner_conv_type,
                use_channel_gate=self.final_skip_refiner_use_channel_gate,
                use_spatial_gate=self.final_skip_refiner_use_spatial_gate,
                device=device,
                dtype=dtype,
            )

        self.initialize_weights()

    def _resolve_stage_scan_type(self, resolution):
        return self.stage_scan_type_overrides.get(resolution, self.scan_type)

    def _resolve_local_window_size(self, resolution):
        if resolution not in self.local_mamba_stage_resolutions:
            return 0
        window_size = self.local_mamba_window_size_overrides.get(
            resolution, min(8, int(resolution))
        )
        if window_size > resolution:
            raise ValueError(
                "local mamba window size must not exceed the stage resolution, "
                f"got {window_size} > {resolution}"
            )
        return int(window_size)

    def _resolve_local_shift_size(self, resolution):
        if resolution not in self.local_mamba_shift_resolutions:
            return 0
        window_size = self._resolve_local_window_size(resolution)
        if window_size <= 1:
            return 0
        return max(1, window_size // 2)

    def _resolve_encoder_mamba_size_presets(self, resolution):
        return self.encoder_mamba_size_preset_overrides.get(
            int(resolution), self.encoder_mamba_size_presets
        )

    def _resolve_fusion_mamba_size_presets(self, resolution):
        return self.fusion_mamba_size_preset_overrides.get(
            int(resolution), self.fusion_mamba_size_presets
        )

    def _build_stage_processor(self, resolution, depth):
        processor_scan_type = self._resolve_stage_scan_type(resolution)
        window_size = self._resolve_local_window_size(resolution)
        shift_size = self._resolve_local_shift_size(resolution)
        if (
            self.use_encoder_mamba_depth_router
            and int(resolution) in self.encoder_mamba_depth_router_stages
        ):
            return RoutedEncoderMambaDepthProcessor(
                dim=self.embed_dim,
                resolution=resolution,
                base_depth=depth,
                has_text=self.has_text,
                ssm_cfg=self.ssm_cfg,
                norm_epsilon=self.norm_epsilon,
                rms_norm=self.rms_norm,
                residual_in_fp32=self.residual_in_fp32,
                fused_add_norm=self.fused_add_norm,
                drop_path_rate=self.drop_path_rate,
                scan_type=self.scan_type,
                processor_scan_type=processor_scan_type,
                processor_window_size=window_size,
                processor_shift_size=shift_size,
                router_top_k=self.encoder_mamba_depth_router_top_k,
                use_jit=self.use_jit,
                use_checkpoint=self.use_checkpoint,
                device=self.factory_kwargs["device"],
                dtype=self.factory_kwargs["dtype"],
            )
        if self.use_mamba_size_router and int(resolution) in self.mamba_size_router_stages:
            return RoutedMambaSizeProcessor(
                dim=self.embed_dim,
                resolution=resolution,
                preset_specs=self._resolve_encoder_mamba_size_presets(resolution),
                has_text=self.has_text,
                ssm_cfg=self.ssm_cfg,
                norm_epsilon=self.norm_epsilon,
                rms_norm=self.rms_norm,
                residual_in_fp32=self.residual_in_fp32,
                fused_add_norm=self.fused_add_norm,
                drop_path_rate=self.drop_path_rate,
                scan_type=self.scan_type,
                processor_scan_type=processor_scan_type,
                processor_window_size=window_size,
                processor_shift_size=shift_size,
                router_top_k=self.mamba_size_router_top_k,
                router_weight_mode=self.mamba_size_router_weight_mode,
                use_jit=self.use_jit,
                use_checkpoint=self.use_checkpoint,
                device=self.factory_kwargs["device"],
                dtype=self.factory_kwargs["dtype"],
            )
        processor_cls = (
            MapWindowMambaResidualProcessor
            if window_size > 0
            else MapMambaResidualProcessor
        )
        processor_kwargs = dict(
            dim=self.embed_dim,
            depth=depth,
            resolution=resolution,
            has_text=self.has_text,
            ssm_cfg=self.ssm_cfg,
            norm_epsilon=self.norm_epsilon,
            rms_norm=self.rms_norm,
            residual_in_fp32=self.residual_in_fp32,
            fused_add_norm=self.fused_add_norm,
            drop_path_values=torch.linspace(0, self.drop_path_rate, depth).tolist(),
            scan_type=processor_scan_type,
            use_jit=self.use_jit,
            use_checkpoint=self.use_checkpoint,
            device=self.factory_kwargs["device"],
            dtype=self.factory_kwargs["dtype"],
        )
        if processor_cls is MapWindowMambaResidualProcessor:
            processor_kwargs.update(window_size=window_size, shift_size=shift_size)
        return processor_cls(**processor_kwargs)

    def initialize_weights(self):
        if self.use_pe == 1:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def _get_condition(self, hidden_states, t, y):
        t = (t * 1000.0).to(hidden_states)
        timestep_condition = self.t_embedder(t)
        raw_text = y
        if self.has_text:
            y = self.y_embedder(raw_text)
            c = timestep_condition + y.mean(dim=1)
        elif self.num_classes > 0:
            c = timestep_condition + self.y_embedder(y, self.training)
        else:
            c = timestep_condition
        return c, y, timestep_condition, raw_text

    def _apply_decoder_stage(
        self,
        resolution,
        decoder_map,
        skip_map,
        c,
        decoder_text,
        stage_router_weight=None,
    ):
        disable_stage = resolution in self.disabled_stage_resolutions
        force_stage_gate_value = self.force_stage_gate_values.get(resolution)
        stage = self.decoder_stages[self.decoder_stage_resolutions.index(resolution)]
        return stage(
            decoder_map,
            skip_map,
            c,
            text=decoder_text,
            disable_stage=disable_stage,
            force_stage_gate_value=force_stage_gate_value,
            gate_floor=self.gate_floor,
            stage_router_weight=stage_router_weight,
        )

    def _select_dynamic_bottleneck_resolutions(
        self,
        stage_router_weights,
        batch_size,
        device,
    ):
        selected_masks = {}
        for resolution in self.dynamic_bottleneck_candidates_ascending:
            weights = stage_router_weights.get(resolution)
            if weights is None:
                selected = torch.zeros(batch_size, dtype=torch.bool, device=device)
            else:
                selected = weights.detach().reshape(batch_size, -1).amax(dim=1) > 0
            selected_masks[resolution] = selected

        selected_stack = torch.stack(
            [
                selected_masks[resolution]
                for resolution in self.dynamic_bottleneck_candidates_ascending
            ],
            dim=1,
        )
        selected_counts = selected_stack.sum(dim=1)
        candidate_tensor = torch.tensor(
            self.dynamic_bottleneck_candidates_ascending,
            dtype=torch.long,
            device=device,
        )
        first_selected = selected_stack.to(torch.int64).argmax(dim=1)
        bottleneck_resolutions = candidate_tensor[first_selected]
        fallback_mask = selected_counts == 0
        if bool(fallback_mask.any().item()):
            bottleneck_resolutions = torch.where(
                fallback_mask,
                candidate_tensor[0],
                bottleneck_resolutions,
            )
        return bottleneck_resolutions, selected_masks, selected_counts

    def _apply_decoder_anchor_processor_for_dynamic(
        self,
        decoder_map,
        c,
        decoder_text,
        decoder_stage_stats,
    ):
        if self.decoder_anchor_processor is None:
            return decoder_map
        anchor_stats = {
            "resolution": int(self.decoder_anchor_resolution),
            "input_dim": int(self.decoder_stage_dims[self.decoder_anchor_resolution]),
            "output_dim": int(self.decoder_stage_dims[self.decoder_anchor_resolution]),
            "skip_used": 0.0,
            "is_anchor_stage": 1.0,
            "uses_decoder_anchor_processor": 1.0,
            "dynamic_bottleneck_anchor_processor": 1.0,
        }
        if decoder_text is not None:
            anchor_stats["decoder_text_input_dim"] = int(decoder_text.shape[-1])
        if self.decoder_anchor_resolution in self.disabled_stage_resolutions:
            anchor_stats["disabled"] = 1.0
        else:
            anchor_condition = self.decoder_anchor_condition_proj(c)
            anchor_text = (
                self.decoder_anchor_text_proj(decoder_text)
                if decoder_text is not None
                else None
            )
            if anchor_text is not None:
                anchor_stats["decoder_text_dim"] = int(anchor_text.shape[-1])
            decoder_map = self.decoder_anchor_pre_mamba_blocks(decoder_map)
            decoder_map, block_stats = self.decoder_anchor_processor(
                decoder_map, anchor_condition, text=anchor_text
            )
            decoder_map = self.decoder_anchor_post_mamba_blocks(decoder_map)
            anchor_stats.update(block_stats)
        decoder_stage_stats.append(anchor_stats)
        return decoder_map

    def _start_decoder_from_dynamic_bottleneck(
        self,
        bottleneck_resolution,
        skip_maps,
        stage_router_weights,
        c,
        decoder_text,
        decoder_stage_stats,
    ):
        router_weight = stage_router_weights.get(bottleneck_resolution)
        if bottleneck_resolution == self.fusion_anchor_resolution:
            decoder_map = self.bottleneck_proj(skip_maps[bottleneck_resolution])
            start_stats = {
                "resolution": int(bottleneck_resolution),
                "dynamic_bottleneck_start": 1.0,
                "started_from_static_bottleneck": 1.0,
            }
            if router_weight is not None:
                decoder_map = decoder_map * router_weight
                start_stats["stage_router_weight"] = router_weight.mean().detach()
            decoder_stage_stats.append(start_stats)
            if bottleneck_resolution == self.decoder_anchor_resolution:
                decoder_map = self._apply_decoder_anchor_processor_for_dynamic(
                    decoder_map,
                    c,
                    decoder_text,
                    decoder_stage_stats,
                )
            return decoder_map, bottleneck_resolution

        if bottleneck_resolution == self.decoder_anchor_resolution and self.anchor_builder is not None:
            stage = self.anchor_builder
        else:
            if bottleneck_resolution not in self.decoder_stage_resolutions:
                raise RuntimeError(
                    "dynamic bottleneck resolution is not reachable by decoder "
                    f"stages: {bottleneck_resolution}; "
                    f"decoder_stage_resolutions={self.decoder_stage_resolutions}"
                )
            stage = self.decoder_stages[
                self.decoder_stage_resolutions.index(bottleneck_resolution)
            ]
        decoder_map, stage_stats = stage.forward_from_skip(
            skip_maps[bottleneck_resolution],
            c,
            text=decoder_text,
            disable_stage=bottleneck_resolution in self.disabled_stage_resolutions,
            force_stage_gate_value=self.force_stage_gate_values.get(
                bottleneck_resolution
            ),
            gate_floor=self.gate_floor,
            stage_router_weight=router_weight,
        )
        stage_stats["dynamic_bottleneck_start"] = 1.0
        stage_stats["dynamic_bottleneck_resolution"] = int(bottleneck_resolution)
        decoder_stage_stats.append(stage_stats)
        return decoder_map, bottleneck_resolution

    def _run_dynamic_bottleneck_decoder(
        self,
        bottleneck_map,
        skip_maps,
        c,
        decoder_text,
        stage_router_weights,
    ):
        batch_size = bottleneck_map.shape[0]
        bottleneck_resolutions, selected_masks, selected_counts = (
            self._select_dynamic_bottleneck_resolutions(
                stage_router_weights,
                batch_size=batch_size,
                device=bottleneck_map.device,
            )
        )
        final_maps = None
        decoder_stage_stats = []
        dynamic_bottleneck_metrics = []

        for bottleneck_resolution in self.dynamic_bottleneck_candidates_ascending:
            group_mask = bottleneck_resolutions == bottleneck_resolution
            group_count = int(group_mask.sum().item())
            if group_count == 0:
                continue
            group_indices = group_mask.nonzero(as_tuple=False).flatten()
            group_c = c.index_select(0, group_indices)
            group_decoder_text = (
                decoder_text.index_select(0, group_indices)
                if decoder_text is not None
                else None
            )
            group_skip_maps = {
                resolution: stage_map.index_select(0, group_indices)
                for resolution, stage_map in skip_maps.items()
            }
            group_router_weights = {
                resolution: weights.index_select(0, group_indices)
                for resolution, weights in stage_router_weights.items()
            }

            decoder_map, decoder_start_resolution = (
                self._start_decoder_from_dynamic_bottleneck(
                    bottleneck_resolution,
                    group_skip_maps,
                    group_router_weights,
                    group_c,
                    group_decoder_text,
                    decoder_stage_stats,
                )
            )
            for resolution in self.decoder_stage_resolutions:
                if resolution <= decoder_start_resolution:
                    continue
                decoder_map, stage_stats = self._apply_decoder_stage(
                    resolution,
                    decoder_map,
                    group_skip_maps.get(resolution),
                    group_c,
                    group_decoder_text,
                    stage_router_weight=group_router_weights.get(resolution),
                )
                stage_stats["dynamic_bottleneck_resolution"] = int(
                    bottleneck_resolution
                )
                decoder_stage_stats.append(stage_stats)

            if final_maps is None:
                final_maps = torch.zeros(
                    batch_size,
                    decoder_map.shape[1],
                    decoder_map.shape[2],
                    decoder_map.shape[3],
                    device=decoder_map.device,
                    dtype=decoder_map.dtype,
                )
            final_maps.index_copy_(0, group_indices, decoder_map)
            start_weight = group_router_weights.get(bottleneck_resolution)
            dynamic_bottleneck_metrics.append(
                {
                    "resolution": int(bottleneck_resolution),
                    "sample_fraction": group_mask.float().mean().detach(),
                    "start_weight": (
                        start_weight.mean().detach()
                        if start_weight is not None
                        else bottleneck_map.new_tensor(0.0).detach()
                    ),
                    "output_norm": decoder_map.norm(dim=-1).mean().detach(),
                }
            )

        if final_maps is None:
            raise RuntimeError("dynamic bottleneck decoder produced no decoder groups")

        selected_fractions = {
            resolution: selected_masks[resolution].float().mean().detach()
            for resolution in self.dynamic_bottleneck_candidates_ascending
        }
        bottleneck_fractions = {
            resolution: (bottleneck_resolutions == resolution).float().mean().detach()
            for resolution in self.dynamic_bottleneck_candidates_ascending
        }
        dynamic_stats = {
            "dynamic_bottleneck_enabled": True,
            "dynamic_bottleneck_candidates": list(self.dynamic_bottleneck_candidates),
            "dynamic_selected_fractions": selected_fractions,
            "dynamic_bottleneck_fractions": bottleneck_fractions,
            "dynamic_selected_count_mean": selected_counts.float().mean().detach(),
            "dynamic_selected_count_min": selected_counts.float().amin().detach(),
            "dynamic_selected_count_max": selected_counts.float().amax().detach(),
            "dynamic_bottleneck_metrics": dynamic_bottleneck_metrics,
        }
        return final_maps, decoder_stage_stats, dynamic_stats

    def _apply_encoder_processor(
        self,
        processor,
        context_map,
        c,
        text,
        resolution,
        base_depth,
        controller_scale_prior,
    ):
        if isinstance(processor, RoutedEncoderMambaDepthProcessor):
            controller_depth = None
            if self.integrated_router_controller is not None:
                controller_depth = self.integrated_router_controller.compute_encoder_depth(
                    c,
                    context_map,
                    resolution,
                    base_depth,
                    controller_scale_prior,
                )
            return processor(
                context_map,
                c,
                text=text,
                router_logits=(
                    None if controller_depth is None else controller_depth["logits"]
                ),
                controller_stats=(
                    None if controller_depth is None else controller_depth["stats"]
                ),
            )
        return processor(context_map, c=c, text=text)

    def forward_backbone(self, hidden_states, t, y=None):
        hidden_states = self.x_embedder(hidden_states)
        c, y, timestep_condition, raw_text = self._get_condition(hidden_states, t, y)
        decoder_text = raw_text if self.has_text else y
        controller_scale_prior = None
        if self.integrated_router_controller is not None:
            controller_scale_prior = self.integrated_router_controller.compute_scale_prior(c)

        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        initial_map = tokens_to_map(hidden_states, self.hierarchy_input_size)
        initial_map = self.highres_local_stem(initial_map)
        highres_map, highres_stats = self._apply_encoder_processor(
            self.highres_processor,
            initial_map,
            c,
            y,
            self.patch_side_len,
            self.highres_stage_depth,
            controller_scale_prior,
        )
        encoder_stage_stats = [dict(stage_idx=0, **highres_stats)]
        skip_maps = {self.patch_side_len: highres_map}
        compression_stage_stats = []

        current_map = highres_map
        for stage_idx, (layout, downsampler) in enumerate(
            zip(self.hierarchy_stage_layout, self.downsamplers), start=1
        ):
            out_resolution = layout["output_resolution"][0]
            if isinstance(downsampler, RoutedCompressionDownsample2d):
                controller_compression = None
                if self.integrated_router_controller is not None:
                    controller_compression = self.integrated_router_controller.compute_compression(
                        c,
                        current_map,
                        out_resolution,
                        self.hierarchy_stage_depth,
                        controller_scale_prior,
                    )
                current_map, compression_stats = downsampler(
                    current_map,
                    c,
                    external_logits=(
                        None
                        if controller_compression is None
                        else controller_compression["logits"]
                    ),
                    channel_gate=(
                        None
                        if controller_compression is None
                        else controller_compression["channel_gate"]
                    ),
                )
                if controller_compression is not None:
                    compression_stats.update(controller_compression["stats"])
                compression_stats["output_resolution"] = int(out_resolution)
                compression_stage_stats.append(compression_stats)
            else:
                current_map = downsampler(current_map)
                compression_stats = None
            if out_resolution > self.fusion_anchor_resolution:
                current_map, processor_stats = self._apply_encoder_processor(
                    self.encoder_processors[str(out_resolution)],
                    current_map,
                    c,
                    y,
                    out_resolution,
                    self.hierarchy_stage_depth,
                    controller_scale_prior,
                )
                if out_resolution in self.required_skip_resolutions:
                    skip_maps[out_resolution] = current_map
            else:
                current_map, processor_stats = self._apply_encoder_processor(
                    self.bottleneck_processor,
                    current_map,
                    c,
                    y,
                    out_resolution,
                    self.bottleneck_stage_depth,
                    controller_scale_prior,
                )
            encoder_stage_stats.append(
                {
                    "stage_idx": stage_idx,
                    "input_resolution": layout["input_resolution"],
                    "output_resolution": layout["output_resolution"],
                    "stride": layout["stride"],
                    **processor_stats,
                }
            )

        bottleneck_map = current_map
        skip_maps[self.fusion_anchor_resolution] = bottleneck_map
        stage_router_weights = {}
        stage_router_stats = {}
        if self.integrated_router_controller is not None:
            stage_router_weights, stage_router_stats = (
                self.integrated_router_controller.compute_stage_weights(
                    c,
                    skip_maps,
                    controller_scale_prior,
                )
            )
        elif self.stage_contribution_router is not None:
            stage_router_weights, stage_router_stats = self.stage_contribution_router(
                c,
                skip_maps,
            )
        aux_4x4_stats = None
        if self.use_dynamic_bottleneck:
            decoder_map, decoder_stage_stats, dynamic_bottleneck_stats = (
                self._run_dynamic_bottleneck_decoder(
                    bottleneck_map,
                    skip_maps,
                    c,
                    decoder_text,
                    stage_router_weights,
                )
            )
            stage_router_stats.update(dynamic_bottleneck_stats)
        else:
            decoder_map = self.bottleneck_proj(bottleneck_map)
            if not isinstance(decoder_map, torch.Tensor):
                decoder_map = bottleneck_map
            anchor_router_weight = stage_router_weights.get(self.fusion_anchor_resolution)
            if anchor_router_weight is not None:
                decoder_map = decoder_map * anchor_router_weight
            decoder_stage_stats = []
            if self.anchor_builder is not None:
                anchor_skip_map = skip_maps.get(self.decoder_anchor_resolution)
                decoder_map, anchor_stats = self.anchor_builder(
                    decoder_map,
                    anchor_skip_map,
                    c,
                    text=decoder_text,
                    disable_stage=self.decoder_anchor_resolution in self.disabled_stage_resolutions,
                    force_stage_gate_value=self.force_stage_gate_values.get(
                        self.decoder_anchor_resolution
                    ),
                    gate_floor=self.gate_floor,
                    stage_router_weight=stage_router_weights.get(
                        self.decoder_anchor_resolution
                    ),
                )
                anchor_stats["is_anchor_stage"] = 1.0
                decoder_stage_stats.append(anchor_stats)
            if self.decoder_anchor_processor is not None:
                anchor_stats = {
                    "resolution": int(self.decoder_anchor_resolution),
                    "input_dim": int(
                        self.decoder_stage_dims[self.decoder_anchor_resolution]
                    ),
                    "output_dim": int(
                        self.decoder_stage_dims[self.decoder_anchor_resolution]
                    ),
                    "skip_used": 0.0,
                    "is_anchor_stage": 1.0,
                    "uses_decoder_anchor_processor": 1.0,
                    "uses_decoder_text_proj": 1.0,
                }
                if decoder_text is not None:
                    anchor_stats["decoder_text_input_dim"] = int(decoder_text.shape[-1])
                if self.decoder_anchor_resolution in self.disabled_stage_resolutions:
                    anchor_stats["disabled"] = 1.0
                else:
                    anchor_condition = self.decoder_anchor_condition_proj(c)
                    anchor_text = (
                        self.decoder_anchor_text_proj(decoder_text)
                        if decoder_text is not None
                        else None
                    )
                    if anchor_text is not None:
                        anchor_stats["decoder_text_dim"] = int(anchor_text.shape[-1])
                    decoder_map = self.decoder_anchor_pre_mamba_blocks(decoder_map)
                    decoder_map, block_stats = self.decoder_anchor_processor(
                        decoder_map, anchor_condition, text=anchor_text
                    )
                    decoder_map = self.decoder_anchor_post_mamba_blocks(decoder_map)
                    anchor_stats.update(block_stats)
                decoder_stage_stats.append(anchor_stats)
            if (
                self.aux_4x4_context_adapter is not None
                and self.aux_4x4_inject_target == "decoder_anchor"
            ):
                decoder_map, aux_4x4_stats = self.aux_4x4_context_adapter(
                    bottleneck_map,
                    decoder_map,
                    c,
                )
            for resolution in self.decoder_stage_resolutions:
                skip_map = skip_maps.get(resolution)
                decoder_map, stage_stats = self._apply_decoder_stage(
                    resolution,
                    decoder_map,
                    skip_map,
                    c,
                    decoder_text,
                    stage_router_weight=stage_router_weights.get(resolution),
                )
                decoder_stage_stats.append(stage_stats)

        final_map_pre_norm = decoder_map.norm(dim=-1).mean().detach()
        final_tokens = map_to_tokens(decoder_map)
        if not self.fused_add_norm:
            final_tokens = self.norm_f(final_tokens.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            final_tokens = fused_add_norm_fn(
                final_tokens,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=None,
                prenorm=False,
                residual_in_fp32=False,
            )
        final_map = tokens_to_map(final_tokens, decoder_map.shape[1:3])

        self.latest_hierarchy_stats = {
            "enabled": True,
            "num_stages": len(encoder_stage_stats),
            "final_h": int(self.fusion_anchor_resolution),
            "final_w": int(self.fusion_anchor_resolution),
            "reached_global_context": float(self.hierarchy_reaches_global_context),
            "stage_metrics": encoder_stage_stats,
            "compression_metrics": compression_stage_stats,
        }
        self.latest_factorized_router_stats = {
            "enabled": float(
                self.stage_contribution_router is not None
                or self.integrated_router_controller is not None
            ),
            "use_integrated_router_controller": float(
                self.integrated_router_controller is not None
            ),
            "use_encoder_mamba_depth_router": float(
                self.use_encoder_mamba_depth_router
            ),
            "encoder_mamba_depth_router_stages": sorted(
                self.encoder_mamba_depth_router_stages
            ),
            "use_mamba_size_router": float(self.use_mamba_size_router),
            "use_compression_router": float(self.use_compression_router),
            "use_dynamic_bottleneck": float(self.use_dynamic_bottleneck),
            "routed_stage_resolutions": list(self.routed_stage_resolutions),
            "routed_stage_count": int(self.routed_stage_count),
            "include_anchor_in_stage_router": float(
                self.include_anchor_in_stage_router
            ),
            "stage_router": stage_router_stats,
            "compression_metrics": compression_stage_stats,
        }
        self.latest_fusion_stats = {
            "enabled": True,
            "selected_stage_resolutions": list(self.fusion_selected_stages),
            "anchor_resolution": int(self.decoder_anchor_resolution),
            "bottleneck_resolution": int(self.fusion_anchor_resolution),
            "dynamic_bottleneck_enabled": float(self.use_dynamic_bottleneck),
            "dynamic_bottleneck_candidates": list(self.dynamic_bottleneck_candidates),
            "stage_dim": int(self.fusion_stage_dim),
            "stage_dims": {
                resolution: int(dim)
                for resolution, dim in sorted(self.decoder_stage_dims.items())
            },
            "fusion_mode": self.fusion_mode,
            "gate_type": self.fusion_gate_type,
            "pos_embed_type": self.fusion_pos_embed_type,
            "block_depth": int(self.fusion_block_depth),
            "fusion_conv_depth": int(self.fusion_conv_depth),
            "fusion_pre_mamba_conv_depth": int(self.fusion_pre_mamba_conv_depth),
            "fusion_post_mamba_conv_depth": int(self.fusion_post_mamba_conv_depth),
            "anchor_builder_depth": int(self.anchor_builder_depth),
            "anchor_builder_stage_depth": int(self.anchor_builder_stage_depth),
            "channel_gate_stages": sorted(self.fusion_channel_gate_stages),
            "local_mamba_stage_resolutions": sorted(self.local_mamba_stage_resolutions),
            "spatial_gate_enabled": float(self.fusion_use_spatial_gate),
            "final_skip_refiner_enabled": float(self.final_skip_refiner is not None),
            "final_skip_refiner_depth": int(self.final_skip_refiner_depth),
            "final_skip_refiner_conv_type": self.final_skip_refiner_conv_type,
            "final_skip_refiner_use_channel_gate": float(
                self.final_skip_refiner_use_channel_gate
            ),
            "final_skip_refiner_use_spatial_gate": float(
                self.final_skip_refiner_use_spatial_gate
            ),
            "aux_4x4_context_enabled": float(self.aux_4x4_context_adapter is not None),
            "aux_4x4_inject_target": self.aux_4x4_inject_target,
            "aux_4x4_context_resolution": int(self.aux_4x4_context_resolution),
            "aux_4x4_target_resolution": int(self.aux_4x4_target_resolution),
            "aux_4x4_prepool_depth": int(self.aux_4x4_prepool_depth),
            "aux_4x4_context_depth": int(self.aux_4x4_context_depth),
            "aux_4x4_fusion_depth": int(self.aux_4x4_fusion_depth),
            "aux_4x4_use_timestep_gate": float(self.aux_4x4_use_timestep_gate),
            "aux_4x4_residual_scale_multiplier": float(
                self.aux_4x4_residual_scale_multiplier
            ),
            "stage_depths": {
                resolution: int(depth)
                for resolution, depth in sorted(self.decoder_stage_depths.items())
            },
            "prediction_head_type": self.fusion_prediction_head_type,
            "decoder_stage_metrics": decoder_stage_stats,
            "runtime_overrides": self.get_fusion_runtime_overrides(),
            "factorized_router": self.latest_factorized_router_stats,
        }
        if aux_4x4_stats is not None:
            self.latest_fusion_stats["aux_4x4_context_metrics"] = aux_4x4_stats
        for stage_stats in decoder_stage_stats:
            resolution = stage_stats["resolution"]
            if "gate_mean" in stage_stats:
                self.latest_fusion_stats.setdefault("stage_gate_means", {})[resolution] = stage_stats[
                    "gate_mean"
                ]
                self.latest_fusion_stats.setdefault("stage_gate_stds", {})[resolution] = stage_stats[
                    "gate_std"
                ]
                self.latest_fusion_stats.setdefault("stage_raw_gate_means", {})[resolution] = stage_stats[
                    "raw_gate_mean"
                ]
                self.latest_fusion_stats.setdefault("stage_raw_gate_stds", {})[resolution] = stage_stats[
                    "raw_gate_std"
                ]
                self.latest_fusion_stats.setdefault("stage_feature_norms", {})[resolution] = stage_stats[
                    "output_norm"
                ]

        self.latest_backbone_stats = {
            "final_map_pre_norm": final_map_pre_norm,
            "final_map_post_norm": final_map.norm(dim=-1).mean().detach(),
            "timestep_condition_norm": timestep_condition.norm(dim=-1).mean().detach(),
            "highres_processor_norm": highres_stats["output_norm"],
            "downsample_premix_depth": float(self.downsample_premix_depth),
            "downsample_uses_separable_conv": float(
                self.downsample_conv_type == "separable"
            ),
            "highres_local_uses_separable_conv": float(
                self.highres_local_conv_type == "separable"
            ),
        }
        return {
            "final_map": final_map,
            "bottleneck_map": bottleneck_map,
            "skip_maps": skip_maps,
            "condition": c,
            "timestep_condition": timestep_condition,
            "stage_router_weights": stage_router_weights,
        }

    def forward_prediction_head(self, backbone_output):
        final_skip_refiner_stats = None
        aux_4x4_stats = None
        if isinstance(backbone_output, dict):
            final_map = backbone_output["final_map"]
            if self.final_skip_refiner is not None:
                skip_map = backbone_output["skip_maps"].get(self.patch_side_len)
                condition = backbone_output["condition"]
                final_map, final_skip_refiner_stats = self.final_skip_refiner(
                    final_map,
                    skip_map,
                    condition,
                    disable_skip=self.patch_side_len in self.disabled_stage_resolutions,
                    force_skip_gate_value=self.force_stage_gate_values.get(self.patch_side_len),
                    gate_floor=self.gate_floor,
                    stage_router_weight=backbone_output.get(
                        "stage_router_weights", {}
                    ).get(self.patch_side_len),
                )
            if (
                self.aux_4x4_context_adapter is not None
                and self.aux_4x4_inject_target == "final_map"
            ):
                final_map, aux_4x4_stats = self.aux_4x4_context_adapter(
                    backbone_output["bottleneck_map"],
                    final_map,
                    backbone_output["condition"],
                )
        else:
            final_map = backbone_output
        if final_skip_refiner_stats is not None:
            self.latest_backbone_stats["final_skip_refined_norm"] = (
                final_map.norm(dim=-1).mean().detach()
            )
            self.latest_fusion_stats["final_skip_refiner_metrics"] = final_skip_refiner_stats
        if aux_4x4_stats is not None:
            self.latest_backbone_stats["final_map_aux_refined_norm"] = (
                final_map.norm(dim=-1).mean().detach()
            )
            self.latest_fusion_stats["aux_4x4_context_metrics"] = aux_4x4_stats
        predicted_tokens = self.prediction_head(final_map)
        return self.unpatchify(predicted_tokens)

    def forward_transport(self, hidden_states, t, y=None):
        backbone_output = self.forward_backbone(hidden_states, t, y=y)
        return self.forward_prediction_head(backbone_output)

    def forward(self, hidden_states, t, y=None):
        backbone_output = self.forward_backbone(hidden_states, t, y=y)
        if self.hierarchical_output_mode == "context":
            final_map = (
                backbone_output["final_map"]
                if isinstance(backbone_output, dict)
                else backbone_output
            )
            return map_to_tokens(final_map)
        return self.forward_prediction_head(backbone_output)

    def get_hierarchy_logging_metrics(self):
        metrics = {
            "hierarchy/enabled": 1.0,
            "hierarchy/window_size": float(self.hierarchy_window_size),
            "hierarchy/stride": float(self.hierarchy_stride),
            "hierarchy/first_layer_stride": float(self.first_layer_stride),
            "hierarchy/stage_depth": float(self.hierarchy_stage_depth),
            "hierarchy/highres_depth": float(self.highres_stage_depth),
            "hierarchy/bottleneck_depth": float(self.bottleneck_stage_depth),
            "hierarchy/downsample_use_premix": float(self.downsample_use_premix),
            "hierarchy/downsample_premix_depth": float(self.downsample_premix_depth),
            "hierarchy/downsample_uses_separable_conv": float(
                self.downsample_conv_type == "separable"
            ),
            "hierarchy/highres_local_conv_depth": float(self.highres_local_conv_depth),
            "hierarchy/highres_local_uses_separable_conv": float(
                self.highres_local_conv_type == "separable"
            ),
            "hierarchy/local_mamba_stage_count": float(
                len(self.local_mamba_stage_resolutions)
            ),
            "hierarchy/num_stages": float(self.latest_hierarchy_stats.get("num_stages", 0)),
            "hierarchy/final_h": float(self.latest_hierarchy_stats.get("final_h", self.fusion_anchor_resolution)),
            "hierarchy/final_w": float(self.latest_hierarchy_stats.get("final_w", self.fusion_anchor_resolution)),
            "hierarchy/reached_global_context": float(
                self.latest_hierarchy_stats.get(
                    "reached_global_context", self.hierarchy_reaches_global_context
                )
            ),
            "fusion/enabled": float(self.use_multiscale_fusion_head),
            "fusion/anchor_resolution": float(self.decoder_anchor_resolution),
            "fusion/bottleneck_resolution": float(self.fusion_anchor_resolution),
            "fusion/stage_dim": float(self.fusion_stage_dim),
            "fusion/block_depth": float(self.fusion_block_depth),
            "fusion/conv_depth": float(self.fusion_conv_depth),
            "fusion/pre_mamba_conv_depth": float(self.fusion_pre_mamba_conv_depth),
            "fusion/post_mamba_conv_depth": float(self.fusion_post_mamba_conv_depth),
            "fusion/anchor_builder_depth": float(self.anchor_builder_depth),
            "fusion/channel_gate_stage_count": float(len(self.fusion_channel_gate_stages)),
            "fusion/selected_stage_count": float(len(self.fusion_selected_stages)),
            "fusion/spatial_gate_enabled": float(self.fusion_use_spatial_gate),
            "fusion/final_skip_refiner_enabled": float(self.final_skip_refiner is not None),
            "fusion/final_skip_refiner_depth": float(self.final_skip_refiner_depth),
            "fusion/final_skip_refiner_uses_separable_conv": float(
                self.final_skip_refiner_conv_type == "separable"
            ),
            "aux4x4/enabled": float(self.aux_4x4_context_adapter is not None),
            "aux4x4/inject_target_decoder_anchor": float(
                self.aux_4x4_inject_target == "decoder_anchor"
            ),
            "aux4x4/inject_target_final_map": float(
                self.aux_4x4_inject_target == "final_map"
            ),
            "aux4x4/context_resolution": float(self.aux_4x4_context_resolution),
            "aux4x4/target_resolution": float(self.aux_4x4_target_resolution),
            "aux4x4/prepool_depth": float(self.aux_4x4_prepool_depth),
            "aux4x4/context_depth": float(self.aux_4x4_context_depth),
            "aux4x4/fusion_depth": float(self.aux_4x4_fusion_depth),
            "aux4x4/use_timestep_gate": float(self.aux_4x4_use_timestep_gate),
            "aux4x4/residual_scale_multiplier_config": float(
                self.aux_4x4_residual_scale_multiplier
            ),
            "head/conv_block_depth": float(self.prediction_head_conv_depth),
            "head/uses_separable_conv": float(self.prediction_head_conv_type == "separable"),
            "router/factorized_top4_enabled": float(self.use_factorized_top4_router),
            "router/mamba_size_enabled": float(self.use_mamba_size_router),
            "router/encoder_mamba_depth_enabled": float(
                self.use_encoder_mamba_depth_router
            ),
            "router/encoder_mamba_depth_top_k": float(
                self.encoder_mamba_depth_router_top_k
            ),
            "router/encoder_mamba_depth_stage_count": float(
                len(self.encoder_mamba_depth_router_stages)
            ),
            "router/mamba_size_top_k": float(self.mamba_size_router_top_k),
            "router/mamba_size_weight_mode_selection": float(
                self.mamba_size_router_weight_mode == "selection"
            ),
            "router/compression_enabled": float(self.use_compression_router),
            "router/integrated_controller_enabled": float(
                self.use_integrated_router_controller
            ),
            "router/integrated_controller_stage_top_k": float(
                self.integrated_controller_stage_top_k
            ),
            "router/integrated_controller_stage_select_mode_adaptive": float(
                self.integrated_controller_stage_select_mode
                in {
                    "adaptive",
                    "adaptive_topk",
                    "relu",
                    "threshold",
                    "sigmoid",
                    "independent",
                }
            ),
            "router/integrated_controller_stage_select_mode_relu": float(
                self.integrated_controller_stage_select_mode == "relu"
            ),
            "router/integrated_controller_stage_select_threshold": float(
                self.integrated_controller_stage_select_threshold_value
            ),
            "router/integrated_controller_stage_select_threshold_margin": float(
                self.integrated_controller_stage_select_threshold_margin
            ),
            "router/integrated_controller_stage_select_warmup_steps": float(
                self.integrated_controller_stage_select_warmup_steps
            ),
            "router/integrated_controller_stage_select_warmup_min_selected": float(
                self.integrated_controller_stage_select_warmup_min_selected
            ),
            "router/integrated_controller_stage_select_warmup_threshold_margin": float(
                self.integrated_controller_stage_select_warmup_threshold_margin
            ),
            "router/integrated_controller_stage_use_scale_prior_context": float(
                self.integrated_controller_stage_use_scale_prior_context
            ),
            "router/integrated_controller_stage_select_threshold_prob_mean": float(
                self.integrated_controller_stage_select_threshold_mode
                == "cumulative_prob_mean"
            ),
            "router/integrated_controller_stage_select_threshold_cumulative_prob_mean": float(
                self.integrated_controller_stage_select_threshold_mode
                == "cumulative_prob_mean"
            ),
            "router/integrated_controller_stage_min_selected": float(
                self.integrated_controller_stage_min_selected
            ),
            "router/integrated_controller_stage_balance_batch": float(
                self.integrated_controller_stage_balance_mode
                in {"batch", "balanced", "balance"}
            ),
            "router/integrated_controller_stage_balance_batch_center": float(
                self.integrated_controller_stage_balance_mode == "batch_center"
            ),
            "router/integrated_controller_stage_balance_logit_center": float(
                self.integrated_controller_stage_balance_mode == "logit_center"
            ),
            "router/integrated_controller_channel_gate": float(
                self.integrated_controller_use_channel_gate
            ),
            "router/integrated_controller_channel_gate_scale": float(
                self.integrated_controller_channel_gate_scale
            ),
            "router/include_anchor": float(self.include_anchor_in_stage_router),
            "router/routed_stage_count": float(self.routed_stage_count),
            "router/stage_top_k": float(self.stage_router_top_k),
            "router/stage_weight_mode_equal_selection": float(
                self.stage_router_weight_mode == "equal_selection"
            ),
            "router/stage_weight_floor": float(self.stage_router_weight_floor),
            "router/stage_max_weight": float(
                0.0
                if self.stage_router_max_weight is None
                else self.stage_router_max_weight
            ),
            "dynamic_bottleneck/enabled": float(self.use_dynamic_bottleneck),
            "dynamic_bottleneck/candidate_count": float(
                len(self.dynamic_bottleneck_candidates)
            ),
        }
        for resolution in self.dynamic_bottleneck_candidates:
            metrics[f"dynamic_bottleneck/candidate_{resolution}"] = 1.0
        for resolution in self.fusion_selected_stages:
            metrics[f"fusion/selected_stage_{resolution}"] = 1.0
            if resolution == self.decoder_anchor_resolution:
                stage_depth = self.anchor_builder_stage_depth
            else:
                stage_depth = self.decoder_stage_depths.get(
                    resolution, self.fusion_block_depth
                )
            metrics[f"fusion/stage_{resolution}_depth"] = float(stage_depth)
            stage_dim = self.decoder_stage_dims.get(
                resolution, self.decoder_stage_dims[self.decoder_anchor_resolution]
            )
            metrics[f"fusion/stage_{resolution}_dim"] = float(stage_dim)
            metrics[f"fusion/stage_{resolution}_content_channel_gate"] = float(
                resolution in self.fusion_channel_gate_stages
            )
        if self.latest_backbone_stats:
            for key, value in self.latest_backbone_stats.items():
                metrics[f"backbone/{key}"] = float(value)
        if self.latest_fusion_stats:
            for resolution, value in self.latest_fusion_stats.get("stage_gate_means", {}).items():
                metrics[f"fusion/stage_{resolution}_gate_mean"] = float(value)
            for resolution, value in self.latest_fusion_stats.get("stage_gate_stds", {}).items():
                metrics[f"fusion/stage_{resolution}_gate_std"] = float(value)
            final_skip_metrics = self.latest_fusion_stats.get("final_skip_refiner_metrics", {})
            if "gate_mean" in final_skip_metrics:
                metrics["fusion/final_skip_refiner_gate_mean"] = float(
                    final_skip_metrics["gate_mean"]
                )
            if "gate_std" in final_skip_metrics:
                metrics["fusion/final_skip_refiner_gate_std"] = float(
                    final_skip_metrics["gate_std"]
                )
            aux_4x4_metrics = self.latest_fusion_stats.get(
                "aux_4x4_context_metrics", {}
            )
            for key, value in aux_4x4_metrics.items():
                metrics[f"aux4x4/{key}"] = float(value)
        if self.latest_factorized_router_stats:
            stage_router = self.latest_factorized_router_stats.get("stage_router", {})
            if "entropy" in stage_router:
                metrics["router/stage_entropy"] = float(stage_router["entropy"])
            if "stage_select_threshold_mean" in stage_router:
                metrics["router/stage_select_threshold_mean"] = float(
                    stage_router["stage_select_threshold_mean"]
                )
            if "stage_select_warmup_updates" in stage_router:
                metrics["router/stage_select_warmup_updates"] = float(
                    stage_router["stage_select_warmup_updates"]
                )
            if "selected_count_mean" in stage_router:
                metrics["router/stage_selected_count_mean"] = float(
                    stage_router["selected_count_mean"]
                )
            if "selected_count_min" in stage_router:
                metrics["router/stage_selected_count_min"] = float(
                    stage_router["selected_count_min"]
                )
            if "selected_count_max" in stage_router:
                metrics["router/stage_selected_count_max"] = float(
                    stage_router["selected_count_max"]
                )
            if "max_weight" in stage_router:
                metrics["router/stage_weight_max_observed"] = float(
                    stage_router["max_weight"]
                )
            if "min_weight" in stage_router:
                metrics["router/stage_weight_min_observed"] = float(
                    stage_router["min_weight"]
                )
            for resolution, value in stage_router.get("weights", {}).items():
                metrics[f"router/stage_{resolution}_weight"] = float(value)
            for resolution, value in stage_router.get("probabilities", {}).items():
                metrics[f"router/stage_{resolution}_prob"] = float(value)
            for resolution, value in stage_router.get("raw_probabilities", {}).items():
                metrics[f"router/stage_{resolution}_raw_prob"] = float(value)
            for resolution, value in stage_router.get("activations", {}).items():
                metrics[f"router/stage_{resolution}_activation"] = float(value)
            for resolution, value in stage_router.get("thresholds", {}).items():
                metrics[f"router/stage_{resolution}_threshold"] = float(value)
            for resolution, value in stage_router.get("selected_fractions", {}).items():
                metrics[f"router/stage_{resolution}_selected"] = float(value)
            for resolution, value in stage_router.get("logits", {}).items():
                metrics[f"router/stage_{resolution}_logit"] = float(value)
            for resolution, value in stage_router.get("dynamic_selected_fractions", {}).items():
                metrics[f"dynamic_bottleneck/stage_{resolution}_selected"] = float(value)
            for resolution, value in stage_router.get("dynamic_bottleneck_fractions", {}).items():
                metrics[f"dynamic_bottleneck/resolution_{resolution}_fraction"] = float(value)
            if "dynamic_selected_count_mean" in stage_router:
                metrics["dynamic_bottleneck/selected_count_mean"] = float(
                    stage_router["dynamic_selected_count_mean"]
                )
            for compression_stats in self.latest_factorized_router_stats.get(
                "compression_metrics", []
            ):
                resolution = compression_stats.get("output_resolution")
                if resolution is None:
                    continue
                metrics[f"router/compress_{resolution}_stride0"] = float(
                    compression_stats["stride0_weight"]
                )
                metrics[f"router/compress_{resolution}_stride1"] = float(
                    compression_stats["stride1_weight"]
                )
                metrics[f"router/compress_{resolution}_stride2"] = float(
                    compression_stats["stride2_weight"]
                )
                for key, value in compression_stats.items():
                    if not str(key).startswith("integrated_"):
                        continue
                    metrics[f"router/compress_{resolution}_{key}"] = float(value)
        for stage_stats in self.latest_hierarchy_stats.get("stage_metrics", []):
            resolution = stage_stats.get("resolution")
            if resolution is None:
                continue
            if "preset_weights" in stage_stats:
                for preset_name, value in stage_stats["preset_weights"].items():
                    metrics[f"router/encoder_mamba_{resolution}_{preset_name}_weight"] = float(value)
                for preset_name, value in stage_stats.get("preset_probabilities", {}).items():
                    metrics[f"router/encoder_mamba_{resolution}_{preset_name}_prob"] = float(value)
                for preset_name, value in stage_stats.get("preset_selected_fractions", {}).items():
                    metrics[f"router/encoder_mamba_{resolution}_{preset_name}_selected"] = float(value)
            if "depth_weights" in stage_stats:
                metrics[f"router/encoder_depth_{resolution}_selected_depth_mean"] = float(
                    stage_stats.get("selected_depth_mean", 0.0)
                )
                for depth_name, value in stage_stats["depth_weights"].items():
                    metrics[f"router/encoder_depth_{resolution}_{depth_name}_weight"] = float(value)
                for depth_name, value in stage_stats.get("depth_probabilities", {}).items():
                    metrics[f"router/encoder_depth_{resolution}_{depth_name}_prob"] = float(value)
                for depth_name, value in stage_stats.get("depth_selected_fractions", {}).items():
                    metrics[f"router/encoder_depth_{resolution}_{depth_name}_selected"] = float(value)
                for depth_name, value in stage_stats.get("depth_branch_depths", {}).items():
                    metrics[f"router/encoder_depth_{resolution}_{depth_name}_depth"] = float(value)
                for key, value in stage_stats.items():
                    if not str(key).startswith("integrated_depth_"):
                        continue
                    metrics[f"router/encoder_depth_{resolution}_{key}"] = float(value)
        for stage_stats in self.latest_fusion_stats.get("decoder_stage_metrics", []):
            resolution = stage_stats.get("resolution")
            if resolution is None or "preset_weights" not in stage_stats:
                continue
            for preset_name, value in stage_stats["preset_weights"].items():
                metrics[f"router/fusion_mamba_{resolution}_{preset_name}_weight"] = float(value)
            for preset_name, value in stage_stats.get("preset_probabilities", {}).items():
                metrics[f"router/fusion_mamba_{resolution}_{preset_name}_prob"] = float(value)
            for preset_name, value in stage_stats.get("preset_selected_fractions", {}).items():
                metrics[f"router/fusion_mamba_{resolution}_{preset_name}_selected"] = float(value)
        return metrics

    def set_fusion_runtime_overrides(
        self,
        *,
        disabled_stage_resolutions=None,
        force_stage_gate_values=None,
        gate_floor=None,
    ):
        self.disabled_stage_resolutions = {
            int(resolution) for resolution in (disabled_stage_resolutions or [])
        }
        self.force_stage_gate_values = (
            {}
            if force_stage_gate_values is None
            else {
                int(resolution): float(value)
                for resolution, value in force_stage_gate_values.items()
            }
        )
        self.gate_floor = None if gate_floor is None else float(gate_floor)

    def clear_fusion_runtime_overrides(self):
        self.set_fusion_runtime_overrides()

    def get_fusion_runtime_overrides(self):
        return {
            "disabled_stage_resolutions": sorted(self.disabled_stage_resolutions),
            "force_stage_gate_values": dict(self.force_stage_gate_values),
            "gate_floor": self.gate_floor,
        }

    def load_state_dict(self, state_dict, strict=True):
        remapped_state_dict = dict(state_dict)
        if self.anchor_builder is not None and not any(
            key.startswith("anchor_builder.") for key in remapped_state_dict.keys()
        ):
            remapped = {}
            for key, value in remapped_state_dict.items():
                match = re.match(r"decoder_stages\.(\d+)\.(.+)", key)
                if match is None:
                    remapped[key] = value
                    continue

                stage_idx = int(match.group(1))
                suffix = match.group(2)
                if stage_idx == 0:
                    remapped[f"anchor_builder.{suffix}"] = value
                else:
                    remapped[f"decoder_stages.{stage_idx - 1}.{suffix}"] = value
            remapped_state_dict = remapped
        current_state_dict = super().state_dict()
        for key in list(remapped_state_dict.keys()):
            if key.startswith(
                "integrated_router_controller.prior_probability_running_"
            ):
                remapped_state_dict.pop(key)
        for key, value in current_state_dict.items():
            if key in remapped_state_dict:
                continue
            if (
                "absolute_pos_embed.pos_embed" in key
                or "child_pos_embed.pos_embed" in key
                or key.startswith(
                    "integrated_router_controller.stage_probability_running_"
                )
                or key == "integrated_router_controller.stage_selection_warmup_updates"
                or key.startswith(
                    "integrated_router_controller.compression_stage_selection_proj."
                )
                or key.startswith(
                    "integrated_router_controller.encoder_depth_stage_selection_proj."
                )
            ):
                remapped_state_dict[key] = value
        return super().load_state_dict(remapped_state_dict, strict=strict)


class StageContextProjector(nn.Module):
    """Project a stage map to a shared context resolution and channel dimension."""

    def __init__(
        self,
        input_dim,
        context_dim,
        input_resolution,
        context_resolution,
        depth,
        conv_type,
        device,
        dtype,
    ):
        super().__init__()
        self.input_resolution = int(input_resolution)
        self.context_resolution = int(context_resolution)
        self.input_proj = nn.Linear(input_dim, context_dim, device=device, dtype=dtype)
        self.refiner = MapConvResidualStack(
            context_dim,
            depth=int(depth),
            conv_type=conv_type,
            device=device,
            dtype=dtype,
        )
        self.norm = nn.LayerNorm(
            context_dim,
            elementwise_affine=False,
            eps=1e-6,
            device=device,
            dtype=dtype,
        )

    def _resize(self, stage_map):
        height, width = stage_map.shape[1:3]
        if height == self.context_resolution and width == self.context_resolution:
            return stage_map
        x = stage_map.permute(0, 3, 1, 2).contiguous()
        if height > self.context_resolution or width > self.context_resolution:
            x = F.adaptive_avg_pool2d(
                x,
                (self.context_resolution, self.context_resolution),
            )
        else:
            x = F.interpolate(
                x,
                size=(self.context_resolution, self.context_resolution),
                mode="nearest",
            )
        return x.permute(0, 2, 3, 1).contiguous()

    def forward(self, stage_map):
        context = self.input_proj(self._resize(stage_map))
        context = self.refiner(context)
        return self.norm(context)


class ContextConditionedDenoisingHead(nn.Module):
    """Denoise full-resolution tokens conditioned on the aggregated hierarchy context."""

    def __init__(
        self,
        input_dim,
        context_dim,
        denoiser_dim,
        condition_dim,
        text_dim,
        resolution,
        depth,
        conv_depth,
        conv_type,
        has_text,
        ssm_cfg,
        norm_epsilon,
        rms_norm,
        residual_in_fp32,
        fused_add_norm,
        scan_type,
        use_jit,
        use_checkpoint,
        device,
        dtype,
    ):
        super().__init__()
        self.resolution = int(resolution)
        self.has_text = bool(has_text)
        self.input_proj = nn.Linear(
            input_dim + context_dim,
            denoiser_dim,
            device=device,
            dtype=dtype,
        )
        self.condition_proj = (
            nn.Identity()
            if condition_dim == denoiser_dim
            else nn.Linear(condition_dim, denoiser_dim, device=device, dtype=dtype)
        )
        self.text_proj = None
        if self.has_text:
            self.text_proj = (
                nn.Identity()
                if text_dim == denoiser_dim
                else nn.Linear(text_dim, denoiser_dim, device=device, dtype=dtype)
            )
        self.pre_blocks = MapConvResidualStack(
            denoiser_dim,
            depth=int(conv_depth),
            conv_type=conv_type,
            device=device,
            dtype=dtype,
        )
        self.processor = MapMambaResidualProcessor(
            dim=denoiser_dim,
            depth=int(depth),
            resolution=resolution,
            has_text=has_text,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            drop_path_values=torch.linspace(0, 0.0, int(depth)).tolist(),
            scan_type=scan_type,
            use_jit=use_jit,
            use_checkpoint=use_checkpoint,
            device=device,
            dtype=dtype,
        )
        self.post_blocks = MapConvResidualStack(
            denoiser_dim,
            depth=int(conv_depth),
            conv_type=conv_type,
            device=device,
            dtype=dtype,
        )

    def forward(self, highres_map, context_map, condition, text=None):
        if context_map.shape[1:3] != highres_map.shape[1:3]:
            context = context_map.permute(0, 3, 1, 2).contiguous()
            context = F.interpolate(
                context,
                size=highres_map.shape[1:3],
                mode="nearest",
            )
            context_map = context.permute(0, 2, 3, 1).contiguous()
        hidden = self.input_proj(torch.cat([highres_map, context_map], dim=-1))
        hidden = self.pre_blocks(hidden)
        denoiser_condition = self.condition_proj(condition)
        denoiser_text = self.text_proj(text) if self.text_proj is not None else None
        hidden, stats = self.processor(
            hidden,
            denoiser_condition,
            text=denoiser_text,
        )
        hidden = self.post_blocks(hidden)
        stats = dict(stats)
        stats["context_conditioned_denoiser"] = 1.0
        stats["context_resolution"] = float(context_map.shape[1])
        stats["denoiser_resolution"] = float(self.resolution)
        stats["denoiser_dim"] = float(hidden.shape[-1])
        return hidden, stats


class HierarchyContextDenoiserV1(HierarchicalMambaHybrid):
    """Integrated-router hierarchy that denoises from an aggregated stage context."""

    def __init__(
        self,
        *args,
        context_dim=None,
        context_resolution=None,
        context_projector_depth=1,
        context_projector_conv_type="standard",
        context_aggregation="weighted_sum",
        denoiser_dim=None,
        denoiser_depth=None,
        denoiser_conv_depth=1,
        denoiser_scan_type=None,
        **kwargs,
    ):
        text_context_dim = int(kwargs.get("d_context", 0))
        super().__init__(*args, **kwargs)
        self.d_context = text_context_dim
        self.context_dim = int(context_dim or self.fusion_stage_dim)
        self.context_resolution = int(context_resolution or self.fusion_anchor_resolution)
        self.context_projector_depth = int(context_projector_depth)
        self.context_projector_conv_type = context_projector_conv_type
        self.context_aggregation = str(context_aggregation).strip().lower()
        if self.context_aggregation not in {"weighted_sum", "mean"}:
            raise ValueError(
                "context_aggregation must be 'weighted_sum' or 'mean', "
                f"got {context_aggregation!r}"
            )
        self.denoiser_dim = int(denoiser_dim or self.context_dim)
        self.denoiser_depth = int(denoiser_depth or self.fusion_block_depth)
        self.denoiser_conv_depth = int(denoiser_conv_depth)
        self.denoiser_scan_type = denoiser_scan_type or self.scan_type

        # Remove decoder-fusion modules from the active parameter tree.  The
        # encoder hierarchy and integrated SSR/CMR/EDR controller remain intact.
        self.anchor_builder = None
        self.decoder_stages = nn.ModuleList()
        self.decoder_anchor_condition_proj = None
        self.decoder_anchor_text_proj = None
        self.decoder_anchor_pre_mamba_blocks = None
        self.decoder_anchor_processor = None
        self.decoder_anchor_post_mamba_blocks = None
        self.final_skip_refiner = None
        self.use_multiscale_fusion_head = False

        device = self.factory_kwargs["device"]
        dtype = self.factory_kwargs["dtype"]
        self.context_projectors = nn.ModuleDict(
            {
                str(resolution): StageContextProjector(
                    input_dim=self.embed_dim,
                    context_dim=self.context_dim,
                    input_resolution=resolution,
                    context_resolution=self.context_resolution,
                    depth=self.context_projector_depth,
                    conv_type=self.context_projector_conv_type,
                    device=device,
                    dtype=dtype,
                )
                for resolution in self.routed_stage_resolutions
            }
        )
        self.context_denoiser = ContextConditionedDenoisingHead(
            input_dim=self.embed_dim,
            context_dim=self.context_dim,
            denoiser_dim=self.denoiser_dim,
            condition_dim=self.embed_dim,
            text_dim=self.d_context,
            resolution=self.patch_side_len,
            depth=self.denoiser_depth,
            conv_depth=self.denoiser_conv_depth,
            conv_type=self.prediction_head_conv_type,
            has_text=self.has_text,
            ssm_cfg=self.ssm_cfg,
            norm_epsilon=self.norm_epsilon,
            rms_norm=self.rms_norm,
            residual_in_fp32=self.residual_in_fp32,
            fused_add_norm=self.fused_add_norm,
            scan_type=self.denoiser_scan_type,
            use_jit=self.use_jit,
            use_checkpoint=self.use_checkpoint,
            device=device,
            dtype=dtype,
        )
        self.prediction_head = MapPredictionHead(
            hidden_size=self.denoiser_dim,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
            prediction_head_type=self.fusion_prediction_head_type,
            conv_block_depth=self.prediction_head_conv_depth,
            conv_type=self.prediction_head_conv_type,
            device=device,
            dtype=dtype,
        )

    def _aggregate_stage_contexts(self, stage_maps, stage_router_weights):
        available = [
            resolution
            for resolution in self.routed_stage_resolutions
            if resolution in stage_maps and str(resolution) in self.context_projectors
        ]
        if not available:
            raise RuntimeError("HierarchyContextDenoiserV1 has no available stage maps")

        projected = {}
        if not stage_router_weights or self.context_aggregation == "mean":
            fallback_weight = 1.0 / float(len(available))
        else:
            fallback_weight = None
        aggregated = None
        weight_stats = {}
        for resolution in available:
            context = self.context_projectors[str(resolution)](stage_maps[resolution])
            projected[resolution] = context
            if fallback_weight is None:
                weight = stage_router_weights.get(resolution)
                if weight is None:
                    weight = context.new_zeros((context.shape[0], 1, 1, 1))
            else:
                weight = context.new_full((context.shape[0], 1, 1, 1), fallback_weight)
            weighted = context * weight.to(dtype=context.dtype)
            aggregated = weighted if aggregated is None else aggregated + weighted
            weight_stats[resolution] = weight.mean().detach()
        return aggregated, projected, weight_stats

    def forward_backbone(self, hidden_states, t, y=None):
        hidden_states = self.x_embedder(hidden_states)
        c, y, timestep_condition, raw_text = self._get_condition(hidden_states, t, y)
        processor_text = raw_text if self.has_text else y
        controller_scale_prior = None
        if self.integrated_router_controller is not None:
            controller_scale_prior = self.integrated_router_controller.compute_scale_prior(c)

        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        initial_map = tokens_to_map(hidden_states, self.hierarchy_input_size)
        initial_map = self.highres_local_stem(initial_map)
        highres_map, highres_stats = self._apply_encoder_processor(
            self.highres_processor,
            initial_map,
            c,
            processor_text,
            self.patch_side_len,
            self.highres_stage_depth,
            controller_scale_prior,
        )
        stage_maps = {self.patch_side_len: highres_map}
        encoder_stage_stats = [dict(stage_idx=0, **highres_stats)]
        compression_stage_stats = []

        current_map = highres_map
        for stage_idx, (layout, downsampler) in enumerate(
            zip(self.hierarchy_stage_layout, self.downsamplers),
            start=1,
        ):
            out_resolution = layout["output_resolution"][0]
            if isinstance(downsampler, RoutedCompressionDownsample2d):
                controller_compression = None
                if self.integrated_router_controller is not None:
                    controller_compression = (
                        self.integrated_router_controller.compute_compression(
                            c,
                            current_map,
                            out_resolution,
                            self.hierarchy_stage_depth,
                            controller_scale_prior,
                        )
                    )
                current_map, compression_stats = downsampler(
                    current_map,
                    c,
                    external_logits=(
                        None
                        if controller_compression is None
                        else controller_compression["logits"]
                    ),
                    channel_gate=(
                        None
                        if controller_compression is None
                        else controller_compression["channel_gate"]
                    ),
                )
                if controller_compression is not None:
                    compression_stats.update(controller_compression["stats"])
                compression_stats["output_resolution"] = int(out_resolution)
                compression_stage_stats.append(compression_stats)
            else:
                current_map = downsampler(current_map)

            if out_resolution > self.fusion_anchor_resolution:
                current_map, processor_stats = self._apply_encoder_processor(
                    self.encoder_processors[str(out_resolution)],
                    current_map,
                    c,
                    processor_text,
                    out_resolution,
                    self.hierarchy_stage_depth,
                    controller_scale_prior,
                )
            else:
                current_map, processor_stats = self._apply_encoder_processor(
                    self.bottleneck_processor,
                    current_map,
                    c,
                    processor_text,
                    out_resolution,
                    self.bottleneck_stage_depth,
                    controller_scale_prior,
                )
            stage_maps[out_resolution] = current_map
            encoder_stage_stats.append(
                {
                    "stage_idx": stage_idx,
                    "input_resolution": layout["input_resolution"],
                    "output_resolution": layout["output_resolution"],
                    "stride": layout["stride"],
                    **processor_stats,
                }
            )

        if self.integrated_router_controller is not None:
            stage_router_weights, stage_router_stats = (
                self.integrated_router_controller.compute_stage_weights(
                    c,
                    stage_maps,
                    controller_scale_prior,
                )
            )
        elif self.stage_contribution_router is not None:
            stage_router_weights, stage_router_stats = self.stage_contribution_router(
                c,
                stage_maps,
            )
        else:
            stage_router_weights, stage_router_stats = {}, {}

        unified_context, projected_contexts, context_weight_stats = (
            self._aggregate_stage_contexts(stage_maps, stage_router_weights)
        )
        final_map, denoiser_stats = self.context_denoiser(
            highres_map,
            unified_context,
            c,
            text=processor_text,
        )
        self.latest_backbone_stats = {
            "context_norm": unified_context.norm(dim=-1).mean().detach(),
            "denoiser_output_norm": final_map.norm(dim=-1).mean().detach(),
        }
        self.latest_fusion_stats = {
            "context_weight_means": context_weight_stats,
            "denoiser_metrics": denoiser_stats,
        }
        self.latest_hierarchy_stats = {
            "num_stages": len(self.hierarchy_stage_layout) + 1,
            "final_h": final_map.shape[1],
            "final_w": final_map.shape[2],
            "reached_global_context": False,
            "stage_metrics": encoder_stage_stats,
            "context_projected_resolutions": list(projected_contexts.keys()),
        }
        self.latest_factorized_router_stats = {
            "enabled": float(bool(stage_router_stats)),
            "use_mamba_size_router": float(self.use_mamba_size_router),
            "use_compression_router": float(self.use_compression_router),
            "use_integrated_router_controller": float(
                self.use_integrated_router_controller
            ),
            "compression_metrics": compression_stage_stats,
            "stage_router": stage_router_stats,
        }
        return {
            "final_map": final_map,
            "unified_context": unified_context,
            "projected_contexts": projected_contexts,
            "stage_maps": stage_maps,
            "condition": c,
            "timestep_condition": timestep_condition,
            "stage_router_weights": stage_router_weights,
            "stage_router_stats": stage_router_stats,
        }

    def forward_prediction_head(self, backbone_output):
        final_map = backbone_output["final_map"] if isinstance(backbone_output, dict) else backbone_output
        predicted_tokens = self.prediction_head(final_map)
        return self.unpatchify(predicted_tokens)

    def forward_transport(self, hidden_states, t, y=None):
        backbone_output = self.forward_backbone(hidden_states, t, y=y)
        return self.forward_prediction_head(backbone_output)

    def forward(self, hidden_states, t, y=None):
        backbone_output = self.forward_backbone(hidden_states, t, y=y)
        if self.hierarchical_output_mode == "context":
            return map_to_tokens(backbone_output["final_map"])
        return self.forward_prediction_head(backbone_output)

    def get_hierarchy_logging_metrics(self):
        metrics = super().get_hierarchy_logging_metrics()
        metrics.update(
            {
                "context_denoiser/enabled": 1.0,
                "context_denoiser/context_dim": float(self.context_dim),
                "context_denoiser/context_resolution": float(self.context_resolution),
                "context_denoiser/projector_depth": float(
                    self.context_projector_depth
                ),
                "context_denoiser/denoiser_dim": float(self.denoiser_dim),
                "context_denoiser/denoiser_depth": float(self.denoiser_depth),
                "context_denoiser/denoiser_conv_depth": float(
                    self.denoiser_conv_depth
                ),
            }
        )
        for resolution, value in self.latest_fusion_stats.get(
            "context_weight_means",
            {},
        ).items():
            metrics[f"context_denoiser/stage_{resolution}_weight"] = float(value)
        for key, value in self.latest_fusion_stats.get("denoiser_metrics", {}).items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[f"context_denoiser/{key}"] = float(value)
        return metrics


class DynamicBottleneckHierarchicalMambaHybrid(HierarchicalMambaHybrid):
    """Sparse-route two of three stages and use the lowest selected stage as bottleneck."""

    def __init__(
        self,
        dynamic_bottleneck_candidate_stages="auto",
        dynamic_bottleneck_select_k=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dynamic_bottleneck_candidate_spec = dynamic_bottleneck_candidate_stages
        self.dynamic_bottleneck_select_k = int(dynamic_bottleneck_select_k)

        if self.stage_contribution_router is None:
            raise ValueError(
                "DynamicBottleneckHierarchicalMambaHybrid requires "
                "use_factorized_top4_router=true so sparse stage selection is active."
            )
        if self.dynamic_bottleneck_select_k != 2:
            raise ValueError(
                "dynamic_bottleneck_select_k must be 2 for the requested "
                f"two-of-three sparse router, got {dynamic_bottleneck_select_k}"
            )
        if getattr(self.stage_contribution_router, "top_k", None) != 2:
            raise ValueError(
                "stage_router_top_k must be 2 so the router selects exactly two "
                f"stages, got {getattr(self.stage_contribution_router, 'top_k', None)}"
            )
        if self.decoder_anchor_resolution != self.fusion_anchor_resolution:
            raise ValueError(
                "dynamic bottleneck routing currently expects "
                "decoder_anchor_resolution == fusion_anchor_resolution."
            )
        if self.anchor_builder is not None:
            raise ValueError(
                "dynamic bottleneck routing currently does not support a separate "
                "anchor_builder path; set decoder_anchor_resolution to "
                "fusion_anchor_resolution."
            )
        if self.aux_4x4_context_adapter is not None:
            raise ValueError(
                "dynamic bottleneck routing currently does not support aux_4x4 "
                "context injection."
            )

        if is_auto_stage_spec(dynamic_bottleneck_candidate_stages):
            candidates = list(self.routed_stage_resolutions)
        else:
            candidates = parse_stage_resolution_spec(dynamic_bottleneck_candidate_stages)
        candidates = [int(resolution) for resolution in candidates]
        if len(candidates) != 3:
            raise ValueError(
                "dynamic bottleneck routing expects exactly three candidate "
                f"stages, got {candidates}"
            )
        if len(set(candidates)) != len(candidates):
            raise ValueError(
                "dynamic bottleneck candidate stages must be unique, "
                f"got {candidates}"
            )
        if self.patch_side_len not in candidates:
            raise ValueError(
                "dynamic bottleneck candidate stages must include the highest "
                f"token resolution {self.patch_side_len}, got {candidates}"
            )
        if min(candidates) != self.fusion_anchor_resolution:
            raise ValueError(
                "For dynamic bottleneck routing, FUSION_ANCHOR_RESOLUTION must be "
                "the minimum candidate stage so the full candidate pyramid exists. "
                f"got fusion_anchor_resolution={self.fusion_anchor_resolution}, "
                f"candidates={candidates}"
            )

        required = set(candidates)
        routed = set(self.routed_stage_resolutions)
        missing_routed = sorted(required - routed)
        if missing_routed:
            raise ValueError(
                "dynamic bottleneck candidates must be included in "
                f"routed_stage_resolutions; missing={missing_routed}, "
                f"routed={self.routed_stage_resolutions}"
            )
        missing_skips = sorted(required - set(self.required_skip_resolutions))
        if missing_skips:
            raise ValueError(
                "dynamic bottleneck candidates must be available as skip maps; "
                f"missing={missing_skips}, required_skip_resolutions="
                f"{self.required_skip_resolutions}"
            )

        self.dynamic_bottleneck_candidates = sorted(candidates, reverse=True)
        self.dynamic_bottleneck_candidates_ascending = sorted(candidates)
        self.dynamic_bottleneck_projs = nn.ModuleDict()
        for resolution in self.dynamic_bottleneck_candidates:
            if resolution not in self.decoder_stage_dims:
                raise ValueError(
                    "dynamic bottleneck candidate is not reachable by the decoder: "
                    f"{resolution}; decoder_stage_dims={sorted(self.decoder_stage_dims)}"
                )
            stage_dim = self.decoder_stage_dims[resolution]
            self.dynamic_bottleneck_projs[str(resolution)] = (
                nn.Identity()
                if stage_dim == self.embed_dim
                else nn.Linear(
                    self.embed_dim,
                    stage_dim,
                    device=self.factory_kwargs["device"],
                    dtype=self.factory_kwargs["dtype"],
                )
            )

    def _select_dynamic_bottlenecks(self, stage_router_weights, batch_size, device):
        selected_masks = {}
        for resolution in self.dynamic_bottleneck_candidates_ascending:
            weights = stage_router_weights.get(resolution)
            if weights is None:
                selected = torch.zeros(batch_size, dtype=torch.bool, device=device)
            else:
                selected = weights.detach().reshape(batch_size, -1).amax(dim=1) > 0
            selected_masks[resolution] = selected

        selected_stack = torch.stack(
            [
                selected_masks[resolution]
                for resolution in self.dynamic_bottleneck_candidates_ascending
            ],
            dim=1,
        )
        selected_counts = selected_stack.sum(dim=1)
        if not torch.all(selected_counts == self.dynamic_bottleneck_select_k):
            raise RuntimeError(
                "dynamic bottleneck router must select exactly two stages per "
                f"sample; observed counts={selected_counts.detach().cpu().tolist()}"
            )
        candidate_tensor = torch.tensor(
            self.dynamic_bottleneck_candidates_ascending,
            dtype=torch.long,
            device=device,
        )
        first_selected = selected_stack.to(torch.int64).argmax(dim=1)
        return candidate_tensor[first_selected], selected_masks

    def _normalize_final_decoder_map(self, decoder_map):
        final_tokens = map_to_tokens(decoder_map)
        if not self.fused_add_norm:
            final_tokens = self.norm_f(final_tokens.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            final_tokens = fused_add_norm_fn(
                final_tokens,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=None,
                prenorm=False,
                residual_in_fp32=False,
            )
        return tokens_to_map(final_tokens, decoder_map.shape[1:3])

    def _apply_decoder_anchor_processor(
        self,
        decoder_map,
        c,
        decoder_text,
        decoder_stage_stats,
    ):
        if self.decoder_anchor_processor is None:
            return decoder_map

        anchor_stats = {
            "resolution": int(self.decoder_anchor_resolution),
            "input_dim": int(self.decoder_stage_dims[self.decoder_anchor_resolution]),
            "output_dim": int(self.decoder_stage_dims[self.decoder_anchor_resolution]),
            "skip_used": 0.0,
            "is_anchor_stage": 1.0,
            "uses_decoder_anchor_processor": 1.0,
            "uses_decoder_text_proj": 1.0,
        }
        if decoder_text is not None:
            anchor_stats["decoder_text_input_dim"] = int(decoder_text.shape[-1])
        if self.decoder_anchor_resolution in self.disabled_stage_resolutions:
            anchor_stats["disabled"] = 1.0
        else:
            anchor_condition = self.decoder_anchor_condition_proj(c)
            anchor_text = (
                self.decoder_anchor_text_proj(decoder_text)
                if decoder_text is not None
                else None
            )
            if anchor_text is not None:
                anchor_stats["decoder_text_dim"] = int(anchor_text.shape[-1])
            decoder_map = self.decoder_anchor_pre_mamba_blocks(decoder_map)
            decoder_map, block_stats = self.decoder_anchor_processor(
                decoder_map, anchor_condition, text=anchor_text
            )
            decoder_map = self.decoder_anchor_post_mamba_blocks(decoder_map)
            anchor_stats.update(block_stats)
        decoder_stage_stats.append(anchor_stats)
        return decoder_map

    def forward_backbone(self, hidden_states, t, y=None):
        hidden_states = self.x_embedder(hidden_states)
        c, y, timestep_condition, raw_text = self._get_condition(hidden_states, t, y)
        decoder_text = raw_text if self.has_text else y

        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        initial_map = tokens_to_map(hidden_states, self.hierarchy_input_size)
        initial_map = self.highres_local_stem(initial_map)
        highres_map, highres_stats = self.highres_processor(initial_map, c=c, text=y)
        encoder_stage_stats = [dict(stage_idx=0, **highres_stats)]
        skip_maps = {self.patch_side_len: highres_map}
        compression_stage_stats = []

        current_map = highres_map
        for stage_idx, (layout, downsampler) in enumerate(
            zip(self.hierarchy_stage_layout, self.downsamplers), start=1
        ):
            out_resolution = layout["output_resolution"][0]
            if isinstance(downsampler, RoutedCompressionDownsample2d):
                current_map, compression_stats = downsampler(current_map, c)
                compression_stats["output_resolution"] = int(out_resolution)
                compression_stage_stats.append(compression_stats)
            else:
                current_map = downsampler(current_map)
                compression_stats = None
            if out_resolution > self.fusion_anchor_resolution:
                current_map, processor_stats = self.encoder_processors[str(out_resolution)](
                    current_map, c=c, text=y
                )
                if out_resolution in self.required_skip_resolutions:
                    skip_maps[out_resolution] = current_map
            else:
                current_map, processor_stats = self.bottleneck_processor(
                    current_map, c=c, text=y
                )
            encoder_stage_stats.append(
                {
                    "stage_idx": stage_idx,
                    "input_resolution": layout["input_resolution"],
                    "output_resolution": layout["output_resolution"],
                    "stride": layout["stride"],
                    **processor_stats,
                }
            )

        static_lowest_bottleneck_map = current_map
        skip_maps[self.fusion_anchor_resolution] = static_lowest_bottleneck_map
        stage_router_weights, stage_router_stats = self.stage_contribution_router(
            c,
            skip_maps,
        )
        bottleneck_resolutions, selected_masks = self._select_dynamic_bottlenecks(
            stage_router_weights,
            batch_size=hidden_states.shape[0],
            device=hidden_states.device,
        )

        final_maps = None
        decoder_stage_stats = []
        dynamic_bottleneck_stats = []
        pre_norm_accum = static_lowest_bottleneck_map.new_tensor(0.0)
        batch_size = hidden_states.shape[0]

        for bottleneck_resolution in self.dynamic_bottleneck_candidates_ascending:
            group_mask = bottleneck_resolutions == bottleneck_resolution
            group_count = int(group_mask.sum().item())
            if group_count == 0:
                continue
            group_indices = group_mask.nonzero(as_tuple=False).flatten()
            group_c = c.index_select(0, group_indices)
            group_decoder_text = (
                decoder_text.index_select(0, group_indices)
                if decoder_text is not None
                else None
            )
            group_skip_maps = {
                resolution: stage_map.index_select(0, group_indices)
                for resolution, stage_map in skip_maps.items()
            }
            group_router_weights = {
                resolution: weights.index_select(0, group_indices)
                for resolution, weights in stage_router_weights.items()
            }

            decoder_map = self.dynamic_bottleneck_projs[str(bottleneck_resolution)](
                group_skip_maps[bottleneck_resolution]
            )
            bottleneck_weight = group_router_weights.get(bottleneck_resolution)
            if bottleneck_weight is not None:
                decoder_map = decoder_map * bottleneck_weight

            group_decoder_stage_stats = []
            if bottleneck_resolution == self.decoder_anchor_resolution:
                decoder_map = self._apply_decoder_anchor_processor(
                    decoder_map,
                    group_c,
                    group_decoder_text,
                    group_decoder_stage_stats,
                )

            for resolution in self.decoder_stage_resolutions:
                if resolution <= bottleneck_resolution:
                    continue
                decoder_map, stage_stats = self._apply_decoder_stage(
                    resolution,
                    decoder_map,
                    group_skip_maps.get(resolution),
                    group_c,
                    group_decoder_text,
                    stage_router_weight=group_router_weights.get(resolution),
                )
                group_decoder_stage_stats.append(stage_stats)

            final_map_pre_norm = decoder_map.norm(dim=-1).mean().detach()
            pre_norm_accum = pre_norm_accum + final_map_pre_norm * group_count
            group_final_map = self._normalize_final_decoder_map(decoder_map)
            if final_maps is None:
                final_maps = group_final_map.new_empty(
                    (batch_size, *group_final_map.shape[1:])
                )
            final_maps.index_copy_(0, group_indices, group_final_map)

            for stage_stats in group_decoder_stage_stats:
                stage_stats["dynamic_bottleneck_resolution"] = int(
                    bottleneck_resolution
                )
                stage_stats["dynamic_group_fraction"] = float(group_count) / float(
                    batch_size
                )
            decoder_stage_stats.extend(group_decoder_stage_stats)
            dynamic_bottleneck_stats.append(
                {
                    "resolution": int(bottleneck_resolution),
                    "sample_count": int(group_count),
                    "sample_fraction": float(group_count) / float(batch_size),
                    "start_weight": (
                        bottleneck_weight.mean().detach()
                        if bottleneck_weight is not None
                        else static_lowest_bottleneck_map.new_tensor(0.0).detach()
                    ),
                    "pre_norm": final_map_pre_norm,
                }
            )

        if final_maps is None:
            raise RuntimeError("dynamic bottleneck router produced no decoder groups")

        final_map_pre_norm = pre_norm_accum / float(batch_size)
        selected_fractions = {
            resolution: selected_masks[resolution].float().mean().detach()
            for resolution in self.dynamic_bottleneck_candidates_ascending
        }
        bottleneck_fractions = {
            resolution: (bottleneck_resolutions == resolution).float().mean().detach()
            for resolution in self.dynamic_bottleneck_candidates_ascending
        }
        stage_router_stats = dict(stage_router_stats)
        stage_router_stats["dynamic_selected_fractions"] = selected_fractions
        stage_router_stats["dynamic_bottleneck_fractions"] = bottleneck_fractions

        self.latest_hierarchy_stats = {
            "enabled": True,
            "num_stages": len(encoder_stage_stats),
            "final_h": int(self.fusion_anchor_resolution),
            "final_w": int(self.fusion_anchor_resolution),
            "reached_global_context": float(self.hierarchy_reaches_global_context),
            "stage_metrics": encoder_stage_stats,
            "compression_metrics": compression_stage_stats,
            "dynamic_bottleneck_enabled": True,
            "dynamic_bottleneck_candidates": list(self.dynamic_bottleneck_candidates),
            "dynamic_bottleneck_select_k": int(self.dynamic_bottleneck_select_k),
            "dynamic_bottleneck_metrics": dynamic_bottleneck_stats,
            "dynamic_selected_fractions": selected_fractions,
            "dynamic_bottleneck_fractions": bottleneck_fractions,
        }
        self.latest_factorized_router_stats = {
            "enabled": float(self.stage_contribution_router is not None),
            "use_mamba_size_router": float(self.use_mamba_size_router),
            "use_compression_router": float(self.use_compression_router),
            "routed_stage_resolutions": list(self.routed_stage_resolutions),
            "routed_stage_count": int(self.routed_stage_count),
            "include_anchor_in_stage_router": float(
                self.include_anchor_in_stage_router
            ),
            "stage_router": stage_router_stats,
            "compression_metrics": compression_stage_stats,
        }
        self.latest_fusion_stats = {
            "enabled": True,
            "selected_stage_resolutions": list(self.fusion_selected_stages),
            "anchor_resolution": int(self.decoder_anchor_resolution),
            "bottleneck_resolution": int(self.fusion_anchor_resolution),
            "dynamic_bottleneck_enabled": True,
            "dynamic_bottleneck_candidates": list(self.dynamic_bottleneck_candidates),
            "dynamic_bottleneck_metrics": dynamic_bottleneck_stats,
            "stage_dim": int(self.fusion_stage_dim),
            "stage_dims": {
                resolution: int(dim)
                for resolution, dim in sorted(self.decoder_stage_dims.items())
            },
            "fusion_mode": self.fusion_mode,
            "gate_type": self.fusion_gate_type,
            "pos_embed_type": self.fusion_pos_embed_type,
            "block_depth": int(self.fusion_block_depth),
            "fusion_conv_depth": int(self.fusion_conv_depth),
            "fusion_pre_mamba_conv_depth": int(self.fusion_pre_mamba_conv_depth),
            "fusion_post_mamba_conv_depth": int(self.fusion_post_mamba_conv_depth),
            "anchor_builder_depth": int(self.anchor_builder_depth),
            "anchor_builder_stage_depth": int(self.anchor_builder_stage_depth),
            "channel_gate_stages": sorted(self.fusion_channel_gate_stages),
            "local_mamba_stage_resolutions": sorted(self.local_mamba_stage_resolutions),
            "spatial_gate_enabled": float(self.fusion_use_spatial_gate),
            "final_skip_refiner_enabled": float(self.final_skip_refiner is not None),
            "final_skip_refiner_depth": int(self.final_skip_refiner_depth),
            "final_skip_refiner_conv_type": self.final_skip_refiner_conv_type,
            "final_skip_refiner_use_channel_gate": float(
                self.final_skip_refiner_use_channel_gate
            ),
            "final_skip_refiner_use_spatial_gate": float(
                self.final_skip_refiner_use_spatial_gate
            ),
            "aux_4x4_context_enabled": float(self.aux_4x4_context_adapter is not None),
            "aux_4x4_inject_target": self.aux_4x4_inject_target,
            "aux_4x4_context_resolution": int(self.aux_4x4_context_resolution),
            "aux_4x4_target_resolution": int(self.aux_4x4_target_resolution),
            "aux_4x4_prepool_depth": int(self.aux_4x4_prepool_depth),
            "aux_4x4_context_depth": int(self.aux_4x4_context_depth),
            "aux_4x4_fusion_depth": int(self.aux_4x4_fusion_depth),
            "aux_4x4_use_timestep_gate": float(self.aux_4x4_use_timestep_gate),
            "aux_4x4_residual_scale_multiplier": float(
                self.aux_4x4_residual_scale_multiplier
            ),
            "stage_depths": {
                resolution: int(depth)
                for resolution, depth in sorted(self.decoder_stage_depths.items())
            },
            "prediction_head_type": self.fusion_prediction_head_type,
            "decoder_stage_metrics": decoder_stage_stats,
            "runtime_overrides": self.get_fusion_runtime_overrides(),
            "factorized_router": self.latest_factorized_router_stats,
        }
        for stage_stats in decoder_stage_stats:
            resolution = stage_stats["resolution"]
            if "gate_mean" in stage_stats:
                self.latest_fusion_stats.setdefault("stage_gate_means", {})[
                    resolution
                ] = stage_stats["gate_mean"]
                self.latest_fusion_stats.setdefault("stage_gate_stds", {})[
                    resolution
                ] = stage_stats["gate_std"]
                self.latest_fusion_stats.setdefault("stage_raw_gate_means", {})[
                    resolution
                ] = stage_stats["raw_gate_mean"]
                self.latest_fusion_stats.setdefault("stage_raw_gate_stds", {})[
                    resolution
                ] = stage_stats["raw_gate_std"]
                self.latest_fusion_stats.setdefault("stage_feature_norms", {})[
                    resolution
                ] = stage_stats["output_norm"]

        self.latest_backbone_stats = {
            "final_map_pre_norm": final_map_pre_norm,
            "final_map_post_norm": final_maps.norm(dim=-1).mean().detach(),
            "timestep_condition_norm": timestep_condition.norm(dim=-1).mean().detach(),
            "highres_processor_norm": highres_stats["output_norm"],
            "downsample_premix_depth": float(self.downsample_premix_depth),
            "downsample_uses_separable_conv": float(
                self.downsample_conv_type == "separable"
            ),
            "highres_local_uses_separable_conv": float(
                self.highres_local_conv_type == "separable"
            ),
        }
        return {
            "final_map": final_maps,
            "bottleneck_map": static_lowest_bottleneck_map,
            "skip_maps": skip_maps,
            "condition": c,
            "timestep_condition": timestep_condition,
            "stage_router_weights": stage_router_weights,
            "dynamic_bottleneck_resolutions": bottleneck_resolutions,
        }

    def get_hierarchy_logging_metrics(self):
        metrics = super().get_hierarchy_logging_metrics()
        if not self.latest_hierarchy_stats.get("dynamic_bottleneck_enabled", False):
            return metrics
        metrics["dynamic_bottleneck/enabled"] = 1.0
        metrics["dynamic_bottleneck/select_k"] = float(self.dynamic_bottleneck_select_k)
        for resolution in self.dynamic_bottleneck_candidates:
            metrics[f"dynamic_bottleneck/candidate_{resolution}"] = 1.0
        for resolution, value in self.latest_hierarchy_stats.get(
            "dynamic_selected_fractions", {}
        ).items():
            metrics[f"dynamic_bottleneck/stage_{resolution}_selected"] = float(value)
        for resolution, value in self.latest_hierarchy_stats.get(
            "dynamic_bottleneck_fractions", {}
        ).items():
            metrics[f"dynamic_bottleneck/resolution_{resolution}_fraction"] = float(
                value
            )
        for values in self.latest_hierarchy_stats.get(
            "dynamic_bottleneck_metrics", []
        ):
            resolution = values["resolution"]
            metrics[f"dynamic_bottleneck/resolution_{resolution}_start_weight"] = float(
                values["start_weight"]
            )
            metrics[f"dynamic_bottleneck/resolution_{resolution}_pre_norm"] = float(
                values["pre_norm"]
            )
        return metrics

    def load_state_dict(self, state_dict, strict=True):
        remapped_state_dict = dict(state_dict)
        current_state_dict = self.state_dict()
        for key, value in current_state_dict.items():
            if key.startswith("dynamic_bottleneck_projs.") and key not in remapped_state_dict:
                remapped_state_dict[key] = value
        return super().load_state_dict(remapped_state_dict, strict=strict)
