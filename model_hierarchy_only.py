import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from model_zigma import (
    RMSNorm,
    PatchEmbed_Video,
    TimestepEmbedder,
    LabelEmbedder,
    HierarchicalFinalLayer,
    compute_window_grid_size,
    pool_tokens,
    map_to_tokens,
    tokens_to_map,
    create_block,
    build_scan_block_kwargs,
    get_2d_sincos_pos_embed,
    layer_norm_fn,
    rms_norm_fn,
)


def parse_stage_resolution_spec(stage_spec):
    if stage_spec is None:
        return []
    if isinstance(stage_spec, str):
        parts = [part.strip() for part in stage_spec.split(",") if part.strip()]
    elif isinstance(stage_spec, (list, tuple)):
        parts = list(stage_spec)
    else:
        raise ValueError(f"Unsupported fusion_selected_stages type: {type(stage_spec)}")

    normalized = []
    for part in parts:
        if isinstance(part, str):
            token = part.lower().replace("x", "").strip()
            normalized.append(int(token))
        else:
            normalized.append(int(part))
    return normalized


class HierarchicalLocalCompressor(nn.Module):
    def __init__(
        self,
        dim,
        stage_depth,
        window_size,
        stride,
        context_compress_type,
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
        self.window_size = window_size
        self.stride = stride
        self.context_compress_type = context_compress_type
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        factory_kwargs = {"device": device, "dtype": dtype}

        block_kwargs = {"use_jit": use_jit}
        if scan_type != "v2":
            block_kwargs.update(
                build_scan_block_kwargs(
                    scan_type=scan_type,
                    patch_side_len=window_size,
                    depth=stage_depth,
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
                for layer_idx in range(stage_depth)
            ]
        )
        self.stage_norm = (nn.LayerNorm if not rms_norm else RMSNorm)(
            dim, eps=norm_epsilon, **factory_kwargs
        )
        self.output_proj = nn.Linear(dim, dim, **factory_kwargs)
        self.residual_proj = nn.Identity()

    def _extract_windows(self, context_map):
        b, h, w, c = context_map.shape
        out_h, pad_h = compute_window_grid_size(h, self.window_size, self.stride)
        out_w, pad_w = compute_window_grid_size(w, self.window_size, self.stride)
        x = context_map.permute(0, 3, 1, 2)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = x.unfold(2, self.window_size, self.stride).unfold(
            3, self.window_size, self.stride
        )
        x = x.permute(0, 2, 3, 4, 5, 1).contiguous()
        windows = x.reshape(b * out_h * out_w, self.window_size * self.window_size, c)
        return windows, (out_h, out_w)

    def forward(self, context_map, c, text=None):
        windows, out_hw = self._extract_windows(context_map)
        residual = None
        hidden_states = windows
        repeat_factor = out_hw[0] * out_hw[1]
        repeated_c = c.repeat_interleave(repeat_factor, dim=0)
        repeated_text = (
            text.repeat_interleave(repeat_factor, dim=0) if text is not None else None
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

        compressed_update = pool_tokens(stage_hidden, self.context_compress_type)
        compressed_input = pool_tokens(windows, self.context_compress_type)
        # Inter-layer residual:
        # compress the previous layer's 2D context to the current resolution and
        # add the newly summarized update before passing it to the next layer.
        stage_output = self.residual_proj(compressed_input) + self.output_proj(compressed_update)
        stage_output = stage_output.reshape(
            context_map.shape[0], out_hw[0], out_hw[1], self.dim
        )
        stats = {
            "input_resolution": tuple(context_map.shape[1:3]),
            "output_resolution": out_hw,
            "context_count": int(out_hw[0] * out_hw[1]),
        }
        return stage_output, stats


class StageAlignToAnchor(nn.Module):
    def __init__(self, anchor_resolution):
        super().__init__()
        self.anchor_resolution = anchor_resolution

    def forward(self, stage_map):
        b, h, w, c = stage_map.shape
        if h == self.anchor_resolution and w == self.anchor_resolution:
            return stage_map

        x = stage_map.permute(0, 3, 1, 2).contiguous()
        if h > self.anchor_resolution or w > self.anchor_resolution:
            x = F.adaptive_avg_pool2d(x, (self.anchor_resolution, self.anchor_resolution))
        else:
            x = F.interpolate(
                x,
                size=(self.anchor_resolution, self.anchor_resolution),
                mode="nearest",
            )
        return x.permute(0, 2, 3, 1).contiguous()


class AnchorPositionalEmbedding(nn.Module):
    def __init__(self, anchor_resolution, dim, device, dtype):
        super().__init__()
        self.anchor_resolution = anchor_resolution
        self.pos_embed = nn.Parameter(
            torch.zeros(1, anchor_resolution, anchor_resolution, dim, device=device, dtype=dtype)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, stage_map):
        return stage_map + self.pos_embed


class StageTimestepGate(nn.Module):
    def __init__(self, cond_dim, device, dtype):
        super().__init__()
        self.norm = nn.LayerNorm(cond_dim, eps=1e-6, device=device, dtype=dtype)
        self.linear = nn.Linear(cond_dim, 1, device=device, dtype=dtype)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def compute_gate(self, condition, batch_size):
        return torch.sigmoid(self.linear(self.norm(condition))).view(
            batch_size, 1, 1, 1
        )

    def forward(self, stage_map, condition):
        gate = self.compute_gate(condition, stage_map.shape[0])
        return stage_map * gate, gate


class FusionResidualBlock(nn.Module):
    def __init__(self, dim, device, dtype):
        super().__init__()
        hidden_dim = dim * 4
        self.norm = nn.LayerNorm(dim, eps=1e-6, device=device, dtype=dtype)
        self.fc1 = nn.Linear(dim, hidden_dim, device=device, dtype=dtype)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, device=device, dtype=dtype)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class AnchorPredictionHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        anchor_resolution,
        output_tokens,
        patch_size,
        out_channels,
        device,
        dtype,
    ):
        super().__init__()
        self.anchor_resolution = anchor_resolution
        self.output_tokens = output_tokens
        self.patch_dim = patch_size * patch_size * out_channels
        self.norm = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
            device=device,
            dtype=dtype,
        )
        self.linear = nn.Linear(
            anchor_resolution * anchor_resolution * hidden_size,
            output_tokens * self.patch_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        b = x.shape[0]
        x = self.norm(x)
        x = x.reshape(b, self.anchor_resolution * self.anchor_resolution * x.shape[-1])
        x = self.linear(x)
        return x.reshape(b, self.output_tokens, self.patch_dim)


class ConvUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, upsample, device, dtype):
        super().__init__()
        self.upsample = upsample
        self.norm1 = nn.GroupNorm(1, in_channels, eps=1e-6).to(device).to(dtype)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            device=device,
            dtype=dtype,
        )
        self.norm2 = nn.GroupNorm(1, out_channels, eps=1e-6).to(device).to(dtype)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
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
        residual = x
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            residual = F.interpolate(residual, scale_factor=2.0, mode="nearest")

        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.conv2(F.gelu(self.norm2(x)))
        residual = self.skip(residual)
        return x + residual


class AnchorConvUpsampleHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        anchor_resolution,
        output_tokens,
        patch_size,
        out_channels,
        device,
        dtype,
    ):
        super().__init__()
        token_side = int(math.isqrt(output_tokens))
        if token_side * token_side != output_tokens:
            raise ValueError(
                "anchor_conv_upsample requires a square token grid, "
                f"got output_tokens={output_tokens}"
            )

        self.anchor_resolution = anchor_resolution
        self.output_tokens = output_tokens
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.patch_dim = patch_size * patch_size * out_channels
        self.output_token_side = token_side
        self.output_resolution = token_side * patch_size
        self.norm = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
            device=device,
            dtype=dtype,
        )
        self.stem = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=3,
            padding=1,
            device=device,
            dtype=dtype,
        )

        current_channels = hidden_size
        current_resolution = anchor_resolution
        min_decoder_channels = max(32, self.patch_dim)
        blocks = []
        while current_resolution < self.output_resolution:
            next_channels = max(min_decoder_channels, current_channels // 2)
            blocks.append(
                ConvUpsampleBlock(
                    current_channels,
                    next_channels,
                    upsample=True,
                    device=device,
                    dtype=dtype,
                )
            )
            current_channels = next_channels
            current_resolution *= 2
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.GroupNorm(1, current_channels, eps=1e-6).to(device).to(dtype)
        self.out_conv = nn.Conv2d(
            current_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            device=device,
            dtype=dtype,
        )

    def _map_to_patch_tokens(self, image):
        if self.patch_size == 1:
            return image.permute(0, 2, 3, 1).reshape(image.shape[0], -1, self.out_channels)

        patches = F.unfold(
            image,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return patches.transpose(1, 2).contiguous()

    def forward(self, x):
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        if x.shape[-2:] != (self.output_resolution, self.output_resolution):
            x = F.interpolate(
                x,
                size=(self.output_resolution, self.output_resolution),
                mode="nearest",
            )
        x = self.out_conv(F.gelu(self.out_norm(x)))
        return self._map_to_patch_tokens(x)


class MultiScaleFusionHead(nn.Module):
    def __init__(
        self,
        input_dim,
        output_tokens,
        patch_size,
        out_channels,
        selected_stage_resolutions,
        anchor_resolution,
        stage_dim,
        fusion_mode,
        gate_type,
        pos_embed_type,
        block_depth,
        prediction_head_type,
        condition_dim,
        device,
        dtype,
    ):
        super().__init__()
        self.selected_stage_resolutions = [int(res) for res in selected_stage_resolutions]
        self.anchor_resolution = anchor_resolution
        self.stage_dim = stage_dim
        self.fusion_mode = fusion_mode
        self.gate_type = gate_type
        self.pos_embed_type = pos_embed_type
        self.block_depth = block_depth
        self.prediction_head_type = prediction_head_type

        if fusion_mode not in {"concat", "gated_sum"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        if gate_type != "stage_timestep":
            raise ValueError(f"Unsupported fusion_gate_type: {gate_type}")
        if pos_embed_type not in {"anchor_shared", "none"}:
            raise ValueError(f"Unsupported fusion_pos_embed_type: {pos_embed_type}")
        if prediction_head_type not in {"anchor_direct", "anchor_conv_upsample"}:
            raise ValueError(
                f"Unsupported fusion_prediction_head_type: {prediction_head_type}"
            )

        self.projectors = nn.ModuleDict(
            {
                str(res): nn.Linear(input_dim, stage_dim, device=device, dtype=dtype)
                for res in self.selected_stage_resolutions
            }
        )
        self.input_norms = nn.ModuleDict(
            {
                str(res): nn.LayerNorm(
                    input_dim,
                    elementwise_affine=False,
                    eps=1e-6,
                    device=device,
                    dtype=dtype,
                )
                for res in self.selected_stage_resolutions
            }
        )
        self.aligner = StageAlignToAnchor(anchor_resolution)
        self.pos_embed = (
            AnchorPositionalEmbedding(anchor_resolution, stage_dim, device, dtype)
            if pos_embed_type == "anchor_shared"
            else None
        )
        self.gates = nn.ModuleDict(
            {
                str(res): StageTimestepGate(condition_dim, device, dtype)
                for res in self.selected_stage_resolutions
            }
        )
        fused_input_dim = (
            len(self.selected_stage_resolutions) * stage_dim
            if fusion_mode == "concat"
            else stage_dim
        )
        self.fusion_proj = nn.Linear(
            fused_input_dim, stage_dim, device=device, dtype=dtype
        )
        self.fusion_blocks = nn.ModuleList(
            [FusionResidualBlock(stage_dim, device, dtype) for _ in range(block_depth)]
        )
        if prediction_head_type == "anchor_direct":
            self.prediction_head = AnchorPredictionHead(
                hidden_size=stage_dim,
                anchor_resolution=anchor_resolution,
                output_tokens=output_tokens,
                patch_size=patch_size,
                out_channels=out_channels,
                device=device,
                dtype=dtype,
            )
        else:
            self.prediction_head = AnchorConvUpsampleHead(
                hidden_size=stage_dim,
                anchor_resolution=anchor_resolution,
                output_tokens=output_tokens,
                patch_size=patch_size,
                out_channels=out_channels,
                device=device,
                dtype=dtype,
            )
        self.disabled_stage_resolutions = set()
        self.force_stage_gate_values = {}
        self.gate_floor = None

    def set_runtime_overrides(
        self,
        *,
        disabled_stage_resolutions=None,
        force_stage_gate_values=None,
        gate_floor=None,
    ):
        self.disabled_stage_resolutions = {
            int(resolution) for resolution in (disabled_stage_resolutions or [])
        }
        if force_stage_gate_values is None:
            self.force_stage_gate_values = {}
        else:
            self.force_stage_gate_values = {
                int(resolution): float(value)
                for resolution, value in force_stage_gate_values.items()
            }
        self.gate_floor = None if gate_floor is None else float(gate_floor)

    def clear_runtime_overrides(self):
        self.set_runtime_overrides()

    def get_runtime_overrides(self):
        return {
            "disabled_stage_resolutions": sorted(self.disabled_stage_resolutions),
            "force_stage_gate_values": dict(self.force_stage_gate_values),
            "gate_floor": self.gate_floor,
        }

    def forward(self, stage_maps_by_resolution, condition):
        aligned_features = []
        gate_stats = {}
        feature_norms = {}
        raw_gate_stats = {}
        projected_norms = {}
        aligned_norms = {}
        projected_norm_values = {}
        aligned_norm_values = {}
        gated_norm_values = {}
        raw_gate_values = {}
        effective_gate_values = {}

        for resolution in self.selected_stage_resolutions:
            key = str(resolution)
            if resolution not in stage_maps_by_resolution:
                available = sorted(stage_maps_by_resolution.keys())
                raise KeyError(
                    f"Missing selected fusion stage {resolution}x{resolution}. "
                    f"Available stages: {available}"
                )

            stage_map = stage_maps_by_resolution[resolution]
            stage_map = self.input_norms[key](stage_map)
            projected = self.projectors[key](stage_map)
            projected_norm = projected.norm(dim=-1).mean(dim=(1, 2)).detach()
            aligned = self.aligner(projected)
            if self.pos_embed is not None:
                aligned = self.pos_embed(aligned)
            aligned_norm = aligned.norm(dim=-1).mean(dim=(1, 2)).detach()
            raw_gate = self.gates[key].compute_gate(condition, aligned.shape[0])
            gate = raw_gate
            if resolution in self.disabled_stage_resolutions:
                gate = torch.zeros_like(gate)
            if resolution in self.force_stage_gate_values:
                gate = torch.full_like(gate, self.force_stage_gate_values[resolution])
            if self.gate_floor is not None:
                gate = self.gate_floor + (1.0 - self.gate_floor) * gate
            gated = aligned * gate
            gated_norm = gated.norm(dim=-1).mean(dim=(1, 2)).detach()
            aligned_features.append(gated)
            gate_stats[resolution] = gate.squeeze(-1).squeeze(-1).squeeze(-1).detach()
            raw_gate_stats[resolution] = (
                raw_gate.squeeze(-1).squeeze(-1).squeeze(-1).detach()
            )
            feature_norms[resolution] = gated_norm.mean().detach()
            projected_norms[resolution] = projected_norm.mean().detach()
            aligned_norms[resolution] = aligned_norm.mean().detach()
            projected_norm_values[resolution] = projected_norm
            aligned_norm_values[resolution] = aligned_norm
            gated_norm_values[resolution] = gated_norm
            raw_gate_values[resolution] = raw_gate_stats[resolution]
            effective_gate_values[resolution] = gate_stats[resolution]

        if self.fusion_mode == "concat":
            fused = torch.cat(aligned_features, dim=-1)
        else:
            fused = torch.stack(aligned_features, dim=0).mean(dim=0)

        fused = self.fusion_proj(fused)
        for block in self.fusion_blocks:
            fused = block(fused)

        predicted_tokens = self.prediction_head(fused)
        stats = {
            "enabled": True,
            "selected_stage_resolutions": list(self.selected_stage_resolutions),
            "anchor_resolution": int(self.anchor_resolution),
            "stage_dim": int(self.stage_dim),
            "fusion_mode": self.fusion_mode,
            "gate_type": self.gate_type,
            "pos_embed_type": self.pos_embed_type,
            "block_depth": int(self.block_depth),
            "prediction_head_type": self.prediction_head_type,
            "fused_norm": fused.norm(dim=-1).mean().detach(),
            "stage_gate_means": {
                resolution: gate_stats[resolution].mean().detach()
                for resolution in self.selected_stage_resolutions
            },
            "stage_gate_stds": {
                resolution: gate_stats[resolution].std(unbiased=False).detach()
                for resolution in self.selected_stage_resolutions
            },
            "stage_raw_gate_means": {
                resolution: raw_gate_stats[resolution].mean().detach()
                for resolution in self.selected_stage_resolutions
            },
            "stage_raw_gate_stds": {
                resolution: raw_gate_stats[resolution].std(unbiased=False).detach()
                for resolution in self.selected_stage_resolutions
            },
            "stage_feature_norms": feature_norms,
            "stage_projected_norms": projected_norms,
            "stage_aligned_norms": aligned_norms,
            "stage_gate_values": effective_gate_values,
            "stage_raw_gate_values": raw_gate_values,
            "stage_projected_norm_values": projected_norm_values,
            "stage_aligned_norm_values": aligned_norm_values,
            "stage_feature_norm_values": gated_norm_values,
            "runtime_overrides": self.get_runtime_overrides(),
        }
        return predicted_tokens, stats


class HierarchicalMambaLocal(nn.Module):
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
        hierarchy_stride=1,
        first_layer_stride=None,
        context_compress_type="last",
        hierarchy_max_stages=None,
        hierarchy_stage_depth=3,
        hierarchy_allow_partial=False,
        share_stage_processor=False,
        hierarchical_output_mode="prediction",
        use_multiscale_fusion_head=False,
        fusion_mode="concat",
        fusion_selected_stages="8,4,2,1",
        fusion_anchor_resolution=4,
        fusion_stage_dim=256,
        fusion_gate_type="stage_timestep",
        fusion_pos_embed_type="anchor_shared",
        fusion_block_depth=1,
        fusion_prediction_head_type="anchor_direct",
        fusion_logging_verbose=True,
    ):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
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
        self.fusion_stage_dim = fusion_stage_dim
        self.fusion_gate_type = fusion_gate_type
        self.fusion_pos_embed_type = fusion_pos_embed_type
        self.fusion_block_depth = fusion_block_depth
        self.fusion_prediction_head_type = fusion_prediction_head_type
        self.fusion_logging_verbose = fusion_logging_verbose
        self.final_layer = None
        self.latest_hierarchy_stats = {}
        self.latest_fusion_stats = {}
        self.latest_backbone_stats = {}
        if str(device) == "cpu":
            fused_add_norm = False
            rms_norm = False

        if hierarchical_output_mode not in {"prediction", "context"}:
            raise ValueError(
                f"Unsupported hierarchical_output_mode: {hierarchical_output_mode}"
            )
        if context_compress_type not in {"last", "mean"}:
            raise ValueError(
                f"Unsupported context_compress_type: {context_compress_type}"
            )
        if hierarchy_stage_depth <= 0:
            raise ValueError(
                f"hierarchy_stage_depth must be positive, got {hierarchy_stage_depth}"
            )
        if fusion_anchor_resolution <= 0:
            raise ValueError(
                f"fusion_anchor_resolution must be positive, got {fusion_anchor_resolution}"
            )
        if fusion_stage_dim <= 0:
            raise ValueError(f"fusion_stage_dim must be positive, got {fusion_stage_dim}")
        if fusion_block_depth < 0:
            raise ValueError(
                f"fusion_block_depth must be non-negative, got {fusion_block_depth}"
            )
        if self.use_multiscale_fusion_head and len(self.fusion_selected_stages) == 0:
            raise ValueError("fusion_selected_stages must not be empty when fusion is enabled")
        if video_frames != 0:
            raise NotImplementedError("HierarchicalMambaLocal currently supports images only")

        num_patches = (img_dim // patch_size) ** 2
        self.patch_side_len = int(math.sqrt(num_patches))
        self.hierarchy_input_size = (self.patch_side_len, self.patch_side_len)

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
        cur_h, cur_w = self.hierarchy_input_size
        while True:
            stage_stride = (
                self.first_layer_stride
                if len(self.hierarchy_stage_layout) == 0
                else self.hierarchy_stride
            )
            out_h, _ = compute_window_grid_size(
                cur_h, self.hierarchy_window_size, stage_stride
            )
            out_w, _ = compute_window_grid_size(
                cur_w, self.hierarchy_window_size, stage_stride
            )
            self.hierarchy_stage_layout.append(
                {
                    "layer_idx": len(self.hierarchy_stage_layout),
                    "input_resolution": (cur_h, cur_w),
                    "output_resolution": (out_h, out_w),
                    "stride": stage_stride,
                }
            )
            if out_h == 1 and out_w == 1:
                break
            if self.hierarchy_max_stages is not None and len(
                self.hierarchy_stage_layout
            ) >= self.hierarchy_max_stages:
                break
            cur_h, cur_w = out_h, out_w

        self.hierarchy_final_resolution = self.hierarchy_stage_layout[-1][
            "output_resolution"
        ]
        self.hierarchy_reaches_global_context = self.hierarchy_final_resolution == (1, 1)
        if not self.hierarchy_allow_partial and not self.hierarchy_reaches_global_context:
            raise ValueError(
                "Hierarchy-only model must reach a single global context unless "
                "hierarchy_allow_partial=True. "
                f"final_resolution={self.hierarchy_final_resolution}"
            )

        dpr = torch.linspace(0, drop_path_rate, hierarchy_stage_depth).tolist()
        self.first_stage_processor = None
        if self.share_stage_processor:
            if self.first_layer_stride != self.hierarchy_stride:
                self.first_stage_processor = HierarchicalLocalCompressor(
                    dim=embed_dim,
                    stage_depth=hierarchy_stage_depth,
                    window_size=hierarchy_window_size,
                    stride=self.first_layer_stride,
                    context_compress_type=context_compress_type,
                    has_text=has_text,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    drop_path_values=dpr,
                    scan_type=scan_type,
                    use_jit=use_jit,
                    use_checkpoint=use_checkpoint,
                    device=device,
                    dtype=dtype,
                )
            self.shared_processor = HierarchicalLocalCompressor(
                dim=embed_dim,
                stage_depth=hierarchy_stage_depth,
                window_size=hierarchy_window_size,
                stride=hierarchy_stride,
                context_compress_type=context_compress_type,
                has_text=has_text,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                drop_path_values=dpr,
                scan_type=scan_type,
                use_jit=use_jit,
                use_checkpoint=use_checkpoint,
                device=device,
                dtype=dtype,
            )
            self.stage_processors = None
        else:
            self.shared_processor = None
            self.stage_processors = nn.ModuleList(
                [
                    HierarchicalLocalCompressor(
                        dim=embed_dim,
                        stage_depth=hierarchy_stage_depth,
                        window_size=hierarchy_window_size,
                        stride=stage_layout["stride"],
                        context_compress_type=context_compress_type,
                        has_text=has_text,
                        ssm_cfg=ssm_cfg,
                        norm_epsilon=norm_epsilon,
                        rms_norm=rms_norm,
                        residual_in_fp32=residual_in_fp32,
                        fused_add_norm=fused_add_norm,
                        drop_path_values=dpr,
                        scan_type=scan_type,
                        use_jit=use_jit,
                        use_checkpoint=use_checkpoint,
                        device=device,
                        dtype=dtype,
                    )
                    for stage_layout in self.hierarchy_stage_layout
                ]
            )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **self.factory_kwargs
        )
        self.output_head = None
        if not self.use_multiscale_fusion_head:
            self.output_head = (
                HierarchicalFinalLayer(
                    self.embed_dim,
                    output_tokens=num_patches,
                    patch_size=patch_size,
                    out_channels=self.out_channels,
                )
                .to(device)
                .to(dtype)
            )
        self.multiscale_fusion_head = None
        if self.use_multiscale_fusion_head:
            self.multiscale_fusion_head = MultiScaleFusionHead(
                input_dim=self.embed_dim,
                output_tokens=num_patches,
                patch_size=patch_size,
                out_channels=self.out_channels,
                selected_stage_resolutions=self.fusion_selected_stages,
                anchor_resolution=self.fusion_anchor_resolution,
                stage_dim=self.fusion_stage_dim,
                fusion_mode=self.fusion_mode,
                gate_type=self.fusion_gate_type,
                pos_embed_type=self.fusion_pos_embed_type,
                block_depth=self.fusion_block_depth,
                prediction_head_type=self.fusion_prediction_head_type,
                condition_dim=self.embed_dim,
                device=device,
                dtype=dtype,
            )

        self.initialize_weights()

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

        processors = (
            (
                ([self.first_stage_processor] if self.first_stage_processor is not None else [])
                + ([self.shared_processor] if self.shared_processor is not None else [])
            )
            if self.stage_processors is None
            else list(self.stage_processors)
        )
        for processor in processors:
            for block in processor.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

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
        if self.has_text:
            y = self.y_embedder(y)
            c = timestep_condition + y.mean(dim=1)
        elif self.num_classes > 0:
            c = timestep_condition + self.y_embedder(y, self.training)
        else:
            c = timestep_condition
        return c, y, timestep_condition

    def _run_hierarchy(self, initial_map, c, y=None, collect_stage_maps=False):
        current_map = initial_map
        stage_stats = []
        selected_stage_maps = {}

        for stage_idx, layout in enumerate(self.hierarchy_stage_layout):
            if self.stage_processors is not None:
                processor = self.stage_processors[stage_idx]
            elif stage_idx == 0 and self.first_stage_processor is not None:
                processor = self.first_stage_processor
            else:
                processor = self.shared_processor
            current_map, stats = processor(current_map, c=c, text=y)
            stats["stage_idx"] = stage_idx
            stats["expected_input_resolution"] = layout["input_resolution"]
            stats["expected_output_resolution"] = layout["output_resolution"]
            stats["stride"] = layout["stride"]
            stage_stats.append(stats)
            if collect_stage_maps:
                out_h, out_w = current_map.shape[1:3]
                if out_h == out_w and out_h in self.fusion_selected_stages:
                    selected_stage_maps[int(out_h)] = current_map

        final_resolution = tuple(current_map.shape[1:3])
        reached_global_context = final_resolution == (1, 1)
        if not self.hierarchy_allow_partial and not reached_global_context:
            raise AssertionError(
                "Hierarchy-only model did not reach a single global context. "
                f"Got final resolution {final_resolution}."
            )
        self.latest_hierarchy_stats = {
            "enabled": True,
            "num_stages": len(stage_stats),
            "final_h": int(final_resolution[0]),
            "final_w": int(final_resolution[1]),
            "reached_global_context": reached_global_context,
            "stage_metrics": stage_stats,
        }
        if collect_stage_maps:
            return current_map, selected_stage_maps
        return current_map

    def get_hierarchy_logging_metrics(self):
        metrics = {
            "hierarchy/enabled": 1.0,
            "hierarchy/window_size": float(self.hierarchy_window_size),
            "hierarchy/stride": float(self.hierarchy_stride),
            "hierarchy/first_layer_stride": float(self.first_layer_stride),
            "hierarchy/stage_depth": float(self.hierarchy_stage_depth),
            "hierarchy/num_stages": float(
                self.latest_hierarchy_stats.get(
                    "num_stages", len(self.hierarchy_stage_layout)
                )
            ),
            "hierarchy/final_h": float(
                self.latest_hierarchy_stats.get(
                    "final_h", self.hierarchy_final_resolution[0]
                )
            ),
            "hierarchy/final_w": float(
                self.latest_hierarchy_stats.get(
                    "final_w", self.hierarchy_final_resolution[1]
                )
            ),
            "hierarchy/reached_global_context": float(
                self.latest_hierarchy_stats.get(
                    "reached_global_context", self.hierarchy_reaches_global_context
                )
            ),
            "hierarchy/shared_processor": float(self.share_stage_processor),
        }
        if self.latest_backbone_stats:
            final_map_pre_norm = self.latest_backbone_stats.get("final_map_pre_norm")
            if final_map_pre_norm is not None:
                metrics["backbone/norm_f_input_norm"] = float(final_map_pre_norm)
            final_map_post_norm = self.latest_backbone_stats.get("final_map_post_norm")
            if final_map_post_norm is not None:
                metrics["backbone/norm_f_output_norm"] = float(final_map_post_norm)
            timestep_condition_norm = self.latest_backbone_stats.get(
                "timestep_condition_norm"
            )
            if timestep_condition_norm is not None:
                metrics["backbone/timestep_condition_norm"] = float(
                    timestep_condition_norm
                )
            if "stage_1_uses_final_map" in self.latest_backbone_stats:
                metrics["fusion/stage_1_uses_normed_final_map"] = float(
                    self.latest_backbone_stats["stage_1_uses_final_map"]
                )
        for stats in self.latest_hierarchy_stats.get("stage_metrics", []):
            stage_idx = stats["stage_idx"]
            metrics[f"hierarchy/stage_{stage_idx}_input_h"] = float(
                stats["input_resolution"][0]
            )
            metrics[f"hierarchy/stage_{stage_idx}_input_w"] = float(
                stats["input_resolution"][1]
            )
            metrics[f"hierarchy/stage_{stage_idx}_output_h"] = float(
                stats["output_resolution"][0]
            )
            metrics[f"hierarchy/stage_{stage_idx}_output_w"] = float(
                stats["output_resolution"][1]
            )
            metrics[f"hierarchy/stage_{stage_idx}_stride"] = float(stats["stride"])
        metrics["fusion/enabled"] = float(self.use_multiscale_fusion_head)
        metrics["fusion/anchor_resolution"] = float(self.fusion_anchor_resolution)
        metrics["fusion/stage_dim"] = float(self.fusion_stage_dim)
        metrics["fusion/block_depth"] = float(self.fusion_block_depth)
        metrics["fusion/selected_stage_count"] = float(len(self.fusion_selected_stages))
        for resolution in self.fusion_selected_stages:
            metrics[f"fusion/selected_stage_{resolution}"] = 1.0
        if self.latest_fusion_stats:
            fused_norm = self.latest_fusion_stats.get("fused_norm")
            if fused_norm is not None:
                metrics["fusion/fused_norm"] = float(fused_norm)
            for resolution, value in self.latest_fusion_stats.get(
                "stage_gate_means", {}
            ).items():
                metrics[f"fusion/stage_{resolution}_gate_mean"] = float(value)
            for resolution, value in self.latest_fusion_stats.get(
                "stage_gate_stds", {}
            ).items():
                metrics[f"fusion/stage_{resolution}_gate_std"] = float(value)
            for resolution, value in self.latest_fusion_stats.get(
                "stage_raw_gate_means", {}
            ).items():
                metrics[f"fusion/stage_{resolution}_raw_gate_mean"] = float(value)
            for resolution, value in self.latest_fusion_stats.get(
                "stage_raw_gate_stds", {}
            ).items():
                metrics[f"fusion/stage_{resolution}_raw_gate_std"] = float(value)
            for resolution, value in self.latest_fusion_stats.get(
                "stage_feature_norms", {}
            ).items():
                metrics[f"fusion/stage_{resolution}_feature_norm"] = float(value)
            for resolution, value in self.latest_fusion_stats.get(
                "stage_projected_norms", {}
            ).items():
                metrics[f"fusion/stage_{resolution}_projected_norm"] = float(value)
            for resolution, value in self.latest_fusion_stats.get(
                "stage_aligned_norms", {}
            ).items():
                metrics[f"fusion/stage_{resolution}_aligned_norm"] = float(value)
            runtime_overrides = self.latest_fusion_stats.get("runtime_overrides", {})
            metrics["fusion/runtime_gate_floor"] = float(
                runtime_overrides.get("gate_floor") or 0.0
            )
        return metrics

    def set_fusion_runtime_overrides(
        self,
        *,
        disabled_stage_resolutions=None,
        force_stage_gate_values=None,
        gate_floor=None,
    ):
        if self.multiscale_fusion_head is None:
            raise AssertionError("Fusion head is not enabled on this model.")
        self.multiscale_fusion_head.set_runtime_overrides(
            disabled_stage_resolutions=disabled_stage_resolutions,
            force_stage_gate_values=force_stage_gate_values,
            gate_floor=gate_floor,
        )

    def clear_fusion_runtime_overrides(self):
        if self.multiscale_fusion_head is None:
            return
        self.multiscale_fusion_head.clear_runtime_overrides()

    def get_fusion_runtime_overrides(self):
        if self.multiscale_fusion_head is None:
            return {}
        return self.multiscale_fusion_head.get_runtime_overrides()

    def forward_backbone(self, hidden_states, t, y=None):
        hidden_states = self.x_embedder(hidden_states)
        c, y, timestep_condition = self._get_condition(hidden_states, t, y)

        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        initial_map = tokens_to_map(hidden_states, self.hierarchy_input_size)
        if self.use_multiscale_fusion_head:
            final_map, selected_stage_maps = self._run_hierarchy(
                initial_map, c=c, y=y, collect_stage_maps=True
            )
        else:
            final_map = self._run_hierarchy(initial_map, c=c, y=y)
            selected_stage_maps = None

        final_map_pre_norm = final_map.norm(dim=-1).mean().detach()
        final_tokens = map_to_tokens(final_map)
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
        final_map = tokens_to_map(final_tokens, final_map.shape[1:3])
        self.latest_backbone_stats = {
            "final_map_pre_norm": final_map_pre_norm,
            "final_map_post_norm": final_map.norm(dim=-1).mean().detach(),
            "timestep_condition_norm": timestep_condition.norm(dim=-1).mean().detach(),
            "stage_1_uses_final_map": float(
                self.use_multiscale_fusion_head
                and 1 in self.fusion_selected_stages
                and final_map.shape[1] == 1
                and final_map.shape[2] == 1
            ),
        }
        if self.use_multiscale_fusion_head:
            if (
                1 in self.fusion_selected_stages
                and final_map.shape[1] == 1
                and final_map.shape[2] == 1
            ):
                selected_stage_maps[1] = final_map
            return {
                "final_map": final_map,
                "selected_stage_maps": selected_stage_maps,
                "timestep_condition": timestep_condition,
            }
        return final_map

    def forward_prediction_head(self, backbone_output):
        if isinstance(backbone_output, dict):
            final_map = backbone_output["final_map"]
            selected_stage_maps = backbone_output.get("selected_stage_maps")
            timestep_condition = backbone_output.get("timestep_condition")
        else:
            final_map = backbone_output
            selected_stage_maps = None
            timestep_condition = None

        if self.use_multiscale_fusion_head:
            if self.multiscale_fusion_head is None:
                raise AssertionError("Fusion head is enabled but not initialized.")
            predicted_tokens, fusion_stats = self.multiscale_fusion_head(
                selected_stage_maps,
                timestep_condition,
            )
            self.latest_fusion_stats = fusion_stats
            return self.unpatchify(predicted_tokens)

        self.latest_fusion_stats = {}
        backbone_output = final_map
        hidden_states = map_to_tokens(backbone_output)
        if hidden_states.shape[1] != 1:
            raise AssertionError(
                "Prediction head expects exactly one final context token, "
                f"got {hidden_states.shape[1]}"
            )
        if self.output_head is None:
            raise AssertionError("Output head is not initialized for non-fusion mode.")
        hidden_states = self.output_head(hidden_states)
        return self.unpatchify(hidden_states)

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

    def load_state_dict(self, state_dict, strict=True):
        filtered_state_dict = dict(state_dict)
        if self.output_head is None:
            for key in list(filtered_state_dict.keys()):
                if key.startswith("output_head."):
                    filtered_state_dict.pop(key)

        incompatible = super().load_state_dict(filtered_state_dict, strict=False)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                "Error(s) in loading state_dict for HierarchicalMambaLocal:\n"
                f"Missing key(s): {incompatible.missing_keys}\n"
                f"Unexpected key(s): {incompatible.unexpected_keys}"
            )
        return incompatible
