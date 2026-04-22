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
        }
        processor_condition = self.condition_proj(condition)
        if skip_map is None:
            if self.absolute_pos_embed is not None:
                low = self.absolute_pos_embed(low)
            low = self.low_only_refinement(low)
            low = self.pre_mamba_blocks(low)
            refined, block_stats = self.processor(low, processor_condition, text=text)
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

        gated_skip = skip * gate
        if self.fusion_mode == "concat":
            fused = self.local_fusion(low, gated_skip)
        else:
            fused = low + gated_skip
            fused = self.gated_sum_refinement(fused)
        if self.absolute_pos_embed is not None:
            fused = self.absolute_pos_embed(fused)
        fused = self.pre_mamba_blocks(fused)

        refined, block_stats = self.processor(fused, processor_condition, text=text)
        refined = self.post_mamba_blocks(refined)
        stage_stats.update(block_stats)
        stage_stats["raw_gate_mean"] = raw_gate.mean().detach()
        stage_stats["raw_gate_std"] = raw_gate.std(unbiased=False).detach()
        stage_stats["gate_mean"] = gate.mean().detach()
        stage_stats["gate_std"] = gate.std(unbiased=False).detach()
        stage_stats["skip_norm"] = gated_skip.norm(dim=-1).mean().detach()
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
        self.latest_hierarchy_stats = {}
        self.latest_fusion_stats = {}
        self.latest_backbone_stats = {}
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

        def make_dpr(depth):
            return torch.linspace(0, drop_path_rate, depth).tolist()

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

        self.bottleneck_processor = self._build_stage_processor(
            self.fusion_anchor_resolution, bottleneck_stage_depth
        )

        selected_skip_resolutions = set(self.fusion_selected_stages) | {self.patch_side_len}
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

    def _build_stage_processor(self, resolution, depth):
        processor_scan_type = self._resolve_stage_scan_type(resolution)
        window_size = self._resolve_local_window_size(resolution)
        shift_size = self._resolve_local_shift_size(resolution)
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
        if self.has_text:
            y = self.y_embedder(y)
            c = timestep_condition + y.mean(dim=1)
        elif self.num_classes > 0:
            c = timestep_condition + self.y_embedder(y, self.training)
        else:
            c = timestep_condition
        return c, y, timestep_condition

    def _apply_decoder_stage(self, resolution, decoder_map, skip_map, c, y):
        disable_stage = resolution in self.disabled_stage_resolutions
        force_stage_gate_value = self.force_stage_gate_values.get(resolution)
        stage = self.decoder_stages[self.decoder_stage_resolutions.index(resolution)]
        return stage(
            decoder_map,
            skip_map,
            c,
            text=y,
            disable_stage=disable_stage,
            force_stage_gate_value=force_stage_gate_value,
            gate_floor=self.gate_floor,
        )

    def forward_backbone(self, hidden_states, t, y=None):
        hidden_states = self.x_embedder(hidden_states)
        c, y, timestep_condition = self._get_condition(hidden_states, t, y)

        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        initial_map = tokens_to_map(hidden_states, self.hierarchy_input_size)
        initial_map = self.highres_local_stem(initial_map)
        highres_map, highres_stats = self.highres_processor(initial_map, c=c, text=y)
        encoder_stage_stats = [dict(stage_idx=0, **highres_stats)]
        skip_maps = {self.patch_side_len: highres_map}

        current_map = highres_map
        for stage_idx, (layout, downsampler) in enumerate(
            zip(self.hierarchy_stage_layout, self.downsamplers), start=1
        ):
            current_map = downsampler(current_map)
            out_resolution = layout["output_resolution"][0]
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

        bottleneck_map = current_map
        decoder_map = self.bottleneck_proj(bottleneck_map)
        if not isinstance(decoder_map, torch.Tensor):
            decoder_map = bottleneck_map
        decoder_stage_stats = []
        aux_4x4_stats = None
        if self.anchor_builder is not None:
            anchor_skip_map = skip_maps.get(self.decoder_anchor_resolution)
            decoder_map, anchor_stats = self.anchor_builder(
                decoder_map,
                anchor_skip_map,
                c,
                text=y,
                disable_stage=self.decoder_anchor_resolution in self.disabled_stage_resolutions,
                force_stage_gate_value=self.force_stage_gate_values.get(
                    self.decoder_anchor_resolution
                ),
                gate_floor=self.gate_floor,
            )
            anchor_stats["is_anchor_stage"] = 1.0
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
                y,
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
        }
        self.latest_fusion_stats = {
            "enabled": True,
            "selected_stage_resolutions": list(self.fusion_selected_stages),
            "anchor_resolution": int(self.decoder_anchor_resolution),
            "bottleneck_resolution": int(self.fusion_anchor_resolution),
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
        }
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
        for key, value in current_state_dict.items():
            if key in remapped_state_dict:
                continue
            if (
                "absolute_pos_embed.pos_embed" in key
                or "child_pos_embed.pos_embed" in key
            ):
                remapped_state_dict[key] = value
        return super().load_state_dict(remapped_state_dict, strict=strict)
