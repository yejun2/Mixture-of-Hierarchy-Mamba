#!/usr/bin/env python3
import argparse
import contextlib
import json
import statistics
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_zigma import selective_scan_flop_jit
from transport import Sampler, create_transport
from utils.train_utils import get_model, requires_grad


def percentile(values, q):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * (q / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def autocast_context(device, amp):
    if device.type != "cuda" or amp == "none":
        return contextlib.nullcontext()
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_model_from_checkpoint(checkpoint_path, device, weights_key):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" not in checkpoint:
        raise KeyError(f"{checkpoint_path} does not contain training args")
    args = checkpoint["args"]
    args.is_latent = True
    args.use_latent = True

    model, in_channels, input_size = get_model(args, device)
    state_dict = checkpoint[weights_key]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    requires_grad(model, False)
    return model, args, in_channels, input_size


def make_dummy_inputs(batch_size, in_channels, input_size, device):
    x = torch.randn(batch_size, in_channels, input_size, input_size, device=device)
    t = torch.rand(batch_size, device=device)
    return x, t, None


class ForwardWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model_fn = getattr(model, "forward_transport", model.forward)

    def forward(self, x, t):
        return self.model_fn(x, t, y=None)


def parameter_summary(model):
    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "params": params,
        "buffers": buffers,
        "param_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "param_mb": param_bytes / (1024.0**2),
        "buffer_mb": buffer_bytes / (1024.0**2),
    }


def cuda_memory_snapshot(device):
    if device.type != "cuda":
        return {}
    return {
        "cuda_allocated_mb": torch.cuda.memory_allocated(device) / (1024.0**2),
        "cuda_reserved_mb": torch.cuda.memory_reserved(device) / (1024.0**2),
        "cuda_max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        "cuda_max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0**2),
    }


def set_mamba_fast_path(model, enabled):
    return set_module_attr(model, "use_fast_path", enabled)


def set_module_attr(model, attr_name, value):
    previous = []
    for module in model.modules():
        if hasattr(module, attr_name):
            previous.append((module, attr_name, getattr(module, attr_name)))
            setattr(module, attr_name, value)
    return previous


def restore_module_attr(previous):
    for module, attr_name, value in previous:
        setattr(module, attr_name, value)


def torch_rms_norm_forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
    if residual is not None:
        x = x + residual
    residual_out = x.to(torch.float32) if residual_in_fp32 else x
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
    y = y * self.weight
    if self.bias is not None:
        y = y + self.bias
    if prenorm:
        return y, residual_out
    return y


def patch_rms_norm_forward(model):
    previous = []
    for module in model.modules():
        if module.__class__.__name__ == "RMSNorm" and hasattr(module, "weight"):
            previous.append((module, module.forward))
            module.forward = types.MethodType(torch_rms_norm_forward, module)
    return previous


def restore_module_forward(previous):
    for module, forward in previous:
        module.forward = forward


def causal_conv1d_flop_jit(inputs, outputs):
    batch, dim, seqlen = inputs[0].type().sizes()
    _, width = inputs[1].type().sizes()
    flops = batch * dim * seqlen * width
    if len(inputs) > 2 and inputs[2].type().sizes() is not None:
        flops += batch * dim * seqlen
    return flops


@torch.inference_mode()
def measure_forward_flops(model, batch_size, in_channels, input_size, device):
    from fvcore.nn import flop_count

    x, t, _ = make_dummy_inputs(batch_size, in_channels, input_size, device)
    wrapper = ForwardWrapper(model).eval()
    supported_ops = {
        "aten::silu": None,
        "aten::neg": None,
        "aten::exp": None,
        "aten::flip": None,
        "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
        "prim::PythonOp.CausalConv1dFn": causal_conv1d_flop_jit,
    }
    previous_fast_path = set_mamba_fast_path(model, False)
    previous_fused_norm = set_module_attr(model, "fused_add_norm", False)
    previous_rms_norm = patch_rms_norm_forward(model)
    try:
        gflops_by_op, unsupported = flop_count(
            model=wrapper,
            inputs=(x, t),
            supported_ops=supported_ops,
        )
    finally:
        restore_module_forward(previous_rms_norm)
        restore_module_attr(previous_fused_norm)
        restore_module_attr(previous_fast_path)
    total_gflops = float(sum(gflops_by_op.values()))
    return {
        "forward_gflops": total_gflops,
        "forward_flops": total_gflops * 1e9,
        "forward_gflops_by_op": dict(gflops_by_op),
        "flops_unsupported_ops": dict(unsupported),
    }


def make_sample_fn(args, mode, num_steps):
    transport = create_transport(
        args.train.path_type,
        args.train.prediction,
        args.train.loss_weight,
        args.train.train_eps,
        args.train.sample_eps,
    )
    sampler = Sampler(transport)
    mode = mode.upper()
    if mode == "ODE":
        sampling_method = OmegaConf.select(args, "ode.sampling_method", default="dopri5")
        atol = OmegaConf.select(args, "ode.atol", default=1e-6)
        rtol = OmegaConf.select(args, "ode.rtol", default=1e-3)
        reverse = OmegaConf.select(args, "ode.reverse", default=False)
        return sampler.sample_ode(
            sampling_method=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            reverse=reverse,
        )
    if mode == "SDE":
        return sampler.sample_sde(
            sampling_method=OmegaConf.select(args, "sde.sampling_method", default="Euler"),
            diffusion_form=OmegaConf.select(args, "sde.diffusion_form", default="sigma"),
            diffusion_norm=OmegaConf.select(args, "sde.diffusion_norm", default=1.0),
            last_step=OmegaConf.select(args, "sde.last_step", default="Mean"),
            last_step_size=OmegaConf.select(args, "sde.last_step_size", default=0.04),
            num_steps=num_steps,
        )
    raise ValueError(f"Unknown sampling mode: {mode}")


@torch.inference_mode()
def benchmark_forward(model, batch_size, in_channels, input_size, device, warmup, iters, amp):
    x, t, y = make_dummy_inputs(batch_size, in_channels, input_size, device)
    model_fn = getattr(model, "forward_transport", model.forward)

    for _ in range(warmup):
        with autocast_context(device, amp):
            _ = model_fn(x, t, y=y)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    memory_start = cuda_memory_snapshot(device)

    latencies_ms = []
    for _ in range(iters):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast_context(device, amp):
                _ = model_fn(x, t, y=y)
            end.record()
            torch.cuda.synchronize()
            latencies_ms.append(start.elapsed_time(end))
        else:
            start_time = time.perf_counter()
            _ = model_fn(x, t, y=y)
            latencies_ms.append((time.perf_counter() - start_time) * 1000.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    memory_end = cuda_memory_snapshot(device)
    if memory_start and memory_end:
        memory_end["cuda_forward_extra_peak_allocated_mb"] = (
            memory_end["cuda_max_allocated_mb"] - memory_start["cuda_allocated_mb"]
        )
        memory_end["cuda_forward_extra_peak_reserved_mb"] = (
            memory_end["cuda_max_reserved_mb"] - memory_start["cuda_reserved_mb"]
        )
    return latencies_ms, memory_start, memory_end


@torch.inference_mode()
def benchmark_sampling(
    model,
    train_args,
    batch_size,
    in_channels,
    input_size,
    device,
    sample_mode,
    num_steps,
    warmup,
    iters,
    amp,
):
    z, _, y = make_dummy_inputs(batch_size, in_channels, input_size, device)
    model_fn = getattr(model, "forward_transport", model.forward)
    sample_fn = make_sample_fn(train_args, sample_mode, num_steps)

    for _ in range(warmup):
        with autocast_context(device, amp):
            _ = sample_fn(z, model_fn, y=y)[-1]
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    memory_start = cuda_memory_snapshot(device)

    latencies_ms = []
    for _ in range(iters):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast_context(device, amp):
                _ = sample_fn(z, model_fn, y=y)[-1]
            end.record()
            torch.cuda.synchronize()
            latencies_ms.append(start.elapsed_time(end))
        else:
            start_time = time.perf_counter()
            _ = sample_fn(z, model_fn, y=y)[-1]
            latencies_ms.append((time.perf_counter() - start_time) * 1000.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    memory_end = cuda_memory_snapshot(device)
    if memory_start and memory_end:
        memory_end["cuda_sampling_extra_peak_allocated_mb"] = (
            memory_end["cuda_max_allocated_mb"] - memory_start["cuda_allocated_mb"]
        )
        memory_end["cuda_sampling_extra_peak_reserved_mb"] = (
            memory_end["cuda_max_reserved_mb"] - memory_start["cuda_reserved_mb"]
        )
    return latencies_ms, memory_start, memory_end


def summarize(latencies_ms, batch_size):
    mean_ms = statistics.fmean(latencies_ms)
    return {
        "iters": len(latencies_ms),
        "mean_ms": mean_ms,
        "median_ms": statistics.median(latencies_ms),
        "p90_ms": percentile(latencies_ms, 90),
        "p95_ms": percentile(latencies_ms, 95),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "per_sample_mean_ms": mean_ms / float(batch_size),
        "throughput_samples_per_sec": 1000.0 * float(batch_size) / mean_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference latency.")
    parser.add_argument("--ckpt", required=True, help="Path to a .pt checkpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mode", choices=["forward", "sampling"], default="forward")
    parser.add_argument("--sample-mode", choices=["ODE", "SDE"], default="ODE")
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16")
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--measure-flops", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    model, train_args, in_channels, input_size = load_model_from_checkpoint(
        checkpoint_path,
        device,
        args.weights,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    model_memory = cuda_memory_snapshot(device)
    params = parameter_summary(model)

    if args.mode == "forward":
        latencies_ms, memory_start, memory_end = benchmark_forward(
            model,
            args.batch_size,
            in_channels,
            input_size,
            device,
            args.warmup,
            args.iters,
            args.amp,
        )
    else:
        latencies_ms, memory_start, memory_end = benchmark_sampling(
            model,
            train_args,
            args.batch_size,
            in_channels,
            input_size,
            device,
            args.sample_mode,
            args.num_steps,
            args.warmup,
            args.iters,
            args.amp,
        )
    flops_result = {}
    if args.measure_flops:
        if args.mode != "forward":
            raise ValueError("--measure-flops is currently supported for --mode forward only")
        flops_result = measure_forward_flops(
            model,
            args.batch_size,
            in_channels,
            input_size,
            device,
        )

    result = {
        "checkpoint": str(checkpoint_path),
        "model": OmegaConf.select(train_args, "model.name", default="unknown"),
        "mode": args.mode,
        "sample_mode": args.sample_mode if args.mode == "sampling" else None,
        "num_steps": args.num_steps if args.mode == "sampling" else None,
        "batch_size": args.batch_size,
        "input_shape": [args.batch_size, in_channels, input_size, input_size],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": args.amp,
        "weights": args.weights,
        **params,
        "model_memory_after_load": model_memory,
        "memory_before_timed_loop": memory_start,
        "memory_after_timed_loop": memory_end,
        **flops_result,
        **summarize(latencies_ms, args.batch_size),
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
