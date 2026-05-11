#!/usr/bin/env python3
"""
Analyze timestep-wise compress type router selections from a trained checkpoint.

The script loads a trained model, hooks compression/type router modules,
aggregates top-1 selections by timestep, and saves a stacked bar plot plus raw
data.

Example:
    python analyze/analyze_compress_type_router.py \
        --checkpoint /path/to/checkpoint.pt \
        --model-factory train:create_model \
        --data-factory train:create_val_loader \
        --output-dir analyze/outputs/compress

If the automatic router search hooks the wrong module, first inspect candidates:
    python analyze/analyze_compress_type_router.py --model-factory train:create_model --list-routers

Then run with:
    --router-name exact_or_partial_module_name
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COMPRESS_KEYWORDS = ("compress", "compression", "type_router", "compress_router", "compress_type")
ROUTER_KEYWORDS = ("router", "gate", "gating", "type")


def import_object(path: str) -> Any:
    if ":" not in path:
        raise ValueError(f"Expected import path like 'module:function', got {path!r}")
    module_name, object_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in object_name.split("."):
        obj = getattr(obj, part)
    return obj


def apply_factory_kwargs(args: argparse.Namespace) -> None:
    if not args.factory_kwargs:
        return
    values = json.loads(args.factory_kwargs)
    if not isinstance(values, dict):
        raise ValueError("--factory-kwargs must be a JSON object")
    for key, value in values.items():
        setattr(args, key, value)


def call_factory(factory_path: str, args: argparse.Namespace) -> Any:
    factory = import_object(factory_path)
    try:
        return factory(args)
    except TypeError:
        return factory()


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict", "ema", "module", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and any(torch.is_tensor(v) for v in value.values()):
                return value
        if any(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Could not find a tensor state_dict inside checkpoint")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)

    cleaned = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        new_key = key
        for prefix in ("module.", "model.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys: {len(unexpected)}")


def looks_like_compress_router(name: str, module: torch.nn.Module) -> bool:
    lower = name.lower()
    if not any(word in lower for word in COMPRESS_KEYWORDS):
        return False
    if not any(word in lower for word in ROUTER_KEYWORDS):
        return False
    return isinstance(module, (torch.nn.Linear, torch.nn.Sequential)) or any(
        word in lower for word in ("router", "gate", "gating")
    )


def tensor_from_output(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("weights", "weight", "prob", "probs", "logits", "router_logits", "router_probs", "type_logits"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
        for value in output.values():
            tensor = tensor_from_output(value)
            if tensor is not None:
                return tensor
    if isinstance(output, (tuple, list)):
        tensors = [item for item in output if torch.is_tensor(item)]
        if tensors:
            return tensors[-1]
    return None


def normalize_router_tensor(tensor: torch.Tensor) -> torch.Tensor | None:
    """Convert a router output to probabilities shaped [batch, num_types]."""
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim > 2:
        reduce_dims = tuple(range(1, tensor.ndim - 1))
        tensor = tensor.mean(dim=reduce_dims)

    if tensor.ndim != 2 or tensor.shape[-1] < 2:
        return None

    tensor = tensor.detach().float()
    row_sums = tensor.sum(dim=-1, keepdim=True)
    is_nonnegative = bool(torch.all(tensor >= -1e-6))
    sums_to_one = bool(torch.allclose(row_sums.mean(), torch.ones_like(row_sums.mean()), atol=1e-2))
    if not (is_nonnegative and sums_to_one):
        tensor = F.softmax(tensor, dim=-1)
    return tensor


def merge_captured(captured: list[tuple[str, torch.Tensor]]) -> torch.Tensor | None:
    if not captured:
        return None
    groups: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
    for _name, tensor in captured:
        groups[tuple(tensor.shape)].append(tensor)
    shape, tensors = max(groups.items(), key=lambda item: (len(item[1]), item[0][-1]))
    if len(groups) > 1:
        print(f"[warn] Multiple router output shapes captured; using shape {shape}")
    return torch.stack(tensors, dim=0).mean(dim=0)


def extract_timesteps(batch: Any, device: torch.device) -> torch.Tensor:
    if isinstance(batch, dict):
        for key in ("t", "timesteps", "timestep", "time", "diffusion_step"):
            if key in batch:
                return torch.as_tensor(batch[key], device=device).long().flatten()
    if isinstance(batch, (tuple, list)):
        for item in batch:
            if torch.is_tensor(item) and item.ndim <= 1 and item.dtype in (torch.int32, torch.int64, torch.long):
                return item.to(device).long().flatten()
    raise ValueError("Could not infer timestep tensor from batch. Use dict batches with key 't' or 'timesteps'.")


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def call_model(model: torch.nn.Module, batch: Any, device: torch.device) -> Any:
    batch = move_to_device(batch, device)
    if isinstance(batch, dict):
        try:
            return model(**batch)
        except TypeError:
            pass
        for keys in (("x", "t"), ("x", "timesteps"), ("input", "t"), ("images", "t")):
            if all(key in batch for key in keys):
                return model(*(batch[key] for key in keys))
    if isinstance(batch, (tuple, list)):
        return model(*batch)
    return model(batch)


def aggregate(
    records: list[tuple[int, int, np.ndarray]], num_bins: int | None
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    timesteps = np.asarray([record[0] for record in records], dtype=np.int64)
    choices = np.asarray([record[1] for record in records], dtype=np.int64)
    probs = np.stack([record[2] for record in records], axis=0)
    num_types = probs.shape[1]

    if num_bins is None or num_bins <= 0:
        x_values = np.unique(timesteps)
        labels = [str(int(value)) for value in x_values]
        choice_rates = []
        mean_probs = []
        for value in x_values:
            mask = timesteps == value
            choice_rates.append(np.bincount(choices[mask], minlength=num_types) / mask.sum())
            mean_probs.append(probs[mask].mean(axis=0))
        return x_values, labels, np.stack(choice_rates, axis=0), np.stack(mean_probs, axis=0)

    lo, hi = int(timesteps.min()), int(timesteps.max())
    edges = np.linspace(lo, hi + 1, num_bins + 1).astype(int)
    x_values = []
    labels = []
    choice_rates = []
    mean_probs = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (timesteps >= start) & (timesteps < end)
        x_values.append((start + end - 1) / 2)
        labels.append(f"{start}-{end - 1}")
        if mask.any():
            choice_rates.append(np.bincount(choices[mask], minlength=num_types) / mask.sum())
            mean_probs.append(probs[mask].mean(axis=0))
        else:
            choice_rates.append(np.full(num_types, np.nan))
            mean_probs.append(np.full(num_types, np.nan))
    return np.asarray(x_values), labels, np.stack(choice_rates, axis=0), np.stack(mean_probs, axis=0)


def plot_choice_rates(
    x_values: np.ndarray, labels: list[str], choice_rates: np.ndarray, output_path: Path, type_names: list[str] | None
) -> None:
    num_types = choice_rates.shape[1]
    names = type_names or [f"Type {idx}" for idx in range(num_types)]
    plt.figure(figsize=(10, 5))
    bottom = np.zeros(choice_rates.shape[0])
    width = 0.8 if len(x_values) <= 30 else max(1.0, (x_values.max() - x_values.min()) / len(x_values) * 0.8)
    for type_idx in range(num_types):
        values = np.nan_to_num(choice_rates[:, type_idx])
        label = names[type_idx] if type_idx < len(names) else f"Type {type_idx}"
        plt.bar(x_values, values, bottom=bottom, width=width, label=label)
        bottom += values
    plt.xlabel("Timestep")
    plt.ylabel("Top-1 selection ratio")
    plt.title("Timestep-wise Compress Type Router Selection")
    plt.ylim(0, 1.02)
    plt.legend(frameon=False, ncol=min(num_types, 4))
    if len(labels) <= 20:
        plt.xticks(x_values, labels, rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model-factory", required=True, help="Import path, e.g. train:create_model")
    parser.add_argument("--data-factory", default="", help="Import path returning a validation/test dataloader")
    parser.add_argument("--factory-kwargs", default="", help='JSON object added to the args namespace, e.g. \'{"image_size": 256}\'')
    parser.add_argument("--output-dir", default="analyze/outputs/compress")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument("--router-name", default="", help="Exact/partial module name to hook")
    parser.add_argument("--type-names", default="", help="Comma-separated compress type labels")
    parser.add_argument("--list-routers", action="store_true", help="Print candidate compress router modules and exit")
    return parser


def main() -> None:
    parser = build_parser()
    args, _ = parser.parse_known_args()
    apply_factory_kwargs(args)

    device = torch.device(args.device)
    model = call_factory(args.model_factory, args)
    model.to(device)

    candidates = [(name, module.__class__.__name__) for name, module in model.named_modules() if looks_like_compress_router(name, module)]
    if args.list_routers:
        for name, class_name in candidates:
            print(f"{name}\t{class_name}")
        return

    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --list-routers is set")
    if not args.data_factory:
        raise ValueError("--data-factory is required unless --list-routers is set")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    loader = call_factory(args.data_factory, args)

    captured: list[tuple[str, torch.Tensor]] = []
    handles = []

    def make_hook(module_name: str) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], None]:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = tensor_from_output(output)
            if tensor is None:
                return
            normalized = normalize_router_tensor(tensor)
            if normalized is not None:
                captured.append((module_name, normalized.detach().cpu()))

        return hook

    for name, module in model.named_modules():
        use_module = args.router_name and args.router_name.lower() in name.lower()
        use_module = use_module or (not args.router_name and looks_like_compress_router(name, module))
        if use_module:
            handles.append(module.register_forward_hook(make_hook(name)))
            print(f"[hook] {name}")

    if not handles:
        raise RuntimeError("No compress type router module was hooked. Try --list-routers or --router-name.")

    records: list[tuple[int, int, np.ndarray]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            captured.clear()
            timesteps = extract_timesteps(batch, device).detach().cpu().numpy()
            _ = call_model(model, batch, device)
            probs = merge_captured(captured)
            if probs is None:
                continue
            probs_np = probs.numpy()
            choices = probs_np.argmax(axis=-1)
            for timestep, choice, prob in zip(timesteps, choices, probs_np):
                records.append((int(timestep), int(choice), prob))

    for handle in handles:
        handle.remove()

    if not records:
        raise RuntimeError("No router records were collected. Check --router-name and dataloader batch format.")

    type_names = [item.strip() for item in args.type_names.split(",") if item.strip()] or None
    x_values, labels, choice_rates, mean_probs = aggregate(records, args.num_bins)
    np.save(output_dir / "compress_choice_rates.npy", choice_rates)
    np.save(output_dir / "compress_mean_probs.npy", mean_probs)
    with (output_dir / "compress_type_router.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "x_values": x_values.tolist(),
                "x_labels": labels,
                "choice_rates": choice_rates.tolist(),
                "mean_probs": mean_probs.tolist(),
                "type_names": type_names,
                "num_records": len(records),
            },
            f,
            indent=2,
        )
    plot_choice_rates(x_values, labels, choice_rates, output_dir / "compress_type_selection.png", type_names)
    print(f"[done] saved to {output_dir}")


if __name__ == "__main__":
    main()
