#!/usr/bin/env python3
import argparse
import io
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import webdataset as wds
from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm import tqdm


try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # Pillow<9.1
    BICUBIC = Image.BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build FacesHQ1024-style WebDataset shards with raw SD-VAE latents from "
            "a directory of numbered image folders such as 00000/, 01000/, ..."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory that contains numbered folders with images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write tar shards into.",
    )
    parser.add_argument(
        "--shard-prefix",
        type=str,
        default="train",
        help="Output shard prefix, e.g. train or test.",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=1000,
        help="Max samples per tar shard.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images staged before VAE encoding.",
    )
    parser.add_argument(
        "--vae-micro-batch-size",
        type=int,
        default=1,
        help="Micro batch size for VAE encode on GPU to avoid OOM.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Resize resolution before VAE encoding.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional cap for total samples written by this worker.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".png,.jpg,.jpeg,.webp,.bmp",
        help="Comma-separated list of image filename extensions to include.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search within each numbered folder for images.",
    )
    parser.add_argument(
        "--vae-model",
        type=str,
        default="stabilityai/sd-vae-ft-ema",
        help="VAE model id for diffusers AutoencoderKL.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="VAE model dtype.",
    )
    parser.add_argument(
        "--num-workers-total",
        type=int,
        default=1,
        help="Total parallel writer workers over the ordered image list.",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Current worker id in [0, num-workers-total).",
    )
    parser.add_argument(
        "--empty-cache",
        action="store_true",
        help="Call torch.cuda.empty_cache() every micro batch (safer, slower).",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> Tuple[object, ...]:
    text = path.name
    parts = re.findall(r"\d+|\D+", text)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return tuple(key)


def parse_extensions(ext_string: str) -> Sequence[str]:
    extensions = []
    for item in ext_string.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        extensions.append(item)
    if not extensions:
        raise ValueError("At least one image extension is required.")
    return tuple(sorted(set(extensions)))


def list_numbered_image_paths(
    input_root: Path, extensions: Sequence[str], recursive: bool
) -> List[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    child_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()], key=natural_sort_key)
    if not child_dirs:
        raise RuntimeError(f"No subdirectories found under {input_root}")

    image_paths: List[Path] = []
    for child_dir in child_dirs:
        iterator: Iterable[Path]
        iterator = child_dir.rglob("*") if recursive else child_dir.iterdir()
        files = [
            p
            for p in iterator
            if p.is_file() and p.suffix.lower() in extensions
        ]
        files.sort(key=natural_sort_key)
        image_paths.extend(files)

    if not image_paths:
        raise RuntimeError(
            f"No images with extensions {extensions} found under {input_root}"
        )
    return image_paths


def encode_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def encode_npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def build_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
        ]
    )


def open_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("RGB")


def validate_latent_shape(latents: np.ndarray, image_size: int) -> None:
    expected_hw = image_size // 8
    expected_shape = (4, expected_hw, expected_hw)
    actual_shape = tuple(latents.shape[1:])
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"Unexpected latent shape {actual_shape}; expected {expected_shape} for "
            f"image_size={image_size}."
        )


def encode_with_micro_batch(
    vae: AutoencoderKL,
    batch: torch.Tensor,
    micro_bs: int,
    device: str,
    dtype: torch.dtype,
    empty_cache: bool,
) -> np.ndarray:
    if micro_bs <= 0:
        raise ValueError("--vae-micro-batch-size must be >= 1")

    out = []
    with torch.inference_mode():
        for i in range(0, batch.shape[0], micro_bs):
            chunk = batch[i : i + micro_bs].to(device, dtype=dtype, non_blocking=True)
            lat = vae.encode(chunk).latent_dist.sample()
            out.append(lat.detach().cpu())
            if empty_cache and device.startswith("cuda"):
                torch.cuda.empty_cache()
    return torch.cat(out, dim=0).numpy()


def selected_indices(total_count: int, num_workers_total: int, worker_id: int) -> List[int]:
    return list(range(worker_id, total_count, num_workers_total))


def shard_pattern(output_dir: Path, shard_prefix: str, num_workers_total: int, worker_id: int) -> str:
    if num_workers_total == 1:
        return str(output_dir / f"{shard_prefix}-%06d.tar")
    return str(output_dir / f"{shard_prefix}-w{worker_id}-%06d.tar")


def main() -> None:
    args = parse_args()
    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be >= 1")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers_total <= 0:
        raise ValueError("--num-workers-total must be >= 1")
    if args.worker_id < 0 or args.worker_id >= args.num_workers_total:
        raise ValueError("--worker-id must be in [0, num-workers-total).")

    extensions = parse_extensions(args.extensions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    image_paths = list_numbered_image_paths(
        input_root=args.input_root,
        extensions=extensions,
        recursive=args.recursive,
    )
    worker_indices = selected_indices(
        total_count=len(image_paths),
        num_workers_total=args.num_workers_total,
        worker_id=args.worker_id,
    )
    if args.num_samples is not None:
        worker_indices = worker_indices[: args.num_samples]

    if not worker_indices:
        raise RuntimeError(
            f"No images assigned to worker {args.worker_id} under {args.input_root}"
        )

    vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=dtype).to(args.device)
    vae.eval()

    transform = build_transform(args.image_size)
    writer = wds.ShardWriter(
        shard_pattern(args.output_dir, args.shard_prefix, args.num_workers_total, args.worker_id),
        maxcount=args.samples_per_shard,
    )

    keys: List[str] = []
    png_bytes_batch: List[bytes] = []
    pixel_batch: List[torch.Tensor] = []
    validated_shape = False
    total_written = 0

    pbar = tqdm(total=len(worker_indices), desc="Encoding FacesHQ1024", unit="img")

    try:
        for image_idx in worker_indices:
            image_path = image_paths[image_idx]
            pil = open_rgb_image(image_path)
            resized = pil.resize((args.image_size, args.image_size), resample=BICUBIC)

            png_bytes_batch.append(encode_png_bytes(resized))
            pixel_batch.append(transform(resized))
            keys.append(f"{image_idx:09d}")

            if len(keys) < args.batch_size:
                continue

            batch = torch.stack(pixel_batch, dim=0)
            batch = batch * 2.0 - 1.0
            latents = encode_with_micro_batch(
                vae=vae,
                batch=batch,
                micro_bs=args.vae_micro_batch_size,
                device=args.device,
                dtype=dtype,
                empty_cache=args.empty_cache,
            )

            if not validated_shape:
                validate_latent_shape(latents, args.image_size)
                validated_shape = True

            for i in range(len(keys)):
                writer.write(
                    {
                        "__key__": keys[i],
                        "image.png": png_bytes_batch[i],
                        # Keep raw VAE latents; training code multiplies by 0.18215.
                        "latent.npy": encode_npy_bytes(latents[i].astype(np.float32)),
                    }
                )
                total_written += 1
                pbar.update(1)

            keys.clear()
            png_bytes_batch.clear()
            pixel_batch.clear()

        if keys:
            batch = torch.stack(pixel_batch, dim=0)
            batch = batch * 2.0 - 1.0
            latents = encode_with_micro_batch(
                vae=vae,
                batch=batch,
                micro_bs=args.vae_micro_batch_size,
                device=args.device,
                dtype=dtype,
                empty_cache=args.empty_cache,
            )

            if not validated_shape:
                validate_latent_shape(latents, args.image_size)

            for i in range(len(keys)):
                writer.write(
                    {
                        "__key__": keys[i],
                        "image.png": png_bytes_batch[i],
                        "latent.npy": encode_npy_bytes(latents[i].astype(np.float32)),
                    }
                )
                total_written += 1
                pbar.update(1)
    finally:
        writer.close()
        pbar.close()

    print(f"Done. Wrote {total_written} samples to {args.output_dir}")
    print(
        "Shard pattern: "
        f"{shard_pattern(args.output_dir, args.shard_prefix, args.num_workers_total, args.worker_id)} "
        f"(maxcount={args.samples_per_shard})"
    )
    print(
        "Stored latent shape is expected to be "
        f"(4, {args.image_size // 8}, {args.image_size // 8}) for image_size={args.image_size}."
    )


if __name__ == "__main__":
    main()
