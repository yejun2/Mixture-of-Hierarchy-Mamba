#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2014.zip",
    "val_images": "http://images.cocodataset.org/zips/val2014.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}


@dataclass
class TextLatentEncoders:
    vae: Any
    tokenizer: Any
    text_encoder: Any


def import_runtime_dependencies() -> None:
    global AutoencoderKL, BICUBIC, CLIPTextModel, CLIPTokenizer
    global Image, TF, np, torch, tqdm, wds

    import numpy as np  # noqa: F401
    import torch  # noqa: F401
    import torchvision.transforms.functional as TF  # noqa: F401
    import webdataset as wds  # noqa: F401
    from diffusers.models import AutoencoderKL  # noqa: F401
    from PIL import Image  # noqa: F401
    from tqdm import tqdm  # noqa: F401
    from transformers import CLIPTextModel, CLIPTokenizer  # noqa: F401

    try:
        BICUBIC = Image.Resampling.BICUBIC
    except AttributeError:  # Pillow<9.1
        BICUBIC = Image.BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download MS COCO 2014 and build WebDataset shards expected by "
            "config/data/coco.yaml: image.jpg, img_feature256.npy, "
            "caption_feature.npy, caption.json."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory for downloaded/extracted COCO 2014 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write WebDataset shards.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="COCO splits to prepare.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing files under --raw-dir and skip downloading/extraction.",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=1000,
        help="Maximum samples per tar shard.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Images per batch before VAE/text encoding.",
    )
    parser.add_argument(
        "--vae-micro-batch-size",
        type=int,
        default=4,
        help="Micro batch size for VAE encoding.",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=64,
        help="Caption strings per CLIP text encoder batch.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square image size used for stored image.jpg and VAE latents.",
    )
    parser.add_argument(
        "--captions-per-image",
        type=int,
        default=5,
        help="Number of captions stored per image. Extra captions are trimmed; missing captions are padded.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Diffusers Stable Diffusion model used for VAE and CLIP text features.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for feature extraction.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Model inference dtype.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional cap for quick smoke-test shard creation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing split shards in --output-dir before writing.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the Stable Diffusion model only from the local Hugging Face cache.",
    )
    return parser.parse_args()


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            bad_member = zf.testzip()
    except zipfile.BadZipFile:
        return False
    return bad_member is None


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_valid_zip(path):
        print(f"[download] exists: {path}")
        return
    if path.exists():
        print(f"[download] removing invalid/incomplete zip: {path}")
        path.unlink()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"[download] {url} -> {path}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        total = response.headers.get("Content-Length")
        total_int = int(total) if total is not None else None
        with tqdm(total=total_int, unit="B", unit_scale=True, desc=path.name) as pbar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                pbar.update(len(chunk))
    tmp_path.replace(path)
    if not is_valid_zip(path):
        raise zipfile.BadZipFile(
            f"Downloaded file is not a valid zip: {path}. "
            "This usually means the server returned an error page or the download was interrupted."
        )


def extract_zip(zip_path: Path, raw_dir: Path, expected_member_dir: str) -> None:
    expected_path = raw_dir / expected_member_dir
    if expected_path.exists():
        print(f"[extract] exists: {expected_path}")
        return
    if not is_valid_zip(zip_path):
        raise zipfile.BadZipFile(
            f"Invalid zip file: {zip_path}. Remove it and rerun, or rerun without --skip-download."
        )
    print(f"[extract] {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)


def download_and_extract(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = raw_dir / "downloads"
    zip_specs = [
        ("train_images", "train2014.zip", "train2014"),
        ("val_images", "val2014.zip", "val2014"),
        ("annotations", "annotations_trainval2014.zip", "annotations"),
    ]
    for key, filename, expected_dir in zip_specs:
        zip_path = downloads_dir / filename
        download_file(COCO_URLS[key], zip_path)
        extract_zip(zip_path, raw_dir, expected_dir)


def encode_npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def encode_jpg_bytes(image: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def center_crop_resize(image: Image.Image, image_size: int) -> Image.Image:
    width, height = image.size
    scale = image_size / min(width, height)
    resized = image.resize(
        (round(width * scale), round(height * scale)),
        resample=BICUBIC,
    )
    left = (resized.width - image_size) // 2
    top = (resized.height - image_size) // 2
    return resized.crop((left, top, left + image_size, top + image_size))


def image_to_vae_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return tensor * 2.0 - 1.0


def read_coco_split(raw_dir: Path, split: str) -> Tuple[List[dict], Dict[int, List[str]]]:
    ann_path = raw_dir / "annotations" / f"captions_{split}2014.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing COCO caption annotation: {ann_path}")
    with ann_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    captions_by_image_id: Dict[int, List[str]] = defaultdict(list)
    for ann in data["annotations"]:
        captions_by_image_id[int(ann["image_id"])].append(str(ann["caption"]))

    images = sorted(data["images"], key=lambda item: int(item["id"]))
    images = [img for img in images if captions_by_image_id.get(int(img["id"]))]
    return images, captions_by_image_id


def normalize_captions(captions: Sequence[str], captions_per_image: int) -> List[str]:
    if captions_per_image <= 0:
        raise ValueError("--captions-per-image must be >= 1")
    cleaned = [caption.strip() for caption in captions if caption.strip()]
    if not cleaned:
        cleaned = [""]
    if len(cleaned) >= captions_per_image:
        return cleaned[:captions_per_image]
    return cleaned + [cleaned[-1]] * (captions_per_image - len(cleaned))


def encode_images(
    vae,
    pixel_batch: torch.Tensor,
    micro_bs: int,
    device: str,
    dtype: torch.dtype,
) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in range(0, pixel_batch.shape[0], micro_bs):
            chunk = pixel_batch[start : start + micro_bs].to(
                device=device, dtype=dtype, non_blocking=True
            )
            latent = vae.encode(chunk).latent_dist.sample()
            chunks.append(latent.detach().cpu())
    return torch.cat(chunks, dim=0).numpy()


def encode_captions(
    tokenizer,
    text_encoder,
    captions_batch: Sequence[Sequence[str]],
    text_batch_size: int,
    device: str,
    dtype: torch.dtype,
) -> np.ndarray:
    flat = [caption for captions in captions_batch for caption in captions]
    features = []
    with torch.inference_mode():
        for start in range(0, len(flat), text_batch_size):
            texts = flat[start : start + text_batch_size]
            tokens = tokenizer(
                texts,
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(device)
            text_encoder_kwargs = {"input_ids": input_ids}
            if getattr(text_encoder.config, "use_attention_mask", False):
                text_encoder_kwargs["attention_mask"] = tokens.attention_mask.to(device)
            hidden = text_encoder(**text_encoder_kwargs).last_hidden_state
            features.append(hidden.detach().to(torch.float32).cpu())
    encoded = torch.cat(features, dim=0).numpy()
    n_images = len(captions_batch)
    n_caps = len(captions_batch[0])
    return encoded.reshape(n_images, n_caps, encoded.shape[1], encoded.shape[2])


def shard_pattern(output_dir: Path, split: str) -> str:
    return str(output_dir / f"{split}-%06d.tar")


def remove_existing_split_shards(output_dir: Path, split: str) -> None:
    for path in output_dir.glob(f"{split}-*.tar"):
        path.unlink()


def build_split(
    args: argparse.Namespace,
    split: str,
    encoders: TextLatentEncoders,
    dtype: torch.dtype,
) -> int:
    images, captions_by_image_id = read_coco_split(args.raw_dir, split)
    if args.max_samples_per_split is not None:
        images = images[: args.max_samples_per_split]
    if not images:
        raise RuntimeError(f"No COCO {split} images found.")

    image_dir = args.raw_dir / f"{split}2014"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    if args.overwrite:
        remove_existing_split_shards(args.output_dir, split)

    writer = wds.ShardWriter(
        shard_pattern(args.output_dir, split),
        maxcount=args.samples_per_shard,
    )

    keys: List[str] = []
    jpg_bytes_batch: List[bytes] = []
    pixels: List[torch.Tensor] = []
    captions_batch: List[List[str]] = []
    total_written = 0
    expected_latent_hw = args.image_size // 8

    def flush() -> None:
        nonlocal total_written
        if not keys:
            return
        pixel_batch = torch.stack(pixels, dim=0)
        img_features = encode_images(
            encoders.vae,
            pixel_batch,
            args.vae_micro_batch_size,
            args.device,
            dtype,
        )
        caption_features = encode_captions(
            encoders.tokenizer,
            encoders.text_encoder,
            captions_batch,
            args.text_batch_size,
            args.device,
            dtype,
        )
        if img_features.shape[1:] != (4, expected_latent_hw, expected_latent_hw):
            raise RuntimeError(
                f"Unexpected VAE latent shape {img_features.shape[1:]}; "
                f"expected {(4, expected_latent_hw, expected_latent_hw)}"
            )
        if caption_features.shape[-1] != 768:
            print(
                "[warn] caption_feature last dim is "
                f"{caption_features.shape[-1]}, so set model.params.d_context to that value."
            )

        for idx, key in enumerate(keys):
            writer.write(
                {
                    "__key__": key,
                    "image.jpg": jpg_bytes_batch[idx],
                    # Raw SD-VAE latent. train_acc.py multiplies this by 0.18215.
                    "img_feature256.npy": encode_npy_bytes(img_features[idx].astype(np.float32)),
                    "caption_feature.npy": encode_npy_bytes(caption_features[idx].astype(np.float32)),
                    "caption.json": json.dumps(
                        captions_batch[idx],
                        ensure_ascii=False,
                    ).encode("utf-8"),
                }
            )
            total_written += 1

        keys.clear()
        jpg_bytes_batch.clear()
        pixels.clear()
        captions_batch.clear()

    try:
        for image_info in tqdm(images, desc=f"Preparing COCO {split}2014", unit="img"):
            image_id = int(image_info["id"])
            filename = str(image_info["file_name"])
            image_path = image_dir / filename
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image referenced by COCO annotations: {image_path}")

            image = center_crop_resize(load_rgb(image_path), args.image_size)
            captions = normalize_captions(
                captions_by_image_id[image_id],
                args.captions_per_image,
            )

            keys.append(f"{split}_{image_id:012d}")
            jpg_bytes_batch.append(encode_jpg_bytes(image))
            pixels.append(image_to_vae_tensor(image))
            captions_batch.append(captions)

            if len(keys) >= args.batch_size:
                flush()
        flush()
    finally:
        writer.close()

    shard_count = math.ceil(total_written / args.samples_per_shard)
    print(f"[done] {split}: wrote {total_written} samples into {shard_count} shard(s)")
    if shard_count > 0:
        print(
            f"[config] data.{ 'train' if split == 'train' else 'validation' }.shards="
            f"'{split}-{{000000..{shard_count - 1:06d}}}.tar'"
        )
    return total_written


def main() -> None:
    args = parse_args()
    import_runtime_dependencies()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be >= 1")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.vae_micro_batch_size <= 0:
        raise ValueError("--vae-micro-batch-size must be >= 1")
    if args.text_batch_size <= 0:
        raise ValueError("--text-batch-size must be >= 1")
    if args.image_size % 8 != 0:
        raise ValueError("--image-size must be divisible by 8")
    if args.output_dir.exists() and args.overwrite:
        for split in args.splits:
            remove_existing_split_shards(args.output_dir, split)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_and_extract(args.raw_dir)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"[model] loading VAE and CLIP text encoder from {args.model_id}")
    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(args.device)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id,
        subfolder="tokenizer",
        local_files_only=args.local_files_only,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(args.device)
    vae.eval()
    text_encoder.eval()
    encoders = TextLatentEncoders(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
    )
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    totals = {}
    for split in args.splits:
        totals[split] = build_split(args, split, encoders, dtype)

    print("[summary]")
    print(f"  output_dir: {args.output_dir}")
    print(f"  samples: {totals}")
    print("  shard keys: image.jpg, img_feature256.npy, caption_feature.npy, caption.json")
    print("  training overrides: data=coco is_latent=true use_latent=true")
    print("  model overrides: model.params.has_text=true model.params.d_context=768")


if __name__ == "__main__":
    main()
