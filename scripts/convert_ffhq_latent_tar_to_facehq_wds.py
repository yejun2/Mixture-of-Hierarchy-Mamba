#!/usr/bin/env python3
"""Convert FFHQ latent-only tar shards into FaceHQ WebDataset shards.

Expected output sample members:
  000000000.image.png
  000000000.latent.npy
  000000000.filename.txt

The converter keeps the existing latent values, optionally fixes latent layout
from HWC to CHW, and adds PNG images resolved from filename.txt.
"""

from __future__ import annotations

import argparse
import io
import re
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # Pillow<9.1
    BICUBIC = Image.BICUBIC


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert latent-only FFHQ tar shards into WebDataset shards with "
            "<key>.image.png, <key>.latent.npy, and <key>.filename.txt."
        )
    )
    parser.add_argument(
        "--input-tar-dir",
        type=Path,
        required=True,
        help="Directory containing source tar shards.",
    )
    parser.add_argument(
        "--input-shards",
        type=str,
        default="train-*.tar",
        help="Glob pattern inside --input-tar-dir.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Root containing source images. Can be passed more than once. If a "
            "source tar already contains image.png entries, this is optional."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where converted tar shards will be written.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="train",
        help="Output shard prefix, e.g. train or test.",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=1000,
        help="Number of samples per output tar shard.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional square resize before storing image.png. Omit to keep source size.",
    )
    parser.add_argument(
        "--latent-layout",
        choices=("auto", "chw", "hwc", "preserve"),
        default="auto",
        help=(
            "How to store latent.npy. auto converts common HWC VAE latents to CHW; "
            "preserve writes the source array unchanged."
        ),
    )
    parser.add_argument(
        "--expected-latent-size",
        type=int,
        default=None,
        help="Optional validation for CHW latents, e.g. 32 or 128.",
    )
    parser.add_argument(
        "--latent-dtype",
        choices=("preserve", "float32", "float16"),
        default="preserve",
        help="Optional dtype conversion for stored latent.npy.",
    )
    parser.add_argument(
        "--preserve-keys",
        action="store_true",
        help="Keep source sample keys instead of renumbering output keys from --start-index.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First output key when not using --preserve-keys.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to convert.",
    )
    parser.add_argument(
        "--skip-samples",
        type=int,
        default=0,
        help="Skip this many ordered input samples before converting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume into an existing output dir. Complete output shards are kept, "
            "the last partial shard is removed, and already-written samples are skipped."
        ),
    )
    parser.add_argument(
        "--recursive-image-lookup",
        action="store_true",
        help="Build a basename index under image roots if direct candidates miss.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip samples whose image cannot be resolved instead of failing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output shards with the same prefix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input samples and image resolution without writing output tars.",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> Tuple[object, ...]:
    parts = re.findall(r"\d+|\D+", path.name)
    key: List[object] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


def list_input_tars(input_tar_dir: Path, pattern: str) -> List[Path]:
    if not input_tar_dir.is_dir():
        raise NotADirectoryError(f"Input tar dir does not exist: {input_tar_dir}")
    paths = sorted(input_tar_dir.glob(pattern), key=natural_sort_key)
    if not paths:
        raise FileNotFoundError(f"No input tar shards matched {input_tar_dir / pattern}")
    return paths


def output_shard_path(output_dir: Path, prefix: str, shard_index: int) -> Path:
    return output_dir / f"{prefix}-{shard_index:06d}.tar"


def count_complete_samples(tar_path: Path) -> int:
    with tarfile.open(tar_path, "r:*") as tar:
        groups = group_members(tar)
    count = 0
    for _, members in groups:
        if {"image.png", "latent.npy", "filename.txt"}.issubset(members):
            count += 1
    return count


def parse_output_shard_index(path: Path, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}-(\d{{6}})\.tar", path.name)
    if match is None:
        raise ValueError(f"Unexpected output shard name: {path.name}")
    return int(match.group(1))


def inspect_resume_outputs(
    output_dir: Path,
    prefix: str,
    samples_per_shard: int,
    dry_run: bool,
) -> Tuple[int, int]:
    existing = sorted(output_dir.glob(f"{prefix}-*.tar"), key=natural_sort_key)
    completed_samples = 0
    next_shard_index = 0

    for expected_index, path in enumerate(existing):
        shard_index = parse_output_shard_index(path, prefix)
        if shard_index != expected_index:
            raise ValueError(
                "Resume requires contiguous output shards starting at 0; "
                f"expected {prefix}-{expected_index:06d}.tar but found {path.name}."
            )

        try:
            sample_count = count_complete_samples(path)
        except tarfile.TarError as exc:
            if path != existing[-1]:
                raise RuntimeError(f"Non-last shard is unreadable: {path}") from exc
            if dry_run:
                print(f"Resume would remove unreadable partial shard: {path.name}")
            else:
                path.unlink()
                print(f"Removed unreadable partial shard: {path.name}")
            break

        if sample_count == samples_per_shard:
            completed_samples += sample_count
            next_shard_index = shard_index + 1
            continue

        if path != existing[-1]:
            raise RuntimeError(
                f"Non-last shard is partial: {path.name} has {sample_count} samples."
            )
        if dry_run:
            print(
                f"Resume would remove partial shard: {path.name} "
                f"({sample_count}/{samples_per_shard} samples)"
            )
        else:
            path.unlink()
            print(
                f"Removed partial shard: {path.name} "
                f"({sample_count}/{samples_per_shard} samples)"
            )
        break

    return completed_samples, next_shard_index


def prepare_output_dir(
    output_dir: Path,
    prefix: str,
    overwrite: bool,
    resume: bool,
    samples_per_shard: int,
    dry_run: bool,
) -> Tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together.")

    if resume:
        return inspect_resume_outputs(output_dir, prefix, samples_per_shard, dry_run)

    existing = sorted(output_dir.glob(f"{prefix}-*.tar"), key=natural_sort_key)
    if not existing:
        return 0, 0
    if dry_run:
        return 0, 0
    if not overwrite:
        examples = ", ".join(path.name for path in existing[:3])
        raise FileExistsError(
            f"Output shards already exist in {output_dir}: {examples}. "
            "Use --overwrite or choose another --output-dir/--output-prefix."
        )
    for path in existing:
        path.unlink()
    return 0, 0


def split_member_name(name: str) -> Optional[Tuple[str, str]]:
    base = Path(name).name
    if "." not in base:
        return None
    key, field = base.split(".", 1)
    if not key:
        return None
    return key, field


def group_members(tar: tarfile.TarFile) -> List[Tuple[str, Dict[str, tarfile.TarInfo]]]:
    groups: Dict[str, Dict[str, tarfile.TarInfo]] = {}
    order: List[str] = []
    for member in tar.getmembers():
        if not member.isfile():
            continue
        parsed = split_member_name(member.name)
        if parsed is None:
            continue
        key, field = parsed
        if field not in {"image.png", "latent.npy", "filename.txt"}:
            continue
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key][field] = member
    return [(key, groups[key]) for key in order]


def read_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    fileobj = tar.extractfile(member)
    if fileobj is None:
        raise RuntimeError(f"Could not read tar member: {member.name}")
    with fileobj:
        return fileobj.read()


def encode_npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def normalize_latent(
    latent_bytes: bytes,
    layout: str,
    expected_size: Optional[int],
    dtype: str,
) -> Tuple[bytes, Tuple[int, ...], Tuple[int, ...]]:
    arr = np.load(io.BytesIO(latent_bytes))
    original_shape = tuple(arr.shape)

    if layout == "auto":
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            pass
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(
                f"Cannot infer latent layout for shape {original_shape}; "
                "use --latent-layout preserve/chw/hwc explicitly."
            )
    elif layout == "hwc":
        if arr.ndim != 3:
            raise ValueError(f"--latent-layout hwc requires a 3D array, got {original_shape}")
        arr = np.transpose(arr, (2, 0, 1))
    elif layout == "chw":
        if arr.ndim != 3:
            raise ValueError(f"--latent-layout chw requires a 3D array, got {original_shape}")
    elif layout == "preserve":
        pass
    else:
        raise ValueError(f"Unknown latent layout: {layout}")

    stored_shape = tuple(arr.shape)
    if expected_size is not None:
        expected = (4, expected_size, expected_size)
        if stored_shape != expected:
            raise ValueError(
                f"Stored latent shape {stored_shape} does not match expected {expected}."
            )

    if dtype == "float32":
        arr = arr.astype(np.float32, copy=False)
    elif dtype == "float16":
        arr = arr.astype(np.float16, copy=False)

    return encode_npy(arr), original_shape, tuple(arr.shape)


def expanded_roots(image_roots: Sequence[Path]) -> List[Path]:
    roots: List[Path] = []
    seen = set()
    for root in image_roots:
        for candidate in (root, root / "images1024x1024"):
            resolved = candidate.expanduser()
            key = str(resolved)
            if key not in seen and resolved.is_dir():
                roots.append(resolved)
                seen.add(key)
    return roots


def numeric_stem(path: Path) -> Optional[int]:
    if path.stem.isdigit():
        return int(path.stem)
    return None


def candidate_image_paths(filename: str, roots: Sequence[Path]) -> Iterator[Path]:
    clean = filename.strip().replace("\\", "/")
    if not clean:
        return

    raw_path = Path(clean)
    suffix = raw_path.suffix.lower() or ".png"
    basename = raw_path.name

    if raw_path.is_absolute():
        yield raw_path

    for root in roots:
        yield root / clean
        yield root / basename

        number = numeric_stem(raw_path)
        if number is None:
            continue

        padded_names = [
            f"{number:05d}{suffix}",
            f"{number:05d}.png",
            f"{number:09d}{suffix}",
            f"{number:09d}.png",
        ]
        group = f"{(number // 1000) * 1000:05d}"
        for padded_name in padded_names:
            yield root / padded_name
            yield root / group / padded_name


def build_image_index(roots: Sequence[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.name, path)
    return index


def resolve_image_path(
    filename: str,
    roots: Sequence[Path],
    recursive_index: Optional[Dict[str, Path]],
) -> Optional[Path]:
    seen = set()
    for candidate in candidate_image_paths(filename, roots):
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate

    if recursive_index is not None:
        basename = Path(filename.strip()).name
        match = recursive_index.get(basename)
        if match is not None:
            return match

    return None


def encode_image_png(image_path: Path, image_size: Optional[int]) -> bytes:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if image_size is not None:
            img = img.resize((image_size, image_size), resample=BICUBIC)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


class TarShardWriter:
    def __init__(
        self,
        output_dir: Path,
        prefix: str,
        samples_per_shard: int,
        start_shard_index: int = 0,
    ):
        if samples_per_shard <= 0:
            raise ValueError("--samples-per-shard must be >= 1")
        self.output_dir = output_dir
        self.prefix = prefix
        self.samples_per_shard = samples_per_shard
        self.shard_index = start_shard_index
        self.samples_in_shard = 0
        self.tar: Optional[tarfile.TarFile] = None

    def _open_next(self) -> None:
        self.close()
        path = output_shard_path(self.output_dir, self.prefix, self.shard_index)
        self.tar = tarfile.open(path, "w")
        self.samples_in_shard = 0
        self.shard_index += 1

    def write_file(self, name: str, data: bytes) -> None:
        if self.tar is None:
            self._open_next()
        assert self.tar is not None
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mode = 0o644
        info.mtime = 0
        self.tar.addfile(info, io.BytesIO(data))

    def write_sample(
        self,
        key: str,
        image: bytes,
        latent: bytes,
        filename: bytes,
    ) -> None:
        if self.tar is None or self.samples_in_shard >= self.samples_per_shard:
            self._open_next()
        self.write_file(f"{key}.image.png", image)
        self.write_file(f"{key}.latent.npy", latent)
        self.write_file(f"{key}.filename.txt", filename)
        self.samples_in_shard += 1

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def make_output_key(
    source_key: str,
    sample_index: int,
    preserve_keys: bool,
    start_index: int,
) -> str:
    if preserve_keys:
        return source_key
    return f"{start_index + sample_index:09d}"


def convert(args: argparse.Namespace) -> None:
    input_tars = list_input_tars(args.input_tar_dir, args.input_shards)
    roots = expanded_roots(args.image_root)
    if args.image_root and not roots:
        raise FileNotFoundError(f"No valid image roots found from: {args.image_root}")

    recursive_index = build_image_index(roots) if args.recursive_image_lookup else None
    resume_skip, output_start_shard = prepare_output_dir(
        args.output_dir,
        args.output_prefix,
        args.overwrite,
        args.resume,
        args.samples_per_shard,
        args.dry_run,
    )
    total_skip = args.skip_samples + resume_skip

    writer = None if args.dry_run else TarShardWriter(
        args.output_dir,
        args.output_prefix,
        args.samples_per_shard,
        start_shard_index=output_start_shard,
    )

    written = 0
    seen = 0
    skipped_missing_images = 0
    first_shape: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None

    try:
        for tar_path in input_tars:
            with tarfile.open(tar_path, "r:*") as tar:
                for source_key, members in group_members(tar):
                    if args.limit is not None and written >= args.limit:
                        break
                    if seen < total_skip:
                        seen += 1
                        continue

                    if "latent.npy" not in members:
                        raise KeyError(f"{tar_path.name}:{source_key} is missing latent.npy")
                    if "filename.txt" not in members:
                        raise KeyError(f"{tar_path.name}:{source_key} is missing filename.txt")

                    filename_bytes = read_member(tar, members["filename.txt"])
                    filename_text = filename_bytes.decode("utf-8").strip()

                    if "image.png" in members:
                        image_bytes = read_member(tar, members["image.png"])
                    else:
                        image_path = resolve_image_path(filename_text, roots, recursive_index)
                        if image_path is None:
                            if args.skip_missing_images:
                                skipped_missing_images += 1
                                continue
                            raise FileNotFoundError(
                                f"Could not resolve image for {tar_path.name}:{source_key} "
                                f"from filename.txt={filename_text!r}"
                            )
                        image_bytes = encode_image_png(image_path, args.image_size)

                    latent_bytes, original_shape, stored_shape = normalize_latent(
                        read_member(tar, members["latent.npy"]),
                        layout=args.latent_layout,
                        expected_size=args.expected_latent_size,
                        dtype=args.latent_dtype,
                    )
                    if first_shape is None:
                        first_shape = (original_shape, stored_shape)

                    out_key = make_output_key(
                        source_key=source_key,
                        sample_index=total_skip + written,
                        preserve_keys=args.preserve_keys,
                        start_index=args.start_index,
                    )

                    if writer is not None:
                        writer.write_sample(out_key, image_bytes, latent_bytes, filename_bytes)
                    written += 1
                    seen += 1

                    if written % 1000 == 0:
                        print(f"Converted {written} samples...")

                if args.limit is not None and written >= args.limit:
                    break
    finally:
        if writer is not None:
            writer.close()

    print(f"Input shards: {len(input_tars)} from {args.input_tar_dir}")
    print(f"Output dir: {args.output_dir}")
    if total_skip:
        print(
            f"Skipped ordered input samples: {total_skip} "
            f"(resume={resume_skip}, manual={args.skip_samples})"
        )
        print(f"Output starts at shard index: {output_start_shard}")
    print(f"Samples converted: {written}")
    if skipped_missing_images:
        print(f"Samples skipped due to missing images: {skipped_missing_images}")
    if first_shape is not None:
        original_shape, stored_shape = first_shape
        print(f"First latent shape: source={original_shape}, stored={stored_shape}")
    if args.dry_run:
        print("Dry run only; no output shards were written.")


def main() -> None:
    args = parse_args()
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be >= 1")
    if args.skip_samples < 0:
        raise ValueError("--skip-samples must be >= 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be >= 1")
    if args.image_size is not None and args.image_size <= 0:
        raise ValueError("--image-size must be >= 1")
    convert(args)


if __name__ == "__main__":
    main()
