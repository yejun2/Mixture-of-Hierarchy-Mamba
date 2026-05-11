#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect one COCO text-latent WebDataset shard sample."
    )
    parser.add_argument("shard", type=Path, help="Path to a train/val .tar shard.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import webdataset as wds

    if not args.shard.exists():
        raise FileNotFoundError(args.shard)

    dataset = wds.WebDataset(str(args.shard)).decode()
    sample = next(iter(dataset))
    print("keys:", sorted(sample.keys()))

    image = sample.get("image.jpg")
    img_feature = sample.get("img_feature256.npy")
    caption_feature = sample.get("caption_feature.npy")
    caption = sample.get("caption.json")

    if image is None:
        raise KeyError("image.jpg missing")
    if img_feature is None:
        raise KeyError("img_feature256.npy missing")
    if caption_feature is None:
        raise KeyError("caption_feature.npy missing")
    if caption is None:
        raise KeyError("caption.json missing")

    print("image:", image.size)
    print("img_feature256.npy:", img_feature.shape, img_feature.dtype)
    print("caption_feature.npy:", caption_feature.shape, caption_feature.dtype)
    print("caption.json length:", len(caption))
    print("first caption:", caption[0] if caption else "")

    if img_feature.shape != (4, 32, 32):
        raise RuntimeError(f"Expected img_feature shape (4, 32, 32), got {img_feature.shape}")
    if len(caption_feature.shape) != 3 or caption_feature.shape[-1] != 768:
        raise RuntimeError(
            "Expected caption_feature shape [N, 77, 768]-like, "
            f"got {caption_feature.shape}"
        )

    print("OK")


if __name__ == "__main__":
    main()
