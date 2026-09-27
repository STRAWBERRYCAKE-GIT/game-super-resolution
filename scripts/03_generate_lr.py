from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from common import list_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paired LR images using Bicubic downsampling.")
    parser.add_argument("--hr-dir", type=Path, required=True)
    parser.add_argument("--lr-dir", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    if args.scale < 2:
        raise SystemExit("--scale must be at least 2.")
    images = list_images(args.hr_dir)
    if not images:
        raise SystemExit(f"No HR images found under: {args.hr_dir}")

    for hr_path in tqdm(images, desc="Generating LR"):
        relative = hr_path.relative_to(args.hr_dir).with_suffix(".png")
        lr_path = args.lr_dir / relative
        lr_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(hr_path) as image:
            image = image.convert("RGB")
            width = image.width - image.width % args.scale
            height = image.height - image.height % args.scale
            if (width, height) != image.size:
                image = image.crop((0, 0, width, height))
            lr = image.resize((width // args.scale, height // args.scale), Image.Resampling.BICUBIC)
            lr.save(lr_path, format="PNG", compress_level=3)

    print(f"Generated {len(images)} LR images -> {args.lr_dir}")


if __name__ == "__main__":
    main()

