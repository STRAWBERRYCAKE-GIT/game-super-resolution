from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import read_csv


def labelled(image: Image.Image, label: str, bar_height: int = 48) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + bar_height), "white")
    canvas.paste(image, (0, bar_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=24) if hasattr(ImageFont, "load_default") else None
    draw.text((12, 10), label, fill="black", font=font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Create LR/Bicubic/HR comparison panels.")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--sr-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    rows = [row for row in read_csv(args.splits) if row["split"] == args.split][:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        hr_path, lr_path = Path(row["hr_path"]), Path(row["lr_path"])
        sr_path = args.sr_dir / f"{hr_path.stem}.png"
        with Image.open(hr_path) as hr_raw, Image.open(lr_path) as lr_raw, Image.open(sr_path) as sr_raw:
            hr = hr_raw.convert("RGB")
            lr = lr_raw.convert("RGB").resize(hr.size, Image.Resampling.NEAREST)
            sr = sr_raw.convert("RGB")
            panels = [labelled(lr, "LR (nearest preview)"), labelled(sr, "Bicubic SR"), labelled(hr, "HR reference")]
            canvas = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white")
            x = 0
            for panel in panels:
                canvas.paste(panel, (x, 0))
                x += panel.width
            canvas.save(args.output / f"{hr_path.stem}_comparison.jpg", quality=95, subsampling=0)
    print(f"Saved {len(rows)} comparison panels -> {args.output}")


if __name__ == "__main__":
    main()

