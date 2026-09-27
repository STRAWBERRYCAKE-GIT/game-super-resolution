from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from common import read_csv, write_csv


def to_rgb_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SR output against HR reference.")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--sr-dir", type=Path, required=True)
    parser.add_argument("--method", default="bicubic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shave", type=int, default=0,
                        help="Ignore this many border pixels before measuring.")
    args = parser.parse_args()

    selected = [row for row in read_csv(args.splits) if row["split"] == args.split]
    metric_rows = []
    for row in tqdm(selected, desc=f"Evaluating {args.method}"):
        hr_path = Path(row["hr_path"])
        sr_path = args.sr_dir / f"{hr_path.stem}.png"
        if not sr_path.exists():
            raise FileNotFoundError(f"Missing SR output: {sr_path}")
        hr = to_rgb_array(hr_path)
        sr = to_rgb_array(sr_path)
        if hr.shape != sr.shape:
            raise ValueError(f"Shape mismatch: {hr_path.name}: HR={hr.shape}, SR={sr.shape}")
        if args.shave > 0:
            s = args.shave
            hr, sr = hr[s:-s, s:-s], sr[s:-s, s:-s]

        psnr = peak_signal_noise_ratio(hr, sr, data_range=255)
        ssim = structural_similarity(hr, sr, channel_axis=2, data_range=255)
        metric_rows.append({"image": hr_path.name, "clip_id": row["clip_id"],
                            "method": args.method, "psnr": round(float(psnr), 6),
                            "ssim": round(float(ssim), 6)})

    write_csv(args.output, metric_rows, ["image", "clip_id", "method", "psnr", "ssim"])
    mean_psnr = float(np.mean([row["psnr"] for row in metric_rows]))
    mean_ssim = float(np.mean([row["ssim"] for row in metric_rows]))
    print(f"{args.method}: PSNR={mean_psnr:.4f} dB, SSIM={mean_ssim:.6f}")
    print(f"Per-image metrics -> {args.output}")


if __name__ == "__main__":
    main()

