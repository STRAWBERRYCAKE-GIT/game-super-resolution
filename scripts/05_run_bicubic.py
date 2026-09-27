from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from common import read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bicubic upscaling on a dataset split.")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timings", type=Path, default=Path("results/metrics/bicubic_timings.csv"))
    args = parser.parse_args()

    selected = [row for row in read_csv(args.splits) if row["split"] == args.split]
    if not selected:
        raise SystemExit(f"No rows for split '{args.split}' in {args.splits}")
    args.output.mkdir(parents=True, exist_ok=True)

    timing_rows = []
    for row in tqdm(selected, desc="Bicubic"):
        lr_path = Path(row["lr_path"])
        hr_path = Path(row["hr_path"])
        with Image.open(lr_path) as lr_image, Image.open(hr_path) as hr_image:
            lr_image = lr_image.convert("RGB")
            start = time.perf_counter()
            sr_image = lr_image.resize(hr_image.size, Image.Resampling.BICUBIC)
            elapsed_ms = (time.perf_counter() - start) * 1000
            output_path = args.output / f"{hr_path.stem}.png"
            sr_image.save(output_path, format="PNG", compress_level=3)
        timing_rows.append({"image": hr_path.name, "method": "bicubic", "time_ms": round(elapsed_ms, 4)})

    write_csv(args.timings, timing_rows, ["image", "method", "time_ms"])
    print(f"Saved {len(selected)} SR images -> {args.output}")
    print(f"Timings -> {args.timings}")


if __name__ == "__main__":
    main()

