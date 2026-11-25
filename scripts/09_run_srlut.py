from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(items, desc="SR-LUT"):
        for index, item in enumerate(items, 1):
            print(f"{desc}: {index}/{len(items)}", flush=True)
            yield item

from srlut_numpy import load_lut, upscale


def resolve_csv_path(value: str, root: Path) -> Path:
    path = Path(value.replace("\\", "/"))
    if not path.is_absolute() and not (len(value) > 1 and value[1] == ":"):
        path = root / path
    if not path.is_file():
        raise ValueError(
            f"Cannot find image: {path}. CSV paths must point to this machine. "
            "Use --project-root for project-relative paths; remap path columns "
            "if they contain absolute paths from another machine, preserving splits."
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run supplied Ours-S LUTs on your existing paired-data CSV."
    )
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--scale", type=int, choices=(2, 4), required=True)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--sampling-bits", type=int, default=4,
                        help="Official interval exponent: 4 means sample spacing 16.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--method", default="srlut_pretrained")
    parser.add_argument("--tile-rows", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.tile_rows < 1 or args.warmup < 0 or (
        args.limit is not None and args.limit < 1
    ):
        parser.error("tile-rows and limit must be positive; warmup must be nonnegative.")

    with args.splits.open(encoding="utf-8-sig", newline="") as file:
        selected = [
            row for row in csv.DictReader(file) if row.get("split") == args.split
        ]
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        parser.error(f"No rows for split {args.split!r}: {args.splits}")

    root = args.project_root.resolve()
    prepared = []
    names = set()
    for row in selected:
        lr_path = resolve_csv_path(row["lr_path"], root)
        hr_path = resolve_csv_path(row["hr_path"], root)
        name = hr_path.stem + ".png"
        if name.casefold() in names:
            raise ValueError(f"Duplicate output name: {name}")
        names.add(name.casefold())
        with Image.open(lr_path) as lr, Image.open(hr_path) as hr:
            if (lr.width * args.scale, lr.height * args.scale) != hr.size:
                raise ValueError(
                    f"Scale/size mismatch for {name}: LR={lr.size}, HR={hr.size}, "
                    f"scale={args.scale}. Images are not cropped or padded to hide this."
                )
        prepared.append((row, lr_path, name))

    lut = load_lut(args.lut, args.scale, args.sampling_bits)
    args.output.mkdir(parents=True, exist_ok=True)
    timings = []
    for index, (row, lr_path, name) in enumerate(tqdm(prepared, desc="SR-LUT")):
        with Image.open(lr_path) as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8)
        if index == 0:
            for _ in range(args.warmup):
                upscale(array, lut, args.scale, args.sampling_bits, args.tile_rows)
        start = time.perf_counter()
        sr = upscale(array, lut, args.scale, args.sampling_bits, args.tile_rows)
        milliseconds = (time.perf_counter() - start) * 1000
        Image.fromarray(sr).save(args.output / name, format="PNG", compress_level=3)
        timings.append({
            "image": name,
            "game_id": row.get("game_id", row.get("clip_id", "").split("_clip")[0]),
            "clip_id": row.get("clip_id", ""),
            "method": args.method,
            "scale": args.scale,
            "time_ms": round(milliseconds, 6),
        })

    args.timings.parent.mkdir(parents=True, exist_ok=True)
    with args.timings.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(timings[0]))
        writer.writeheader()
        writer.writerows(timings)
    provenance = {
        "implementation": "CPU NumPy Ours-S 4D simplex, tiled vectorization",
        "splits": str(args.splits), "split": args.split, "images": len(timings),
        "method": args.method, "scale": args.scale, "sampling_bits": args.sampling_bits,
        "tile_rows": args.tile_rows, "warmup": args.warmup,
        "lut": str(args.lut), "lut_sha256": hashlib.sha256(args.lut.read_bytes()).hexdigest(),
        "numpy": np.__version__,
        "timing_scope": "four LUT passes plus RGB output conversion; excludes LUT load, image decode and PNG save",
    }
    args.timings.with_suffix(".json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(timings)} SR images -> {args.output}")
    print(f"Timings -> {args.timings}")
    print(f"Mean reconstruction time: {np.mean([r['time_ms'] for r in timings]):.3f} ms")
    print("Compute quality with your existing 06_evaluate.py using the same CSV and shave.")


if __name__ == "__main__":
    main()

