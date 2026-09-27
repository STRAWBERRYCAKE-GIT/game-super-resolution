from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import clip_id_from_frame, list_images, write_csv


def allocate_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if total == 1:
        return 1, 0
    if total == 2:
        return 1, 0
    train_count = max(1, round(total * train_ratio))
    val_count = max(1, round(total * val_ratio))
    if train_count + val_count >= total:
        train_count = total - 2
        val_count = 1
    return train_count, val_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Split paired data by whole video clip.")
    parser.add_argument("--hr-dir", type=Path, required=True)
    parser.add_argument("--lr-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metadata/splits.csv"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("Ratios must satisfy train > 0, val >= 0, train + val < 1.")

    hr_images = list_images(args.hr_dir)
    clips = sorted({clip_id_from_frame(path) for path in hr_images})
    if not clips:
        raise SystemExit(f"No HR images found under: {args.hr_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(clips)
    train_count, val_count = allocate_counts(len(clips), args.train_ratio, args.val_ratio)
    train_clips = set(clips[:train_count])
    val_clips = set(clips[train_count:train_count + val_count])

    rows = []
    missing = []
    for hr_path in hr_images:
        relative = hr_path.relative_to(args.hr_dir).with_suffix(".png")
        lr_path = args.lr_dir / relative
        if not lr_path.exists():
            missing.append(str(lr_path))
            continue
        clip_id = clip_id_from_frame(hr_path)
        split = "train" if clip_id in train_clips else "val" if clip_id in val_clips else "test"
        rows.append({"split": split, "clip_id": clip_id,
                     "lr_path": str(lr_path), "hr_path": str(hr_path)})

    if missing:
        preview = "\n".join(missing[:5])
        raise SystemExit(f"Missing {len(missing)} LR pairs. First entries:\n{preview}")

    write_csv(args.output, rows, ["split", "clip_id", "lr_path", "hr_path"])
    counts = {name: sum(row["split"] == name for row in rows) for name in ("train", "val", "test")}
    print(f"Saved {len(rows)} pairs -> {args.output}; frame counts: {counts}")
    print(f"Clip counts: train={len(train_clips)}, val={len(val_clips)}, test={len(clips)-train_count-val_count}")


if __name__ == "__main__":
    main()

