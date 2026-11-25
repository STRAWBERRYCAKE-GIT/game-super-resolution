from __future__ import annotations

import argparse
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import list_images, write_csv


# person01 = game ID; clip001 = original video within that game.
FRAME_PATTERN = re.compile(
    r"(?P<game>person\d+)_(?P<clip>clip\d+)__f(?P<frame>\d+)",
    re.IGNORECASE,
)


def parse_frame_id(path: Path) -> tuple[str, str]:
    """Return game ID and a video ID that is unique across games."""
    match = FRAME_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(
            f"Unsupported frame name: {path.name}. "
            "Expected person01_clip001__f00001320.png "
            "(person01 = game, clip001 = original video)."
        )
    game_id = match.group("game").lower()
    clip_id = f"{game_id}_{match.group('clip').lower()}"
    return game_id, clip_id


def allocate_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    """Allocate whole videos within one game, keeping every enabled split nonempty."""
    if (
        not math.isfinite(train_ratio)
        or not math.isfinite(val_ratio)
        or train_ratio <= 0
        or val_ratio < 0
        or train_ratio + val_ratio >= 1
    ):
        raise ValueError(
            "Ratios must be finite and satisfy "
            "train > 0, val >= 0, train + val < 1."
        )
    minimum = 3 if val_ratio > 0 else 2
    if total < minimum:
        message = (
            f"Need at least {minimum} independent original videos per game; "
            f"found {total}. Frames of one video cannot be split across datasets."
        )
        if total == 2 and val_ratio > 0:
            message += " Use --val-ratio 0 only for a train/test split without validation."
        raise ValueError(message)

    val_count = (
        min(max(1, round(total * val_ratio)), total - 2)
        if val_ratio > 0
        else 0
    )
    train_count = min(
        max(1, round(total * train_ratio)),
        total - val_count - 1,
    )
    return train_count, val_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split paired frames within EACH game by whole original video. "
            "Read game/video IDs from person01_clip001__f00001320.png; "
            "game folders are not required."
        )
    )
    parser.add_argument("--hr-dir", type=Path, required=True)
    parser.add_argument("--lr-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metadata/splits.csv"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hr_images = list_images(args.hr_dir)
    if not hr_images:
        raise SystemExit(f"No HR images found under: {args.hr_dir}")

    games: dict[str, set[str]] = defaultdict(set)
    records: list[dict[str, str]] = []
    seen_frames: dict[tuple[str, str, int], Path] = {}
    missing: list[str] = []

    for hr_path in hr_images:
        try:
            game_id, clip_id = parse_frame_id(hr_path)
        except ValueError as error:
            raise SystemExit(str(error)) from error

        match = FRAME_PATTERN.fullmatch(hr_path.stem)
        assert match is not None
        frame_key = (game_id, clip_id, int(match.group("frame")))
        if frame_key in seen_frames:
            raise SystemExit(
                "Duplicate game/video/frame ID found:\n"
                f"{seen_frames[frame_key]}\n{hr_path}\n"
                "Each game/video/frame ID must identify exactly one HR image."
            )
        seen_frames[frame_key] = hr_path

        relative = hr_path.relative_to(args.hr_dir).with_suffix(".png")
        lr_path = args.lr_dir / relative
        if not lr_path.is_file():
            missing.append(str(lr_path))

        games[game_id].add(clip_id)
        records.append(
            {
                "game_id": game_id,
                "clip_id": clip_id,
                "lr_path": str(lr_path),
                "hr_path": str(hr_path),
            }
        )

    if missing:
        preview = "\n".join(missing[:5])
        raise SystemExit(f"Missing {len(missing)} LR pairs. First entries:\n{preview}")

    assignments: dict[str, str] = {}
    for game_id in sorted(games):
        clips = sorted(games[game_id])
        try:
            train_count, val_count = allocate_counts(
                len(clips), args.train_ratio, args.val_ratio
            )
        except ValueError as error:
            raise SystemExit(f"Game {game_id}: {error}") from error

        # Independent per-game RNG: adding another game does not reshuffle this one.
        random.Random(f"{args.seed}:{game_id}").shuffle(clips)
        for index, clip_id in enumerate(clips):
            assignments[clip_id] = (
                "train" if index < train_count
                else "val" if index < train_count + val_count
                else "test"
            )

    rows = [{"split": assignments[row["clip_id"]], **row} for row in records]
    write_csv(
        args.output, rows,
        ["split", "game_id", "clip_id", "lr_path", "hr_path"],
    )

    print(f"Saved {len(rows)} pairs -> {args.output}")
    print("Ratios apply to original-video counts within each game, not frame counts.")
    for game_id in sorted(games):
        frame_counts = Counter(
            row["split"] for row in rows if row["game_id"] == game_id
        )
        print(f"Game {game_id}:")
        for split in ("train", "val", "test"):
            selected_clips = sorted(
                clip_id for clip_id in games[game_id]
                if assignments[clip_id] == split
            )
            print(
                f"  {split}: videos={len(selected_clips)}, "
                f"frames={frame_counts[split]}; "
                f"video IDs: {', '.join(selected_clips) or '(none)'}"
            )
    totals = Counter(row["split"] for row in rows)
    print(f"Total frame counts: {dict(totals)}")


if __name__ == "__main__":
    main()