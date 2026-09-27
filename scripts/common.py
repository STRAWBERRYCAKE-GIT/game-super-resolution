from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(path))]


def list_images(directory: Path) -> list[Path]:
    return sorted(
        (p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )


def list_videos(directory: Path) -> list[Path]:
    return sorted(
        (p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES),
        key=natural_key,
    )


def safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", value.strip())
    return value.strip("_") or "clip"


def clip_id_from_frame(path: Path) -> str:
    """Extract clip id from '<clip>__f000001.png'."""
    return path.stem.rsplit("__f", 1)[0]


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))

