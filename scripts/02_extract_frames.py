from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from common import list_videos, safe_name


def extract(video: Path, output: Path, sample_fps: float, png_compression: int) -> int:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS: {video}")

    interval = max(1, round(source_fps / sample_fps))
    clip_id = safe_name(video.stem)
    saved = 0

    for frame_index in tqdm(range(total_frames), desc=video.name, leave=False):
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % interval != 0:
            continue
        destination = output / f"{clip_id}__f{frame_index:08d}.png"
        if not cv2.imwrite(str(destination), frame,
                           [cv2.IMWRITE_PNG_COMPRESSION, png_compression]):
            raise RuntimeError(f"Failed to write: {destination}")
        saved += 1

    capture.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract lossless PNG HR frames.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--png-compression", type=int, choices=range(0, 10), default=3)
    args = parser.parse_args()

    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps must be greater than 0.")
    args.output.mkdir(parents=True, exist_ok=True)

    videos = list_videos(args.input)
    if not videos:
        raise SystemExit(f"No videos found under: {args.input}")

    total = 0
    for video in videos:
        total += extract(video, args.output, args.sample_fps, args.png_compression)
    print(f"Saved {total} HR frames -> {args.output}")


if __name__ == "__main__":
    main()

