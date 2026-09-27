from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from common import list_videos, write_csv


def probe_video(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration,size",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    fmt = payload.get("format", {})

    def fps_of(value: str) -> float:
        numerator, denominator = value.split("/")
        return float(numerator) / float(denominator) if float(denominator) else 0.0

    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    return {
        "path": str(path),
        "file_name": path.name,
        "codec": stream.get("codec_name", ""),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": round(fps_of(stream.get("avg_frame_rate", "0/1")), 3),
        "duration_s": round(duration, 3),
        "frames": stream.get("nb_frames", ""),
        "size_mb": round(int(fmt.get("size", path.stat().st_size)) / 1024 / 1024, 2),
        "is_1080p": int(stream.get("width", 0)) == 1920 and int(stream.get("height", 0)) == 1080,
        "status": "ok",
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check video metadata with ffprobe.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metadata/videos.csv"))
    args = parser.parse_args()

    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe not found. Install FFmpeg and add it to PATH.")

    videos = list_videos(args.input)
    if not videos:
        raise SystemExit(f"No videos found under: {args.input}")

    rows = []
    for video in videos:
        try:
            rows.append(probe_video(video))
        except Exception as exc:
            rows.append({
                "path": str(video), "file_name": video.name, "codec": "",
                "width": "", "height": "", "fps": "", "duration_s": "",
                "frames": "", "size_mb": round(video.stat().st_size / 1024 / 1024, 2),
                "is_1080p": False, "status": "error", "error": str(exc),
            })

    fields = ["path", "file_name", "codec", "width", "height", "fps",
              "duration_s", "frames", "size_mb", "is_1080p", "status", "error"]
    write_csv(args.output, rows, fields)
    print(f"Checked {len(rows)} videos -> {args.output}")


if __name__ == "__main__":
    main()

