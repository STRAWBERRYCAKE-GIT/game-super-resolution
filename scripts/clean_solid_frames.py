from __future__ import annotations

import argparse
import csv
import io
import json
import math
import ntpath
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
RELATED_DIRS = (
    "data/frames/LR_bicubic_X2",
    "data/frames/LR_bicubic_X4",
    "results/images/bicubic_X2",
    "results/images/bicubic_X4",
)


@dataclass
class CsvEdit:
    path: Path
    original: bytes
    replacement: bytes
    removed: int
    remaining: int
    means: list[dict]
    split_counts: dict[str, int]


@dataclass
class CleanPlan:
    root: Path
    backup_root: Path
    pure_frames: list[dict]
    images: list[Path]
    edits: list[CsvEdit]
    summary_tables: list[str]


def under_root(root: Path, value: Path) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path must stay inside the project: {path}")
    return path


def is_inside(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def pure_rgb_color(path: Path) -> tuple[int, int, int] | None:
    """A solid color has zero range in EVERY RGB channel, including red/blue."""
    with Image.open(path) as source:
        with source.convert("RGB") as image:
            extrema = image.getextrema()
    if all(low == high for low, high in extrema):
        return tuple(low for low, _ in extrema)
    return None


def cell_matches_frame(value: str, stems: set[str]) -> bool:
    # ntpath handles both Windows and POSIX paths, including Windows paths in CSV.
    name = ntpath.basename(value.strip().strip("\"'")).casefold()
    if name in stems:
        return True
    stem, suffix = os.path.splitext(name)
    return suffix in IMAGE_SUFFIXES and stem in stems


def decode_csv(raw: bytes) -> tuple[str, str]:
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16"), "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("gb18030"), "gb18030"


def display_number(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def remaining_means(header: list[str], rows: list[list[str]]) -> list[dict]:
    columns = {name.strip().casefold(): index for index, name in enumerate(header)}
    metrics = [name for name in ("psnr", "ssim", "time_ms") if name in columns]
    if not metrics:
        return []
    method_index = columns.get("method")
    groups = defaultdict(list)
    for row in rows:
        method = row[method_index] if method_index is not None else "all"
        groups[method].append(row)
    results = []
    for method, group in sorted(groups.items()):
        entry = {"method": method, "rows": len(group)}
        for metric in metrics:
            values = []
            for row in group:
                try:
                    values.append(float(row[columns[metric]]))
                except ValueError:
                    continue
            if values:
                entry[f"mean_{metric}"] = display_number(sum(values) / len(values))
                entry[f"{metric}_count"] = len(values)
        results.append(entry)
    return results


def plan_csv(path: Path, stems: set[str]) -> tuple[CsvEdit | None, bool]:
    raw = path.read_bytes()
    text, encoding = decode_csv(raw)
    if not text.strip():
        return None, False
    try:
        dialect = csv.Sniffer().sniff(text[:65536], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text, newline=""), dialect=dialect, strict=True)
    table = list(reader)
    if not table:
        return None, False
    header, rows = table[0], table[1:]
    # Refuse malformed CSV before any image or metadata is changed.
    for index, row in enumerate(rows, 2):
        if row and len(row) != len(header):
            raise ValueError(f"Malformed CSV row {index}: {path}")

    columns = {name.strip().casefold(): index for index, name in enumerate(header)}
    is_summary = (
        "summary" in path.stem.casefold()
        or any(name.startswith(("mean_", "avg_", "average_")) for name in columns)
    )
    kept = []
    removed = 0
    for row in rows:
        if any(cell_matches_frame(value, stems) for value in row):
            removed += 1
        else:
            kept.append(row)
    if not removed:
        return None, is_summary

    newline = "\r\n" if "\r\n" in text else "\n"
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, dialect=dialect, lineterminator=newline)
    writer.writerow(header)
    writer.writerows(kept)
    split_index = columns.get("split")
    splits = Counter(
        row[split_index] for row in kept if row and split_index is not None
    )
    return CsvEdit(
        path, raw, buffer.getvalue().encode(encoding),
        removed, len(kept), remaining_means(header, [row for row in kept if row]),
        dict(splits),
    ), is_summary


def build_plan(
    root: Path, hr_dir: Path, backup_root: Path, extra_csv_dirs: list[Path]
) -> CleanPlan:
    root = root.resolve()
    hr_dir = under_root(root, hr_dir)
    backup_root = under_root(root, backup_root)
    if not hr_dir.is_dir():
        raise ValueError(f"HR directory does not exist: {hr_dir}")
    if is_inside(backup_root, hr_dir) or is_inside(hr_dir, backup_root):
        raise ValueError("The backup directory must be separate from the HR tree.")

    paths_by_stem = defaultdict(list)
    pure_frames = []
    for path in sorted(hr_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        path = under_root(root, path)
        paths_by_stem[path.stem.casefold()].append(path)
        try:
            color = pure_rgb_color(path)
        except Exception as error:
            raise ValueError(f"Cannot inspect HR image {path}: {error}") from error
        if color is not None:
            pure_frames.append({
                "hr_path": path.relative_to(root).as_posix(),
                "stem": path.stem.casefold(),
                "rgb": list(color),
            })

    stems = {frame["stem"] for frame in pure_frames}
    for stem in stems:
        if len(paths_by_stem[stem]) != 1:
            raise ValueError(
                f"Ambiguous frame name '{stem}' appears in multiple HR files. "
                "Give each frame a unique name before cleaning filename-based CSV rows."
            )
    images = {
        root / frame["hr_path"] for frame in pure_frames
    }
    if not stems:
        return CleanPlan(root, backup_root, [], [], [], [])

    for directory in RELATED_DIRS:
        target = under_root(root, Path(directory))
        if not target.is_dir():
            continue
        for path in target.rglob("*"):
            if (
                path.is_file()
                and path.suffix.casefold() in IMAGE_SUFFIXES
                and path.stem.casefold() in stems
            ):
                images.add(under_root(root, path))

    csv_paths = set()
    for directory in [Path("metadata"), Path("results"), *extra_csv_dirs]:
        directory = under_root(root, directory)
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.casefold() == ".csv":
                path = under_root(root, path)
                if not is_inside(path, backup_root):
                    csv_paths.add(path)

    edits, summaries = [], []
    for path in sorted(csv_paths):
        edit, is_summary = plan_csv(path, stems)
        if edit is not None:
            edits.append(edit)
        if is_summary:
            summaries.append(path.relative_to(root).as_posix())
    return CleanPlan(
        root, backup_root, pure_frames, sorted(images), edits, summaries
    )


def atomic_write(path: Path, content: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix="cleanup_", suffix=".tmp", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def apply_plan(plan: CleanPlan) -> Path:
    # Validate metadata once more before applying a plan if another process edited it.
    for edit in plan.edits:
        if edit.path.read_bytes() != edit.original:
            raise RuntimeError(f"CSV changed during preview; run again: {edit.path}")

    run_dir = plan.backup_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "in_progress",
        "project_root": str(plan.root),
        "pure_frames": plan.pure_frames,
        "removed_image_paths": [p.relative_to(plan.root).as_posix() for p in plan.images],
        "csv_changes": [{
            "path": edit.path.relative_to(plan.root).as_posix(),
            "removed_rows": edit.removed,
            "remaining_rows": edit.remaining,
            "remaining_means": edit.means,
            "split_frame_counts": edit.split_counts,
        } for edit in plan.edits],
        "summary_tables_to_regenerate": plan.summary_tables,
    }
    report_path = run_dir / "cleanup_report.json"

    def save_report():
        atomic_write(
            report_path, json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
        )

    save_report()
    moved = []
    changed_csvs = []
    try:
        # Back up every table before removing any image.
        for edit in plan.edits:
            destination = run_dir / "csv" / edit.path.relative_to(plan.root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(edit.path, destination)
        for source in plan.images:
            destination = run_dir / "images" / source.relative_to(plan.root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
        for edit in plan.edits:
            atomic_write(edit.path, edit.replacement)
            changed_csvs.append(edit)
        report["status"] = "completed"
        save_report()
    except Exception as error:
        rollback_errors = []
        for edit in reversed(changed_csvs):
            try:
                atomic_write(edit.path, edit.original)
            except Exception as rollback_error:
                rollback_errors.append(f"{edit.path}: {rollback_error}")
        for source, destination in reversed(moved):
            try:
                shutil.move(str(destination), str(source))
            except Exception as rollback_error:
                rollback_errors.append(f"{source}: {rollback_error}")
        report["status"] = "rollback_incomplete" if rollback_errors else "rolled_back"
        report["error"] = str(error)
        report["rollback_errors"] = rollback_errors
        save_report()
        raise RuntimeError(f"Cleanup failed; inspect {report_path}: {error}") from error
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean exactly solid-color HR frames, paired images, and CSV rows."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--hr-dir", type=Path, default=Path("data/frames/HR"))
    parser.add_argument(
        "--csv-dir", type=Path, action="append", default=[],
        help="Additional CSV directory; metadata/ and results/ are always scanned.",
    )
    parser.add_argument(
        "--backup-dir", type=Path, default=Path("results/cleanup_runs")
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply cleanup; without this flag only print the plan.",
    )
    args = parser.parse_args()
    try:
        plan = build_plan(
            args.project_root, args.hr_dir, args.backup_dir, args.csv_dir
        )
        print(f"Exactly solid-color HR frames: {len(plan.pure_frames)}")
        for frame in plan.pure_frames:
            print(f"  {frame['hr_path']}  RGB={frame['rgb']}")
        print(f"Images to remove from active directories: {len(plan.images)}")
        for path in plan.images:
            print(f"  {path.relative_to(plan.root).as_posix()}")
        print(f"CSV files to update: {len(plan.edits)}")
        for edit in plan.edits:
            print(
                f"  {edit.path.relative_to(plan.root).as_posix()}: "
                f"remove {edit.removed}, keep {edit.remaining} rows"
            )
        if plan.summary_tables:
            print("Existing aggregate tables must be regenerated from remaining samples:")
            for name in plan.summary_tables:
                print(f"  {name}")
        if not plan.pure_frames:
            print("Nothing to clean.")
        elif not args.apply:
            print("Preview only. Run the same command with --apply to clean.")
        else:
            report = apply_plan(plan)
            print(f"Cleanup completed. Backups and report: {report.parent}")
            print("Existing train/val/test assignments were preserved.")
            print("Re-run 06_evaluate.py to refresh displayed averages.")
    except (ValueError, OSError, csv.Error, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()

