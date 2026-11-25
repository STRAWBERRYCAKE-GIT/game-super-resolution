"""Save mod-cropped HR references and repair x4 CSV pairs without resplitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image


def resolve(value, root):
    path = Path(value.replace("\\", "/"))
    if not path.is_absolute() and not (len(value) > 1 and value[1] == ":"):
        path = root / path
    return path.resolve()


def csv_path(path, root):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def align_pairs(splits, output, output_dir, project_root):
    root = project_root.resolve()
    splits, output, output_dir = splits.resolve(), output.resolve(), output_dir.resolve()
    report_path = output.with_suffix(".alignment.json")
    for target in (output, report_path, output_dir):
        if target.exists():
            raise ValueError(f"Output already exists: {target}. Choose a new output CSV/directory.")
    with splits.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        if not {"split", "lr_path", "hr_path"}.issubset(fields or []):
            raise ValueError("CSV must contain split, lr_path and hr_path columns.")
        rows = list(reader)
    if not rows:
        raise ValueError("The input CSV is empty.")
    plans, names = [], set()
    # Preflight all train/val/test pairs before creating output files.
    for row in rows:
        high = resolve(row["hr_path"], root)
        low = resolve(row["lr_path"], root)
        if not high.is_file():
            raise FileNotFoundError(f"Missing HR reference: {high}")
        name = high.stem + ".png"
        if name.casefold() in names:
            raise ValueError(f"Duplicate output frame name: {name}")
        names.add(name.casefold())
        if low.stem != high.stem:
            raise ValueError(f"LR/HR filename mismatch: {low.name}, {high.name}")
        if row["split"] not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {row['split']}")
        with Image.open(high) as image:
            before = image.size
        aligned = (before[0] // 4 * 4, before[1] // 4 * 4)
        if min(aligned) < 8:
            raise ValueError(f"HR too small for x4 and reflection padding: {high}, {before}")
        expected_low = (aligned[0] // 4, aligned[1] // 4)
        low_size = None
        if low.is_file():
            with Image.open(low) as image:
                low_size = image.size
        plans.append({"hr": high, "lr": low, "name": name, "before": before,
                      "aligned": aligned, "lr_before": low_size,
                      "needs_repair": before != aligned or low_size != expected_low})
    count = sum(plan["needs_repair"] for plan in plans)
    print(f"Pairs: {len(rows)}; pairs needing alignment: {count}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".x4_align_", dir=output_dir.parent))
    published = csv_created = report_created = False
    changes = []
    try:
        for index, (row, plan) in enumerate(zip(rows, plans), 1):
            row = dict(row)
            high, low = plan["hr"], plan["lr"]
            row["hr_path"], row["lr_path"] = csv_path(high, root), csv_path(low, root)
            if plan["needs_repair"]:
                with Image.open(high) as source:
                    reference = source.convert("RGB").crop((0, 0, *plan["aligned"]))
                expected = reference.resize((reference.width // 4, reference.height // 4),
                                            Image.Resampling.BICUBIC)
                same_low = False
                if plan["lr_before"] == expected.size:
                    with Image.open(low) as source:
                        same_low = source.convert("RGB").tobytes() == expected.tobytes()
                if plan["before"] != plan["aligned"]:
                    relative = Path("HR") / plan["name"]
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    reference.save(target, format="PNG", compress_level=3)
                    row["hr_path"] = csv_path(output_dir / relative, root)
                if not same_low:
                    relative = Path("LR_bicubic_X4") / plan["name"]
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    expected.save(target, format="PNG", compress_level=3)
                    row["lr_path"] = csv_path(output_dir / relative, root)
                changes.append({"image": plan["name"], "split": row["split"],
                                "hr_before": plan["before"], "hr_after": plan["aligned"],
                                "lr_before": plan["lr_before"], "lr_after": expected.size,
                                "lr_reused_after_pixel_check": same_low,
                                "hr_path": row["hr_path"], "lr_path": row["lr_path"]})
            rows[index - 1] = row
            if index == 1 or index % 100 == 0 or index == len(rows):
                print(f"Aligning: {index}/{len(rows)}", flush=True)
        # Only publish after all pairs have been successfully decoded/repaired.
        os.rename(staging, output_dir)
        published = True
        with output.open("x", encoding="utf-8-sig", newline="") as file:
            csv_created = True
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        report = {"input_csv": str(splits), "input_csv_sha256": hashlib.sha256(splits.read_bytes()).hexdigest(),
                  "output_csv": str(output), "scale": 4, "pairs": len(rows), "repaired_pairs": count,
                  "crop": "top-left anchored; remove width%4 on right and height%4 on bottom",
                  "degradation": "Pillow RGB Bicubic from mod-cropped HR",
                  "pillow": Image.__version__, "split_assignments_changed": False,
                  "original_images_overwritten": False, "changes": changes}
        with report_path.open("x", encoding="utf-8") as file:
            report_created = True
            json.dump(report, file, ensure_ascii=False, indent=2)
    except BaseException:
        if csv_created:
            output.unlink(missing_ok=True)
        if report_created:
            report_path.unlink(missing_ok=True)
        if published:
            shutil.rmtree(output_dir)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    reused = sum(change["lr_reused_after_pixel_check"] for change in changes)
    print(f"Saved {len(rows)} pairs -> {output}", flush=True)
    print(f"Repaired: {count}; existing LR verified/reused: {reused}; regenerated LR: {count - reused}", flush=True)
    print("Train/val/test assignments and original images retained. Use this NEW CSV for every x4 method.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Align x4 HR/LR references without overwriting data or resplitting.")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metadata/splits_X4_aligned.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/frames/aligned_X4"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    align_pairs(args.splits, args.output, args.output_dir, args.project_root)


if __name__ == "__main__":
    main()
