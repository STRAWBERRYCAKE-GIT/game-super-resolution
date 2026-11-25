from __future__ import annotations

import csv
import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
aligner = importlib.import_module("12_align_x4_pairs")


class AlignmentTests(unittest.TestCase):
    def fixture(self, root, size=(28, 18), mode="cropped"):
        name = "person02_clip001__f00000000.png"
        hr, lr = root / "HR" / name, root / "LR" / name
        hr.parent.mkdir()
        lr.parent.mkdir()
        image = Image.fromarray(np.random.default_rng(42).integers(
            0, 256, (size[1], size[0], 3), dtype=np.uint8))
        image.save(hr)
        aligned = (size[0] // 4 * 4, size[1] // 4 * 4)
        expected = image.crop((0, 0, *aligned)).resize(
            (aligned[0] // 4, aligned[1] // 4), Image.Resampling.BICUBIC)
        if mode == "cropped":
            expected.save(lr)
        elif mode == "full_resize":
            image.resize(expected.size, Image.Resampling.BICUBIC).save(lr)
        elif mode == "wrong_size":
            Image.new("RGB", (9, 9)).save(lr)
        elif mode != "missing":
            raise ValueError(mode)
        rows = [{"split": "train", "game_id": "person02", "clip_id": "person02_clip001",
                 "lr_path": "LR\\" + name, "hr_path": "HR\\" + name, "note": "keep this metadata"}]
        path = root / "splits.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path, hr, lr, expected

    def repair(self, root, path):
        output = root / "metadata" / "splits_X4_aligned.csv"
        directory = root / "data" / "aligned_X4"
        report = aligner.align_pairs(path, output, directory, root)
        with output.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        return output, directory, report, rows

    def test_reported_2880x1610_case_keeps_correct_lr_and_saves_1608_hr(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, hr, lr, expected = self.fixture(root, (2880, 1610))
            originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, hr, lr)}
            output, directory, report, rows = self.repair(root, path)
            with Image.open(root / rows[0]["hr_path"]) as fixed:
                self.assertEqual(fixed.size, (2880, 1608))
                with Image.open(hr) as original:
                    self.assertEqual(fixed.tobytes(), original.crop((0, 0, 2880, 1608)).tobytes())
            self.assertEqual(rows[0]["lr_path"], "LR/" + lr.name)
            self.assertTrue(report["changes"][0]["lr_reused_after_pixel_check"])
            self.assertFalse((directory / "LR_bicubic_X4").exists())
            self.assertEqual(report["repaired_pairs"], 1)
            for p, digest in originals.items():
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)
            self.assertEqual(rows[0]["split"], "train")
            self.assertEqual(rows[0]["game_id"], "person02")
            self.assertEqual(rows[0]["clip_id"], "person02_clip001")
            self.assertEqual(rows[0]["note"], "keep this metadata")
            self.assertFalse(json.loads(output.with_suffix(".alignment.json").read_text())["split_assignments_changed"])

    def test_lr_from_full_odd_hr_is_regenerated_to_match_cropped_hr(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, hr, lr, expected = self.fixture(root, mode="full_resize")
            _, _, report, rows = self.repair(root, path)
            self.assertFalse(report["changes"][0]["lr_reused_after_pixel_check"])
            self.assertNotEqual(rows[0]["lr_path"], "LR/" + lr.name)
            with Image.open(root / rows[0]["lr_path"]) as fixed:
                self.assertEqual(fixed.tobytes(), expected.tobytes())

    def test_aligned_pairs_are_reused_and_missing_or_wrong_lr_rebuilt(self):
        for mode in ("cropped", "missing", "wrong_size"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, hr, lr, expected = self.fixture(root, size=(28, 20), mode=mode)
                _, _, report, rows = self.repair(root, path)
                self.assertEqual(rows[0]["hr_path"], "HR/" + hr.name)
                self.assertEqual(report["repaired_pairs"], 0 if mode == "cropped" else 1)
                with Image.open(root / rows[0]["lr_path"]) as fixed:
                    self.assertEqual(fixed.size, expected.size)
                    self.assertEqual(fixed.tobytes(), expected.tobytes())

    def test_existing_outputs_and_missing_hr_fail_before_mutating_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, hr, lr, _ = self.fixture(root)
            output, directory, _, _ = self.repair(root, path)
            saved = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                aligner.align_pairs(path, output, directory, root)
            self.assertEqual(output.read_bytes(), saved)
            hr.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing HR"):
                aligner.align_pairs(path, root / "new.csv", root / "new_images", root)
            self.assertFalse((root / "new.csv").exists())
            self.assertFalse((root / "new_images").exists())

    def test_decode_failure_cleans_staged_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, _, _ = self.fixture(root)
            original_open = aligner.Image.open
            calls = 0

            def fail_during_decode(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls > 2:
                    raise OSError("simulated failed image decode")
                return original_open(*args, **kwargs)

            from unittest.mock import patch
            with patch.object(aligner.Image, "open", side_effect=fail_during_decode):
                with self.assertRaisesRegex(OSError, "simulated"):
                    self.repair(root, path)
            self.assertFalse((root / "metadata/splits_X4_aligned.csv").exists())
            self.assertFalse((root / "data/aligned_X4").exists())
            self.assertFalse(list((root / "data").glob(".x4_align_*")))

    def test_new_csv_is_accepted_by_existing_x4_lut_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, _, _ = self.fixture(root)
            output, _, _, _ = self.repair(root, path)
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/09_run_srlut.py"),
                "--splits", str(output), "--split", "train", "--project-root", str(root),
                "--scale", "4", "--lut", str(ROOT / "weights/srlut_official/Model_S_x4_4bit_int8.npy"),
                "--output", str(root / "SR"), "--timings", str(root / "timings.csv"), "--warmup", "0",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            files = list((root / "SR").glob("*.png"))
            self.assertEqual(len(files), 1)
            with Image.open(files[0]) as image:
                self.assertEqual(image.size, (28, 16))


if __name__ == "__main__":
    unittest.main()
