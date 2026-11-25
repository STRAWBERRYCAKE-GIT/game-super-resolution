from __future__ import annotations

import ast
import csv
import itertools
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
from srlut_numpy import load_lut, simplex_direction, upscale

# Compile only the supplied pure NumPy interpolation function. Importing the
# entire original script would run its dataset loop and import training helpers.
tree = ast.parse((ROOT / "tests/reference/Test_Model_S.py").read_text())
function = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "FourSimplexInterp")
namespace = {"np": np, "SAMPLING_INTERVAL": 4}
exec(compile(ast.fix_missing_locations(ast.Module(
    body=[function], type_ignores=[]
)), "supplied_official_interpolation", "exec"), namespace)
REFERENCE = namespace["FourSimplexInterp"]


def reference_direction(image, table, scale, rotation=0):
    h, w = image.shape[:2]
    padded = np.pad(
        image.astype(np.float32), ((0, 1), (0, 1), (0, 0)), mode="reflect"
    ).transpose(2, 0, 1)
    return REFERENCE(
        table.astype(np.float32), padded, h, w, 16, rotation, scale
    ).transpose(1, 2, 0)


def reference_upscale(image, table, scale):
    total = sum(
        reference_direction(np.rot90(image, k), table, scale, (-k) % 4)
        for k in range(4)
    )
    return np.round(np.clip(total / 255.0, 0, 1) * 255).astype(np.uint8)


class SrlutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = {
            scale: load_lut(
                ROOT / f"weights/srlut_official/Model_S_x{scale}_4bit_int8.npy", scale
            )
            for scale in (2, 4)
        }

    def test_all_24_orderings_match_supplied_reference(self):
        for scale in (2, 4):
            for order in itertools.permutations((12, 9, 6, 3)):
                with self.subTest(scale=scale, residuals=order):
                    values = (np.array(order) + [32, 64, 96, 128]).reshape(2, 2)
                    image = np.repeat(values[..., None], 3, axis=2).astype(np.uint8)
                    actual = simplex_direction(self.tables[scale], image, scale, tile_rows=1)
                    expected = reference_direction(image, self.tables[scale], scale)
                    np.testing.assert_array_equal(actual, expected)

    def test_ties_quantization_boundaries_and_extremes(self):
        for scale in (2, 4):
            for values in (
                (0, 0, 0, 0), (255, 255, 255, 255), (15, 16, 17, 31),
                (32, 32, 32, 48), (255, 254, 240, 239), (0, 255, 0, 255),
            ):
                with self.subTest(scale=scale, values=values):
                    image = np.repeat(np.array(values).reshape(2, 2, 1), 3, 2).astype(np.uint8)
                    np.testing.assert_array_equal(
                        upscale(image, self.tables[scale], scale, tile_rows=1),
                        reference_upscale(image, self.tables[scale], scale),
                    )

    def test_non_square_rgb_and_rotation_ensemble(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (7, 11, 3), dtype=np.uint8)
        for scale in (2, 4):
            np.testing.assert_array_equal(
                upscale(image, self.tables[scale], scale, tile_rows=3),
                reference_upscale(image, self.tables[scale], scale),
            )

    def test_tile_sizes_do_not_change_pixels(self):
        image = np.random.default_rng(7).integers(0, 256, (13, 9, 3), dtype=np.uint8)
        for scale in (2, 4):
            expected = upscale(image, self.tables[scale], scale, tile_rows=1)
            for tile in (3, 8, 32):
                np.testing.assert_array_equal(
                    expected, upscale(image, self.tables[scale], scale, tile_rows=tile)
                )

    def test_five_supplied_x4_example_outputs_match_pixel_for_pixel(self):
        test_table = load_lut(
            ROOT / "weights/srlut_official/Model_S_x4_test_directory_4bit_int8.npy", 4
        )
        for name in ("baby", "bird", "butterfly", "head", "woman"):
            with self.subTest(name=name):
                with Image.open(ROOT / f"tests/reference/LR/{name}.png") as image:
                    lr = np.array(image.convert("RGB"))
                with Image.open(ROOT / f"tests/reference/SR/{name}.png") as image:
                    expected = np.array(image.convert("RGB"))
                actual = upscale(lr, test_table, 4)
                # Official Set5 code crops its outputs to the supplied HR bounds.
                actual = actual[:expected.shape[0], :expected.shape[1]]
                np.testing.assert_array_equal(actual, expected)

    def test_wrong_lut_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "LUT size mismatch"):
            load_lut(ROOT / "weights/srlut_official/Model_S_x4_4bit_int8.npy", 2)

    def test_invalid_input_and_tile_settings(self):
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            upscale(image.astype(np.float32), self.tables[2], 2)
        with self.assertRaises(ValueError):
            upscale(image, self.tables[2], 2, tile_rows=0)
        with self.assertRaises(ValueError):
            upscale(np.zeros((1, 4, 3), dtype=np.uint8), self.tables[2], 2)

    def fixture(self, root, scale):
        rows = []
        for index, split in enumerate(("train", "val", "test", "test")):
            name = f"person01_clip{index + 1:03d}__f00000001.png"
            lr_path = root / "data" / f"LR_X{scale}" / name
            hr_path = root / "data" / "HR" / name
            lr_path.parent.mkdir(parents=True, exist_ok=True)
            hr_path.parent.mkdir(parents=True, exist_ok=True)
            lr = Image.fromarray(np.random.default_rng(index).integers(
                0, 256, (5, 7, 3), dtype=np.uint8))
            lr.save(lr_path)
            lr.resize((7 * scale, 5 * scale)).save(hr_path)
            rows.append({
                "split": split, "game_id": "person01", "clip_id": f"person01_clip{index + 1:03d}",
                "lr_path": str(lr_path.relative_to(root)).replace("/", "\\"),
                "hr_path": str(hr_path.relative_to(root)).replace("/", "\\"),
            })
        csv_path = root / "splits.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return csv_path, rows

    def cli(self, root, scale, csv_path, *extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/09_run_srlut.py"),
             "--splits", str(csv_path), "--scale", str(scale),
             "--lut", str(ROOT / f"weights/srlut_official/Model_S_x{scale}_4bit_int8.npy"),
             "--output", str(root / "output"), "--timings", str(root / "timings.csv"),
             "--project-root", str(root), "--warmup", "0", *extra],
            capture_output=True, text=True,
        )

    def test_csv_cli_x2_and_x4_uses_only_test_rows_and_keeps_names(self):
        for scale in (2, 4):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                csv_path, rows = self.fixture(root, scale)
                result = self.cli(root, scale, csv_path)
                self.assertEqual(result.returncode, 0, result.stderr)
                expected_names = {Path(row["hr_path"].replace("\\", "/")).name
                                  for row in rows if row["split"] == "test"}
                self.assertEqual({p.name for p in (root / "output").glob("*.png")}, expected_names)
                for name in expected_names:
                    with Image.open(root / "output" / name) as image:
                        self.assertEqual(image.size, (7 * scale, 5 * scale))
                with (root / "timings.csv").open(encoding="utf-8-sig", newline="") as file:
                    timing_rows = list(csv.DictReader(file))
                self.assertEqual(len(timing_rows), 2)
                self.assertTrue(all(float(row["time_ms"]) >= 0 for row in timing_rows))
                self.assertTrue(all(row["game_id"] == "person01" for row in timing_rows))
                metadata = json.loads((root / "timings.json").read_text())
                self.assertEqual(metadata["scale"], scale)
                self.assertEqual(metadata["images"], 2)

    def test_limit_runs_requested_number_of_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path, _ = self.fixture(root, 2)
            result = self.cli(root, 2, csv_path, "--limit", "1")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list((root / "output").glob("*.png"))), 1)

    def test_size_mismatch_is_not_hidden_by_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path, rows = self.fixture(root, 2)
            path = root / rows[2]["hr_path"].replace("\\", "/")
            Image.new("RGB", (15, 10)).save(path)
            result = self.cli(root, 2, csv_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Scale/size mismatch", result.stderr)
            self.assertFalse((root / "output").exists())

    def test_empty_test_set_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path, _ = self.fixture(root, 2)
            with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["split", "lr_path", "hr_path"])
            result = self.cli(root, 2, csv_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No rows", result.stderr)


if __name__ == "__main__":
    unittest.main()

