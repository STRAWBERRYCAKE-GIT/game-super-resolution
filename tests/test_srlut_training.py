from __future__ import annotations

import csv
import importlib.util
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
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from srlut_training import (
        RandomPatches, SRNet, ensemble, load_parameters, predict_image,
        read_training_pairs, torch,
    )
    import importlib
    exporter = importlib.import_module("11_export_srlut")

INITIAL = ROOT / "weights/srlut_official/Model_S_x4_original_weights.npz"


@unittest.skipUnless(HAS_TORCH, "Install PyTorch to exercise training/export.")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def network(self):
        model = SRNet()
        load_parameters(model, INITIAL)
        return model

    def fixture(self, root):
        rows = []
        for index, split in enumerate(("train", "val", "test"), 1):
            name = f"person01_clip{index:03d}__f00000001.png"
            row = {"split": split, "game_id": "person01", "clip_id": f"person01_clip{index:03d}",
                   "lr_path": f"LR\\{name}", "hr_path": f"HR\\{name}"}
            rows.append(row)
            if split == "test":
                continue  # Prove the trainer never tries to open test images.
            low = np.random.default_rng(index).integers(0, 256, (8, 10, 3), dtype=np.uint8)
            high = np.repeat(np.repeat(low, 4, axis=0), 4, axis=1)
            for folder, image in (("LR", low), ("HR", high)):
                (root / folder).mkdir(exist_ok=True)
                Image.fromarray(image).save(root / folder / name)
        path = root / "splits.csv"
        self.write_csv(path, rows)
        return path, rows

    def write_csv(self, path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def run_script(self, name, *args, expected=0):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, args)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_network_matches_independent_numpy_forward_from_official_weights(self):
        patches = np.random.default_rng(5).random((31, 1, 2, 2), dtype=np.float32)
        with np.load(INITIAL, allow_pickle=False) as z:
            value = patches.reshape(31, 4) @ z["conv1.weight"].reshape(64, 4).T + z["conv1.bias"]
            for index in range(2, 7):
                value = np.maximum(value, 0) @ z[f"conv{index}.weight"].reshape(-1, 64).T + z[f"conv{index}.bias"]
        with torch.no_grad():
            actual = self.network()(torch.from_numpy(patches)).numpy()
        np.testing.assert_allclose(actual.reshape(31, 16), value, rtol=1e-5, atol=2e-6)

    def test_patch_alignment_and_deterministic_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.fixture(root)
            pairs = read_training_pairs(path, root, 3, 0)
            dataset = RandomPatches(pairs["train"], 3, 40, 42, 1)
            for index in range(16):
                low, high = dataset[index]
                np.testing.assert_array_equal(high.numpy(), low.numpy().repeat(4, axis=1).repeat(4, axis=2))
                np.testing.assert_array_equal(dataset[index][0].numpy(), low.numpy())

    def test_video_leakage_and_mismatched_dimensions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, rows = self.fixture(root)
            rows[2]["clip_id"] = "person01_clip001"
            rows[2]["hr_path"] = "HR/person01_clip001__f00000002.png"
            self.write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "leakage"):
                read_training_pairs(path, root, 3, 0)
            path, rows = self.fixture(root)
            name = Path(rows[0]["hr_path"].replace("\\", "/")).name
            Image.new("RGB", (41, 32)).save(root / "HR" / name)
            with self.assertRaisesRegex(ValueError, "x4 dimensions"):
                read_training_pairs(path, root, 3, 0)

    def test_tiled_validation_matches_full_network_with_four_rotations(self):
        model = self.network().eval()
        low = np.random.default_rng(8).integers(0, 256, (9, 13, 3), dtype=np.uint8)
        with torch.no_grad():
            tensor = torch.from_numpy(low.transpose(2, 0, 1).copy()).unsqueeze(0).float() / 255
            full = ensemble(model, tensor)[0].permute(1, 2, 0).numpy()
        expected = np.round(np.clip(full, 0, 1) * 255).astype(np.uint8)
        for tile in (1, 3, 5, 64):
            np.testing.assert_array_equal(predict_image(model, low, torch.device("cpu"), tile), expected)

    def test_export_grid_order_quantization_and_original_lut(self):
        model = self.network().eval()
        actual = exporter.export_table(model, torch.device("cpu"), 4, 2048)
        original = np.load(ROOT / "weights/srlut_official/Model_S_x4_4bit_int8.npy")
        delta = np.abs(actual.astype(np.int16) - original.astype(np.int16))
        self.assertEqual(actual.shape, (83521, 1, 4, 4))
        self.assertLessEqual(delta.max(), 1)
        self.assertLess(np.count_nonzero(delta), 1000)
        # Selected lattice entries also catch swapped a/b/c/d order.
        for index in (0, 1, 17, 4913, 83520):
            coordinates = [(index // 17 ** power) % 17 for power in (3, 2, 1, 0)]
            patch = np.minimum(np.array(coordinates) * 16, 255).astype(np.float32).reshape(1, 1, 2, 2)
            with torch.no_grad():
                value = torch.round(model(torch.from_numpy(patch) / 255).clamp(-1, 1) * 127).to(torch.int8).numpy()
            np.testing.assert_array_equal(actual[index], value[0])

    def test_train_resume_export_and_existing_lut_inference_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.fixture(root)
            output = root / "run"
            base = ["--splits", path, "--project-root", root, "--init", INITIAL,
                    "--output", output, "--batch-size", "2", "--patch-size", "3", "--device", "cpu",
                    "--val-every", "1", "--log-every", "1", "--threads", "1", "--val-tile", "5"]
            self.run_script("10_finetune_srlut.py", *base, "--steps", "2")
            last = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(last["step"], 2)
            with np.load(INITIAL) as initial:
                self.assertTrue(any(not np.array_equal(last["model_state_dict"][key].numpy(), initial[key])
                                    for key in initial.files))
            self.run_script("10_finetune_srlut.py", *base, "--steps", "3", "--resume", output / "last.pt")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["completed_updates"], 3)
            self.assertFalse(summary["test_used"])
            best = self.network()
            metadata = load_parameters(best, output / "best.pt")
            self.assertEqual(metadata["step"], summary["best_step"])
            lut = root / "finetuned.npy"
            self.run_script("11_export_srlut.py", "--checkpoint", output / "best.pt", "--output", lut,
                            "--device", "cpu", "--threads", "1")
            self.assertEqual(np.load(lut).shape, (83521, 1, 4, 4))
            self.run_script("09_run_srlut.py", "--splits", path, "--split", "val", "--project-root", root,
                            "--scale", "4", "--lut", lut, "--output", root / "SR", "--timings", root / "timings.csv",
                            "--warmup", "0", "--method", "srlut_finetuned")
            self.assertEqual(len(list((root / "SR").glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
