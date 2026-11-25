"""Shared x4 network, paired-data reader and validation for game fine-tuning."""
from __future__ import annotations

import csv
import hashlib
import math
import random
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as error:
    raise SystemExit(
        "PyTorch is required for training/export. See README_X4微调操作.md. "
        "The existing NumPy LUT test does not require PyTorch."
    ) from error


class SRNet(nn.Module):
    """Same six layers and channel sharing as the supplied official Ours-S."""
    def __init__(self):
        super().__init__()
        self.upscale = 4
        self.conv1 = nn.Conv2d(1, 64, 2)
        for index in range(2, 6):
            setattr(self, f"conv{index}", nn.Conv2d(64, 64, 1))
        self.conv6 = nn.Conv2d(64, 16, 1)
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, image):
        batch, channels, height, width = image.shape
        value = self.conv1(image.reshape(batch * channels, 1, height, width))
        for index in range(2, 7):
            value = getattr(self, f"conv{index}")(F.relu(value))
        return self.pixel_shuffle(value).reshape(
            batch, channels, 4 * (height - 1), 4 * (width - 1)
        )


def ensemble(model, image):
    directions = []
    for rotation in range(4):
        rotated = torch.rot90(image, rotation, (2, 3))
        value = model(F.pad(rotated, (0, 1, 0, 1), mode="reflect"))
        value = torch.rot90(value, -rotation, (2, 3))
        directions.append(torch.clamp(value, -1, 1) * 127)
    # Preserve the original grouping and normalization; no extra division by 4.
    value = directions[0] + directions[1]
    value = value + (directions[2] + directions[3])
    return value / 255.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_parameters(model, path):
    """NPZ initial weights or this adapter's state-dict-only checkpoint."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            expected = model.state_dict()
            if set(arrays.files) != set(expected):
                raise ValueError("Initial NPZ must contain exactly conv1--conv6 weights/biases.")
            parameters = {}
            for key, target in expected.items():
                value = arrays[key]
                if value.shape != tuple(target.shape) or value.dtype != np.float32:
                    raise ValueError(f"Wrong shape/dtype for {key}: {value.shape}, {value.dtype}")
                if not np.isfinite(value).all():
                    raise ValueError(f"Non-finite initial parameter: {key}")
                parameters[key] = torch.from_numpy(value.copy())
        metadata = None
    else:
        # Do not load the legacy whole-object Model_S.pth here.
        metadata = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(metadata, dict) or metadata.get("format") != "srlut_game_x4_v1":
            raise ValueError("Use the supplied NPZ or a checkpoint produced by 10_finetune_srlut.py.")
        if metadata.get("scale") != 4:
            raise ValueError("Expected an x4 network checkpoint.")
        parameters = metadata["model_state_dict"]
    model.load_state_dict(parameters, strict=True)
    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
        raise ValueError("Checkpoint contains non-finite parameters.")
    return metadata


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment. Use --device cpu or a CUDA PyTorch environment.")
    return torch.device(name)


def resolve_image(value, root):
    path = Path(value.replace("\\", "/"))
    if not path.is_absolute() and not (len(value) > 1 and value[1] == ":"):
        path = root / path
    if not path.is_file():
        raise ValueError(f"Missing image: {path}. Remap CSV paths when moving machines; preserve split assignments.")
    return path


def read_training_pairs(csv_path, root, patch_size, shave):
    """Check all split identities, but open image files only for train/val."""
    groups = {"train": [], "val": []}
    assignments, names = {}, set()
    pattern = re.compile(r"(?P<game>person\d+)_(?P<video>clip\d+)__f\d+", re.I)
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"split", "lr_path", "hr_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV needs columns: {sorted(required)}")
        for line, row in enumerate(reader, 2):
            split = row["split"]
            if split not in ("train", "val", "test"):
                raise ValueError(f"Unknown split at CSV line {line}: {split!r}")
            name = Path(row["hr_path"].replace("\\", "/")).stem
            match = pattern.fullmatch(name)
            if not match:
                raise ValueError(f"Expected person01_clip001__f00000001 naming: {name}")
            game = match["game"]
            video = game + "_" + match["video"]
            if row.get("game_id") and row["game_id"] != game:
                raise ValueError(f"game_id contradicts filename at line {line}")
            if row.get("clip_id") and row["clip_id"] != video:
                raise ValueError(f"clip_id must identify the original video ({video}), line {line}")
            if assignments.setdefault(video.casefold(), split) != split:
                raise ValueError(f"Original-video leakage across splits: {video}")
            if name.casefold() in names:
                raise ValueError(f"Duplicate frame in CSV: {name}")
            names.add(name.casefold())
            if split == "test":
                continue
            lr = resolve_image(row["lr_path"], root)
            hr = resolve_image(row["hr_path"], root)
            if lr.stem != name:
                raise ValueError(f"LR/HR filename mismatch at line {line}")
            with Image.open(lr) as low, Image.open(hr) as high:
                if high.size != (4 * low.width, 4 * low.height):
                    raise ValueError(
                        f"Expected exact x4 dimensions: {name}, LR={low.size}, HR={high.size}. "
                        "Run 12_align_x4_pairs.py on the original CSV, then use the aligned CSV for every x4 method."
                    )
                if min(low.size) < 2:
                    raise ValueError(f"LR image must be at least 2x2: {name}")
                if split == "train" and min(low.size) < patch_size:
                    raise ValueError(f"Training frame smaller than LR patch size {patch_size}: {name}")
                if split == "val" and min(high.size) <= 2 * shave:
                    raise ValueError(f"Validation shave removes the entire image: {name}")
            groups[split].append({
                "lr_path": lr, "hr_path": hr, "image": name + ".png",
                "game_id": game, "clip_id": video,
            })
    if not groups["train"] or not groups["val"]:
        raise ValueError("Both train and val rows are required. Test images are not used for training/selection.")
    return groups


def read_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


class RandomPatches(torch.utils.data.Dataset):
    """An index fixes frame/crop/augmentation, including after resume and on Windows."""
    def __init__(self, pairs, patch_size, samples, seed, cache_pairs=8):
        self.pairs, self.patch_size = pairs, patch_size
        self.samples, self.seed, self.cache_pairs = samples, seed, cache_pairs
        self.cache = OrderedDict()

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = random.Random(f"{self.seed}:{index}")
        frame = rng.randrange(len(self.pairs))
        if frame in self.cache:
            low, high = self.cache.pop(frame)
        else:
            pair = self.pairs[frame]
            low, high = read_rgb(pair["lr_path"]), read_rgb(pair["hr_path"])
        if self.cache_pairs:
            self.cache[frame] = (low, high)
            while len(self.cache) > self.cache_pairs:
                self.cache.popitem(last=False)
        size = self.patch_size
        top = rng.randrange(low.shape[0] - size + 1)
        left = rng.randrange(low.shape[1] - size + 1)
        low = low[top:top + size, left:left + size]
        high = high[top * 4:(top + size) * 4, left * 4:(left + size) * 4]
        if rng.random() < 0.5:
            low, high = low[:, ::-1], high[:, ::-1]
        rotation = rng.randrange(4)
        low, high = np.rot90(low, rotation), np.rot90(high, rotation)
        return tuple(torch.from_numpy(np.ascontiguousarray(
            image.transpose(2, 0, 1), dtype=np.float32
        )) / 255.0 for image in (low, high))


@torch.inference_mode()
def predict_image(model, low, device, tile_size=64):
    """One-pixel halo covers all four rotations of the 2x2 receptive field."""
    height, width = low.shape[:2]
    output = np.empty((height * 4, width * 4, 3), dtype=np.float32)
    for top in range(0, height, tile_size):
        bottom = min(height, top + tile_size)
        y0, y1 = max(0, top - 1), min(height, bottom + 1)
        for left in range(0, width, tile_size):
            right = min(width, left + tile_size)
            x0, x1 = max(0, left - 1), min(width, right + 1)
            array = np.ascontiguousarray(low[y0:y1, x0:x1].transpose(2, 0, 1), dtype=np.float32)
            tensor = torch.from_numpy(array).unsqueeze(0).to(device) / 255.0
            value = ensemble(model, tensor)[0].permute(1, 2, 0).cpu().numpy()
            dy, dx = 4 * (top - y0), 4 * (left - x0)
            output[top * 4:bottom * 4, left * 4:right * 4] = value[
                dy:dy + 4 * (bottom - top), dx:dx + 4 * (right - left)
            ]
    return np.round(np.clip(output, 0, 1) * 255).astype(np.uint8)


def validate(model, pairs, device, tile_size, shave):
    model.eval()
    rows = []
    for index, pair in enumerate(pairs, 1):
        low, high = read_rgb(pair["lr_path"]), read_rgb(pair["hr_path"])
        prediction = predict_image(model, low, device, tile_size)
        if shave:
            high = high[shave:-shave, shave:-shave]
            prediction = prediction[shave:-shave, shave:-shave]
        mse = float(np.mean((high.astype(np.float64) - prediction.astype(np.float64)) ** 2))
        psnr = math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)
        rows.append({"image": pair["image"], "game_id": pair["game_id"],
                     "clip_id": pair["clip_id"], "psnr": psnr})
        if index == 1 or index % 10 == 0 or index == len(pairs):
            print(f"Validation: {index}/{len(pairs)}", flush=True)
    return float(np.mean([row["psnr"] for row in rows])), rows


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
