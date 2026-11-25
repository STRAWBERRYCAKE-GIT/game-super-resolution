from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from srlut_training import SRNet, load_parameters, select_device, sha256, torch


@torch.inference_mode()
def export_table(model, device, sampling_bits=4, batch_size=2048):
    q = 2 ** sampling_bits
    levels = 2 ** (8 - sampling_bits) + 1
    count = levels ** 4
    table = np.empty((count, 1, 4, 4), dtype=np.int8)
    model.eval()
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        index = np.arange(start, end, dtype=np.int64)
        # Same lexicographic a,b,c,d order and final 255 sample as official export.
        coordinates = np.stack([(index // levels ** power) % levels
                                for power in (3, 2, 1, 0)], axis=1)
        patches = np.minimum(coordinates * q, 255).astype(np.float32).reshape(-1, 1, 2, 2)
        tensor = torch.from_numpy(patches).to(device) / 255.0
        table[start:end] = torch.round(torch.clamp(model(tensor), -1, 1) * 127).to(torch.int8).cpu().numpy()
    return table


def main():
    parser = argparse.ArgumentParser(description="Export the x4 network to an int8 LUT for 09_run_srlut.py.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling-bits", type=int, choices=range(3, 9), default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive.")
    if args.output.suffix.lower() != ".npy":
        parser.error("--output must end with .npy")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Output already exists. Choose a new filename to retain prior LUTs and provenance.")
    torch.set_num_threads(args.threads)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = select_device(args.device)
    model = SRNet()
    checkpoint = load_parameters(model, args.checkpoint)
    model.to(device)
    print(f"Exporting x4 network on {device}, spacing={2 ** args.sampling_bits}", flush=True)
    table = export_table(model, device, args.sampling_bits, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".npy.tmp")
    with temporary.open("wb") as file:
        np.save(file, table, allow_pickle=False)
    os.replace(temporary, args.output)
    metadata = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
                "checkpoint_step": None if checkpoint is None else checkpoint["step"],
                "scale": 4, "sampling_bits": args.sampling_bits, "spacing": 2 ** args.sampling_bits,
                "shape": list(table.shape), "dtype": "int8", "lut_sha256": sha256(args.output),
                "device": str(device), "torch": str(torch.__version__),
                "quantization": "round(clamp(single-direction network output,-1,1)*127)",
                "test_used": False}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {table.shape}, int8 -> {args.output}", flush=True)
    if checkpoint is not None and checkpoint["step"] == 0:
        print("This is the unchanged initialization (best step 0), not an improved fine-tuned model.", flush=True)


if __name__ == "__main__":
    main()
