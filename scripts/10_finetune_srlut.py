from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from srlut_training import (
    RandomPatches, SRNet, ensemble, load_parameters, read_training_pairs,
    select_device, sha256, torch, validate, write_rows,
)


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the supplied x4 SR-LUT network on game CSV pairs.")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--init", type=Path, default=Path("weights/srlut_official/Model_S_x4_original_weights.npz"))
    parser.add_argument("--output", type=Path, default=Path("weights/srlut_game_X4"))
    parser.add_argument("--steps", type=int, default=5000, help="Total updates, including completed updates when resuming.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=48, help="LR crop width/height; HR crop is four times larger.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-tile", type=int, default=64)
    parser.add_argument("--val-limit", type=int, help="Smoke checks only: validate this many val frames. Omit for the formal run.")
    parser.add_argument("--shave", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-pairs", type=int, default=8, help="Decoded LR/HR pairs cached per data worker.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--resume", type=Path, help="Resume this run's last.pt; reuse the same training options.")
    args = parser.parse_args()
    if any(v < 1 for v in (args.steps, args.batch_size, args.val_every, args.log_every, args.val_tile, args.threads)):
        parser.error("Steps, batch-size, intervals, val-tile and threads must be positive.")
    if args.patch_size < 2 or args.shave < 0 or args.workers < 0 or args.cache_pairs < 0 or not (0 < args.lr < math.inf):
        parser.error("Invalid patch-size, shave, workers, cache-pairs or learning rate.")
    if args.val_limit is not None and args.val_limit < 1:
        parser.error("val-limit must be positive.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    # These 1x1/2x2 convolutions do not need TF32; full float32 aids reference comparison.
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = select_device(args.device)
    print(f"Device: {device}; PyTorch {torch.__version__}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    else:
        print("CPU training selected. Use a CUDA PyTorch environment for faster training.", flush=True)
    pairs = read_training_pairs(args.splits, args.project_root.resolve(), args.patch_size, args.shave)
    if args.val_limit is not None:
        pairs["val"] = pairs["val"][:args.val_limit]
        print("Validation subset selected for a smoke check; omit --val-limit for formal experiments.", flush=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"splits_sha256": sha256(args.splits), "scale": 4,
                   "train_frames": len(pairs["train"]), "val_frames": len(pairs["val"]),
                   "selection": "mean per-frame RGB uint8 CNN validation PSNR",
                   "torch": str(torch.__version__), "actual_device": str(device)})
    output = args.output.resolve()
    config_path = output / "config.json"
    if args.resume is None and output.exists() and any(output.iterdir()):
        parser.error(f"Output is not empty: {output}. Use a new directory, or --resume with this run's last.pt.")
    if args.resume is not None and not config_path.is_file():
        parser.error("Resume requires the original run directory in --output, including config.json and best.pt.")
    model = SRNet()
    if args.resume:
        checkpoint = load_parameters(model, args.resume)
        old = checkpoint["config"]
        for key in ("splits_sha256", "batch_size", "patch_size", "lr", "seed", "shave", "val_limit"):
            if config[key] != old[key]:
                parser.error(f"Resume option mismatch: {key}. Keep original training/data options.")
        if args.resume.resolve() != output / "last.pt" or not (output / "best.pt").is_file():
            parser.error("Resume this output directory's last.pt and retain its best.pt.")
        config["init_sha256"] = old["init_sha256"]
    else:
        checkpoint = load_parameters(model, args.init)
        if checkpoint is not None:
            parser.error("For a new run use the supplied initial NPZ. Use --resume to continue last.pt.")
        config["init_sha256"] = sha256(args.init)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step = 0 if checkpoint is None else checkpoint["step"]
    if step >= args.steps:
        parser.error(f"--steps must exceed completed updates ({step}).")
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    output.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Train: {len(pairs['train'])} frames; val: {len(pairs['val'])}; test pixels are not read.", flush=True)

    def save(path, best_psnr, best_step):
        atomic_save({"format": "srlut_game_x4_v1", "scale": 4, "step": step,
                     "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                     "optimizer_state_dict": optimizer.state_dict(), "best_psnr": best_psnr,
                     "best_step": best_step, "config": config}, path)

    if checkpoint is None:
        print("Validating the unchanged initial network (step 0)...", flush=True)
        best_psnr, rows = validate(model, pairs["val"], device, args.val_tile, args.shave)
        best_step = 0
        write_rows(output / "val_step000000.csv", rows)
        save(output / "best.pt", best_psnr, best_step)
        save(output / "last.pt", best_psnr, best_step)
        print(f"Initial validation PSNR: {best_psnr:.6f} dB", flush=True)
    else:
        best_psnr, best_step = checkpoint["best_psnr"], checkpoint["best_step"]
        print(f"Resuming after step {step}; best step: {best_step}", flush=True)
    dataset = RandomPatches(pairs["train"], args.patch_size, args.steps * args.batch_size,
                            args.seed, args.cache_pairs)
    subset = torch.utils.data.Subset(dataset, range(step * args.batch_size, len(dataset)))
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=device.type == "cuda")
    log_path = output / "training.csv"
    with log_path.open("a" if args.resume else "w", encoding="utf-8", newline="") as file:
        logger = csv.DictWriter(file, fieldnames=["step", "train_mse", "val_psnr", "best_step"])
        if not args.resume:
            logger.writeheader()
            logger.writerow({"step": 0, "val_psnr": best_psnr, "best_step": 0})
            file.flush()
        for low, high in loader:
            model.train()
            low, high = low.to(device, non_blocking=True), high.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((ensemble(model, low) - high) ** 2)
            if not torch.isfinite(loss).item():
                raise ValueError(f"Non-finite loss at update {step + 1}; previous last.pt remains available.")
            loss.backward()
            optimizer.step()
            step += 1
            value = loss.item()
            should_validate = step % args.val_every == 0 or step == args.steps
            val_score = ""
            if should_validate:
                val_score, rows = validate(model, pairs["val"], device, args.val_tile, args.shave)
                write_rows(output / "val_latest.csv", rows)
                if val_score > best_psnr:
                    best_psnr, best_step = val_score, step
                    save(output / "best.pt", best_psnr, best_step)
                    write_rows(output / "val_best.csv", rows)
                print(f"Step {step}: MSE={value:.8f}; val PSNR={val_score:.6f}; best step={best_step}", flush=True)
            elif step == 1 or step % args.log_every == 0:
                print(f"Step {step}/{args.steps}: MSE={value:.8f}", flush=True)
            if should_validate or step == 1 or step % args.log_every == 0:
                logger.writerow({"step": step, "train_mse": value, "val_psnr": val_score, "best_step": best_step})
                file.flush()
                save(output / "last.pt", best_psnr, best_step)
    summary = {"best_step": best_step, "best_val_psnr": str(best_psnr) if not math.isfinite(best_psnr) else best_psnr,
               "completed_updates": step, "selection": config["selection"],
               "validation_subset": args.val_limit is not None,
               "test_used": False, "best_checkpoint": str(output / "best.pt")}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Finished. Export: {output / 'best.pt'}", flush=True)
    if best_step == 0:
        print("Validation did not improve over initialization: best.pt retains step 0. Do not claim a fine-tuning gain.", flush=True)


if __name__ == "__main__":
    main()
