"""Train the 48M byte student with EOT, Marginalize-It or byte-CE."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .model import ByteLM, ModelConfig, parameter_count
from .shards import EOT, TargetDataset


@dataclass
class TrainConfig:
    targets: str
    out: str
    objective: str = "eot"
    steps: int = 0
    epochs: int = 12
    batch: int = 16
    seq_len: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_fraction: float = 0.05
    min_lr_fraction: float = 0.1
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    max_windows: int = 0
    overfit: bool = False
    amp: bool = True
    seed: int = 1337
    eval_every: int = 100
    save_every: int = 500
    device: str = "cuda"


class WindowData(Dataset):
    def __init__(self, base: TargetDataset, indices: np.ndarray) -> None:
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        symbols, targets, nexts = self.base[int(self.indices[i])]
        return torch.from_numpy(symbols), torch.from_numpy(targets), torch.from_numpy(nexts)


def objective_loss(logits: torch.Tensor, targets: torch.Tensor, nexts: torch.Tensor, objective: str):
    logp = F.log_softmax(logits.float(), dim=-1)
    t = targets.float().clamp_min(0)
    if objective == "eot":
        return (torch.special.xlogy(t, t) - t * logp).sum(-1).mean()
    if objective == "ce":
        return -logp.gather(-1, nexts.unsqueeze(-1)).squeeze(-1).mean()
    if objective == "mi":
        shift = torch.zeros_like(t)
        shift[:, :-1] = t[:, 1:]
        use_shift = (nexts == EOT).unsqueeze(-1)
        t_mi = torch.where(use_shift, shift, t)
        b = t_mi[..., :256]
        total = b.sum(-1, keepdim=True)
        b = b / total.clamp_min(1e-30)
        kl = (torch.special.xlogy(b, b) - b * logp[..., :256]).sum(-1)
        mask = total.squeeze(-1) > 0
        if not bool(mask.any()):
            return logits.sum() * 0.0
        return kl[mask].mean()
    raise ValueError(objective)


@torch.no_grad()
def evaluate(model: ByteLM, loader: DataLoader, objective: str, device: str) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    bits = 0.0
    eot_bits = 0.0
    n_bytes = 0
    top1 = 0
    n_tokens = 0
    for symbols, targets, nexts in loader:
        symbols, targets, nexts = symbols.to(device), targets.to(device), nexts.to(device)
        logits = model(symbols)
        total_loss += float(objective_loss(logits, targets, nexts, objective))
        n_batches += 1
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(-1, nexts.unsqueeze(-1)).squeeze(-1)
        byte_mask = nexts < 256
        bits += float(nll[byte_mask].sum())
        eot_bits += float(nll[~byte_mask].sum())
        n_bytes += int(byte_mask.sum())
        top1 += int((logits.argmax(-1) == nexts).sum())
        n_tokens += nexts.numel()
    model.train()
    return {
        "loss": total_loss / max(n_batches, 1),
        "bpb": bits / math.log(2) / max(n_bytes, 1),
        "bpb_eot_inclusive": (bits + eot_bits) / math.log(2) / max(n_bytes, 1),
        "next_top1": top1 / max(n_tokens, 1),
        "bytes": n_bytes,
    }


def make_loader(dataset: WindowData, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch,
        shuffle=shuffle,
        num_workers=0,
        drop_last=shuffle,
        pin_memory=False,
    )


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    warmup = max(1, int(total * cfg.warmup_fraction))
    min_lr = cfg.lr * cfg.min_lr_fraction
    if step < warmup:
        return cfg.lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + (cfg.lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objective", default="eot", choices=["eot", "mi", "ce"])
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-fraction", type=float, default=0.05)
    ap.add_argument("--min-lr-fraction", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = TrainConfig(
        targets=args.targets,
        out=args.out,
        objective=args.objective,
        steps=args.steps,
        epochs=args.epochs,
        batch=args.batch,
        seq_len=args.seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction,
        min_lr_fraction=args.min_lr_fraction,
        grad_clip=args.grad_clip,
        val_fraction=args.val_fraction,
        max_windows=args.max_windows,
        overfit=args.overfit,
        amp=not args.no_amp,
        seed=args.seed,
        eval_every=args.eval_every,
        save_every=args.save_every,
        device=args.device,
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    base = TargetDataset(cfg.targets, seq_len=cfg.seq_len)
    if len(base) == 0:
        raise SystemExit("no full windows in target shards")
    n = len(base) if not cfg.max_windows else min(len(base), cfg.max_windows)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(base))[:n]
    if cfg.overfit:
        train_idx = order
        val_idx = order
    else:
        n_val = max(1, int(n * cfg.val_fraction))
        val_idx, train_idx = order[:n_val], order[n_val:]
    train_ds = WindowData(base, train_idx)
    val_ds = WindowData(base, val_idx)
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)

    model = ByteLM(ModelConfig(max_seq_len=cfg.seq_len)).to(cfg.device)
    n_params = parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and cfg.device.startswith("cuda"))

    total_steps = cfg.steps if cfg.steps else cfg.epochs * max(1, len(train_loader))
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt["step"])
        print(f"resumed at step {start_step}")

    log_path = out / "train.jsonl"
    log = log_path.open("a")
    model.train()
    step = start_step
    t0 = time.time()
    tokens_seen = 0
    running = 0.0
    running_n = 0
    print(json.dumps({"params": n_params, "windows": len(base), "train": len(train_ds),
                      "val": len(val_ds), "total_steps": total_steps}, indent=2))

    while step < total_steps:
        for symbols, targets, nexts in train_loader:
            if step >= total_steps:
                break
            lr = lr_at(step, total_steps, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            symbols = symbols.to(cfg.device, non_blocking=True)
            targets = targets.to(cfg.device, non_blocking=True)
            nexts = nexts.to(cfg.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=cfg.amp and cfg.device.startswith("cuda")):
                logits = model(symbols)
            loss = objective_loss(logits, targets, nexts, cfg.objective)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            tokens_seen += symbols.numel()
            running += float(loss.detach())
            running_n += 1

            if step % 20 == 0:
                elapsed = time.time() - t0
                rec = {
                    "step": step,
                    "loss": running / running_n,
                    "lr": lr,
                    "tokens_per_s": round(tokens_seen / elapsed, 1),
                    "vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                    if torch.cuda.is_available() else None,
                }
                print(json.dumps(rec))
                log.write(json.dumps(rec) + "\n")
                log.flush()
                running = 0.0
                running_n = 0

            if step % cfg.eval_every == 0 or step == total_steps:
                metrics = evaluate(model, val_loader, cfg.objective, cfg.device)
                rec = {"step": step, "eval": metrics}
                print(json.dumps(rec))
                log.write(json.dumps(rec) + "\n")
                log.flush()

            if step % cfg.save_every == 0 or step == total_steps:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step": step,
                        "config": asdict(cfg),
                    },
                    out / "ckpt.pt",
                )

    model.eval()
    torch.save({"model": model.state_dict(), "step": step, "config": asdict(cfg)}, out / "final.pt")
    log.close()
    print(json.dumps({"done": step, "params": n_params, "elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
