import argparse
import csv
import itertools
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from dataset import RewardPairCollator, RewardPairDataset, WINNER_TO_LABEL, load_fold_split
from reward_model import RewardModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        # explicit override from config, e.g. "cpu" to work around the
        # DeBERTa/MPS crash (MPSNDArrayMatrixMultiplication dtype assert)
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_param_groups(model: RewardModel, lr: float, head_lr: float, delta_lr: float, weight_decay: float):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    encoder_decay, encoder_no_decay = [], []

    for name, param in model.encoder.named_parameters():
        if not param.requires_grad:
            continue
        (encoder_no_decay if any(nd in name for nd in no_decay) else encoder_decay).append(param)

    return [
        {"params": encoder_decay, "lr": lr, "weight_decay": weight_decay},
        {"params": encoder_no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": list(model.reward_head.parameters()), "lr": head_lr, "weight_decay": 0.0},
        {"params": [model.delta], "lr": delta_lr, "weight_decay": 0.0},
    ]


def multiclass_log_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], eps, 1 - eps)
    return float(-np.log(p).mean())


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_loss):
    """Epoch-boundary checkpoint: full training state, saved once per completed epoch."""
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt["scaler_state"] is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["epoch"], ckpt["best_val_loss"]


def run_epoch(
    model, loader, device, optimizer, scheduler, scaler, precision, max_grad_norm,
    log_every_n_steps, train: bool, grad_accum_steps: int = 1,
    skip_batches: int = 0, step_checkpoint_path=None,
    checkpoint_every_n_steps: int = 0, epoch_idx: int = 0, progress_fns=None,
):
    """progress_fns: optional (load_progress, save_progress) pair, only needed
    if a caller wants to touch progress.json from inside the loop; unused here
    but kept pluggable since the notebook cell leaves a hook for it.
    """
    model.train(mode=train)
    total_loss, n_batches = 0.0, 0
    all_probs, all_labels = [], []

    use_amp = precision == "fp16" and device.type == "cuda"
    autocast_dtype = torch.float16 if precision == "fp16" else torch.float32

    if train:
        optimizer.zero_grad(set_to_none=True)

    iterator = enumerate(loader)
    if skip_batches > 0:
        print(f"    [resume] skipping {skip_batches} already-processed batches")
        iterator = itertools.islice(iterator, skip_batches, None)

    for step, batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                out = model(input_ids, attention_mask)
                loss = F.cross_entropy(out["class_logits"], labels)

        if train:
            loss_to_backward = loss / grad_accum_steps
            if use_amp:
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            if (step + 1) % grad_accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if (checkpoint_every_n_steps and step_checkpoint_path
                    and (step + 1) % checkpoint_every_n_steps == 0):
                torch.save({
                    "epoch": epoch_idx, "next_batch_idx": step + 1,
                    "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler is not None else None,
                }, step_checkpoint_path)
                print(f"    [checkpoint] saved mid-epoch state at batch {step + 1}")

        total_loss += loss.item()
        n_batches += 1

        probs = torch.stack([out["p_a"], out["p_b"], out["p_tie"]], dim=1).detach()
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

        if train and log_every_n_steps and (step + 1) % log_every_n_steps == 0:
            print(f"    step {step + 1}/{len(loader)} | loss {total_loss / n_batches:.4f} "
                  f"| delta {model.delta.item():.4f}")

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return total_loss / max(n_batches, 1), all_probs, all_labels


def run_fold(cfg: dict, fold: int, tokenizer, device,
             resume_epoch: int = 0, resume_skip_batches: int = 0,
             metrics_path=None, progress_path=None):
    print(f"\n=== Fold {fold} ===")
    train_df, val_df = load_fold_split(cfg["data"]["path"], fold, cfg["data"]["fold_col"])
    print(f"[data] train {len(train_df)} rows | val {len(val_df)} rows")

    col_kwargs = dict(
        prompt_col=cfg["data"]["prompt_col"],
        response_a_col=cfg["data"]["response_a_col"],
        response_b_col=cfg["data"]["response_b_col"],
        winner_col=cfg["data"]["winner_col"],
    )
    train_ds = RewardPairDataset(train_df, tokenizer, cfg["model"]["max_length"], **col_kwargs)
    val_ds = RewardPairDataset(val_df, tokenizer, cfg["model"]["max_length"], **col_kwargs)
    collator = RewardPairCollator(tokenizer)

    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["eval_batch_size"], shuffle=False,
        collate_fn=collator, num_workers=cfg["train"]["num_workers"],
    )

    model = RewardModel(
        backbone=cfg["model"]["backbone"],
        pooling=cfg["model"]["pooling"],
        dropout=cfg["model"]["dropout"],
        delta_init=cfg["model"]["delta_init"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg["train"]["lr"], cfg["train"]["head_lr"], cfg["train"]["delta_lr"], cfg["train"]["weight_decay"])
    )

    n_steps_per_epoch = len(train_ds) // cfg["train"]["batch_size"] // cfg["train"]["grad_accum_steps"]
    n_steps = n_steps_per_epoch * cfg["train"]["epochs"]
    n_warmup = int(n_steps * cfg["train"]["warmup_ratio"])
    scheduler_fn = get_cosine_schedule_with_warmup if cfg["train"]["scheduler"] == "cosine" \
        else get_linear_schedule_with_warmup
    scheduler = scheduler_fn(optimizer, num_warmup_steps=n_warmup, num_training_steps=n_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg["train"]["precision"] == "fp16" and device.type == "cuda"))

    fold_dir = Path(cfg["output"]["dir"]) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_dir / "checkpoint.pt"            # epoch-boundary checkpoint
    step_checkpoint_path = fold_dir / "step_checkpoint.pt"  # mid-epoch checkpoint
    best_path = fold_dir / "best.pt"

    best_val_loss = float("inf")
    start_epoch = 0
    skip_batches = 0

    if resume_epoch > 0 and checkpoint_path.exists():
        loaded_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device)
        start_epoch = loaded_epoch + 1
        print(f"[resume] fold {fold}: resuming from epoch boundary {start_epoch} "
              f"(best_val_loss so far: {best_val_loss:.4f})")

    if resume_skip_batches > 0 and step_checkpoint_path.exists():
        ckpt = torch.load(step_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if scaler is not None and ckpt["scaler_state"] is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"]
        skip_batches = ckpt["next_batch_idx"]
        print(f"[resume] fold {fold}: resuming mid-epoch {start_epoch + 1}, skipping first {skip_batches} batches")

    best_val_probs, best_val_labels = None, None

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        t0 = time.time()
        epoch_skip = skip_batches if epoch == start_epoch else 0

        # Seeded per (fold, epoch) so a resumed run can reproduce the exact
        # same shuffle order and safely skip the batches already trained on.
        train_gen = torch.Generator().manual_seed(1000 + fold * 100 + epoch)
        train_loader = DataLoader(
            train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
            collate_fn=collator, num_workers=cfg["train"]["num_workers"],
            drop_last=True, generator=train_gen,
        )

        train_loss, _, _ = run_epoch(
            model, train_loader, device, optimizer, scheduler, scaler,
            cfg["train"]["precision"], cfg["train"]["max_grad_norm"], cfg["train"]["log_every_n_steps"], train=True,
            grad_accum_steps=cfg["train"]["grad_accum_steps"], skip_batches=epoch_skip,
            step_checkpoint_path=step_checkpoint_path,
            checkpoint_every_n_steps=cfg["train"]["checkpoint_every_n_steps"], epoch_idx=epoch,
        )
        val_loss, val_probs, val_labels = run_epoch(
            model, val_loader, device, optimizer=None, scheduler=None, scaler=None,
            precision=cfg["train"]["precision"], max_grad_norm=None,
            log_every_n_steps=None, train=False,
        )
        val_ll = multiclass_log_loss(val_probs, val_labels)
        dt = time.time() - t0

        print(f"[fold {fold}][epoch {epoch + 1}/{cfg['train']['epochs']}] "
              f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_log_loss {val_ll:.4f} "
              f"| delta {model.delta.item():.4f} | {dt:.0f}s")

        append_metrics_row(metrics_path, {
            "fold": fold, "epoch": epoch + 1, "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6), "val_log_loss": round(val_ll, 6),
            "delta": round(model.delta.item(), 6), "elapsed_sec": round(dt, 1),
        })

        if val_ll < best_val_loss:
            best_val_loss = val_ll
            best_val_probs, best_val_labels = val_probs, val_labels
            torch.save(model.state_dict(), best_path)

        save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch, best_val_loss)
        if step_checkpoint_path.exists():
            step_checkpoint_path.unlink()  # epoch finished cleanly, mid-epoch checkpoint is now stale

        progress = load_progress(progress_path)
        progress["current_fold"] = fold
        progress["current_fold_completed_epochs"] = epoch + 1
        save_progress(progress_path, progress)

    final_delta = model.delta.item()
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return {
        "fold": fold,
        "val_log_loss": best_val_loss,
        "val_probs": best_val_probs,
        "val_labels": best_val_labels,
        "val_ids": val_df["id"].to_numpy() if "id" in val_df.columns else None,
        "delta": final_delta,
    }


def append_metrics_row(metrics_path: Path, row: dict):
    is_new = not metrics_path.exists()
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {"completed_folds": [], "current_fold": None, "current_fold_completed_epochs": 0}


def save_progress(progress_path: Path, progress: dict):
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    print(f"[setup] device={device} | config={config_path}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_history.csv"
    progress_path = out_dir / "progress.json"

    resume_from = cfg.get("resume_from")
    if resume_from:
        resume_from = Path(resume_from)
        if resume_from.exists():
            print(f"[resume] copying previous progress from {resume_from} into {out_dir}")
            shutil.copytree(resume_from, out_dir, dirs_exist_ok=True)
        else:
            print(f"[resume] resume_from={resume_from} does not exist — starting fresh")
    else:
        print("[resume] no resume_from set — starting fresh")

    folds = cfg["folds_to_run"] or list(range(cfg["data"]["n_folds"]))
    print(f"[setup] running folds: {folds}")

    progress = load_progress(progress_path)
    print(f"[main] progress so far: {progress}")
    completed_folds = set(progress["completed_folds"])
    results = []

    for fold in folds:
        if fold in completed_folds:
            print(f"[main] fold {fold} already completed, loading its saved val predictions")
            fold_dir = out_dir / f"fold{fold}"
            saved = torch.load(fold_dir / "checkpoint.pt", map_location=device)
            best_ll = saved["best_val_loss"]
            _, val_df_f = load_fold_split(cfg["data"]["path"], fold, cfg["data"]["fold_col"])
            col_kwargs = dict(
                prompt_col=cfg["data"]["prompt_col"], response_a_col=cfg["data"]["response_a_col"],
                response_b_col=cfg["data"]["response_b_col"], winner_col=cfg["data"]["winner_col"],
            )
            val_ds_f = RewardPairDataset(val_df_f, tokenizer, cfg["model"]["max_length"], **col_kwargs)
            val_loader_f = DataLoader(
                val_ds_f, batch_size=cfg["train"]["eval_batch_size"], shuffle=False,
                collate_fn=RewardPairCollator(tokenizer), num_workers=cfg["train"]["num_workers"],
            )
            m = RewardModel(
                cfg["model"]["backbone"], cfg["model"]["pooling"], cfg["model"]["dropout"], cfg["model"]["delta_init"]
            ).to(device)
            m.load_state_dict(torch.load(fold_dir / "best.pt", map_location=device))
            _, val_probs_f, val_labels_f = run_epoch(
                m, val_loader_f, device, None, None, None, cfg["train"]["precision"], None, None, train=False,
            )
            del m
            torch.cuda.empty_cache()
            results.append({
                "fold": fold, "val_log_loss": best_ll, "val_probs": val_probs_f,
                "val_labels": val_labels_f, "val_ids": val_df_f["id"].to_numpy(),
            })
            continue

        fold_dir = out_dir / f"fold{fold}"
        step_ckpt = fold_dir / "step_checkpoint.pt"
        resume_skip = 0
        if step_ckpt.exists():
            resume_skip = torch.load(step_ckpt, map_location="cpu")["next_batch_idx"]
            print(f"[main] fold {fold} has a mid-epoch checkpoint, will skip first {resume_skip} batches")

        resume_epoch = progress["current_fold_completed_epochs"] if progress.get("current_fold") == fold else 0
        r = run_fold(
            cfg, fold, tokenizer, device,
            resume_epoch=resume_epoch, resume_skip_batches=resume_skip,
            metrics_path=metrics_path, progress_path=progress_path,
        )
        results.append(r)

        progress["completed_folds"] = list(set(progress["completed_folds"]) | {fold})
        progress["current_fold"] = None
        progress["current_fold_completed_epochs"] = 0
        save_progress(progress_path, progress)

    fold_summary = {r["fold"]: r["val_log_loss"] for r in results}
    print("\n=== Per-fold val log loss ===")
    for fold, ll in fold_summary.items():
        print(f"  fold {fold}: {ll:.4f}")

    if cfg["output"]["save_oof"] and all(r["val_ids"] is not None for r in results):
        oof_probs = np.concatenate([r["val_probs"] for r in results])
        oof_labels = np.concatenate([r["val_labels"] for r in results])
        oof_ids = np.concatenate([r["val_ids"] for r in results])
        oof_ll = multiclass_log_loss(oof_probs, oof_labels)
        print(f"\n=== OOF log loss (folds run: {folds}) === {oof_ll:.4f}")

        oof_df = pd.DataFrame(oof_probs, columns=["p_a", "p_b", "p_tie"])
        oof_df["id"] = oof_ids
        oof_df["label"] = oof_labels
        oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

        with open(out_dir / "summary.json", "w") as f:
            json.dump({
                "folds_run": folds,
                "per_fold_log_loss": fold_summary,
                "oof_log_loss": oof_ll,
            }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
