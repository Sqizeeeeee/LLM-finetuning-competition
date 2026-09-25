import argparse
import json
import random
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


def build_param_groups(model: RewardModel, lr: float, head_lr: float, weight_decay: float):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    encoder_decay, encoder_no_decay, head_params = [], [], []

    for name, param in model.encoder.named_parameters():
        if not param.requires_grad:
            continue
        (encoder_no_decay if any(nd in name for nd in no_decay) else encoder_decay).append(param)

    head_params += list(model.reward_head.parameters())
    head_params.append(model.delta)

    return [
        {"params": encoder_decay, "lr": lr, "weight_decay": weight_decay},
        {"params": encoder_no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
    ]


def run_epoch(
    model, loader, device, optimizer, scheduler, scaler, precision, max_grad_norm,
    log_every_n_steps, train: bool,
):
    model.train(mode=train)
    total_loss, n_batches = 0.0, 0
    all_probs, all_labels = [], []

    autocast_dtype = torch.float16 if precision == "fp16" else torch.float32
    use_amp = precision == "fp16" and device.type == "cuda"

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                out = model(input_ids, attention_mask)
                loss = F.cross_entropy(out["class_logits"], labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

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


def multiclass_log_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], eps, 1 - eps)
    return float(-np.log(p).mean())


def run_fold(cfg: dict, fold: int, tokenizer, device) -> dict:
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

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        collate_fn=collator, num_workers=cfg["train"]["num_workers"], drop_last=True,
    )
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
        build_param_groups(model, cfg["train"]["lr"], cfg["train"]["head_lr"], cfg["train"]["weight_decay"])
    )

    n_steps = len(train_loader) * cfg["train"]["epochs"] // cfg["train"]["grad_accum_steps"]
    n_warmup = int(n_steps * cfg["train"]["warmup_ratio"])
    scheduler_fn = get_cosine_schedule_with_warmup if cfg["train"]["scheduler"] == "cosine" \
        else get_linear_schedule_with_warmup
    scheduler = scheduler_fn(optimizer, num_warmup_steps=n_warmup, num_training_steps=n_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["train"]["precision"] == "fp16" and device.type == "cuda"))

    best_val_loss = float("inf")
    best_val_probs, best_val_labels = None, None

    for epoch in range(cfg["train"]["epochs"]):
        t0 = time.time()
        train_loss, _, _ = run_epoch(
            model, train_loader, device, optimizer, scheduler, scaler,
            cfg["train"]["precision"], cfg["train"]["max_grad_norm"],
            cfg["train"]["log_every_n_steps"], train=True,
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

        if val_ll < best_val_loss:
            best_val_loss = val_ll
            best_val_probs, best_val_labels = val_probs, val_labels
            if cfg["output"]["save_checkpoints"]:
                out_dir = Path(cfg["output"]["dir"]) / f"fold{fold}"
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out_dir / "best.pt")

    return {
        "fold": fold,
        "val_log_loss": best_val_loss,
        "val_probs": best_val_probs,
        "val_labels": best_val_labels,
        "val_ids": val_df["id"].to_numpy() if "id" in val_df.columns else None,
        "delta": model.delta.item(),
    }


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    print(f"[setup] device={device} | config={config_path}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])

    folds = cfg["folds_to_run"] or list(range(cfg["data"]["n_folds"]))
    print(f"[setup] running folds: {folds}")

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [run_fold(cfg, fold, tokenizer, device) for fold in folds]

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
                "delta_per_fold": {r["fold"]: r["delta"] for r in results},
            }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
