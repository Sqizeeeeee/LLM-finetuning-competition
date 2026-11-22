"""Compare exp2 with one exp1 encoder on that encoder's held-out fold."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
EXP0 = EXP_DIR.parent / "00_manual_features"
EXP1 = EXP_DIR.parent / "01_bradley_terry_deberta_base"
EXP2 = EXP_DIR.parent / "02_frozen_deberta"
CLASSES = ["a", "b", "tie"]
TEXT_COLUMNS = ["prompt_text", "response_a_text", "response_b_text"]


def load_rows(config, fold, train_limit, val_limit):
    data = config["data"]
    df = pd.read_parquet(ROOT / data["path"])
    bt = pd.read_parquet(ROOT / "data/processed/BT/train_bt.parquet", columns=["id", "fold"])
    bt_folds = bt.set_index("id")["fold"].reindex(df["id"])
    if len(bt) != len(df) or not np.array_equal(bt_folds, df[data["fold_col"]]):
        raise ValueError("exp1 and exp2 fold assignments differ. Check for leakage before scoring.")
    val = df[data["fold_col"]].eq(fold)
    if not val.any():
        raise ValueError(f"Fold {fold} is missing.")
    if train_limit is not None or val_limit is not None:
        train_rows, val_rows = df[~val], df[val]
        if train_limit is not None:
            train_rows = train_rows.sample(train_limit, random_state=42)
        if val_limit is not None:
            val_rows = val_rows.sample(val_limit, random_state=43)
        df = pd.concat([train_rows, val_rows])
    return df.reset_index(drop=True)


def resumable_buffer(df, checkpoint, backbone, max_length, device, shape, output):
    if output is None:
        return np.empty(shape, dtype=np.float32), 0, None

    buffer = output / "features.partial.npy"
    progress = output / "progress.json"
    rows = pd.util.hash_pandas_object(df[["id", *TEXT_COLUMNS]], index=False).to_numpy()
    run = {"rows": hashlib.sha256(rows.tobytes()).hexdigest(), "backbone": backbone,
           "max_length": max_length, "device": device,
           "checkpoint_size": checkpoint.stat().st_size,
           "checkpoint_mtime": checkpoint.stat().st_mtime_ns}
    if progress.exists():
        saved = json.loads(progress.read_text())
        if saved["run"] != run:
            raise ValueError(f"Partial cache belongs to a different run: {output}")
        embeddings = np.lib.format.open_memmap(buffer, mode="r+")
        done = saved["encoded"]
        if embeddings.shape != shape or embeddings.dtype != np.float32 or not 0 <= done <= len(df):
            raise ValueError(f"Invalid partial cache: {output}")
        print(f"Resuming from {done}/{len(df)} pairs", flush=True)
    else:
        embeddings = np.lib.format.open_memmap(buffer, mode="w+", dtype=np.float32, shape=shape)
        done = 0

    def save(encoded):
        embeddings.flush()
        temp = progress.with_suffix(".tmp")
        temp.write_text(json.dumps({"run": run, "encoded": encoded, "total": len(df)}, indent=2) + "\n")
        temp.replace(progress)

    return embeddings, done, save


def encode(df, checkpoint, backbone, max_length, batch_size, device, output=None):
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(backbone, local_files_only=True)
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(backbone, local_files_only=True))
    state = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    encoder.load_state_dict({
        key.removeprefix("encoder."): value
        for key, value in state.items() if key.startswith("encoder.")
    })
    del state
    encoder = encoder.requires_grad_(False).eval().to(device)
    response_lengths = df[TEXT_COLUMNS[1:]].apply(lambda column: column.str.len())
    lengths = df[TEXT_COLUMNS[0]].str.len() + response_lengths.max(axis=1)
    order = np.argsort(lengths.to_numpy(), kind="stable")
    shape = (len(df), encoder.config.hidden_size + 2)
    embeddings, resume_at, save_progress = resumable_buffer(
        df, checkpoint, backbone, max_length, device, shape, output)
    start_time = time.monotonic()
    pad_multiple = 32 if max_length % 32 == 0 else None
    with torch.inference_mode():
        for start in range(resume_at, len(df), batch_size):
            rows = order[start:start + batch_size]
            prompts, a, b = (df[column].iloc[rows].tolist() for column in TEXT_COLUMNS)
            batch = tokenizer(prompts + prompts, a + b, padding=True, truncation="longest_first",
                              max_length=max_length, pad_to_multiple_of=pad_multiple,
                              return_tensors="pt").to(device)
            hidden = encoder(**batch).last_hidden_state.float()
            mask = batch.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            h_a, h_b = pooled.chunk(2)
            delta = h_a - h_b
            features = torch.column_stack((F.cosine_similarity(h_a, h_b), delta.norm(dim=1), delta))
            embeddings[rows] = features.cpu().numpy()
            done = start + len(rows)
            if start == 0 or start // 256 != done // 256 or done == len(df):
                if save_progress is not None:
                    save_progress(done)
                print(f"Encoded {done}/{len(df)} pairs in {time.monotonic() - start_time:.1f}s", flush=True)
    return embeddings


def metrics(y, predictions):
    tie = y == CLASSES.index("tie")
    return {
        "log_loss": log_loss(y, predictions, labels=range(3)),
        "tie_log_loss": log_loss(y[tie], predictions[tie], labels=range(3)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-limit", type=int, help="Sample training rows; keep the full validation fold.")
    parser.add_argument("--val-limit", type=int, help="Sample validation rows for an exploratory pilot.")
    parser.add_argument("--encode-limit", type=int, help="Benchmark extraction only; do not save or score.")
    parser.add_argument("--baseline-cache", type=Path, default=EXP2 / "res" / "features.npz")
    args = parser.parse_args()
    limits = (args.train_limit, args.val_limit, args.encode_limit)
    if args.batch_size < 1 or any(n is not None and n < 1 for n in limits):
        parser.error("Batch size and limits must be positive.")

    config = yaml.safe_load((EXP0 / "config.yaml").read_text())
    data = config["data"]
    df = load_rows(config, args.fold, args.train_limit, args.val_limit)
    train_tag = "full" if args.train_limit is None else f"train{args.train_limit}"
    if args.val_limit is not None:
        train_tag += f"-val{args.val_limit}"
    output = EXP_DIR / "res" / f"fold{args.fold}-{train_tag}"
    checkpoint = EXP1 / "res" / f"fold{args.fold}" / "best.pt"
    with np.load(args.baseline_cache) as cache:
        backbone, max_length = str(cache["model"]), int(cache["max_length"])
        if args.encode_limit:
            sample = df.sample(min(len(df), args.encode_limit), random_state=42).reset_index(drop=True)
            result = encode(sample, checkpoint, backbone, max_length, args.batch_size, args.device)
            if not np.isfinite(result).all():
                raise ValueError("Encoder returned non-finite features.")
            print(f"Shape: {result.shape}, mean difference norm: {result[:, 1].mean():.5g}")
            return
        order = pd.Index(cache["ids"]).get_indexer(pd.Index(df["id"]))
        if (order < 0).any():
            raise ValueError("Missing exp2 features for comparison.")
        frozen = np.column_stack((cache["similarity"][order], cache["difference"][order]))

    val = df[data["fold_col"]].eq(args.fold).to_numpy()
    labels = df[data["target_col"]].map({label: i for i, label in enumerate(CLASSES)}).to_numpy()
    if set(labels[val]) != set(range(3)) or set(labels[~val]) != set(range(3)):
        raise ValueError("Training and validation must each contain all three target classes.")

    output.mkdir(parents=True, exist_ok=True)
    local_cache = output / "features.npz"
    if local_cache.exists():
        with np.load(local_cache) as cache:
            if not np.array_equal(cache["ids"], df["id"]):
                raise ValueError("Saved exp3 feature IDs differ from the requested rows.")
            finetuned = cache["features"]
    else:
        finetuned = encode(df, checkpoint, backbone, max_length, args.batch_size, args.device, output)
        if not np.isfinite(finetuned).all():
            raise ValueError("Encoder returned non-finite features.")
        temp = output / "features.tmp.npz"
        np.savez(temp, ids=df["id"].to_numpy(), features=finetuned)
        temp.replace(local_cache)
        (output / "progress.json").unlink()
        (output / "features.partial.npy").unlink()
        if sys.platform == "darwin":
            # Keep PyTorch and LightGBM OpenMP runtimes in separate processes.
            os.execv(sys.executable, [sys.executable, "-u", *sys.argv])

    sys.path.insert(0, str(EXP0))
    from boosting import BoostingModel

    manual = df[data["feature_cols"]].to_numpy()
    report = {"fold": args.fold, "train_rows": int((~val).sum()), "val_rows": int(val.sum()),
              "backbone": backbone, "max_length": max_length}
    predictions = {}
    for name, embeddings in (("frozen", frozen), ("finetuned", finetuned)):
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{name} embeddings contain non-finite values.")
        matrix = np.column_stack((manual, embeddings))
        print(f"Training LightGBM: {name}", flush=True)
        model = BoostingModel(config["lightgbm"])
        model.fit(matrix[~val], labels[~val], matrix[val], labels[val])
        booster = model.model
        if booster is None:
            raise RuntimeError(f"LightGBM did not produce a model for {name}.")
        predictions[name] = model.predict_proba(matrix[val])
        booster.save_model(str(output / f"{name}.txt"))
        report[name] = {**metrics(labels[val], predictions[name]), "best_iteration": booster.best_iteration,
                        "median_val_difference_norm": float(np.median(embeddings[val, 1]))}
        print(f"{name}: {report[name]}", flush=True)
    blend = (predictions["frozen"] + predictions["finetuned"]) / 2
    report["blend_equal"] = metrics(labels[val], blend)
    report["delta_log_loss"] = report["finetuned"]["log_loss"] - report["frozen"]["log_loss"]
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
