"""Cache frozen DeBERTa pair features from the exp0 training rows."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
DEFAULT_CACHE = EXP_DIR / "res" / "features.npz"
TEXT_COLUMNS = ["prompt_text", "response_a_text", "response_b_text"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8, help="Pairs of responses per batch")
    parser.add_argument("--limit", type=int, help="Only for smoke tests. Requires --output.")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 4 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size and limit must be positive, and max length must be at least four.")
    if args.limit is not None and args.output.resolve() == DEFAULT_CACHE:
        parser.error("Use a separate --output for smoke tests.")

    df = pd.read_parquet(
        ROOT / "data/processed/classical/train_features.parquet",
        columns=["id", *TEXT_COLUMNS],
    )
    if args.limit is not None:
        df = df.head(args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    encoder = AutoModel.from_pretrained(args.model, local_files_only=args.local_files_only)
    encoder = encoder.requires_grad_(False).eval().to(args.device)

    difference = np.empty((len(df), encoder.config.hidden_size), dtype=np.float32)
    similarity = np.empty((len(df), 2), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(df), args.batch_size):
            stop = min(start + args.batch_size, len(df))
            prompts, responses_a, responses_b = (
                df[column].iloc[start:stop].tolist() for column in TEXT_COLUMNS
            )
            batch = tokenizer(
                prompts + prompts,
                responses_a + responses_b,
                padding=True,
                truncation="longest_first",
                max_length=args.max_length,
                return_tensors="pt",
            ).to(args.device)
            hidden = encoder(**batch).last_hidden_state.float()
            mask = batch.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            h_a, h_b = pooled.chunk(2)
            delta = h_a - h_b
            difference[start:stop] = delta.cpu().numpy()
            similarity[start:stop, 0] = F.cosine_similarity(h_a, h_b).cpu().numpy()
            similarity[start:stop, 1] = delta.norm(dim=1).cpu().numpy()
            if start % (100 * args.batch_size) == 0 or stop == len(df):
                print(f"Encoded {stop}/{len(df)} pairs", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(".tmp.npz")
    np.savez(
        temp_path, ids=df["id"].to_numpy(), similarity=similarity, difference=difference,
        model=args.model, max_length=args.max_length,
    )
    os.replace(temp_path, args.output)


if __name__ == "__main__":
    main()
