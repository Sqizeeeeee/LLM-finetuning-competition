from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

COMMON_PATH = ROOT / "data/processed/common/train_clean.parquet"
OUT_DIR = ROOT / "data/processed/BT"
OUT_PATH = OUT_DIR / "train_bt.parquet"

KEEP_COLS = [
    "id",
    "prompt_text", "response_a_text", "response_b_text",
    "winner", "fold",
]


def main():
    df = pd.read_parquet(COMMON_PATH)
    print(f"[load] {len(df)} rows from {COMMON_PATH}")

    df = df[KEEP_COLS]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[save] wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()