import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data/raw/train.csv"
OUT_DIR = ROOT / "data/processed/common"
OUT_PATH = OUT_DIR / "train_clean.parquet"

N_FOLDS = 5
RANDOM_STATE = 42

TEXT_COLS = ["prompt", "response_a", "response_b"]
DEDUP_COLS = ["prompt", "response_a", "response_b", "model_a", "model_b"]
EMPTY_TURN = "[EMPTY]"


def parse_turns(x: str) -> list[str]:
    """Parse a JSON-serialized list of turns (i-th item = i-th dialog turn).

    The raw columns are JSON, not Python literals: json.loads handles `null`
    and JSON-only escapes that literal_eval chokes on. None / blank turns are
    replaced with EMPTY_TURN so rows are kept. Raises if the value is not a
    JSON list: the data is known to parse fully, so a failure should be loud.
    """
    parsed = json.loads(x)
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON list, got {type(parsed).__name__}: {x[:100]!r}")
    return [t if isinstance(t, str) and t.strip() else EMPTY_TURN for t in parsed]



def sanitize_text(text: str) -> str:
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")


def parse_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    parsed = {col: df[col].apply(parse_turns) for col in TEXT_COLS}

    n_turns = parsed["prompt"].apply(len)
    for col in ("response_a", "response_b"):
        assert (parsed[col].apply(len) == n_turns).all(), \
            f"{col}: number of turns differs from prompt"

    for col in TEXT_COLS:
        n_empty = parsed[col].apply(lambda turns: EMPTY_TURN in turns).sum()
        print(f"[parse] {col}: {n_empty} rows with an empty turn replaced by {EMPTY_TURN}")
        df[f"{col}_text"] = parsed[col].apply(
            lambda turns: sanitize_text(" ".join(turns).strip())
        )
    return df

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully duplicated rows (same input, same target).

    Note: this also happens to catch the one known label-conflict pair
    (same input, different target) as non-duplicates on the target-
    inclusive check below, so we keep both rows of that pair — they
    represent real disagreement between judges, not redundant data.
    """
    before = len(df)

    target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
    full_dup_mask = df.duplicated(subset=DEDUP_COLS + target_cols, keep="first")
    df = df[~full_dup_mask].reset_index(drop=True)

    dropped = before - len(df)
    print(f"[dedup] dropped {dropped} fully duplicated rows "
          f"(same input AND same target), {len(df)} rows remaining")
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df["winner"] = np.select(
        [df["winner_model_a"] == 1, df["winner_model_b"] == 1, df["winner_tie"] == 1],
        ["a", "b", "tie"],
        default="unknown",
    )
    n_unknown = (df["winner"] == "unknown").sum()
    if n_unknown:
        print(f"[warn] {n_unknown} rows have no target set across the three winner columns")
    return df


def assign_folds(df: pd.DataFrame, n_folds: int = N_FOLDS) -> pd.DataFrame:
    """GroupKFold by prompt, so the same prompt never appears in both
    train and validation within a fold.
    """
    gkf = GroupKFold(n_splits=n_folds)
    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(gkf.split(df, groups=df["prompt"])):
        df.loc[val_idx, "fold"] = fold_idx

    assert (df["fold"] >= 0).all(), "some rows did not get a fold assigned"

    # sanity check: no prompt should span multiple folds
    prompt_fold_counts = df.groupby("prompt")["fold"].nunique()
    leaking_prompts = (prompt_fold_counts > 1).sum()
    if leaking_prompts:
        print(f"[warn] {leaking_prompts} prompts span multiple folds, GroupKFold may not have worked as expected")
    else:
        print(f"[folds] {n_folds} folds assigned, no prompt leakage across folds")

    return df


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    print(f"[load] {len(df)} rows from {RAW_PATH}")

    df = deduplicate(df)
    df = parse_text_columns(df)
    df = build_target(df)
    df = assign_folds(df)

    keep_cols = [
        "id",
        "prompt_text", "response_a_text", "response_b_text",
        "model_a", "model_b",
        "winner_model_a", "winner_model_b", "winner_tie",
        "winner", "fold",
    ]
    df = df[keep_cols]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[save] wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()