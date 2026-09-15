import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

IN_PATH = ROOT / "data/processed/common/train_clean.parquet"
OUT_DIR = ROOT / "data/processed/classical"
OUT_PATH = OUT_DIR / "train_features.parquet"

REL_LEN_CLIP = 50

REFUSAL_PATTERN = re.compile(
    r"\b(?:i cannot|i can't|i'm sorry|i am sorry|as an ai|i'm not able to|"
    r"i don't have the ability|i'm unable to|cannot provide|can't provide)\b",
    flags=re.IGNORECASE,
)


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    df["prompt_len_chars"] = df["prompt_text"].str.len()
    df["response_a_len_chars"] = df["response_a_text"].str.len()
    df["response_b_len_chars"] = df["response_b_text"].str.len()
    df["len_diff"] = df["response_a_len_chars"] - df["response_b_len_chars"]

    df["a_rel_len"] = df["response_a_len_chars"] / (df["prompt_len_chars"] + 1)
    df["b_rel_len"] = df["response_b_len_chars"] / (df["prompt_len_chars"] + 1)
    rel_len_diff = df["a_rel_len"] - df["b_rel_len"]
    df["rel_len_diff"] = rel_len_diff.clip(-REL_LEN_CLIP, REL_LEN_CLIP)
    return df


def count_markdown_features(text: str) -> dict:
    return {
        "n_bullets": len(re.findall(r"(?:^|\n)[\-\*]\s", text)),
        "n_numbered": len(re.findall(r"(?:^|\n)\d+\.\s", text)),
        "n_headers": len(re.findall(r"(?:^|\n)#{1,6}\s", text)),
        "n_bold": len(re.findall(r"\*\*[^*]+\*\*", text)),
        "n_code_blocks": text.count("```"),
    }


def add_formatting_features(df: pd.DataFrame) -> pd.DataFrame:
    feat_a = df["response_a_text"].apply(count_markdown_features).apply(pd.Series).add_prefix("a_")
    feat_b = df["response_b_text"].apply(count_markdown_features).apply(pd.Series).add_prefix("b_")
    df = pd.concat([df, feat_a, feat_b], axis=1)

    for feat in ["n_bullets", "n_numbered", "n_headers", "n_bold", "n_code_blocks"]:
        df[f"{feat}_diff"] = df[f"a_{feat}"] - df[f"b_{feat}"]
    return df


def add_refusal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["a_refusal"] = df["response_a_text"].str.contains(REFUSAL_PATTERN, regex=True).astype(int)
    df["b_refusal"] = df["response_b_text"].str.contains(REFUSAL_PATTERN, regex=True).astype(int)
    df["refusal_diff"] = df["a_refusal"] - df["b_refusal"]
    return df


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    print(f"[load] {len(df)} rows from {IN_PATH}")

    df = add_length_features(df)
    df = add_formatting_features(df)
    df = add_refusal_features(df)

    feature_cols = [c for c in df.columns if c not in (
        "id", "prompt_text", "response_a_text", "response_b_text",
        "model_a", "model_b",
        "winner_model_a", "winner_model_b", "winner_tie", "winner", "fold",
    )]
    print(f"[features] {len(feature_cols)} feature columns: {feature_cols}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[save] wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()