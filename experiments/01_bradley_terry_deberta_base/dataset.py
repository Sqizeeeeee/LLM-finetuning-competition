
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

WINNER_TO_LABEL = {"a": 0, "b": 1, "tie": 2}
LABEL_TO_WINNER = {v: k for k, v in WINNER_TO_LABEL.items()}


class RewardPairDataset(Dataset):
    """One item = one row of train_bt.parquet = a (prompt, response_a, response_b, winner) tuple.

    Tokenization happens lazily in __getitem__ (not precomputed): the dataset
    is small enough (~57k rows) that on-the-fly tokenization is simple and
    fast enough, at the cost of a bit of per-epoch CPU work.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        prompt_col: str = "prompt_text",
        response_a_col: str = "response_a_text",
        response_b_col: str = "response_b_text",
        winner_col: str = "winner",
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_col = prompt_col
        self.response_a_col = response_a_col
        self.response_b_col = response_b_col
        self.winner_col = winner_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        prompt = row[self.prompt_col]

        enc_a = self.tokenizer(
            text=prompt,
            text_pair=row[self.response_a_col],
            truncation="longest_first",
            max_length=self.max_length,
        )
        enc_b = self.tokenizer(
            text=prompt,
            text_pair=row[self.response_b_col],
            truncation="longest_first",
            max_length=self.max_length,
        )

        return {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "input_ids_b": enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
            "label": WINNER_TO_LABEL[row[self.winner_col]],
        }


class RewardPairCollator:
    """Pads A and B sides together into one (batch * 2, seq_len) tensor,
    so the encoder runs a single forward pass per batch instead of two.
    The model is responsible for splitting the output back into r_a / r_b.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        input_ids, attention_mask = [], []
        for item in batch:
            input_ids.append(item["input_ids_a"])
            input_ids.append(item["input_ids_b"])
            attention_mask.append(item["attention_mask_a"])
            attention_mask.append(item["attention_mask_b"])

        padded = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        return {
            "input_ids": padded["input_ids"],       # (batch * 2, seq_len)
            "attention_mask": padded["attention_mask"],
            "labels": labels,                        # (batch,)
        }


def load_fold_split(
    parquet_path: str | Path, fold: int, fold_col: str = "fold"
        ) -> tuple[pd.DataFrame, pd.DataFrame]:

        """Standard train/val split for one fold: val = rows in `fold`,
        train = everything else. Matches the GroupKFold-by-prompt assignment
        done in preprocessing/common.py.
        """
        df = pd.read_parquet(parquet_path)
        train_df = df[df[fold_col] != fold]
        val_df = df[df[fold_col] == fold]
        return train_df, val_df
