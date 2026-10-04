# D1: Reward collapse check
# Question: does d = r_a - r_b carry any signal, or has the model collapsed to a constant reward?
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path.cwd()  # run from repo root
EXP = ROOT / "experiments/01_bradley_terry_deberta_base"
sys.path.insert(0, str(EXP))
from dataset import RewardPairCollator, RewardPairDataset
from reward_model import RewardModel

FOLD = 1
CKPT = EXP / "res" / f"fold{FOLD}" / "best.pt"
DATA = ROOT / "data/processed/BT/train_bt.parquet"
N_ROWS = 400        # lower it if CPU inference is too slow
MAX_LENGTH = 512    # shorter than the 1024 used in training, to keep CPU inference fast
BATCH = 4

df = pd.read_parquet(DATA)
train_df = df[df["fold"] != FOLD]
val_df = df[df["fold"] == FOLD].sample(N_ROWS, random_state=42).reset_index(drop=True)
prior = train_df["winner"].value_counts(normalize=True).reindex(["a", "b", "tie"]).to_numpy()

tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = RewardModel()  # DeBERTa crashes on MPS, so stay on CPU
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

loader = DataLoader(
    RewardPairDataset(val_df, tok, MAX_LENGTH),
    batch_size=BATCH, shuffle=False, collate_fn=RewardPairCollator(tok),
)

r_a, r_b, probs, labels = [], [], [], []
with torch.inference_mode():
    for batch in loader:
        out = model(batch["input_ids"], batch["attention_mask"])
        r_a.append(out["r_a"])
        r_b.append(out["r_b"])
        probs.append(torch.stack([out["p_a"], out["p_b"], out["p_tie"]], dim=1))
        labels.append(batch["labels"])

r_a = torch.cat(r_a).numpy()
r_b = torch.cat(r_b).numpy()
probs = torch.cat(probs).numpy()
labels = torch.cat(labels).numpy()
d = r_a - r_b
delta = model.delta.item()

print(f"delta={delta:.3f} | std(r_a)={r_a.std():.4f} std(r_b)={r_b.std():.4f}")
print(f"d: mean={d.mean():+.4f} std={d.std():.4f} | share of rows with |d|>delta: {(np.abs(d) > delta).mean():.1%}")
print("mean probs [a, b, tie]:", probs.mean(0).round(3), "| class prior:", prior.round(3))

ll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-15, 1)).mean()
ll_prior = -np.log(prior[labels]).mean()
print(f"log loss: model {ll:.4f} | constant-prior baseline on the same rows {ll_prior:.4f}")

mask = labels != 2  # drop ties for AUC
y = (labels[mask] == 0).astype(int)  # 1 = A won
len_diff = (val_df["response_a_text"].str.len() - val_df["response_b_text"].str.len()).to_numpy()
print(f"AUC of d (A vs B, ties excluded, n={mask.sum()}): {roc_auc_score(y, d[mask]):.3f}")
print(f"AUC of length diff (reference): {roc_auc_score(y, len_diff[mask]):.3f}")
print(f"Spearman(d, length diff): {spearmanr(d, len_diff)[0]:.3f}")
