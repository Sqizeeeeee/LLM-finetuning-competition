"""
Run from repo root:
CPU only. Uses small samples so each test finishes in a couple of minutes.
"""
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/01_bradley_terry_deberta_base"
sys.path.insert(0, str(EXP))
from dataset import RewardPairCollator, RewardPairDataset
from reward_model import RewardModel

DATA = ROOT / "data/processed/BT/train_bt.parquet"
N_ROWS = 32
MAX_LENGTH = 96  # short on purpose
N_STEPS = 80
LR = 2e-5
HEAD_LR = 1e-4

torch.manual_seed(0)


def tiny_loader(tokenizer, n_rows=N_ROWS, seed=0):
    df = pd.read_parquet(DATA).sample(n_rows, random_state=seed).reset_index(drop=True)
    ds = RewardPairDataset(df, tokenizer, MAX_LENGTH)
    return DataLoader(ds, batch_size=8, shuffle=True, collate_fn=RewardPairCollator(tokenizer)), df


def build_optimizer(model):
    head_params = list(model.reward_head.parameters()) + [model.delta]
    head_ids = {id(p) for p in head_params}
    encoder_params = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW([
        {"params": encoder_params, "lr": LR},
        {"params": head_params, "lr": HEAD_LR},
    ])


def run_training_probe(use_checkpointing: bool, tokenizer):
    print(f"\n=== T{'2' if use_checkpointing else '1'}: training probe, "
          f"gradient_checkpointing={use_checkpointing} ===")
    model = RewardModel()
    if use_checkpointing:
        model.encoder.gradient_checkpointing_enable()
    model.train()
    optimizer = build_optimizer(model)

    loader, _ = tiny_loader(tokenizer)
    step = 0
    while step < N_STEPS:
        for batch in loader:
            out = model(batch["input_ids"], batch["attention_mask"])
            loss = F.cross_entropy(out["class_logits"], batch["labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if step == 0:
                enc_grad = sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None)
                head_grad = sum(p.grad.abs().sum().item() for p in model.reward_head.parameters() if p.grad is not None)
                delta_grad = model.delta.grad.item() if model.delta.grad is not None else None
                print(f"  step 0 grad magnitude: encoder={enc_grad:.6f} reward_head={head_grad:.6f} delta_grad={delta_grad}")

            optimizer.step()
            r_a = out["r_a"].detach()
            r_b = out["r_b"].detach()
            if step % 10 == 0 or step == N_STEPS - 1:
                print(f"  step {step:3d} | loss {loss.item():.4f} | delta {model.delta.item():.4f} "
                      f"| std(r_a) {r_a.std().item():.5f} | std(r_b) {r_b.std().item():.5f} "
                      f"| mean(r_a) {r_a.mean().item():+.5f}")
            step += 1
            if step >= N_STEPS:
                break
    return model


def run_alignment_check(tokenizer):
    print("\n=== T3: A/B index alignment check ===")
    model = RewardModel()
    model.eval()

    short_response = "No."
    long_response = ("This is a much longer and more detailed answer that goes on at length, "
                      "repeating itself to make sure it is unambiguously longer and different "
                      "from the other response in every way, padded further to be very long. ") * 3

    rows = []
    for i in range(6):

        if i % 2 == 0:
            rows.append({"prompt_text": f"prompt {i}", "response_a_text": long_response,
                         "response_b_text": short_response, "winner": "a"})
        else:
            rows.append({"prompt_text": f"prompt {i}", "response_a_text": short_response,
                         "response_b_text": long_response, "winner": "b"})
    df = pd.DataFrame(rows)

    ds = RewardPairDataset(df, tokenizer, max_length=256)
    loader = DataLoader(ds, batch_size=6, shuffle=False, collate_fn=RewardPairCollator(tokenizer))
    batch = next(iter(loader))

    with torch.inference_mode():
        out = model(batch["input_ids"], batch["attention_mask"])

    for i in range(len(df)):
        side_with_long = "a" if i % 2 == 0 else "b"
        print(f"  row {i} (long response is side {side_with_long}): "
              f"r_a={out['r_a'][i].item():+.4f}  r_b={out['r_b'][i].item():+.4f}  "
              f"d={out['d'][i].item():+.4f}")

    print("  Expect |r_a - r_b| to be noticeably nonzero and consistent per row (random init, "
          "but same two texts reused -> same encoder should give a repeatable, non-tiny gap "
          "between a clearly different long vs short input). A near-zero gap on every row here "
          "(even before any training) would point at an indexing/alignment bug, not at training "
          "dynamics.")


def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    run_alignment_check(tokenizer)
    run_training_probe(use_checkpointing=False, tokenizer=tokenizer)
    run_training_probe(use_checkpointing=True, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
