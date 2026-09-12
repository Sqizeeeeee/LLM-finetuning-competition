# LLM Classification Finetuning

Kaggle competition: https://www.kaggle.com/competitions/llm-classification-finetuning

Predict which of two LLM responses a human judge preferred (model A, model B, or tie),
given the prompt and both responses. Code competition - the final submission is a
Kaggle notebook run by Kaggle itself against the private test set, not a local CSV.

## Repo structure

```
data/
├── raw/                   # train.csv, test.csv (untouched)
└── processed/
    ├── common/              # parsed, deduplicated, fold-assigned base table
    └── classical/            # + hand-crafted features for classical ML

eda/                        # exploration notebooks + summary.md

preprocessing/
├── common.py               # shared parsing/dedup/fold logic
└── classical.py              # hand-crafted features for classical ML

experiments/
└── 00_manual_features/       # LogReg + LightGBM baseline on hand-crafted features
    ├── config.yaml
    ├── logreg.py / boosting.py / train.py / inference.py
    ├── export_logreg_npz.py    # version-independent export for Kaggle
    ├── notes.md
    └── res/                     # OOF predictions, model weights, plots

```

## Findings (see eda/summary.md for full detail)

- Response length and relative length (response / prompt) correlate with the winner.
- Markdown formatting (lists, bold, code blocks) is a strong signal - judges seem to
  reward visual structure independent of content.
- Refusal/apology language ("I cannot", "I'm sorry") is the strongest single hand-crafted
  signal found so far - near-symmetric ~49% vs ~23% win rate when only one side refuses.
- `model_a`/`model_b` identity is not available in test.csv, so raw model win rate can only
  inform EDA, not be used directly as a feature.
- There is a small amount of irreducible label noise: at least one pair of identical inputs
  received different judgments from different judges.

## Submissions log

Since this is a code competition, the real submission is a Kaggle notebook, not a
local file. This table tracks what was submitted and how it scored.

| date       | experiment          | kaggle notebook                 | version | local OOF | public score | notes |
|------------|----------------------|-----------------------------------|---------|-----------|---------------|-------|
| 2026-09-12 | 00_manual_features    | LLM-finetuning-manual-features     | v2      | 1.04576   | 1.04598       | LogReg + LightGBM ensemble (w=0.95) on hand-crafted features |
