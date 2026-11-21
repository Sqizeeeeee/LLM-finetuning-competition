# Frozen DeBERTa + LightGBM on Kaggle

The notebook uses five LightGBM boosters trained on the original folds with
manual features, cosine similarity, the norm of the embedding difference,
and the full A − B embedding difference. Local OOF log loss was 1.022379.

To rebuild the Kaggle input from the repository root, use the existing
`experiments/02_frozen_deberta/res/features.npz`:

```bash
.venv/bin/python experiments/02_frozen_deberta/compare.py --export-dir build/exp2/boosters
.venv/bin/python - <<'PY'
import json
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

bundle = Path("build/exp2")
model_id = json.loads((bundle / "boosters/config.json").read_text())["model"]
AutoTokenizer.from_pretrained(model_id).save_pretrained(bundle / "encoder")
AutoModel.from_pretrained(model_id).save_pretrained(bundle / "encoder")
PY
(cd build/exp2 && zip -q -r ../exp2-kaggle-input.zip boosters encoder)
```

Create a Kaggle Dataset from `build/exp2-kaggle-input.zip` with the
`boosters/` and `encoder/` directories. Add that dataset and the competition
data to `llm-finetuning-frozen-deberta.ipynb`. Select a GPU accelerator and
run with Internet off. The notebook writes `submission.csv` in the working
directory.
