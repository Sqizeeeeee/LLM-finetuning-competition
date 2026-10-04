.PHONY: prep-data prep-bt train-baseline finetune-ensemble inference-baseline smoke-bt help

help:
	@echo "make prep-data          - regenerate data/processed/common/train_clean.parquet"
	@echo "make prep-bt            - regenerate data/processed/BT/train_bt.parquet (needs prep-data first)"
	@echo "make train-baseline     - train 00_manual_features base models (LogReg + LightGBM)"
	@echo "make finetune-ensemble  - fit ensemble weights on top of train-baseline OOF (needs train-baseline first)"
	@echo "make inference-baseline - build the final ensemble + submission for 00_manual_features"
	@echo "make smoke-bt           - run the 01_bradley_terry_deberta_base smoke test (CPU) locally"

prep-data:
	python preprocessing/common.py

prep-bt: prep-data
	python experiments/01_bradley_terry_deberta_base/bt_prep.py

train-baseline:
	python experiments/00_manual_features/train.py

finetune-ensemble: train-baseline
	python experiments/00_manual_features/finetuning_ensemble.py

inference-baseline: finetune-ensemble
	python experiments/00_manual_features/inference.py

smoke-bt:
	python experiments/01_bradley_terry_deberta_base/train.py \
		--config=experiments/01_bradley_terry_deberta_base/test_config.yaml
