.PHONY: prep-data prep-bt train-baseline smoke-bt help

help:
	@echo "make prep-data      - regenerate data/processed/common/train_clean.parquet"
	@echo "make prep-bt        - regenerate data/processed/BT/train_bt.parquet (needs prep-data first)"
	@echo "make train-baseline - train the 00_manual_features baseline locally"
	@echo "make smoke-bt       - run the 01_bradley_terry_deberta_base smoke test (CPU) locally"

prep-data:
	python preprocessing/common.py

prep-bt: prep-data
	python experiments/01_bradley_terry_deberta_base/bt_prep.py

train-baseline:
	python experiments/00_manual_features/train.py

smoke-bt:
	python experiments/01_bradley_terry_deberta_base/train.py \
		--config=experiments/01_bradley_terry_deberta_base/test_config.yaml