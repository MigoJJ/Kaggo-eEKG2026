# ECG Training Pipeline

PyTorch scaffold for:

- PTB-XL 5-superclass multi-label training
- Metrics-first evaluation
- Class-wise threshold tuning
- MIT-BIH beat pretraining for transferable morphology features

## Layout

- `configs/ptbxl_baseline.json`: PTB-XL baseline config
- `configs/mitbih_pretrain.json`: MIT-BIH beat pretraining config
- `ecg_training/train_ptbxl.py`: PTB-XL train/eval entrypoint
- `ecg_training/pretrain_mitbih.py`: MIT-BIH beat pretraining entrypoint
- `ecg_training/thresholds.py`: threshold search utilities

## Expected archives

Place these files in the repository root:

- `ptb-xl.zip`
- `mit-bih-arrhythmia-database-1.0.0.zip`

The code extracts them into `data/raw/` on first use.

## Install

```bash
python -m venv ecg_training/.venv
source ecg_training/.venv/bin/activate
pip install -r ecg_training/requirements.txt
```

## Phase 1: PTB-XL baseline

```bash
python -m ecg_training.train_ptbxl --config ecg_training/configs/ptbxl_baseline.json
```

Outputs go to `runs/ptbxl_<timestamp>/`.

Artifacts:

- `history.csv`
- `best_model.pt`
- `val_predictions.csv`
- `test_predictions.csv`
- `thresholds.json`
- `report.json`

## Kaggle Checklist

Use this when running the full PTB-XL baseline on Kaggle free GPU.

1. Create a Kaggle Dataset that already contains extracted PTB-XL files.
2. Add that dataset to the notebook instead of uploading `ptb-xl.zip` each session.
3. Copy this repo into `/kaggle/working/` and install requirements there.
4. Set `output_root` or `--run-dir` to `/kaggle/working/runs/...` so checkpoints persist in notebook output.
5. Start with `100 Hz`, mixed precision, and epoch checkpoints enabled.
6. Save `latest_checkpoint.pt` every epoch and resume from it after disconnects.
7. Run threshold tuning only after the training loop finishes.
8. Download the whole run directory after each long session.

Recommended baseline command:

```bash
python -m ecg_training.train_ptbxl \
  --config ecg_training/configs/ptbxl_baseline.json \
  --device cuda \
  --amp \
  --save-every-epoch \
  --run-dir /kaggle/working/runs/ptbxl_full_baseline
```

Resume command:

```bash
python -m ecg_training.train_ptbxl \
  --config ecg_training/configs/ptbxl_baseline.json \
  --device cuda \
  --amp \
  --save-every-epoch \
  --run-dir /kaggle/working/runs/ptbxl_full_baseline \
  --resume /kaggle/working/runs/ptbxl_full_baseline/latest_checkpoint.pt
```

Artifacts useful for recovery:

- `latest_checkpoint.pt`: resume target
- `best_model.pt`: best validation model by `macro_pr_auc`
- `checkpoint_epoch_XXX.pt`: optional per-epoch fallback
- `history.csv`: epoch-wise metrics and learning rate
- `report.json`: final evaluation summary

## Kaggle Notebook Cells

Cell 1: unpack the code bundle into working directory.

```bash
cd /kaggle/working
unzip -q /kaggle/input/ecg-training-code/ecg_training_kaggle_bundle.zip
```

Cell 2: install dependencies.

```bash
cd /kaggle/working
python -m venv .venv
source .venv/bin/activate
pip install -r ecg_training/requirements.txt
```

Cell 3: launch the first training session.

```bash
source /kaggle/working/.venv/bin/activate
cd /kaggle/working
python -m ecg_training.train_ptbxl \
  --config ecg_training/configs/ptbxl_kaggle_100hz.json \
  --device cuda \
  --amp \
  --save-every-epoch \
  --run-dir /kaggle/working/runs/ptbxl_full_baseline
```

Cell 4: resume after reconnect.

```bash
source /kaggle/working/.venv/bin/activate
cd /kaggle/working
python -m ecg_training.train_ptbxl \
  --config ecg_training/configs/ptbxl_kaggle_100hz.json \
  --device cuda \
  --amp \
  --save-every-epoch \
  --run-dir /kaggle/working/runs/ptbxl_full_baseline \
  --resume /kaggle/working/runs/ptbxl_full_baseline/latest_checkpoint.pt
```

## Minimal Upload Bundles

Code bundle:

- `ecg_training/`
- build it locally with `bash ecg_training/scripts/make_kaggle_code_bundle.sh`
- upload `dist/ecg_training_kaggle_bundle.zip` as a Kaggle Dataset

PTB-XL data bundle:

- `ptbxl_database.csv`
- `scp_statements.csv`
- `records100/`

Recommended Kaggle datasets:

- `ecg-training-code`
- `ptb-xl-extracted`

The provided Kaggle config expects:

- `/kaggle/input/ecg-training-code/ecg_training_kaggle_bundle.zip`
- `/kaggle/input/ptb-xl-extracted/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3`

## Phase 2: MIT-BIH pretraining

```bash
python -m ecg_training.pretrain_mitbih --config ecg_training/configs/mitbih_pretrain.json
```

This saves a beat encoder checkpoint you can reuse in PTB-XL config:

```json
{
  "model": {
    "pretrained_encoder_path": "runs/mitbih_pretrain_YYYYMMDD_HHMMSS/best_encoder.pt"
  }
}
```

## Operating principles

- Model selection is based on `macro_pr_auc`
- PTB-XL split uses official folds `1-8 train`, `9 val`, `10 test`
- Thresholds are tuned on validation only
- Critical classes `MI` and `CD` default to recall-favoring `f2` threshold search
