# Swahili Audio Classification: Sequential Models (Formative 2)

12-class spoken Swahili word classification (Zindi dataset).

## Layout
```
src/            reusable code (data_overview.py, later features.py, split.py, baseline.py)
notebooks/      EDA and experiment notebooks
data/raw/       Train.csv + extracted audio (not committed)
data/processed/ index, splits, cached features (not committed)
reports/figures figures used in the report
```

## Setup
```bash
pip install -r requirements.txt
# Linux only: sudo apt install ffmpeg libsndfile1
```
Put `Train.csv` and the extracted audio (from `Swahili_words.zip`) under `data/raw/`.

## Step 1: dataset overview
```bash
python src/data_overview.py --train-csv data/raw/Train.csv --audio-dir data/raw
```
Use `--id-col` / `--label-col` if the column names are not detected correctly.
Paths are relative, so the same commands work on Google Colab.