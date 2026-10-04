# Swahili Audio Classification — Formative Assignment 2

12-class spoken Swahili word classification using sequential and classical machine learning models.  
Dataset: [Zindi Swahili Words Audio Classification](https://zindi.africa/competitions/swahili-words-audio-classification).

---

## Task

Classify 16 kHz mono WAV recordings into one of 12 Swahili words:

> `hapana` · `kumi` · `mbili` · `moja` · `nane` · `ndio` · `nne` · `saba` · `sita` · `tano` · `tatu` · `tisa`

Primary metric: **macro-averaged F1-score** (handles class imbalance).

---

## Project Structure

```
src/                        reusable pipeline scripts
  data_overview.py          dataset statistics and EDA plots
  split.py                  reproducible train/val/test split
  features.py               MFCC + delta feature extraction → .npz
  baseline.py               SVM / RF / LogReg baselines
  eda_audio.py              audio-level EDA (duration, mel spectrograms)

notebooks/
  bilstm_model.ipynb        BiLSTM neural model (TensorFlow/Keras)

models/
  cnn_1d_model.py           1D CNN architecture + trainer (PyTorch)

experiments/
  cnn_1d_experiments.py     1D CNN hyperparameter sweep (5 configs)

data/
  processed/splits.csv      frozen train/val/test split index

reports/
  experiments.csv           all baseline run results
  figures/                  EDA and confusion matrix plots

results/
  cnn_experiments.csv       1D CNN hyperparameter experiment results
  hyperparameter_analysis.png  visual comparison of CNN configs

w1.5_energy_mfcc40_d/       pre-extracted features (not committed to git)
  train.npz / val.npz / test.npz
  labels.json
```

---

## Setup

```bash
pip install -r requirements.txt
```

On Linux/Colab, also install system audio libraries:
```bash
sudo apt install ffmpeg libsndfile1
```

Place `Train.csv` and the extracted audio folder (from `Swahili_words.zip`) under `data/raw/`.

---

## Reproducing the Pipeline

### Step 1 — Dataset overview
```bash
python src/data_overview.py --train-csv data/raw/Train.csv --audio-dir data/raw
```
Prints class distribution, duration statistics, and saves EDA figures to `reports/figures/`.

### Step 2 — Create train/val/test split
```bash
python src/split.py --train-csv data/raw/Train.csv
```
Saves `data/processed/splits.csv` with stratified 70/15/15 splits.

### Step 3 — Extract features
```bash
python src/features.py --train-csv data/raw/Train.csv --audio-dir data/raw \
       --window 1.5 --selection energy --n-mfcc 40 --deltas
```
Saves `w1.5_energy_mfcc40_d/{train,val,test}.npz` and `labels.json`.  
Output shape: `(N, 150, 120)` — 150 timesteps × 120 features (40 MFCCs + Δ + ΔΔ).

### Step 4 — Run baselines
```bash
python src/baseline.py --feat-dir w1.5_energy_mfcc40_d
```
Trains SVM, Random Forest, and Logistic Regression on aggregated features (mean/std/min/max over time). Logs results to `reports/experiments.csv`.

### Step 5 — BiLSTM (Jupyter / Colab)
Open `notebooks/bilstm_model.ipynb` and run all cells.  
The notebook auto-detects Colab vs local and sets paths accordingly.  
For Colab: upload `w1.5_energy_mfcc40_d/` to your Drive root before running.

### Step 6 — 1D CNN experiments
```bash
python experiments/cnn_1d_experiments.py
```
Runs 5 hyperparameter configurations and saves results to `results/cnn_experiments.csv`.

---

## Models

### Classical Baselines (`src/baseline.py`)

Features are aggregated over the time axis (mean, std, min, max) before being passed to sklearn classifiers.

| Model | Val macro-F1 | Test macro-F1 |
|---|---|---|
| Logistic Regression | 0.7155 | 0.7083 |
| SVM (RBF, C=10) | 0.7046 | 0.7027 |
| Random Forest (500 trees) | 0.6847 | 0.6452 |

### BiLSTM (`notebooks/bilstm_model.ipynb`) — TensorFlow/Keras

Processes the full `(150, 120)` MFCC sequence without aggregation, capturing temporal dynamics in both directions.

**Architecture:**
- Input: `(150, 120)` — 150 timesteps, 120 features
- BiLSTM(128 units, return_sequences=True) → BatchNorm → Dropout(0.3)
- BiLSTM(64 units, return_sequences=False) → BatchNorm → Dropout(0.3)
- Dense(128, ReLU) → Dropout(0.15) → Dense(12, softmax)

**Training:** Adam(lr=1e-3), EarlyStopping(patience=12), ReduceLROnPlateau(factor=0.5, patience=6)

**Why BiLSTM?** Speech is a bidirectional temporal signal — phonetic context flows both forward and backward. A BiLSTM captures coarticulation effects that unidirectional models and aggregation-based classifiers miss (Graves & Schmidhuber, 2005).

### 1D CNN (`models/cnn_1d_model.py`) — PyTorch

Applies convolutional filters directly over the time axis to detect local temporal patterns (e.g. onset, nucleus, coda of a syllable).

**Architecture:**
- Conv1D(→32) → BN → ReLU → MaxPool → Dropout
- Conv1D(→64) → BN → ReLU → MaxPool → Dropout
- Conv1D(→128) → BN → ReLU → GlobalAvgPool → Dropout
- Dense(128, ReLU) → Dropout → Dense(12)

**Hyperparameter sweep results (`results/cnn_experiments.csv`):**

| Config | Filters | Kernel | Batch | Dropout | Test Acc | Test F1 |
|---|---|---|---|---|---|---|
| baseline | 32 | 3 | 64 | 0.3 | 0.9111 | 0.9113 |
| increased_filters | 64 | 3 | 64 | 0.3 | 0.9317 | 0.9313 |
| smaller_batch | 64 | 3 | 32 | 0.3 | 0.9381 | 0.9384 |
| **larger_kernel** | **64** | **5** | **32** | **0.3** | **0.9413** | **0.9411** |
| higher_dropout | 64 | 3 | 32 | 0.4 | 0.9349 | 0.9351 |

Best config: `larger_kernel` — 64 filters, kernel size 5, batch 32, dropout 0.3.

---

## Results Summary

| Model | Framework | Test macro-F1 |
|---|---|---|
| Random Forest | sklearn | 0.6452 |
| SVM | sklearn | 0.7027 |
| Logistic Regression | sklearn | 0.7083 |
| BiLSTM | TensorFlow/Keras | *(run notebook)* |
| 1D CNN (best config) | PyTorch | **0.9411** |

---

## Key Design Decisions

- **Energy-based window selection**: 1.5s windows centred on the highest-energy segment of each recording, rather than fixed-start or full-file windows. This removes silence padding and focuses on the actual spoken word.
- **Delta features**: Including Δ and ΔΔ MFCCs (40 → 120 features) captures velocity and acceleration of the spectral envelope, which encodes phonetic transitions.
- **Frozen split**: `data/processed/splits.csv` is committed so all models are evaluated on identical train/val/test partitions — no data leakage between experiments.
- **Macro-F1 as primary metric**: The 12 classes are not perfectly balanced; macro-F1 weights all classes equally and is more informative than accuracy.

---

## Google Colab

All scripts use relative paths and work on Colab without modification.  
For the BiLSTM notebook, mount Drive and upload `w1.5_energy_mfcc40_d/` to your Drive root, then set `FEAT_DIR` in Cell 0 accordingly (the notebook includes instructions).

```python
FEAT_DIR = '/content/drive/MyDrive/w1.5_energy_mfcc40_d'
```
