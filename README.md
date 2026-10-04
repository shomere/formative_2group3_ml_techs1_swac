# Swahili Audio Classification — Formative Assignment 2

**Group 3 | Machine Learning Technologies 1**

12-class spoken Swahili word classification using classical and deep learning models.  
Dataset: [Zindi — Swahili Words Audio Classification](https://zindi.africa/competitions/swahili-words-audio-classification)

---

## Table of Contents

1. [Task Description](#task-description)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [Reproducing the Pipeline](#reproducing-the-pipeline)
6. [Feature Engineering](#feature-engineering)
7. [Models](#models)
8. [Results](#results)
9. [Key Design Decisions](#key-design-decisions)
10. [Google Colab](#google-colab)

---

## Task Description

Classify 16 kHz mono WAV recordings into one of **12 Swahili words**:

> `hapana` · `kumi` · `mbili` · `moja` · `nane` · `ndio` · `nne` · `saba` · `sita` · `tano` · `tatu` · `tisa`

Primary evaluation metric: **macro-averaged F1-score**, which weights all 12 classes equally regardless of class frequency — more informative than accuracy on an imbalanced dataset.

---

## Dataset

- Source: Zindi Swahili Words Audio Classification challenge
- Format: 16 kHz mono WAV files
- Labels: provided in `Train.csv` with columns `filename` and `label`
- Split: stratified 70% train / 15% validation / 15% test (frozen in `data/processed/splits.csv`)

The dataset contains variable-length recordings. Most clips are short (under 3 seconds) but some extend to 10–15 seconds with significant silence padding. EDA figures are saved to `reports/figures/`.

---

## Project Structure

```
formative_2group3_ml_techs1_swac/
│
├── src/                            reusable pipeline scripts
│   ├── data_overview.py            class distribution, duration stats, EDA plots
│   ├── split.py                    stratified train/val/test split (seed=42)
│   ├── features.py                 MFCC + delta feature extraction → .npz files
│   ├── baseline.py                 SVM / Random Forest / Logistic Regression
│   └── eda_audio.py                audio-level EDA (active speech, mel spectrograms)
│
├── notebooks/
│   └── bilstm_model.ipynb          BiLSTM model — TensorFlow/Keras (Colab-ready)
│
├── models/
│   └── cnn_1d_model.py             1D CNN architecture + trainer class — PyTorch
│
├── experiments/
│   └── cnn_1d_experiments.py       1D CNN hyperparameter sweep (5 configurations)
│
├── data/
│   ├── raw/                        Train.csv + audio files (not committed)
│   └── processed/
│       └── splits.csv              frozen train/val/test index (committed)
│
├── reports/
│   ├── experiments.csv             all baseline run logs
│   └── figures/                    EDA plots and confusion matrices
│       ├── class_counts.png
│       ├── duration_by_class.png
│       ├── duration_hist.png
│       ├── examples_mel.png
│       ├── speech_vs_total.png
│       ├── cm_baseline_logreg.png
│       ├── cm_baseline_svm.png
│       └── cm_baseline_rf.png
│
├── results/
│   ├── cnn_experiments.csv         1D CNN hyperparameter sweep results
│   └── hyperparameter_analysis.png visual comparison of CNN configurations
│
├── w1.5_energy_mfcc40_d/           pre-extracted features (not committed to git)
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── labels.json
│
└── requirements.txt
```

---

## Setup & Installation

### Requirements

```bash
pip install -r requirements.txt
```

On Linux or Google Colab, also install system audio libraries:

```bash
sudo apt install ffmpeg libsndfile1
```

### Data

Place the following under `data/raw/`:
- `Train.csv` — label index from the Zindi competition
- Extracted audio folder from `Swahili_words.zip`

The directory should look like:
```
data/raw/
  Train.csv
  audio/
    hapana_001.wav
    kumi_002.wav
    ...
```

---

## Reproducing the Pipeline

Run the steps below in order from the repo root. All paths are relative and work identically on local machines and Google Colab.

### Step 1 — Dataset Overview & EDA

```bash
python src/data_overview.py --train-csv data/raw/Train.csv --audio-dir data/raw
```

Outputs:
- Class distribution counts
- Duration statistics (min, max, mean, std)
- Saves `reports/figures/class_counts.png`, `duration_hist.png`, `duration_by_class.png`

### Step 2 — Audio-Level EDA

```bash
python src/eda_audio.py --index data/processed/train_index.csv
```

Outputs:
- Active speech duration vs total duration scatter plot
- Per-class mel spectrogram examples
- Saves `reports/figures/speech_vs_total.png`, `examples_mel.png`

### Step 3 — Create Train/Val/Test Split

```bash
python src/split.py --index data/processed/train_index.csv
```

Saves `data/processed/splits.csv` with a stratified 70/15/15 split using `random_state=42`.  
This file is committed to the repo so all team members and all models use identical partitions.

### Step 4 — Extract Features

```bash
python src/features.py \
    --splits data/processed/splits.csv \
    --window-sec 1.5 \
    --select energy \
    --n-mfcc 40 \
    --deltas
```

Saves `w1.5_energy_mfcc40_d/{train,val,test}.npz` and `labels.json`.

Output tensor shape: `(N, 150, 120)`
- `N` — number of clips
- `150` — timesteps (1.5s ÷ 10ms hop)
- `120` — features (40 MFCCs × 3: base + Δ + ΔΔ)

### Step 5 — Run Classical Baselines

Run each classifier separately:

```bash
python src/baseline.py --feat-dir w1.5_energy_mfcc40_d --model logreg
python src/baseline.py --feat-dir w1.5_energy_mfcc40_d --model svm
python src/baseline.py --feat-dir w1.5_energy_mfcc40_d --model rf
```

Each run:
- Performs a small grid search over hyperparameters on the validation set
- Evaluates the best config on the test set
- Saves a confusion matrix to `reports/figures/`
- Appends a result row to `reports/experiments.csv`

### Step 6 — BiLSTM (Jupyter / Colab)

Open `notebooks/bilstm_model.ipynb` and run all cells.

The notebook:
- Auto-detects Colab vs local and sets paths accordingly
- Loads `.npz` features, applies z-score normalisation (fitted on train only)
- Builds, trains, and evaluates the BiLSTM model
- Saves learning curves and confusion matrix to `reports/figures/`
- Appends results to `reports/experiments.csv`
- Saves the trained model to `models/bilstm_swahili.keras`

### Step 7 — 1D CNN Hyperparameter Sweep

```bash
python experiments/cnn_1d_experiments.py
```

Runs 5 configurations, saves results to `results/cnn_experiments.csv` and a comparison plot to `results/hyperparameter_analysis.png`.

---

## Feature Engineering

All models share the same pre-extracted features from `src/features.py`.

### Audio Preprocessing

1. **Load** at 16 kHz mono
2. **Peak-normalise** — divide by max absolute amplitude to handle recording level variation
3. **Window selection** — extract a 1.5s segment using energy-based selection (see below)

### Energy-Based Window Selection

Rather than taking the first 1.5s or the full clip, a sliding window finds the 1.5s segment with the highest RMS energy. This centres the window on the actual spoken word, removing leading/trailing silence that would otherwise dominate the feature representation.

### MFCC Extraction

- 40 Mel-Frequency Cepstral Coefficients (MFCCs)
- Frame size: 25ms (400 samples), hop: 10ms (160 samples)
- 64 mel filterbanks

### Delta Features

First-order (Δ) and second-order (ΔΔ) derivatives are appended to the base MFCCs:
- Δ MFCCs capture the velocity of the spectral envelope (how features change over time)
- ΔΔ MFCCs capture acceleration (rate of change of change)

This triples the feature dimension: 40 → 120, encoding phonetic transitions that static MFCCs miss.

### Final Shape

`(N, 150, 120)` — 150 timesteps × 120 features per clip.

---

## Models

### 1. Logistic Regression (`src/baseline.py`)

Features are aggregated over the time axis (mean, std, min, max → 480-dim vector) before classification. Grid search over `C ∈ {0.1, 1, 10}`.

Best params: `C=0.1`

### 2. SVM — RBF Kernel (`src/baseline.py`)

Same 480-dim aggregated features with StandardScaler. Grid search over `C ∈ {1, 10, 100}`.

Best params: `C=10`

### 3. Random Forest (`src/baseline.py`)

Same 480-dim aggregated features. Grid search over `n_estimators ∈ {200, 500}`.

Best params: `n_estimators=500`

### 4. BiLSTM (`notebooks/bilstm_model.ipynb`) — TensorFlow/Keras

Processes the full `(150, 120)` sequence without aggregation, capturing temporal dynamics in both directions.

**Architecture:**

```
Input (150, 120)
  → BiLSTM(128 units, return_sequences=True) → BatchNorm → Dropout(0.3)
  → BiLSTM(64 units,  return_sequences=False) → BatchNorm → Dropout(0.3)
  → Dense(128, ReLU) → Dropout(0.15)
  → Dense(12, softmax)
```

**Training config:**
- Optimiser: Adam (lr=1e-3)
- Loss: sparse categorical cross-entropy
- EarlyStopping: patience=12, monitor=val_accuracy, restore_best_weights=True
- ReduceLROnPlateau: factor=0.5, patience=6, min_lr=1e-6
- Max epochs: 60, batch size: 64

**Why BiLSTM?**  
Speech is a bidirectional temporal signal — the acoustic realisation of a phoneme depends on both preceding and following context (coarticulation). A BiLSTM processes each MFCC frame in both the forward and backward direction, capturing these dependencies. This is well-established in the speech recognition literature (Graves & Schmidhuber, 2005; Graves et al., 2013).

### 5. 1D CNN (`models/cnn_1d_model.py`) — PyTorch

Applies convolutional filters directly over the time axis to detect local temporal patterns such as the onset, nucleus, and coda of a syllable.

**Architecture:**

```
Input (120, 150)  ← features as channels, time as sequence length
  → Conv1D(→32, k=3) → BN → ReLU → MaxPool(2) → Dropout
  → Conv1D(→64, k=3) → BN → ReLU → MaxPool(2) → Dropout
  → Conv1D(→128, k=3) → BN → ReLU → GlobalAvgPool → Dropout
  → Dense(128, ReLU) → Dropout
  → Dense(12)
```

**Training config:**
- Optimiser: Adam (lr=1e-3)
- Loss: CrossEntropyLoss
- ReduceLROnPlateau: factor=0.5, patience=5
- EarlyStopping: patience=10
- Max epochs: 50

---

## Results

### Classical Baselines

| Model | Val Accuracy | Val macro-F1 | Test Accuracy | Test macro-F1 |
|---|---|---|---|---|
| Logistic Regression | 0.7159 | 0.7155 | 0.7095 | 0.7083 |
| SVM (RBF, C=10) | 0.7048 | 0.7046 | 0.7048 | 0.7027 |
| Random Forest (500 trees) | 0.6825 | 0.6847 | 0.6460 | 0.6452 |

### 1D CNN Hyperparameter Sweep

| Config | Filters | Kernel | Batch | Dropout | Test Accuracy | Test F1 |
|---|---|---|---|---|---|---|
| baseline | 32 | 3 | 64 | 0.3 | 0.9111 | 0.9113 |
| increased_filters | 64 | 3 | 64 | 0.3 | 0.9317 | 0.9313 |
| smaller_batch | 64 | 3 | 32 | 0.3 | 0.9381 | 0.9384 |
| **larger_kernel** | **64** | **5** | **32** | **0.3** | **0.9413** | **0.9411** |
| higher_dropout | 64 | 3 | 32 | 0.4 | 0.9349 | 0.9351 |

Best 1D CNN config: `larger_kernel` — 64 filters, kernel size 5, batch 32, dropout 0.3.

### Overall Comparison

| Model | Framework | Test macro-F1 |
|---|---|---|
| Random Forest | sklearn | 0.6452 |
| SVM | sklearn | 0.7027 |
| Logistic Regression | sklearn | 0.7083 |
| BiLSTM | TensorFlow/Keras | *(run notebook)* |
| **1D CNN (larger_kernel)** | **PyTorch** | **0.9411** |

The 1D CNN achieves a **+0.233 macro-F1 improvement** over the best classical baseline (Logistic Regression), demonstrating the advantage of learning directly from the temporal sequence rather than aggregating features.

---

## Key Design Decisions

**Energy-based window selection**  
Fixed-start windowing captures silence at the beginning of recordings. Energy-based selection finds the most acoustically active 1.5s segment, ensuring the model sees the actual spoken word rather than silence padding.

**Delta and delta-delta MFCCs**  
Static MFCCs describe the spectral shape at each frame but not how it changes. Δ and ΔΔ features encode the velocity and acceleration of the spectral envelope, capturing phonetic transitions (e.g. consonant-to-vowel coarticulation). This tripled the feature dimension (40 → 120) and consistently improved all models.

**Frozen train/val/test split**  
`data/processed/splits.csv` is committed to the repository. All models — classical and neural — are evaluated on the exact same partitions, preventing any data leakage and ensuring fair comparison across experiments.

**Macro-F1 as primary metric**  
The 12 classes are not perfectly balanced. Macro-F1 computes F1 per class and averages them equally, penalising models that perform well on frequent classes but poorly on rare ones. Accuracy would be misleading here.

**Aggregation for classical models vs full sequence for neural models**  
Classical models (SVM, RF, LogReg) require fixed-size input, so features are aggregated over the time axis (mean, std, min, max → 480-dim). Neural models (BiLSTM, 1D CNN) consume the full `(150, 120)` sequence, preserving temporal structure — which explains their substantially higher performance.

---

## Google Colab

All scripts use relative paths and work on Colab without modification.

For the BiLSTM notebook:
1. Mount Google Drive
2. Upload the `w1.5_energy_mfcc40_d/` folder to your Drive root
3. The notebook auto-detects Colab and sets paths accordingly

```python
# Cell 0 in bilstm_model.ipynb — set this if running on Colab
FEAT_DIR = '/content/drive/MyDrive/w1.5_energy_mfcc40_d'
```
