# Swahili Audio Classification

Formative Assignment 2, Group 3, Machine Learning Technologies 1

This repo is our attempt at the Zindi *Swahili Words Audio Classification* challenge: given a short recording of someone saying a Swahili word, which of 12 words was it? We wanted to see how much it helps to model the audio as a sequence, so we compare simple classical models against neural ones that read the recording frame by frame.

## Contents

- [The task](#the-task)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Features](#features)
- [Models](#models)
- [Results](#results)
- [Design decisions](#design-decisions)
- [Running on Google Colab](#running-on-google-colab)

## The task

Each clip is a 16 kHz mono WAV file containing one of these words:

`hapana` · `kumi` · `mbili` · `moja` · `nane` · `ndio` · `nne` · `saba` · `sita` · `tano` · `tatu` · `tisa`

We use macro-averaged F1 as our main metric. It treats every class equally, so a model can't hide a weak class behind strong ones.

## Dataset

- **Source:** Zindi, *Swahili Words Audio Classification*
- **Audio:** 16 kHz mono WAV
- **Labels:** `Train.csv`, with columns `filename` and `label`
- **Split:** stratified 70% train, 15% validation, 15% test, saved in `data/processed/splits.csv` so every model sees the same partitions

Clip lengths vary a lot. Most recordings are under 3 seconds, but some run to 10-15 seconds and are mostly silence, which is why we ended up choosing a window around the speech (see [Features](#features)). The EDA plots are in `reports/figures/`.

## Project structure

```
formative_2group3_ml_techs1_swac/
├── src/                            pipeline scripts
│   ├── data_overview.py            class counts, duration stats, EDA plots
│   ├── split.py                    stratified train/val/test split (seed 42)
│   ├── features.py                 MFCC + delta extraction, saved as .npz
│   ├── baseline.py                 SVM / Random Forest / Logistic Regression
│   └── eda_audio.py                active-speech analysis, mel spectrograms
├── notebooks/
│   └── bilstm_model.ipynb          BiLSTM (TensorFlow/Keras, runs on Colab)
├── models/
│   └── cnn_1d_model.py             1D CNN and its trainer (PyTorch)
├── experiments/
│   └── cnn_1d_experiments.py       1D CNN sweep over 5 configurations
├── data/
│   ├── raw/                        Train.csv and audio (not committed)
│   └── processed/splits.csv        frozen split (committed)
├── reports/
│   ├── experiments.csv             baseline run logs
│   └── figures/                    EDA plots, confusion matrices
├── results/
│   ├── cnn_experiments.csv         CNN sweep results
│   └── hyperparameter_analysis.png comparison of CNN configurations
├── w1.5_energy_mfcc40_d/           extracted features (not committed)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

On Linux or Colab you also need the system audio libraries:

```bash
sudo apt install ffmpeg libsndfile1
```

Put the data under `data/raw/`: `Train.csv` from Zindi, plus the audio extracted from `Swahili_words.zip`:

```
data/raw/
  Train.csv
  audio/
    hapana_001.wav
    kumi_002.wav
    ...
```

## Running the pipeline

Run these from the repo root, in order. Paths are relative, so the same commands work locally and on Colab.

**1. Dataset overview**

```bash
python src/data_overview.py --train-csv data/raw/Train.csv --audio-dir data/raw
```

Prints class counts and duration statistics, and saves `class_counts.png`, `duration_hist.png`, and `duration_by_class.png` to `reports/figures/`.

**2. Audio-level EDA**

```bash
python src/eda_audio.py --index data/processed/train_index.csv
```

Plots active speech against total clip length and one mel spectrogram per class (`speech_vs_total.png`, `examples_mel.png`).

**3. Train/val/test split**

```bash
python src/split.py --index data/processed/train_index.csv
```

Writes `data/processed/splits.csv` (stratified 70/15/15, `random_state=42`). The file is committed so the whole team uses the same split.

**4. Feature extraction**

```bash
python src/features.py \
    --splits data/processed/splits.csv \
    --window-sec 1.5 \
    --select energy \
    --n-mfcc 40 \
    --deltas
```

Writes `w1.5_energy_mfcc40_d/{train,val,test}.npz` and `labels.json`. Each clip becomes a tensor of shape `(150, 120)`: 150 time steps (1.5 s at a 10 ms hop) and 120 features (40 MFCCs, their deltas, and their delta-deltas).

**5. Classical baselines**

```bash
python src/baseline.py --feat-dir w1.5_energy_mfcc40_d --model logreg
python src/baseline.py --feat-dir w1.5_energy_mfcc40_d --model svm
python src/baseline.py --feat-dir w1.5_energy_mfcc40_d --model rf
```

Each run does a small grid search on the validation set, evaluates the best setting on the test set, saves a confusion matrix to `reports/figures/`, and appends a row to `reports/experiments.csv`.

**6. BiLSTM**

Open `notebooks/bilstm_model.ipynb` and run all cells. The notebook detects whether it's on Colab or a local machine, loads the `.npz` features, z-scores them using training statistics only, trains and evaluates the model, saves the learning curves and confusion matrix, logs the result to `reports/experiments.csv`, and writes the model to `models/bilstm_swahili.keras`.

**7. 1D CNN sweep**

```bash
python experiments/cnn_1d_experiments.py
```

Trains five configurations and saves the results to `results/cnn_experiments.csv` and a comparison plot to `results/hyperparameter_analysis.png`.

## Features

Every model uses the features from `src/features.py`, so the comparison is about the models and not the preprocessing.

We load each clip at 16 kHz mono and peak-normalize it, so quiet and loud recordings look alike. The next step matters most. Taking the first 1.5 seconds of a clip would often give us silence, so we slide a 1.5 s window over the clip and keep the segment with the most energy (RMS). That puts the window on the spoken word.

From that window we compute 40 MFCCs (25 ms frames, 10 ms hop, 64 mel filterbanks) and add their first- and second-order deltas. A static MFCC describes the spectrum at one instant. The deltas describe how it is changing, which is where a lot of the information about how one sound moves into the next lives. That takes us from 40 to 120 features per frame, and one `(150, 120)` array per clip.

## Models

### Classical baselines (`src/baseline.py`)

These need fixed-size input, so we summarize each clip over time with the mean, standard deviation, minimum, and maximum of every feature. That gives a 480-dimensional vector.

| Model | Grid searched | Best setting |
|---|---|---|
| Logistic Regression | C in {0.1, 1, 10} | C = 0.1 |
| SVM (RBF kernel, with StandardScaler) | C in {1, 10, 100} | C = 10 |
| Random Forest | n_estimators in {200, 500} | 500 |

### BiLSTM (`notebooks/bilstm_model.ipynb`, TensorFlow/Keras)

The BiLSTM reads the full `(150, 120)` sequence with no summarizing. Speech is context-dependent: how a sound is produced depends on what comes before and after it (coarticulation). A bidirectional LSTM sees each frame from both directions, which suits this. The approach is well established in speech recognition (Graves & Schmidhuber, 2005; Graves et al., 2013).

```
Input (150, 120)
  -> BiLSTM(128, return_sequences=True) -> BatchNorm -> Dropout(0.3)
  -> BiLSTM(64)                         -> BatchNorm -> Dropout(0.3)
  -> Dense(128, ReLU) -> Dropout(0.15)
  -> Dense(12, softmax)
```

Training: Adam (lr 1e-3), sparse categorical cross-entropy, batch size 64, up to 60 epochs. Early stopping watches validation accuracy (patience 12, restoring the best weights), and the learning rate halves after 6 epochs without improvement (minimum 1e-6).

### 1D CNN (`models/cnn_1d_model.py`, PyTorch)

The convolutions slide along the time axis, so they pick up short local patterns, such as the onset, vowel, and ending of a syllable.

```
Input (120, 150)   # features as channels, time as length
  -> Conv1D(32, k=3)  -> BN -> ReLU -> MaxPool(2) -> Dropout
  -> Conv1D(64, k=3)  -> BN -> ReLU -> MaxPool(2) -> Dropout
  -> Conv1D(128, k=3) -> BN -> ReLU -> GlobalAvgPool -> Dropout
  -> Dense(128, ReLU) -> Dropout
  -> Dense(12)
```

Training: Adam (lr 1e-3), cross-entropy, up to 50 epochs, learning rate halved after 5 epochs without improvement, early stopping with patience 10.

## Results

### Classical baselines

| Model | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|---|
| Logistic Regression | 0.7159 | 0.7155 | 0.7095 | 0.7083 |
| SVM (RBF, C=10) | 0.7048 | 0.7046 | 0.7048 | 0.7027 |
| Random Forest (500 trees) | 0.6825 | 0.6847 | 0.6460 | 0.6452 |

### 1D CNN sweep

| Config | Filters | Kernel | Batch | Dropout | Test accuracy | Test F1 |
|---|---|---|---|---|---|---|
| baseline | 32 | 3 | 64 | 0.3 | 0.9111 | 0.9113 |
| increased_filters | 64 | 3 | 64 | 0.3 | 0.9317 | 0.9313 |
| smaller_batch | 64 | 3 | 32 | 0.3 | 0.9381 | 0.9384 |
| larger_kernel | 64 | 5 | 32 | 0.3 | 0.9413 | 0.9411 |
| higher_dropout | 64 | 3 | 32 | 0.4 | 0.9349 | 0.9351 |

The best configuration was `larger_kernel` (64 filters, kernel size 5, batch size 32, dropout 0.3).

### Overall

| Model | Framework | Test macro-F1 |
|---|---|---|
| Random Forest | scikit-learn | 0.6452 |
| SVM | scikit-learn | 0.7027 |
| Logistic Regression | scikit-learn | 0.7083 |
| BiLSTM | TensorFlow/Keras | *to be filled in after running the notebook* |
| 1D CNN (`larger_kernel`) | PyTorch | 0.9411 |

The best CNN beats the best classical baseline (Logistic Regression) by 0.233 macro-F1. The most likely reason is that the classical models only see summary statistics, so the CNN keeps information about timing that they lose.

## Design decisions

A few choices shaped everything else:

- **Silence was a real problem**, so we pick the highest-energy window instead of a fixed start.
- **We froze one split** and committed `splits.csv`. Every model, classical or neural, is scored on the same clips, so the numbers can be compared directly.
- **We lead with macro-F1** because the classes aren't perfectly balanced.
- **Classical models get summaries, neural models get sequences.** SVM, Random Forest, and Logistic Regression need fixed-size vectors, so we reduce each clip to its per-feature mean, standard deviation, minimum, and maximum. The BiLSTM and CNN see the whole `(150, 120)` sequence.

## Notes from the team

*Replace this section with your own observations before submitting. Things worth writing down: what you tried that did not work, what surprised you in the results or the confusion matrices, which word pairs the models mix up and what you heard when you listened to those clips, and how you split the work.*

## Running on Google Colab

The scripts use relative paths and run on Colab as they are. For the BiLSTM notebook:

1. Mount Google Drive.
2. Upload the `w1.5_energy_mfcc40_d/` folder to the root of your Drive.
3. In cell 0 of `bilstm_model.ipynb`, set:

```python
FEAT_DIR = '/content/drive/MyDrive/w1.5_energy_mfcc40_d'
```
