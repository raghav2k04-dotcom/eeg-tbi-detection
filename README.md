# EEG-Based TBI Detection using EEGNet + XGBoost

A deep learning pipeline for classifying EEG signals as **Control (Normal)** or **TBI (Traumatic Brain Injury)** using a hybrid approach combining EEGNet (CNN) and XGBoost.

---

## Project Overview

This project processes raw EEG recordings in EEGLAB `.set` format, extracts deep features using a custom EEGNet model alongside traditional bandpower features (delta, theta, alpha, beta), and fuses them to train an XGBoost classifier.

**Achieved Results:**
- Accuracy: ~100% on test set (8550 windows)
- Control samples correctly classified: 4150
- TBI samples correctly classified: 4400

---

## Pipeline Architecture

```
Raw EEG (.set files)
        │
        ▼
  Preprocessing          ← bandpass filter (1–40 Hz), drop EOG/EKG channels
        │
        ▼
  Window Segmentation    ← 1000-sample windows, 63 channels
        │
   ┌────┴────┐
   ▼         ▼
EEGNet    Bandpower      ← delta, theta, alpha, beta
Features  Features
   └────┬────┘
        ▼
  Feature Fusion         ← concatenate EEGNet + bandpower vectors
        │
        ▼
  XGBoost Classifier     ← binary: Control vs TBI
        │
        ▼
  Evaluation & Plots
```

---

## Project Structure

```
eeg-tbi-detection/
│
├── main_pipeline.py          # Runs all steps end-to-end
│
├── src/
│   ├── check_dataset.py          # Scans dataset for .set files
│   ├── load_eeg_test.py          # Sanity-checks one EEG file
│   ├── preprocess_eeg.py         # Bandpass filter + channel cleanup
│   ├── create_labels.py          # Assigns labels from subject IDs
│   ├── segment_windows.py        # Slices EEG into windows, saves .npy
│   ├── train_eegnet_stream.py    # Trains EEGNet using a RAM-safe generator
│   ├── extract_features.py       # Extracts deep features from EEGNet
│   ├── extract_bandpower.py      # Extracts delta/theta/alpha/beta features
│   ├── fuse_features.py          # Concatenates EEGNet + bandpower features
│   ├── train_xgboost.py          # Trains and evaluates XGBoost classifier
│   └── evaluate_eegnet.py        # Evaluates standalone EEGNet performance
│
├── plots/
│   ├── plot_confusion_matrix.py      # Heatmap of predictions
│   ├── plot_feature_importance.py    # Top XGBoost feature gains
│   └── plot_prediction_distribution.py  # Bar chart of Control vs TBI counts
│
├── models/                   # Saved model files (not tracked in Git)
│   ├── eegnet_model.h5
│   └── xgboost_model.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/eeg-tbi-detection.git
cd eeg-tbi-detection
pip install -r requirements.txt
```

---

## Dataset

This project uses EEG recordings in **EEGLAB `.set` format**, organized by subject (e.g. `sub-001`, `sub-002`, ...).

- Subjects 1–10 → **Control (label 0)**
- Subjects 11+ → **TBI (label 1)**

> Update the `dataset_path` variable in each script to point to your local dataset folder before running.

---

## Running the Pipeline

Run everything end-to-end:

```bash
python main_pipeline.py
```

Or run individual steps:

```bash
python src/check_dataset.py          # Step 1: verify dataset
python src/preprocess_eeg.py         # Step 2: preprocess
python src/segment_windows.py        # Step 3: segment windows
python src/train_eegnet_stream.py    # Step 4: train EEGNet
python src/extract_features.py       # Step 5: extract deep features
python src/extract_bandpower.py      # Step 6: extract bandpower features
python src/fuse_features.py          # Step 7: fuse features
python src/train_xgboost.py          # Step 8: train XGBoost
```

---

## Visualizations

```bash
python plots/plot_confusion_matrix.py
python plots/plot_feature_importance.py
python plots/plot_prediction_distribution.py
```

---

## Requirements

See `requirements.txt` for full list. Key dependencies:

- Python 3.8+
- TensorFlow 2.x
- MNE (EEG processing)
- XGBoost
- scikit-learn
- NumPy, SciPy
- Matplotlib, Seaborn

---

## Model Details

**EEGNet (CNN):**
- Input: `(63 channels, 1000 samples, 1)`
- Conv2D → DepthwiseConv2D → SeparableConv2D
- Output: 2-class softmax
- Trained with a RAM-safe streaming generator

**XGBoost Classifier:**
- Input: fused EEGNet deep features + bandpower features
- `n_estimators=25`, `max_depth=2`, `learning_rate=0.22`
- Regularized with `gamma=6`, `reg_lambda=7`, `reg_alpha=5`

---

## Author

**Yogavarshini** — EEG Signal Processing & Deep Learning Project
