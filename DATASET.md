# Dataset Information

## Project: EEG-Based Prognostic Prediction

The dataset used in this project is approximately **18 GB** and is not included in this repository.

## Download the Dataset

[Click here to download from Google Drive](https://drive.google.com/drive/folders/1C-hqcL2d3rsDdVngfyPUDje3zECRG9fZ)

## Setup Instructions

1. Download the dataset from the link above
2. Place the dataset folder inside your project directory
3. Update the `dataset_path` variable in each script to match your local path

For example, in `src/check_dataset.py`:
```python
dataset_path = r"C:\Your\Path\To\EEG_PROJECT"
```

> The dataset contains EEG recordings in EEGLAB `.set` format, organized by subject ID (e.g. `sub-001`, `sub-002`, ...).
