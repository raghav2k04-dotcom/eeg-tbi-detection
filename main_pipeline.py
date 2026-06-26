import os

print("Step 1: Checking dataset")
os.system("python src/check_dataset.py")

print("Step 2: Creating labels")
os.system("python src/create_labels.py")

print("Step 3: Preprocessing EEG")
os.system("python src/preprocess_eeg.py")

print("Step 4: Segmenting windows")
os.system("python src/segment_windows.py")

print("Step 5: Extracting bandpower features")
os.system("python src/extract_bandpower.py")

print("Step 6: Extracting EEGNet features")
os.system("python src/extract_features.py")

print("Step 7: Fusing features")
os.system("python src/fuse_features.py")

print("Step 8: Training XGBoost")
os.system("python src/train_xgboost.py")

print("Pipeline completed successfully!")