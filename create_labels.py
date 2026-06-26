import os
import numpy as np
import mne

dataset_path = r"C:\Users\yogav\OneDrive\Desktop\EEG_PROJECT"

window_size = 1000
step_size = 1000

labels = []

set_files = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".set"):
            set_files.append(os.path.join(root,file))

print("Total EEG files:",len(set_files))

for filepath in set_files:

    print("Processing:",filepath)

    try:
        raw = mne.io.read_raw_eeglab(filepath, preload=False)
    except:
        continue

    samples = raw.n_times

    subject_id = int(filepath.split("sub-")[1][:3])

    # example labeling rule
    if subject_id <= 10:
        label = 0   # control
    else:
        label = 1   # TBI

    for start in range(0, samples-window_size, step_size):
        labels.append(label)

labels = np.array(labels)

print("Label shape:",labels.shape)

np.save("eeg_labels.npy",labels)

print("Labels saved successfully")