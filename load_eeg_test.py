import os
import mne

dataset_path = r"C:\Users\yogav\OneDrive\Desktop\EEG_PROJECT"

set_files = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".set"):
            set_files.append(os.path.join(root, file))

print("Total EEG files found:", len(set_files))

if len(set_files) == 0:
    print("No .set files found")
else:
    filepath = set_files[0]

    print("\nLoading EEG file:")
    print(filepath)

    raw = mne.io.read_raw_eeglab(filepath, preload=True)

    print("\nEEG Information")
    print("Channels:", len(raw.ch_names))
    print("Sampling rate:", raw.info['sfreq'])
    print("Total samples:", raw.n_times)