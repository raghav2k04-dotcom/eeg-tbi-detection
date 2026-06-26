import os
import mne

dataset_path = r"C:\Users\yogav\OneDrive\Desktop\EEG_PROJECT"

set_files = []

# find all EEG files
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".set"):
            set_files.append(os.path.join(root, file))

print("Total EEG recordings:", len(set_files))

# load the first file for testing
filepath = set_files[0]

print("\nLoading EEG file:")
print(filepath)

raw = mne.io.read_raw_eeglab(filepath, preload=True)

print("\nOriginal channels:", len(raw.ch_names))

# bandpass filter
raw.filter(1., 40.)

# remove non-brain channels
remove_channels = []

for ch in raw.ch_names:
    if "EOG" in ch or "VEOG" in ch or "EKG" in ch:
        remove_channels.append(ch)

if len(remove_channels) > 0:
    raw.drop_channels(remove_channels)

print("Channels after cleaning:", len(raw.ch_names))

data = raw.get_data()

print("EEG data shape:", data.shape)