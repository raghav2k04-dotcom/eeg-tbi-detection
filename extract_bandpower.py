import os
import numpy as np
import mne
from scipy.signal import welch

dataset_path = r"C:\Users\yogav\OneDrive\Desktop\EEG_PROJECT"

window_size = 1000
step_size = 1000
target_channels = 63
sfreq = 500

def bandpower(data, sf, band):
    freqs, psd = welch(data, sf, nperseg=256)
    idx = (freqs >= band[0]) & (freqs <= band[1])
    return np.mean(psd[:, idx], axis=1)

bands = {
    "delta": (1,4),
    "theta": (4,8),
    "alpha": (8,13),
    "beta": (13,30)
}

features = []
set_files = []

# find all EEG files
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".set"):
            set_files.append(os.path.join(root,file))

print("Total EEG files:",len(set_files))

for filepath in set_files:

    print("Processing:",filepath)

    try:
        raw = mne.io.read_raw_eeglab(filepath, preload=False)
    except Exception as e:
        print("Skipping corrupted file:", filepath)
        print("Reason:", e)
        continue

    remove_channels = []

    for ch in raw.ch_names:
        if "EOG" in ch or "VEOG" in ch or "EKG" in ch:
            remove_channels.append(ch)

    if len(remove_channels)>0:
        raw.drop_channels(remove_channels)

    samples = raw.n_times

    for start in range(0,samples-window_size,step_size):

        stop = start + window_size

        segment = raw.get_data(start=start, stop=stop)

        if segment.shape[0] >= target_channels:

            segment = segment[:target_channels,:]

            band_features = []

            for band in bands.values():
                bp = bandpower(segment, sfreq, band)
                band_features.extend(bp)

            features.append(band_features)

features = np.array(features)

print("Bandpower feature shape:",features.shape)

np.save("bandpower_features.npy",features)

print("Bandpower features saved successfully.")