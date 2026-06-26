import numpy as np

print("Loading EEGNet features...")
eegnet = np.load("eeg_features.npy")

print("Loading bandpower features...")
band = np.load("bandpower_features.npy")

print("EEGNet shape:", eegnet.shape)
print("Bandpower shape:", band.shape)

# ensure same length
min_len = min(len(eegnet), len(band))

eegnet = eegnet[:min_len]
band = band[:min_len]

# combine features
combined = np.concatenate([eegnet, band], axis=1)

print("Combined feature shape:", combined.shape)

np.save("combined_features.npy", combined)

print("Saved combined_features.npy")