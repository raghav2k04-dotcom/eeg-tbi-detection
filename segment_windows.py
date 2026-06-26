import os
import numpy as np
import mne

# Dataset location
dataset_path = r"D:\EEG_PROJECT"

# Where to save windows
save_folder = r"D:\EEG_WINDOWS"

os.makedirs(save_folder, exist_ok=True)

window_size = 1000
step_size = 1000
target_channels = 63
max_windows = 8000

windows = []
labels = []

set_files = []

# Find all .set files
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".set"):
            set_files.append(os.path.join(root, file))

print("Total EEG files found:", len(set_files))


for filepath in set_files:

    print("Processing:", filepath)

    try:
        raw = mne.io.read_raw_eeglab(filepath, preload=False)
    except Exception as e:
        print("Skipping file due to error:", e)
        continue

    # Remove non-brain channels
    remove_channels = []

    for ch in raw.ch_names:
        if "EOG" in ch or "VEOG" in ch or "EKG" in ch:
            remove_channels.append(ch)

    if len(remove_channels) > 0:
        raw.drop_channels(remove_channels)

    samples = raw.n_times

    # Label based on subject ID
    subject_id = int(filepath.split("sub-")[1][:3])

    if subject_id <= 10:
        label = 0
    else:
        label = 1

    for start in range(0, samples - window_size, step_size):

        stop = start + window_size

        segment = raw.get_data(start=start, stop=stop)

        if segment.shape[0] >= target_channels:

            segment = segment[:target_channels, :]

            windows.append(segment.astype(np.float32))
            labels.append(label)

        if len(windows) >= max_windows:
            break

    if len(windows) >= max_windows:
        break


# Convert to numpy arrays
windows = np.array(windows)
labels = np.array(labels)

print("\nWindows shape:", windows.shape)
print("Labels shape:", labels.shape)


# Save safely
windows_file = os.path.join(save_folder, "windows_small.npy")
labels_file = os.path.join(save_folder, "labels_small.npy")

np.save(windows_file, windows)
np.save(labels_file, labels)

print("\nSaved successfully:")
print(windows_file)
print(labels_file)