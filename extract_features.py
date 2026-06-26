import os
import numpy as np
import mne
import tensorflow as tf
from tensorflow.keras import Model

dataset_path = r"C:\Users\yogav\OneDrive\Desktop\EEG_PROJECT"

window_size = 1000
step_size = 1000
target_channels = 63

print("Loading EEGNet model...")
model = tf.keras.models.load_model("eegnet_model.h5")

dummy = np.zeros((1,63,1000,1),dtype=np.float32)
model.predict(dummy)

feature_model = Model(inputs=model.layers[0].input,
                      outputs=model.layers[-2].output)

features = []
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

    # label from subject id
    if "sub-0" in filepath:
        subject_id = int(filepath.split("sub-")[1][:3])
    else:
        continue

    # example labeling rule (modify if needed)
    if subject_id <= 10:
        label = 0   # control
    else:
        label = 1   # TBI

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

            segment = segment.astype(np.float32)

            segment = np.expand_dims(segment,axis=-1)
            segment = np.expand_dims(segment,axis=0)

            feat = feature_model.predict(segment,verbose=0)

            features.append(feat[0])
            labels.append(label)

features = np.array(features)
labels = np.array(labels)

print("Feature shape:",features.shape)
print("Label shape:",labels.shape)

np.save("eeg_features.npy",features)
np.save("eeg_labels.npy",labels)

print("Saved features and labels")
