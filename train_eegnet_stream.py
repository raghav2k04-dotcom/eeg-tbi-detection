import os
import numpy as np
import mne
import tensorflow as tf
from tensorflow.keras import layers, models

dataset_path = r"C:\Users\yogav\OneDrive\Desktop\EEG_PROJECT"

window_size = 1000
step_size = 1000
target_channels = 63
batch_size = 32

# -------------------------------------------------
# EEGNet model
# -------------------------------------------------

def build_eegnet():

    model = models.Sequential()

    model.add(layers.Input(shape=(target_channels, window_size, 1)))

    model.add(layers.Conv2D(16, (1,64), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())

    model.add(layers.DepthwiseConv2D((target_channels,1), activation="relu"))
    model.add(layers.BatchNormalization())

    model.add(layers.AveragePooling2D((1,4)))
    model.add(layers.Dropout(0.5))

    model.add(layers.SeparableConv2D(32, (1,16), activation="relu"))
    model.add(layers.BatchNormalization())

    model.add(layers.AveragePooling2D((1,8)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())

    model.add(layers.Dense(2, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -------------------------------------------------
# Data generator (RAM-safe)
# -------------------------------------------------

def eeg_generator():

    set_files = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".set"):
                set_files.append(os.path.join(root, file))

    while True:

        X_batch = []
        y_batch = []

        for filepath in set_files:

            try:
                raw = mne.io.read_raw_eeglab(filepath, preload=False)
            except:
                continue

            remove_channels = []

            for ch in raw.ch_names:
                if "EOG" in ch or "VEOG" in ch or "EKG" in ch:
                    remove_channels.append(ch)

            if len(remove_channels) > 0:
                raw.drop_channels(remove_channels)

            samples = raw.n_times

            for start in range(0, samples - window_size, step_size):

                stop = start + window_size

                segment = raw.get_data(start=start, stop=stop)

                if segment.shape[0] >= target_channels:

                    segment = segment[:target_channels,:]

                    segment = segment.astype(np.float32)

                    segment = np.expand_dims(segment, axis=-1)

                    X_batch.append(segment)

                    y_batch.append(0)

                if len(X_batch) == batch_size:

                    yield np.array(X_batch), np.array(y_batch)

                    X_batch = []
                    y_batch = []


# -------------------------------------------------
# Train model
# -------------------------------------------------

print("Building EEGNet model...")

model = build_eegnet()

print("Starting training...")

model.fit(
    eeg_generator(),
    steps_per_epoch=200,
    epochs=5
)

print("Training finished.")

model.save("eegnet_model.h5")
print("Model saved as eegnet_model.h5")
print("Model saved successfully.")