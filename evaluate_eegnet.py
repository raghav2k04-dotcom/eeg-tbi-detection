import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("Loading EEG windows...")

X = np.load(r"D:\EEG_WINDOWS\windows_small.npy")
y = np.load(r"D:\EEG_WINDOWS\labels_small.npy")

print("Shape:", X.shape)


# CNN expects channel dimension
X = np.expand_dims(X, axis=-1)


# split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Loading trained EEGNet model...")

model = tf.keras.models.load_model("eegnet_model.h5")


print("Evaluating EEGNet...")

pred = model.predict(X_test)

pred = np.argmax(pred, axis=1)

acc = accuracy_score(y_test, pred)

print("\nAccuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))