import numpy as np
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

print("Loading combined features and labels...")

X = np.load("combined_features.npy")
y = np.load("eeg_labels.npy")

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

# make lengths match
min_len = min(len(X), len(y))
X = X[:min_len]
y = y[:min_len]

print("Adjusted feature shape:", X.shape)
print("Adjusted label shape:", y.shape)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# add noise to reduce overfitting
X = X + np.random.normal(0, 0.13, X.shape)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Training XGBoost classifier...")

model = xgb.XGBClassifier(
    n_estimators=25,
    max_depth=2,
    learning_rate=0.22,
    subsample=0.45,
    colsample_bytree=0.45,
    gamma=6,
    reg_lambda=7,
    reg_alpha=5,
    tree_method="hist",
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Evaluating model...")

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\nAccuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

joblib.dump(model, "xgboost_model.pkl")

print("\nModel saved as: xgboost_model.pkl")

tbi_count = np.sum(pred == 1)
normal_count = np.sum(pred == 0)

print("\nPrediction Summary:")
print("Normal (Control):", normal_count)
print("TBI:", tbi_count)