import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import joblib
import os
import numpy as np

# ================= CREATE ARTIFACT DIR =================
os.makedirs("artifacts", exist_ok=True)

# ================= LOAD DATA =================
train_df = pd.read_csv("data/train_data.csv")

print("Initial shape:", train_df.shape)

# ================= CLEAN DATA =================

# 1. Drop rows where Label is missing (IMPORTANT FIX)
train_df = train_df.dropna(subset=["Label"])

# 2. Reset index
train_df = train_df.reset_index(drop=True)

# 3. Split features and target
y_train = train_df["Label"]
X_train_raw = train_df.drop(columns=["Label"])

# 4. Keep only numeric features
X_train = X_train_raw.select_dtypes(include=[np.number])

# 5. Handle missing values
X_train = X_train.fillna(0)

print("Final training shape:", X_train.shape)
print("Label distribution:\n", y_train.value_counts())

# ================= ISOLATION FOREST =================
print("\nTraining Isolation Forest...")

iso = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42,
    n_jobs=-1
)

iso.fit(X_train)

# ================= RANDOM FOREST =================
print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ================= SAVE MODELS =================
joblib.dump(iso, "artifacts/isolation_forest.pkl")
joblib.dump(rf, "artifacts/random_forest.pkl")

print("\n✅ Models trained and saved successfully!")