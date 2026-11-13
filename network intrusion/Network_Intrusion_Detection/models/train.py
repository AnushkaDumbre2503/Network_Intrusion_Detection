"""Train Random Forest classifier and benign-baseline Isolation Forest."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"


def load_split(name):
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run feature_engineering/feature_engineering_ml.py first.")
    df = pd.read_csv(path, low_memory=False)
    if "Label" not in df:
        raise ValueError(f"{path} has no Label column")
    metadata = df[[c for c in ["Timestamp", "Flow ID", "day"] if c in df]].copy()
    X = df.drop(columns=["Label", "Timestamp", "Flow ID", "day"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, df["Label"].astype(str), metadata


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    X_train, y_train, _ = load_split("train_data.csv")
    X_val, y_val, _ = load_split("validation_data.csv")
    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Train and validation feature columns/order do not match.")

    print("Training Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_split=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Fit the anomaly detector on benign training traffic only.
    benign_mask = y_train.str.casefold().eq("benign")
    if benign_mask.sum() < 10:
        raise ValueError("Insufficient benign training rows for Isolation Forest.")
    print(f"Training Isolation Forest on {benign_mask.sum():,} benign flows...")
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(X_train.loc[benign_mask])

    # Set the anomaly threshold from validation benign scores to target ~1% validation benign alerts.
    val_benign = X_val.loc[y_val.str.casefold().eq("benign")]
    if len(val_benign):
        val_scores = iso.decision_function(val_benign)
        anomaly_threshold = float(np.quantile(val_scores, 0.01))
    else:
        anomaly_threshold = 0.0
        print("WARNING: validation split has no benign rows; using threshold 0.0.")

    joblib.dump(rf, ARTIFACTS / "random_forest.pkl")
    joblib.dump(iso, ARTIFACTS / "isolation_forest.pkl")
    metadata = {
        "model_version": "2.0",
        "target_classes": rf.classes_.tolist(),
        "feature_names": X_train.columns.tolist(),
        "anomaly_threshold": anomaly_threshold,
        "anomaly_threshold_method": "1st percentile of validation benign decision_function scores",
        "isolation_forest_training_population": "Benign-only training rows",
        "random_state": 42,
    }
    joblib.dump(metadata, ARTIFACTS / "model_metadata.pkl")
    print("Models saved. Validation-derived anomaly threshold:", anomaly_threshold)
    print("Class labels:", rf.classes_.tolist())


if __name__ == "__main__":
    main()
