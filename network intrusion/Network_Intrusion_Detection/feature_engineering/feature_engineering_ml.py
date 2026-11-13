"""Create leakage-safe train/validation/test feature datasets and preprocessing artifact."""
from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.preprocessing import fit_preprocessor, transform_features

INPUT = ROOT / "data" / "processed" / "cleaned_cicids2017.csv"
OUTPUT = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {INPUT}. Run data_prep/data_prep.py first.")
    df = pd.read_csv(INPUT, low_memory=False)
    if "Label" not in df:
        raise ValueError("Cleaned dataset must contain a Label column.")
    df["Label"] = df["Label"].astype(str).str.strip()
    df = df[df["Label"].ne("")].reset_index(drop=True)

    # 70% train, 15% validation, 15% test. All fitted transforms see train only.
    train_df, holdout_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["Label"]
    )
    val_df, test_df = train_test_split(
        holdout_df, test_size=0.50, random_state=42, stratify=holdout_df["Label"]
    )

    X_train, state = fit_preprocessor(train_df)
    X_val = transform_features(val_df, state)
    X_test = transform_features(test_df, state)

    # Assign labels positionally to prevent pandas index alignment errors.
    def save_split(features, source, filename):
        result = features.reset_index(drop=True).copy()
        result["Label"] = source["Label"].to_numpy()
        for metadata in ("Timestamp", "Flow ID", "day"):
            if metadata in source.columns:
                result[metadata] = source[metadata].reset_index(drop=True).to_numpy()
        result.to_csv(OUTPUT / filename, index=False)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    save_split(X_train, train_df, "train_data.csv")
    save_split(X_val, val_df, "validation_data.csv")
    save_split(X_test, test_df, "test_data.csv")
    joblib.dump(state, ARTIFACTS / "preprocessing.pkl")

    print("Leakage-safe datasets generated.")
    print(f"Train: {len(train_df):,}; validation: {len(val_df):,}; test: {len(test_df):,}")
    print("Features:", len(state["feature_names"]))
    print("Classes:", sorted(df["Label"].unique().tolist()))
    print("Saved preprocessing state to", ARTIFACTS / "preprocessing.pkl")


if __name__ == "__main__":
    main()
