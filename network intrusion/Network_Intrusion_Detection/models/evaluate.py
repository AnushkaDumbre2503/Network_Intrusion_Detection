"""Evaluate saved models on the held-out test split."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"


def main():
    rf = joblib.load(ARTIFACTS / "random_forest.pkl")
    iso = joblib.load(ARTIFACTS / "isolation_forest.pkl")
    metadata = joblib.load(ARTIFACTS / "model_metadata.pkl")
    df = pd.read_csv(DATA / "test_data.csv", low_memory=False)
    y = df["Label"].astype(str)
    X = df.drop(columns=["Label", "Timestamp", "Flow ID", "day"], errors="ignore")
    X = X.reindex(columns=metadata["feature_names"]).apply(pd.to_numeric, errors="coerce").fillna(0)

    rf_pred = rf.predict(X)
    rf_prob = rf.predict_proba(X)
    print("\n=== RANDOM FOREST: MULTICLASS ===")
    print("Accuracy:", round(accuracy_score(y, rf_pred), 4))
    print(classification_report(y, rf_pred, labels=rf.classes_, zero_division=0))
    print("Confusion matrix (labels order):", rf.classes_.tolist())
    print(confusion_matrix(y, rf_pred, labels=rf.classes_).tolist())
    try:
        if len(rf.classes_) == 2:
            rf_roc_auc = roc_auc_score(y, rf_prob[:, 1])
            rf_pr_auc = average_precision_score((y == rf.classes_[1]).astype(int), rf_prob[:, 1])
        else:
            y_one_hot = label_binarize(y, classes=rf.classes_)
            rf_roc_auc = roc_auc_score(y_one_hot, rf_prob, average="macro", multi_class="ovr")
            rf_pr_auc = average_precision_score(y_one_hot, rf_prob, average="macro")
        print("Macro ROC-AUC (one-vs-rest):", round(float(rf_roc_auc), 4))
        print("Macro PR-AUC (one-vs-rest):", round(float(rf_pr_auc), 4))
    except ValueError as exc:
        rf_roc_auc = None
        rf_pr_auc = None
        print("Classifier AUC unavailable:", exc)

    # Binary attack-vs-benign evaluation for Isolation Forest.
    y_binary = y.str.casefold().ne("benign").astype(int)
    iso_scores = -iso.decision_function(X)  # larger means more anomalous
    threshold = -float(metadata["anomaly_threshold"])
    iso_pred = (iso_scores > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_binary, iso_pred, average="binary", zero_division=0)
    print("\n=== ISOLATION FOREST: ATTACK VS BENIGN ===")
    print("Precision:", round(float(p), 4), "Recall:", round(float(r), 4), "F1:", round(float(f1), 4))
    print("Confusion matrix:", confusion_matrix(y_binary, iso_pred).tolist())
    try:
        print("ROC-AUC:", round(float(roc_auc_score(y_binary, iso_scores)), 4))
        print("PR-AUC:", round(float(average_precision_score(y_binary, iso_scores)), 4))
    except ValueError as exc:
        print("AUC unavailable:", exc)

    report = {
        "test_rows": int(len(df)),
        "random_forest_accuracy": float(accuracy_score(y, rf_pred)),
        "random_forest_macro_roc_auc_ovr": None if rf_roc_auc is None else float(rf_roc_auc),
        "random_forest_macro_pr_auc_ovr": None if rf_pr_auc is None else float(rf_pr_auc),
        "random_forest_report": classification_report(y, rf_pred, labels=rf.classes_, output_dict=True, zero_division=0),
        "isolation_forest_binary": {"precision": float(p), "recall": float(r), "f1": float(f1)},
        "anomaly_threshold": metadata["anomaly_threshold"],
    }
    (ARTIFACTS / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSaved evaluation report to artifacts/evaluation_report.json")


if __name__ == "__main__":
    main()
