"""Inference for raw CICIDS2017 flow CSV files using saved training artifacts."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def _temporal_summary(frame: pd.DataFrame, anomaly_flags: np.ndarray) -> dict:
    timestamp_col = next((c for c in ("Timestamp", "timestamp", "Flow Start Time") if c in frame.columns), None)
    if timestamp_col is None:
        return {"time_bucket": [], "anomaly_count": [], "available": False,
                "message": "No timestamp column supplied; temporal analysis is unavailable."}
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce", dayfirst=False)
    valid = timestamps.notna()
    if not valid.any():
        return {"time_bucket": [], "anomaly_count": [], "available": False,
                "message": "Timestamp column could not be parsed."}
    temp = pd.DataFrame({"timestamp": timestamps[valid], "is_anomaly": anomaly_flags[valid.to_numpy()]})
    temp = temp.set_index("timestamp").sort_index()
    counts = temp["is_anomaly"].resample("15min").sum()
    return {"time_bucket": [t.strftime("%Y-%m-%d %H:%M") for t in counts.index],
            "anomaly_count": [int(v) for v in counts.values], "available": True}


def run_prediction(csv_path: str) -> dict:
    required_artifacts = ["random_forest.pkl", "isolation_forest.pkl", "preprocessing.pkl", "model_metadata.pkl"]
    missing = [name for name in required_artifacts if not (ARTIFACTS / name).exists()]
    if missing:
        raise RuntimeError("Required model artifacts are missing: " + ", ".join(missing) + ". Run the training workflow first.")

    frame = pd.read_csv(csv_path, low_memory=False)
    # Raw CICIDS2017 CSVs have leading spaces in column names; normalise them.
    frame.columns = frame.columns.str.strip()

    # Log uploaded columns so debugging is straightforward.
    import logging
    logger = logging.getLogger("uvicorn.error")
    logger.info("Uploaded CSV columns (%d): %s", len(frame.columns), frame.columns.tolist())
    print(f"[DEBUG] Uploaded CSV columns ({len(frame.columns)}): {frame.columns.tolist()}", flush=True)

    # Attempt case-insensitive / underscore-tolerant column matching.
    from models.preprocessing import RAW_FEATURES, transform_features
    existing_cols = {c.casefold().replace("_", " "): c for c in frame.columns}
    rename_map = {}
    for expected in RAW_FEATURES:
        if expected not in frame.columns:
            key = expected.casefold().replace("_", " ")
            if key in existing_cols:
                rename_map[existing_cols[key]] = expected
    if rename_map:
        logger.info("Auto-renaming columns: %s", rename_map)
        frame.rename(columns=rename_map, inplace=True)

    if frame.empty:
        raise ValueError("The uploaded CSV contains no rows.")
    if len(frame) > 1_000_000:
        raise ValueError("The uploaded CSV exceeds the 1,000,000-flow processing limit.")

    # Labels are ignored at inference. The input still must contain the raw CICIDS feature schema.
    state = joblib.load(ARTIFACTS / "preprocessing.pkl")
    rf = joblib.load(ARTIFACTS / "random_forest.pkl")
    iso = joblib.load(ARTIFACTS / "isolation_forest.pkl")
    metadata = joblib.load(ARTIFACTS / "model_metadata.pkl")
    model_features = metadata["feature_names"]

    missing_raw = [c for c in RAW_FEATURES if c not in frame.columns]
    if not missing_raw:
        # Raw CSV — run the full preprocessing pipeline.
        X = transform_features(frame, state).reindex(columns=model_features)
    else:
        # Pre-processed CSV — columns already contain engineered features.
        # Map known column-name variants to the names the model expects.
        variant_map = {
            "packets_per_second": "packets_per_second_calculated",
        }
        frame.rename(columns={k: v for k, v in variant_map.items() if k in frame.columns}, inplace=True)

        # Also map case-insensitively to model features.
        frame_lower = {c.casefold(): c for c in frame.columns}
        for feat in model_features:
            if feat not in frame.columns and feat.casefold() in frame_lower:
                frame.rename(columns={frame_lower[feat.casefold()]: feat}, inplace=True)

        logger.info("Pre-processed CSV detected (missing %d raw cols). Using columns directly.", len(missing_raw))

        # Build the feature matrix, filling missing features with 0.
        X = frame.reindex(columns=model_features)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

    class_predictions = rf.predict(X)
    probabilities = rf.predict_proba(X)
    confidence = probabilities.max(axis=1)
    anomaly_decision = iso.decision_function(X)
    anomaly_threshold = float(metadata["anomaly_threshold"])
    is_anomaly = (anomaly_decision < anomaly_threshold).astype(int)

    benign_name = next((str(c) for c in rf.classes_ if str(c).casefold() == "benign"), None)
    is_attack_prediction = np.array([
        int(benign_name is None or str(label).casefold() != "benign")
        for label in class_predictions
    ])

    # Transparent heuristic indicator, not a calibrated probability of compromise.
    risk_score = np.clip(0.70 * is_attack_prediction + 0.30 * is_anomaly, 0.0, 1.0)
    risk_level = np.where(risk_score >= 0.70, "High", np.where(risk_score >= 0.30, "Medium", "Low"))

    # Behaviour indicators are descriptive heuristics, not attack-family predictions.
    raw_features = frame.copy()
    for col in ("Flow Packets/s", "SYN Flag Count", "ACK Flag Count", "Total Fwd Packets", "Total Backward Packets"):
        if col in raw_features:
            raw_features[col] = pd.to_numeric(raw_features[col], errors="coerce")
    pps = raw_features.get("Flow Packets/s", pd.Series(0, index=frame.index)).fillna(0)
    syn = raw_features.get("SYN Flag Count", pd.Series(0, index=frame.index)).fillna(0)
    ack = raw_features.get("ACK Flag Count", pd.Series(0, index=frame.index)).fillna(0)
    packet_ratio = raw_features.get("Total Fwd Packets", pd.Series(0, index=frame.index)).fillna(0) / (raw_features.get("Total Backward Packets", pd.Series(0, index=frame.index)).fillna(0) + 1)
    pps_threshold = float(state.get("behavior_thresholds", {}).get("flow_packets_per_second_p99", np.inf))

    behavior_tags, explanations = [], []
    for i in range(len(frame)):
        indicators = []
        if float(pps.iloc[i]) > pps_threshold:
            indicators.append("Elevated packet rate (heuristic)")
        if float(syn.iloc[i]) > 0 and float(syn.iloc[i]) / (float(ack.iloc[i]) + 1.0) > 0.5:
            indicators.append("High SYN-to-ACK ratio (heuristic)")
        if float(packet_ratio.iloc[i]) > 2.0:
            indicators.append("Forward/backward packet imbalance")
        behavior_tags.append("; ".join(indicators) if indicators else "No configured indicator")

        reasons = []
        if is_attack_prediction[i]:
            reasons.append(f"Classifier predicted {class_predictions[i]}")
        if is_anomaly[i]:
            reasons.append("Flow falls below the benign-baseline anomaly threshold")
        if indicators:
            reasons.extend(indicators)
        explanations.append("; ".join(reasons) if reasons else "Classifier predicted Benign and anomaly detector did not flag the flow")

    anomaly_severity = -anomaly_decision  # larger means more anomalous; not a probability
    flow_ids = frame["Flow ID"] if "Flow ID" in frame.columns else pd.Series([f"row-{i + 1}" for i in range(len(frame))])
    output_rows = []
    display_order = np.argsort(-risk_score)[:500]
    for i in display_order:
        output_rows.append({
            "flow_id": _json_value(flow_ids.iloc[i]),
            "anomaly_score": round(float(anomaly_severity[i]), 6),
            "is_anomaly": int(is_anomaly[i]),
            "attack_type": str(class_predictions[i]),
            "risk_score": round(float(risk_score[i]), 3),
            "risk_level": str(risk_level[i]),
            "behavior_tag": behavior_tags[i],
            "confidence": round(float(confidence[i]), 4),
            "explanation": explanations[i],
        })

    temporal = _temporal_summary(frame, is_anomaly)
    class_counts = pd.Series(class_predictions).value_counts().to_dict()
    return {
        "summary": {
            "total_flows": int(len(frame)),
            "total_anomalies": int(is_anomaly.sum()),
            "high_risk": int(np.sum(risk_level == "High")),
            "predicted_attack_flows": int(is_attack_prediction.sum()),
        },
        "table_data": output_rows,
        "temporal_data": temporal,
        "feature_importance": [
            {"feature": name, "importance": float(value)}
            for name, value in sorted(zip(metadata["feature_names"], rf.feature_importances_), key=lambda item: item[1], reverse=True)[:5]
        ],
        "attack_type": {str(k): int(v) for k, v in class_counts.items()},
        "score_note": "Risk score is a fixed heuristic indicator (0.70 x non-benign classifier prediction + 0.30 x anomaly flag); it is not a calibrated probability.",
        "confidence_note": "Random Forest maximum class probability; not necessarily calibrated.",
        "returned_rows": len(output_rows),
        "truncated": len(frame) > len(output_rows),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NIDS inference on a raw CICIDS2017 CSV")
    parser.add_argument("csv_path")
    args = parser.parse_args()
    print(json.dumps(run_prediction(args.csv_path), indent=2))
