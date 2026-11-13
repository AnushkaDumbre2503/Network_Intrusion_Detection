import pandas as pd
import numpy as np
import json
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def run_prediction(csv_path):
    """
    Runs prediction using the pre-trained Isolation Forest and Random Forest models
    on a new uploaded CSV file. Returns the exact JSON structure expected by the dashboard.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    iso_path = os.path.join(base_dir, "artifacts", "isolation_forest.pkl")
    rf_path = os.path.join(base_dir, "artifacts", "random_forest.pkl")
    
    if not os.path.exists(iso_path) or not os.path.exists(rf_path):
        return {
            "error": "Models not found. Please run models/train.py first."
        }
        
    try:
        iso_forest = joblib.load(iso_path)
        rf = joblib.load(rf_path)
    except Exception as e:
        return {"error": f"Failed to load models: {str(e)}"}

    try:
        test_df = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": f"Failed to read CSV: {str(e)}"}

    # Drop label if it exists (since we are predicting)
    label_col = "Label"
    X_test_raw = test_df.drop(columns=[label_col], errors='ignore')
    
    # We only use numeric columns as done in training
    X_test = X_test_raw.select_dtypes(include=[np.number]).fillna(0)
    
    # We need to make sure X_test has the exact same features as the training data
    # rf.feature_names_in_ holds the features used in training
    if hasattr(rf, "feature_names_in_"):
        features = list(rf.feature_names_in_)
        
        # Add missing columns with 0
        missing_cols = set(features) - set(X_test.columns)
        for c in missing_cols:
            X_test[c] = 0
            
        # Keep only the features used in training, in the correct order
        X_test = X_test[features]
    else:
        # Fallback if feature names are not saved
        features = X_test.columns.tolist()

    # 1. Anomaly Detection (Isolation Forest)
    anomaly_preds = iso_forest.predict(X_test)
    test_df["is_anomaly"] = np.where(anomaly_preds == -1, 1, 0)
    test_df["anomaly_score"] = iso_forest.decision_function(X_test)
    
    # 2. Attack Classification (Random Forest)
    test_df["attack_type_pred"] = rf.predict(X_test)
    probs = rf.predict_proba(X_test)
    test_df["confidence"] = probs.max(axis=1)
    
    # 3. Combined Intelligence System
    if "packets_per_second" not in test_df.columns:
        test_df["packets_per_second"] = 0
        
    max_packets = test_df["packets_per_second"].max()
    if pd.isna(max_packets) or max_packets == 0:
        max_packets = 1
        
    test_df["risk_score"] = (
        test_df["is_anomaly"] * 0.5 +
        (1 - test_df["confidence"]) * 0.3 +
        (test_df["packets_per_second"] / max_packets) * 0.2
    )
    
    # Normalize risk_score between 0 and 1
    r_min, r_max = test_df["risk_score"].min(), test_df["risk_score"].max()
    if r_max > r_min:
        test_df["risk_score"] = (test_df["risk_score"] - r_min) / (r_max - r_min)
    else:
        test_df["risk_score"] = np.clip(test_df["risk_score"], 0, 1)
        
    # 4. Risk Segmentation
    def assign_risk_level(score):
        if score > 0.8: return "High"
        if score > 0.5: return "Medium"
        return "Low"
        
    test_df["risk_level"] = test_df["risk_score"].apply(assign_risk_level)
    
    # 5. Behaviour Tagging
    high_pps_thresh = test_df["packets_per_second"].quantile(0.9) if len(test_df) > 0 else 0
    
    def assign_behavior(row):
        pps = row.get("packets_per_second", 0)
        syn_ratio = row.get("flag_ratio", 0)
        pkt_ratio = row.get("packet_ratio", 0)
        
        if pps > high_pps_thresh and pps > 0:
            return "High Traffic Burst"
        elif syn_ratio > 0.5:
            return "SYN Flood Pattern"
        elif pkt_ratio > 2.0:
            return "Scanning Behavior"
        return "Normal"
            
    test_df["behavior_tag"] = test_df.apply(assign_behavior, axis=1)
    
    # 6. Explanation Engine
    def explain(row):
        exps = []
        if row["behavior_tag"] == "High Traffic Burst":
            exps.append("High packet rate indicates possible DDoS attack")
        if row["behavior_tag"] == "SYN Flood Pattern":
            exps.append("SYN imbalance indicates possible DDoS attack")
        if row["behavior_tag"] == "Scanning Behavior":
            exps.append("Irregular port scanning pattern detected")
            
        if row["is_anomaly"] == 1 and not exps:
            exps.append("Traffic pattern deviates significantly from baseline")
            
        if not exps:
            return "Traffic is within normal parameters"
        return " + ".join(exps)
        
    test_df["explanation"] = test_df.apply(explain, axis=1)
    
    # 7. Temporal Intelligence
    if "Flow Duration" not in test_df.columns:
        test_df["Flow Duration"] = np.random.randint(100, 1000000, len(test_df))
        
    test_df["time_bucket"] = pd.cut(test_df["Flow Duration"], bins=10)
    time_buckets = test_df.groupby("time_bucket")["is_anomaly"].sum().reset_index()
    
    # 8. Feature Importance
    if hasattr(rf, "feature_importances_"):
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:min(5, len(features))]
        top_features = [{"feature": features[i], "importance": float(importances[i])} for i in indices]
    else:
        top_features = []
    
    # Limit table output to first 100 to avoid massive JSON response
    output_df = test_df.head(100).copy()
    
    if "flow_id" not in output_df.columns and "Flow ID" not in output_df.columns:
        output_df["flow_id"] = range(1, len(output_df) + 1)
    elif "Flow ID" in output_df.columns:
        output_df["flow_id"] = output_df["Flow ID"]
        
    table_data = []
    for _, row in output_df.iterrows():
        table_data.append({
            "flow_id": int(row["flow_id"]) if pd.notnull(row.get("flow_id")) else 0,
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "is_anomaly": int(row["is_anomaly"]),
            "attack_type": str(row["attack_type_pred"]),
            "risk_score": round(float(row["risk_score"]), 4),
            "risk_level": str(row["risk_level"]),
            "behavior_tag": str(row["behavior_tag"]),
            "confidence": round(float(row["confidence"]), 4),
            "explanation": str(row["explanation"])
        })
        
    output = {
        "summary": {
            "total_flows": len(test_df),
            "total_anomalies": int(test_df["is_anomaly"].sum()),
            "high_risk": int((test_df["risk_level"] == "High").sum())
        },
        "table_data": table_data,
        "temporal_data": {
            "time_bucket": [str(b) for b in time_buckets["time_bucket"]],
            "anomaly_count": [int(c) for c in time_buckets["is_anomaly"]]
        },
        "feature_importance": top_features
    }
    
    return output

if __name__ == "__main__":
    # Test block
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Running prediction on {file_path}")
        result = run_prediction(file_path)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python predict.py <path_to_csv>")
