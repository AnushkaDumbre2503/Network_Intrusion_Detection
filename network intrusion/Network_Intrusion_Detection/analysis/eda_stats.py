"""Generate dashboard EDA statistics from the cleaned CICIDS2017 dataset."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.preprocessing import build_features

INPUT = ROOT / "data" / "processed" / "cleaned_cicids2017.csv"
OUTPUT = ROOT / "dashboard" / "static" / "data" / "eda_stats.json"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Run data_prep/data_prep.py first.")
    df = pd.read_csv(INPUT, low_memory=False)
    if "Label" not in df:
        raise ValueError("Dataset must contain Label.")
    features = build_features(df)
    for col in features:
        df[col] = features[col]

    labels = df["Label"].astype(str)
    class_counts = labels.value_counts()
    class_percentages = (class_counts / len(df) * 100).round(2)
    top_ports = df["Destination Port"].value_counts().head(10)

    port_counts = df.groupby("Destination Port").size()
    attack_counts = df[labels.ne("Benign")].groupby("Destination Port").size()
    attack_ratio = (attack_counts / port_counts).fillna(0).sort_values(ascending=False).head(10)

    traffic = df.groupby("Label")[["Flow Bytes/s", "Flow Packets/s"]].mean(numeric_only=True)
    imbalance = df.groupby("Label")["traffic_imbalance"].mean()
    timing = df.groupby("Label")[["Flow IAT Mean", "iat_variability"]].mean(numeric_only=True)
    flags = df.groupby("Label")[["SYN Flag Count", "ACK Flag Count", "RST Flag Count", "flag_ratio"]].mean(numeric_only=True)
    day_analysis = (df.groupby(["day", "Label"]).size().unstack(fill_value=0)
                    if "day" in df.columns else pd.DataFrame())

    stats = {
        "data_overview": {"shape": [int(df.shape[0]), int(df.shape[1])]},
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "class_percentage": {str(k): float(v) for k, v in class_percentages.items()},
        "top_ports": {str(k): int(v) for k, v in top_ports.items()},
        "top_suspicious_ports": {str(k): float(v) for k, v in attack_ratio.items()},
        "traffic_intensity": traffic.replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="index"),
        "packet_imbalance": imbalance.replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(),
        "timing_behavior": timing.replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="index"),
        "flag_analysis": flags.replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="index"),
        "day_wise_analysis": {str(k): {str(label): int(value) for label, value in row.items()}
                              for k, row in day_analysis.to_dict(orient="index").items()},
        "attack_type": {str(k): int(v) for k, v in labels[labels.ne("Benign")].value_counts().items()},
        "insights": [
            "Class distribution is imbalanced; evaluate per-class recall rather than accuracy alone.",
            "Port attack ratios are descriptive dataset statistics, not proof that a port is malicious.",
            "Flow Packets/s and Flow Bytes/s summarize recorded flow intensity.",
            "Traffic imbalance and inter-arrival-time variability describe flow characteristics, not user intent.",
            "SYN-to-ACK ratio is a heuristic indicator and cannot independently confirm a SYN flood.",
            "Day-wise counts describe the CICIDS2017 collection period, not live network activity.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(stats, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Dashboard EDA statistics saved to {OUTPUT}")


if __name__ == "__main__":
    main()
