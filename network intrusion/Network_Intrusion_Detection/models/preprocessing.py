"""Shared train/inference feature construction and fitted preprocessing."""
from __future__ import annotations
import numpy as np
import pandas as pd

RAW_FEATURES = [
    "Destination Port", "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Packet Length Mean", "Packet Length Std", "Average Packet Size",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Mean", "Bwd IAT Mean", "SYN Flag Count", "ACK Flag Count",
    "RST Flag Count", "Down/Up Ratio",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required CICIDS2017 columns: " + ", ".join(missing)
        )
    x = frame[RAW_FEATURES].copy()
    for col in RAW_FEATURES:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)

    fwd_packets = x["Total Fwd Packets"].clip(lower=0)
    bwd_packets = x["Total Backward Packets"].clip(lower=0)
    total_packets = fwd_packets + bwd_packets
    fwd_bytes = x["Total Length of Fwd Packets"].clip(lower=0)
    bwd_bytes = x["Total Length of Bwd Packets"].clip(lower=0)
    total_bytes = fwd_bytes + bwd_bytes
    duration_seconds = x["Flow Duration"].clip(lower=0) / 1_000_000.0

    x["total_packets"] = total_packets
    x["total_bytes"] = total_bytes
    x["bytes_per_packet"] = total_bytes.div(total_packets.replace(0, np.nan))
    x["packet_ratio"] = fwd_packets.div(bwd_packets.replace(0, np.nan))
    x["packets_per_second_calculated"] = total_packets.div(duration_seconds.replace(0, np.nan))
    x["traffic_imbalance"] = (fwd_bytes - bwd_bytes).div(total_bytes.replace(0, np.nan))
    x["iat_variability"] = x["Flow IAT Std"].div(x["Flow IAT Mean"].abs().replace(0, np.nan))
    x["flag_ratio"] = x["SYN Flag Count"].div(x["ACK Flag Count"].clip(lower=0) + 1.0)
    x["reset_rate"] = x["RST Flag Count"].clip(lower=0).div(total_packets + 1.0)
    x["packet_size_variability"] = x["Packet Length Std"].div(x["Packet Length Mean"].abs().replace(0, np.nan))
    return x.replace([np.inf, -np.inf], np.nan)


def fit_preprocessor(train_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = build_features(train_frame)
    # Destination-port frequency is fitted only on training observations.
    ports = pd.to_numeric(train_frame["Destination Port"], errors="coerce")
    port_counts = ports.value_counts(dropna=True).to_dict()
    x["destination_port_frequency"] = ports.map(port_counts).fillna(0).astype(float)

    medians = x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    x = x.fillna(medians).fillna(0)
    clip_upper = x.quantile(0.99).to_dict()
    for col, upper in clip_upper.items():
        x[col] = x[col].clip(upper=upper)

    skewed = x.skew(numeric_only=True)
    log_columns = [c for c in x.columns if skewed.get(c, 0) > 1 and (x[c] >= 0).all()]
    for col in log_columns:
        x[col] = np.log1p(x[col])

    benign_rows = train_frame["Label"].astype(str).str.casefold().eq("benign") if "Label" in train_frame else pd.Series(True, index=train_frame.index)
    benign_pps = pd.to_numeric(train_frame.loc[benign_rows, "Flow Packets/s"], errors="coerce").dropna()
    behavior_thresholds = {
        "flow_packets_per_second_p99": float(benign_pps.quantile(0.99)) if len(benign_pps) else float("inf")
    }
    state = {
        "raw_features": RAW_FEATURES,
        "feature_names": x.columns.tolist(),
        "medians": medians.to_dict(),
        "clip_upper": clip_upper,
        "log_columns": log_columns,
        "port_counts": {str(k): int(v) for k, v in port_counts.items()},
        "behavior_thresholds": behavior_thresholds,
    }
    return x.astype(float), state


def transform_features(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    x = build_features(frame)
    ports = pd.to_numeric(frame["Destination Port"], errors="coerce")
    port_counts = {float(k): float(v) for k, v in state["port_counts"].items()}
    x["destination_port_frequency"] = ports.map(port_counts).fillna(0).astype(float)

    feature_names = state["feature_names"]
    medians = pd.Series(state["medians"])
    x = x.reindex(columns=feature_names)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)
    for col, upper in state["clip_upper"].items():
        x[col] = x[col].clip(upper=upper)
    for col in state["log_columns"]:
        x[col] = np.log1p(x[col].clip(lower=0))
    return x.astype(float)
