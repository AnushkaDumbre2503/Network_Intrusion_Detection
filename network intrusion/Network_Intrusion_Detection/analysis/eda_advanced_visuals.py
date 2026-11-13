"""Generate advanced CICIDS2017 EDA figures using shared, valid derived features."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.preprocessing import build_features

INPUT = ROOT / "data" / "processed" / "cleaned_cicids2017.csv"
OUT = ROOT / "dashboard" / "static" / "visuals"


def save(name, title):
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150)
    plt.close()


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Run data_prep/data_prep.py first.")
    df = pd.read_csv(INPUT, low_memory=False)
    df = df.sample(min(100_000, len(df)), random_state=42).copy()
    derived = build_features(df)
    for col in derived:
        df[col] = derived[col]
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df["Label"].value_counts().plot(kind="bar", figsize=(9, 5)); save("class_distribution.png", "Class Distribution")

    plot = df[["Flow Duration", "Flow Bytes/s", "Label"]].replace([np.inf, -np.inf], np.nan).dropna()
    plot = plot[(plot["Flow Duration"] > 0) & (plot["Flow Bytes/s"] > 0)]
    plt.figure(figsize=(10, 6)); sns.scatterplot(data=plot, x="Flow Duration", y="Flow Bytes/s", hue="Label", alpha=.4)
    plt.xscale("log"); plt.yscale("log"); save("flow_behavior.png", "Flow Duration and Byte Rate")

    plt.figure(figsize=(10, 6)); sns.boxplot(data=df, x="Label", y="Flow Bytes/s", showfliers=False)
    plt.yscale("log"); plt.xticks(rotation=35); save("traffic_intensity.png", "Flow Byte Rate by Class")

    plt.figure(figsize=(10, 6)); sns.boxplot(data=df, x="Label", y="packets_per_second_calculated", showfliers=False)
    plt.yscale("log"); plt.xticks(rotation=35); save("burstiness.png", "Calculated Packets per Second by Class")

    timing = df[["Flow IAT Mean", "Label"]].replace([np.inf, -np.inf], np.nan).dropna()
    timing = timing[timing["Flow IAT Mean"] > 0]
    plt.figure(figsize=(10, 6)); sns.boxplot(data=timing, x="Label", y="Flow IAT Mean", showfliers=False)
    plt.yscale("log"); plt.xticks(rotation=35); save("timing_behavior.png", "Flow Inter-Arrival Time")

    plt.figure(figsize=(10, 6)); sns.boxplot(data=df, x="Label", y="packet_ratio", showfliers=False)
    plt.xticks(rotation=35); save("packet_ratio.png", "Forward-to-Backward Packet Ratio")

    flag_cols = ["SYN Flag Count", "ACK Flag Count", "RST Flag Count"]
    df.groupby("Label")[flag_cols].mean().plot(kind="bar", figsize=(10, 6))
    plt.xticks(rotation=35); save("flag_behavior.png", "Mean TCP Flag Counts by Class")

    df["Destination Port"].value_counts().head(10).plot(kind="bar", figsize=(10, 6))
    save("top_ports.png", "Most Frequent Destination Ports")

    port_attack = pd.crosstab(df["Destination Port"], df["Label"])
    top = port_attack.sum(axis=1).nlargest(10).index
    plt.figure(figsize=(11, 6)); sns.heatmap(port_attack.loc[top], annot=True, fmt=".0f", cmap="coolwarm")
    save("port_attack_heatmap.png", "Destination Port vs Label")

    corr_cols = ["Flow Bytes/s", "Flow Packets/s", "packet_ratio", "flag_ratio", "iat_variability", "traffic_imbalance"]
    plt.figure(figsize=(10, 6)); sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", center=0)
    save("correlation_heatmap.png", "Selected Feature Correlations")

    if "day" in df.columns:
        df.groupby(["day", "Label"]).size().unstack(fill_value=0).plot(kind="bar", stacked=True, figsize=(10, 6))
        plt.xticks(rotation=35); save("day_wise_attack.png", "Day-wise CICIDS2017 Class Counts")

    # Keep the historical filename used by the dashboard, but plot a valid timing feature.
    plt.figure(figsize=(10, 6)); sns.boxplot(data=df, x="Label", y="iat_variability", showfliers=False)
    plt.xticks(rotation=35); save("repeat_call_ratio.png", "Inter-Arrival-Time Variability (Not Repeat-Call Detection)")
    print(f"Advanced EDA figures saved to {OUT}")


if __name__ == "__main__":
    main()
