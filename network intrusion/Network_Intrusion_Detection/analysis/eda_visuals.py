"""Generate basic EDA figures from the cleaned CICIDS2017 dataset."""
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

    df["Label"].value_counts().plot(kind="bar", figsize=(9, 5))
    plt.title("CICIDS2017 Class Distribution")
    plt.tight_layout(); plt.savefig(OUT / "basic_class_dist.png", dpi=150); plt.close()

    plot = df[["Flow Duration", "Flow Bytes/s", "Label"]].replace([np.inf, -np.inf], np.nan).dropna()
    plot = plot[(plot["Flow Duration"] > 0) & (plot["Flow Bytes/s"] > 0)]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot, x="Flow Duration", y="Flow Bytes/s", hue="Label", alpha=.35, s=15)
    plt.xscale("log"); plt.yscale("log"); plt.title("Flow Duration vs Flow Bytes/s")
    plt.tight_layout(); plt.savefig(OUT / "basic_scatter.png", dpi=150); plt.close()

    box = df[["packet_ratio", "bytes_per_packet", "iat_variability", "Label"]].replace([np.inf, -np.inf], np.nan).dropna()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, feature in zip(axes, ["packet_ratio", "bytes_per_packet", "iat_variability"]):
        sns.boxplot(data=box, x="Label", y=feature, ax=ax, showfliers=False)
        ax.tick_params(axis="x", rotation=35); ax.set_title(feature)
    plt.tight_layout(); plt.savefig(OUT / "basic_boxplots.png", dpi=150); plt.close()

    timing = df[["Flow IAT Mean", "Label"]].replace([np.inf, -np.inf], np.nan).dropna()
    timing = timing[timing["Flow IAT Mean"] > 0]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=timing, x="Label", y="Flow IAT Mean", showfliers=False)
    plt.yscale("log"); plt.xticks(rotation=35); plt.title("Flow Inter-Arrival Time")
    plt.tight_layout(); plt.savefig(OUT / "basic_timing.png", dpi=150); plt.close()

    numeric = df.select_dtypes(include=np.number)
    numeric.hist(figsize=(16, 11), bins=30)
    plt.tight_layout(); plt.savefig(OUT / "basic_histograms.png", dpi=150); plt.close()

    plt.figure(figsize=(13, 9))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0)
    plt.title("Numeric Feature Correlation")
    plt.tight_layout(); plt.savefig(OUT / "basic_correlation.png", dpi=150); plt.close()
    print(f"Basic EDA figures saved to {OUT}")


if __name__ == "__main__":
    main()
