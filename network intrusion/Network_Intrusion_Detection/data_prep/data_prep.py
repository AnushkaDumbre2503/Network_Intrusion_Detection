"""Load and clean CICIDS2017 daily CSV files without fitting dataset-wide transforms."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "cleaned_cicids2017.csv"

FILES = [
    ("Monday-WorkingHours.pcap_ISCX.csv", "Monday"),
    ("Tuesday-WorkingHours.pcap_ISCX.csv", "Tuesday"),
    ("Wednesday-WorkingHours.pcap_ISCX.csv", "Wednesday"),
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "Thursday"),
    ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Thursday"),
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv", "Friday"),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "Friday"),
    ("Friday-WorkingHours-Afternoon-PortScan.csv", "Friday"),
    ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "Friday"),
]

FEATURE_COLUMNS = [
    "Destination Port", "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Packet Length Mean", "Packet Length Std", "Average Packet Size",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Mean", "Bwd IAT Mean", "SYN Flag Count", "ACK Flag Count",
    "RST Flag Count", "Down/Up Ratio",
]

LABEL_MAP = {
    "BENIGN": "Benign",
    "DoS Hulk": "DDoS", "DoS GoldenEye": "DDoS",
    "DoS slowloris": "DDoS", "DoS Slowhttptest": "DDoS",
    "PortScan": "PortScan",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
    "Bot": "Bot", "Heartbleed": "Heartbleed", "Infiltration": "Infiltration",
}


def main():
    frames, missing = [], []
    for filename, day in FILES:
        path = DATASET_DIR / filename
        if not path.exists():
            missing.append(filename)
            continue
        print(f"Reading {path.name}")
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = frame.columns.astype(str).str.strip()
        required = [c for c in FEATURE_COLUMNS + ["Label"] if c not in frame.columns]
        if required:
            raise ValueError(f"{path.name} is missing required columns: {required}")
        keep = FEATURE_COLUMNS + ["Label"]
        for metadata in ("Timestamp", "Flow ID"):
            if metadata in frame.columns:
                keep.append(metadata)
        frame = frame[keep].copy()
        frame["day"] = day
        frame["Label"] = frame["Label"].astype(str).str.strip().str.replace(r"[–—−�]", "-", regex=True)
        frame["Label"] = frame["Label"].replace(LABEL_MAP)
        frame = frame[frame["Label"].notna() & (frame["Label"] != "")]
        for col in FEATURE_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(subset=["Flow Duration"])
        frame = frame[frame["Flow Duration"] > 0]
        frame = frame.drop_duplicates()
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No CICIDS2017 CSV files found in {DATASET_DIR}. Place the original daily CSVs there."
        )
    result = pd.concat(frames, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(result):,} rows to {OUTPUT_FILE}")
    print("Class distribution:\n", result["Label"].value_counts())
    if missing:
        print("Skipped missing files:", ", ".join(sorted(set(missing))))
    print("No global clipping, scaling, or destination-port frequency fitting was performed.")


if __name__ == "__main__":
    main()
