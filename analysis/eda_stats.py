import pandas as pd
import numpy as np

# ================= LOAD =================
df = pd.read_csv("D:\\Mini projects\\FinalMiniProject(Pyhton)\\data\\cleaned_cicids2017.csv")

target = "Label"

print("\n================ DATA OVERVIEW ================")
print("Shape:", df.shape)

# =========================================================
# 🔴 1. CLASS IMBALANCE (VERY IMPORTANT FOR ML)
# =========================================================

print("\n================ CLASS DISTRIBUTION ================")
class_counts = df[target].value_counts()
print(class_counts)

print("\nClass Percentage:")
print((class_counts / len(df)) * 100)

# =========================================================
# 🔴 2. PORT BEHAVIOR (ZOMBIE API ENTRY POINTS)
# =========================================================

print("\n================ PORT ANALYSIS =================")

port_attack = df.groupby("Destination Port")[target].value_counts().unstack().fillna(0)

# Top ports by total traffic
top_ports = df["Destination Port"].value_counts().head(10)
print("\nTop Ports:\n", top_ports)

# Ports with highest attack ratio
port_attack["attack_ratio"] = (
    port_attack.drop(columns=["Benign"], errors="ignore").sum(axis=1) /
    port_attack.sum(axis=1)
)

print("\nTop Suspicious Ports (High Attack Ratio):")
print(port_attack["attack_ratio"].sort_values(ascending=False).head(10))

# =========================================================
# 🔴 3. TRAFFIC INTENSITY (DDoS SIGNATURE)
# =========================================================

print("\n================ TRAFFIC INTENSITY =================")

traffic_stats = df.groupby(target)[["Flow Bytes/s", "Flow Packets/s"]].mean()
print(traffic_stats)

# =========================================================
# 🔴 4. PACKET BEHAVIOR (ASYMMETRY DETECTION)
# =========================================================

print("\n================ PACKET IMBALANCE =================")

imbalance = df.groupby(target)["traffic_imbalance"].mean()
print(imbalance)

# =========================================================
# 🔴 5. TIMING BEHAVIOR (BOT VS HUMAN)
# =========================================================

print("\n================ TIMING BEHAVIOR =================")

timing = df.groupby(target)[["Flow IAT Mean", "iat_variability"]].mean()
print(timing)

# =========================================================
# 🔴 6. BURST TRAFFIC (ZOMBIE API SIGNATURE)
# =========================================================

print("\n================ BURSTINESS =================")

burst = df.groupby(target)["burstiness"].mean()
print(burst)

# =========================================================
# 🔴 7. FLAG BEHAVIOR (SYN FLOOD DETECTION)
# =========================================================

print("\n================ FLAG ANALYSIS =================")

flags = df.groupby(target)[["SYN Flag Count", "ACK Flag Count", "flag_ratio"]].mean()
print(flags)

# =========================================================
# 🔴 8. HIGH-RISK FLOWS (EXTREME BEHAVIOR)
# =========================================================

print("\n================ EXTREME FLOWS =================")

high_packets = df[df["packets_per_second"] > df["packets_per_second"].quantile(0.99)]
print("High Packet Rate Distribution:")
print(high_packets[target].value_counts())

# =========================================================
# 🔴 9. REPEAT CALL ANALYSIS (ZOMBIE API LOOPING)
# =========================================================

print("\n================ REPEAT CALL BEHAVIOR =================")

repeat = df.groupby(target)["repeat_call_ratio"].mean()
print(repeat)

# =========================================================
# 🔴 10. ENDPOINT USAGE (API DIVERSITY)
# =========================================================

print("\n================ ENDPOINT DIVERSITY =================")

endpoint = df.groupby(target)["endpoint_diversity"].mean()
print(endpoint)

# =========================================================
# 🔴 11. DAY-WISE ATTACK PATTERN
# =========================================================

print("\n================ DAY-WISE ANALYSIS =================")

day_attack = df.groupby(["day", target]).size().unstack().fillna(0)
print(day_attack)

# =========================================================
# 🔴 12. ATTACK TYPE BREAKDOWN
# =========================================================

print("\n================ ATTACK TYPE =================")

print(df["attack_type"].value_counts())

# =========================================================
# 🔴 13. CORRELATION (IMPORTANT FOR ML FEATURE REDUCTION)
# =========================================================

print("\n================ TOP CORRELATIONS =================")

numeric_cols = df.select_dtypes(include=np.number).columns
corr = df[numeric_cols].corr().abs()

top_corr = (
    corr.unstack()
    .sort_values(ascending=False)
    .drop_duplicates()
)

print(top_corr.head(15))

# =========================================================
# 🔴 FINAL INSIGHTS (VERY IMPORTANT FOR REPORT)
# =========================================================

print("\n================ FINAL INSIGHTS =================")

print("""
1. Strong class imbalance exists → Benign traffic dominates (~83%).
2. Ports 53, 80, 443 dominate traffic → critical API exposure points.
3. High attack ratio ports indicate potential Zombie API endpoints.
4. DDoS traffic shows high packets/sec and burstiness.
5. Attack traffic shows asymmetric communication (traffic imbalance).
6. Bot-like behavior seen in low timing variability (iat_variability).
7. High SYN/ACK ratio indicates handshake abuse (SYN flood).
8. High repeat_call_ratio suggests automated API calls (Zombie APIs).
9. Endpoint diversity shows how widely APIs are being targeted.
10. Day-wise patterns help identify attack campaigns.
""")

# ================= EXPORT TO JSON FOR FRONTEND =================
import json
import os

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

stats_dict = {
    "data_overview": {
        "shape": list(df.shape)
    },
    "class_distribution": class_counts.to_dict(),
    "class_percentage": ((class_counts / len(df)) * 100).to_dict(),
    "top_ports": {str(k): v for k, v in top_ports.to_dict().items()},
    "top_suspicious_ports": {str(k): v for k, v in port_attack["attack_ratio"].sort_values(ascending=False).head(10).to_dict().items()},
    "traffic_intensity": traffic_stats.to_dict(orient="index"),
    "packet_imbalance": imbalance.to_dict(),
    "timing_behavior": timing.to_dict(orient="index"),
    "burstiness": burst.to_dict(),
    "flag_analysis": flags.to_dict(orient="index"),
    "repeat_call_behavior": repeat.to_dict(),
    "endpoint_diversity": endpoint.to_dict(),
    "day_wise_analysis": {str(k): v for k, v in day_attack.to_dict(orient="index").items()},
    "attack_type": {str(k): v for k, v in df["attack_type"].value_counts().to_dict().items()},
    "insights": [
        "Strong class imbalance exists → Benign traffic dominates (~83%).",
        "Ports 53, 80, 443 dominate traffic → critical API exposure points.",
        "High attack ratio ports indicate potential Zombie API endpoints.",
        "DDoS traffic shows high packets/sec and burstiness.",
        "Attack traffic shows asymmetric communication (traffic imbalance).",
        "Bot-like behavior seen in low timing variability (iat_variability).",
        "High SYN/ACK ratio indicates handshake abuse (SYN flood).",
        "High repeat_call_ratio suggests automated API calls (Zombie APIs).",
        "Endpoint diversity shows how widely APIs are being targeted.",
        "Day-wise patterns help identify attack campaigns."
    ]
}

stats_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'static', 'data'))
os.makedirs(stats_dir, exist_ok=True)
with open(os.path.join(stats_dir, "eda_stats.json"), "w") as f:
    json.dump(stats_dict, f, indent=4, cls=NpEncoder)
print(f"\\nJSON stats successfully written to {stats_dir}/eda_stats.json")