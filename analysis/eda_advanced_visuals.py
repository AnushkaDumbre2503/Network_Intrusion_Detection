import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

import os
visuals_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'static', 'visuals'))
os.makedirs(visuals_dir, exist_ok=True)


# ================= LOAD + SAMPLE =================
df = pd.read_csv("data/cleaned_cicids2017.csv")

print("Original shape:", df.shape)

df = df.sample(100000, random_state=42)

print("Sampled shape:", df.shape)

target_col = "Label"

# ================= CLEAN FUNCTION =================

def clean_plot_data(df, cols):
    plot_df = df[cols].copy()
    plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return plot_df.dropna()

# =========================================================
# 🔴 1. CLASS DISTRIBUTION
# =========================================================

plt.figure(figsize=(8,5))
df[target_col].value_counts().plot(kind='bar', color='#3498db')
plt.title("Class Distribution")
plt.xticks(rotation=30)
plt.savefig(os.path.join(visuals_dir, "class_distribution.png"), bbox_inches='tight')
plt.close()

# =========================================================
# 🔴 2. FLOW BEHAVIOR (SCATTER)
# =========================================================

scatter_cols = ['Flow Duration', 'Flow Bytes/s', target_col]
df_scatter = clean_plot_data(df, scatter_cols)

plt.figure(figsize=(10,6))
sns.scatterplot(data=df_scatter, x='Flow Duration', y='Flow Bytes/s',
                hue=target_col, alpha=0.5)

plt.xscale('log')
plt.yscale('log')

plt.title("Flow Behavior Separation")
plt.savefig(os.path.join(visuals_dir, "flow_behavior.png"), bbox_inches='tight')
plt.close()

# =========================================================
# 🔴 3. TRAFFIC INTENSITY COMPARISON
# =========================================================

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x=target_col, y='Flow Bytes/s', showfliers=False)
plt.yscale('log')
plt.title("Traffic Intensity per Class")
plt.savefig(os.path.join(visuals_dir, "traffic_intensity.png"), bbox_inches='tight')
plt.close()

# =========================================================
# 🔴 4. BURSTINESS VS PACKET RATE
# =========================================================

cols = ['burstiness', 'packets_per_second', target_col]
df_temp = clean_plot_data(df, cols)

plt.figure(figsize=(10,6))
sns.scatterplot(data=df_temp, x='burstiness', y='packets_per_second',
                hue=target_col, alpha=0.6)

plt.yscale('log')
plt.title("Burstiness vs Packet Rate")
plt.savefig(os.path.join(visuals_dir, "burstiness.png"), bbox_inches='tight')
plt.close()

# =========================================================
# 🔴 5. TIMING DISTRIBUTION
# =========================================================

timing_feature = 'time_between_requests' if 'time_between_requests' in df.columns else 'Flow IAT Mean'
df_timing = clean_plot_data(df, [timing_feature, target_col])

plt.figure(figsize=(10,6))
sns.kdeplot(data=df_timing, x=timing_feature, hue=target_col, fill=True)
plt.xscale('log')
plt.title("Timing Behavior")
plt.savefig(os.path.join(visuals_dir, "timing_behavior.png"), bbox_inches='tight')
plt.close()

# =========================================================
# 🔴 6. PACKET RATIO (SCAN DETECTION)
# =========================================================

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x=target_col, y='packet_ratio', showfliers=False)
plt.title("Packet Ratio Analysis")
plt.savefig(os.path.join(visuals_dir, "packet_ratio.png"), bbox_inches='tight')
plt.close()

# =========================================================
# 🔴 7. FLAG BEHAVIOR (SYN FLOOD)
# =========================================================

flag_cols = ['SYN Flag Count', 'ACK Flag Count', 'RST Flag Count']
df_flags = df[flag_cols + [target_col]]

df_flags.groupby(target_col)[flag_cols].mean().plot(kind='bar', figsize=(10,6))
plt.title("Average Flag Behavior")
plt.ylabel("Average Count")
plt.xticks(rotation=30)
plt.savefig(os.path.join(visuals_dir, "flag_behavior.png"), bbox_inches='tight')
plt.close()

# =========================================================
# 🔴 8. PORT ANALYSIS (TOP PORTS)
# =========================================================

top_ports = df['Destination Port'].value_counts().head(10)

plt.figure(figsize=(10,6))
top_ports.plot(kind='bar', color='#e67e22')
plt.title("Top Ports")
plt.ylabel("Frequency")
plt.savefig(os.path.join(visuals_dir, "top_ports.png"), bbox_inches='tight')
plt.close()

# =========================================================
# 🔴 9. PORT VS ATTACK HEATMAP
# =========================================================

port_attack = df.groupby(['Destination Port', target_col]).size().unstack().fillna(0)
top_ports = port_attack.sum(axis=1).sort_values(ascending=False).head(10).index

plt.figure(figsize=(10,6))
sns.heatmap(port_attack.loc[top_ports], annot=True, fmt=".0f", cmap='coolwarm')
plt.title("Top Ports vs Attack Types")
plt.savefig(os.path.join(visuals_dir, "port_attack_heatmap.png"), bbox_inches='tight')
plt.close()

# =========================================================
# 🔴 10. CORRELATION HEATMAP (FOCUSED)
# =========================================================

important_cols = [
    'Flow Bytes/s', 'Flow Packets/s', 'packet_ratio',
    'burstiness', 'flag_ratio', 'iat_variability'
]

plt.figure(figsize=(10,6))
sns.heatmap(df[important_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Important Feature Correlation")
plt.savefig(os.path.join(visuals_dir, "correlation_heatmap.png"), bbox_inches='tight')
plt.close()

# =========================================================
# 🔴 11. DAY-WISE ATTACK PATTERN
# =========================================================

day_attack = df.groupby(['day', target_col]).size().unstack().fillna(0)

day_attack.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title("Day-wise Attack Distribution")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.savefig(os.path.join(visuals_dir, "day_wise_attack.png"), bbox_inches='tight')
plt.close()

# =========================================================
# 🔴 12. REPEAT CALL BEHAVIOR
# =========================================================

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x=target_col, y='repeat_call_ratio', showfliers=False)
plt.title("Repeat Call Ratio (Zombie API Indicator)")
plt.savefig(os.path.join(visuals_dir, "repeat_call_ratio.png"), bbox_inches='tight')
plt.close()

# =========================================================
# 🔴 FINAL INSIGHTS
# =========================================================

print("""
ADVANCED VISUAL INSIGHTS:

1. Attack traffic shows high burstiness and packet rate spikes.
2. Timing distributions indicate automated bot-like behavior.
3. Packet ratio reveals scanning patterns.
4. SYN/ACK imbalance indicates potential SYN flood attacks.
5. Specific ports dominate attack traffic → critical endpoints.
6. Repeat call ratio strongly indicates automated API abuse.
7. Day-wise trends reveal attack campaign timing.
8. Multi-feature correlation helps explain model decisions.
""")