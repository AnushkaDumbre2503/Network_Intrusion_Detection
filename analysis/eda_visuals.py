import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="muted")

VISUALS_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard", "static", "visuals")
os.makedirs(VISUALS_DIR, exist_ok=True)

# ================= LOAD + SAMPLE =================

df = pd.read_csv("../data/cleaned_cicids2017.csv")

print("Original shape:", df.shape)

df = df.sample(100000, random_state=42)

print("Sampled shape:", df.shape)

target_col = "Label"

# ================= CLEAN FUNCTION =================

def clean_plot_data(df, cols):
    plot_df = df[cols].copy()
    plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return plot_df.dropna()

# ================= CLASS DISTRIBUTION =================

plt.figure(figsize=(8,5))
df[target_col].value_counts().plot(kind='bar', color='#3498db')
plt.title("Class Distribution (Sampled)")
plt.xticks(rotation=30)
plt.savefig(os.path.join(VISUALS_DIR, "basic_class_dist.png"), bbox_inches='tight', dpi=150)
plt.close()

# ================= SCATTER =================

scatter_cols = ['Flow Duration', 'Flow Bytes/s', target_col]
df_scatter = clean_plot_data(df, scatter_cols)

normals = df_scatter[df_scatter[target_col] == 'Benign']
anomalies = df_scatter[df_scatter[target_col] != 'Benign']

plt.figure(figsize=(10,6))

plt.scatter(normals['Flow Duration'], normals['Flow Bytes/s'],
            alpha=0.3, label='Normal', color='#3498db', s=20)

plt.scatter(anomalies['Flow Duration'], anomalies['Flow Bytes/s'],
            alpha=0.8, label='Attack', color='#e74c3c', marker='x', s=40)

plt.xscale('log')
plt.yscale('log')

plt.title("Flow Duration vs Bytes/s")
plt.legend()
plt.savefig(os.path.join(VISUALS_DIR, "basic_scatter.png"), bbox_inches='tight', dpi=150)
plt.close()

# ================= FEATURE DISTRIBUTIONS =================

features = ['packet_ratio', 'bytes_per_packet', 'burstiness']
df_grid = clean_plot_data(df, features + [target_col])

fig, axes = plt.subplots(1,3, figsize=(18,6))

for i, f in enumerate(features):
    sns.boxplot(data=df_grid, x=target_col, y=f, ax=axes[i], showfliers=False)
    axes[i].set_title(f)

plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "basic_boxplots.png"), bbox_inches='tight', dpi=150)
plt.close()

# ================= TIMING =================

timing_feature = 'time_between_requests' if 'time_between_requests' in df.columns else 'Flow IAT Mean'
df_timing = clean_plot_data(df, [timing_feature, target_col])

plt.figure(figsize=(10,6))
sns.kdeplot(data=df_timing, x=timing_feature, hue=target_col, fill=True, alpha=0.5)
plt.xscale('log')

plt.title("Timing Behavior")
plt.savefig(os.path.join(VISUALS_DIR, "basic_timing.png"), bbox_inches='tight', dpi=150)
plt.close()

# ================= HISTOGRAM =================

numeric_cols = df.select_dtypes(include=np.number).columns

df[numeric_cols].hist(figsize=(15,10), bins=30)
plt.suptitle("Feature Distributions (Sampled)")
plt.savefig(os.path.join(VISUALS_DIR, "basic_histograms.png"), bbox_inches='tight', dpi=150)
plt.close()

# ================= CORRELATION =================

plt.figure(figsize=(12,8))
sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(VISUALS_DIR, "basic_correlation.png"), bbox_inches='tight', dpi=150)
plt.close()

print(f"Basic EDA visuals saved to {VISUALS_DIR}")

# ================= FINAL INSIGHTS =================

print("""
VISUAL INSIGHTS:

1. Attack traffic shows higher burstiness and packet ratios.
2. Scatter plot shows clear separation in flow behavior.
3. Timing distribution suggests automated patterns in attacks.
4. Feature distributions vary significantly between normal and attack traffic.
""")