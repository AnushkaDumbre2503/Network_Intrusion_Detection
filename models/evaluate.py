import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score
)

# ================= LOAD MODELS =================
iso = joblib.load("artifacts/isolation_forest.pkl")
rf = joblib.load("artifacts/random_forest.pkl")

print("✅ Models loaded")

# ================= LOAD TEST DATA =================
df = pd.read_csv("data/test_data.csv")

df = df.dropna(subset=["Label"]).reset_index(drop=True)

y_test = df["Label"]

X_test = df.drop(columns=["Label"])
X_test = X_test.select_dtypes(include=[np.number]).fillna(0)

print("Test shape:", X_test.shape)

# ================= PREDICTIONS =================

# Random Forest
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

# Isolation Forest
iso_pred = iso.predict(X_test)
iso_pred = np.where(iso_pred == -1, 1, 0)
iso_scores = -iso.decision_function(X_test)

# ================= RANDOM FOREST METRICS =================

print("\n================ RANDOM FOREST =================")

print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

rf_roc_auc = roc_auc_score(y_test, rf_prob)
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_prob)
rf_pr_auc = auc(rf_recall, rf_precision)

print("ROC-AUC:", rf_roc_auc)
print("PR-AUC:", rf_pr_auc)

# ================= ISOLATION FOREST METRICS =================

print("\n================ ISOLATION FOREST =================")

print(classification_report(y_test, iso_pred))

iso_roc_auc = roc_auc_score(y_test, iso_scores)
iso_precision, iso_recall, _ = precision_recall_curve(y_test, iso_scores)
iso_pr_auc = auc(iso_recall, iso_precision)

print("ROC-AUC:", iso_roc_auc)
print("PR-AUC:", iso_pr_auc)

# ================= CONFUSION MATRIX =================

def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_cm(y_test, rf_pred, "Random Forest Confusion Matrix")
plot_cm(y_test, iso_pred, "Isolation Forest Confusion Matrix")

# ================= ROC CURVE =================

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
iso_fpr, iso_tpr, _ = roc_curve(y_test, iso_scores)

plt.figure()
plt.plot(rf_fpr, rf_tpr, label=f"RF ROC-AUC = {rf_roc_auc:.2f}")
plt.plot(iso_fpr, iso_tpr, label=f"IF ROC-AUC = {iso_roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# ================= PR CURVE =================

plt.figure()
plt.plot(rf_recall, rf_precision, label=f"RF PR-AUC = {rf_pr_auc:.2f}")
plt.plot(iso_recall, iso_precision, label=f"IF PR-AUC = {iso_pr_auc:.2f}")

plt.title("Precision-Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()

print("\n✅ Evaluation Completed Successfully")