"""
Generate metrics_roc.pdf - confusion matrix & ROC curve, banking tanpa `duration`.
Run from slides/ directory: python3 figures/gen_metrics_roc.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "metrics_roc.pdf")

df = pd.read_csv(os.path.join(SCRIPT_DIR, "..", "..", "banking.csv"))
df.columns = df.columns.str.strip()

y = df["y"].astype(int)
X = pd.get_dummies(df.drop(columns=["y", "duration"]), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
clf.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)
y_proba = clf.predict_proba(X_test_s)[:, 1]

cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

# ── Plot ─────────────────────────────────────────────────────────────────
BG_COLOR = "#1A1A2E"
TEXT_COLOR = "white"
ACCENT = "#76B900"
LIGHT = "#A3D944"

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor(BG_COLOR)

# Left: confusion matrix
ax0 = axes[0]
ax0.set_facecolor(BG_COLOR)
im = ax0.imshow(cm, cmap="Greens", alpha=0.85)
labels_cls = ["Tidak (0)", "Ya (1)"]
ax0.set_xticks([0, 1])
ax0.set_yticks([0, 1])
ax0.set_xticklabels(labels_cls, color=TEXT_COLOR, fontsize=9)
ax0.set_yticklabels(labels_cls, color=TEXT_COLOR, fontsize=9)
ax0.set_xlabel("Prediksi", color=TEXT_COLOR, fontsize=10)
ax0.set_ylabel("Aktual", color=TEXT_COLOR, fontsize=10)
ax0.set_title("Confusion Matrix", color=TEXT_COLOR, fontsize=11, fontweight="bold")
vmax = cm.max()
for i in range(2):
    for j in range(2):
        txt_color = "white" if cm[i, j] < vmax * 0.6 else "black"
        ax0.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                 color=txt_color, fontsize=13, fontweight="bold")
for spine in ax0.spines.values():
    spine.set_edgecolor("#444466")

# Right: ROC curve
ax1 = axes[1]
ax1.set_facecolor(BG_COLOR)
ax1.plot(fpr, tpr, color=ACCENT, linewidth=2.5, label=f"AUC = {auc:.3f}")
ax1.plot([0, 1], [0, 1], color="#777799", linestyle="--", linewidth=1.2, label="Tebakan acak")
ax1.set_xlabel("False Positive Rate", color=TEXT_COLOR, fontsize=10)
ax1.set_ylabel("True Positive Rate", color=TEXT_COLOR, fontsize=10)
ax1.set_title("ROC Curve", color=TEXT_COLOR, fontsize=11, fontweight="bold")
ax1.tick_params(colors=TEXT_COLOR, labelsize=8)
for spine in ax1.spines.values():
    spine.set_edgecolor("#444466")
legend = ax1.legend(loc="lower right", framealpha=0.25, facecolor="#2A2A4E",
                     edgecolor="#555577", labelcolor=TEXT_COLOR, fontsize=9)

plt.tight_layout(pad=0.8)
fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved: {OUTPUT_PATH}")
print(f"Confusion matrix:\n{cm}")
print(f"AUC: {auc:.4f}")
