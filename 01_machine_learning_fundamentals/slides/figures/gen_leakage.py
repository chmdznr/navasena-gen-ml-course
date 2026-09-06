"""
Generate leakage.pdf - AUC banking dengan vs tanpa kolom `duration` (leakage).
Run from slides/ directory: python3 figures/gen_leakage.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "leakage.pdf")

df = pd.read_csv(os.path.join(SCRIPT_DIR, "..", "..", "banking.csv"))
df.columns = df.columns.str.strip()

y = df["y"].astype(int)
X_full = pd.get_dummies(df.drop(columns=["y"]), drop_first=True)
X_no_duration = pd.get_dummies(df.drop(columns=["y", "duration"]), drop_first=True)


def fit_auc(X):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)
    proba = clf.predict_proba(X_test_s)[:, 1]
    return roc_auc_score(y_test, proba)


auc_with = fit_auc(X_full)
auc_without = fit_auc(X_no_duration)

# ── Plot ─────────────────────────────────────────────────────────────────
BG_COLOR = "#1A1A2E"
TEXT_COLOR = "white"
LEAK_COLOR = "#EF5350"
CLEAN_COLOR = "#76B900"

labels = ["Dengan duration\n(leakage)", "Tanpa duration"]
values = [auc_with, auc_without]
colors = [LEAK_COLOR, CLEAN_COLOR]

fig, ax = plt.subplots(figsize=(5.5, 4.5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
            ha="center", va="bottom", color=TEXT_COLOR, fontsize=13, fontweight="bold")

ax.set_ylabel("AUC (test set)", color=TEXT_COLOR, fontsize=11)
ax.set_ylim(0, 1.05)
ax.tick_params(colors=TEXT_COLOR, labelsize=10)
for spine in ax.spines.values():
    spine.set_edgecolor(TEXT_COLOR)
    spine.set_alpha(0.4)
ax.yaxis.label.set_color(TEXT_COLOR)
ax.yaxis.grid(True, color="#333355", linestyle="--", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout(pad=0.5)
fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved: {OUTPUT_PATH}")
print(f"AUC dengan duration: {auc_with:.4f}, tanpa duration: {auc_without:.4f}")
