"""
Generate feature_importance.pdf - top-8 feature_importances_ XGBoost pada income_evaluation.
Run from slides/ directory: python3 figures/gen_feature_importance.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "feature_importance.pdf")

df = pd.read_csv(os.path.join(SCRIPT_DIR, "..", "..", "income_evaluation.csv"))
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

y = (df["income"] == ">50K").astype(int)
X = pd.get_dummies(df.drop(columns=["income"]), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = XGBClassifier(random_state=42, eval_metric="logloss")
clf.fit(X_train, y_train)

importances = pd.Series(clf.feature_importances_, index=X.columns)
top8 = importances.sort_values(ascending=False).head(8).sort_values()

# ── Plot ─────────────────────────────────────────────────────────────────
BG_COLOR = "#1A1A2E"
TEXT_COLOR = "white"
BAR_COLOR = "#76B900"

fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

bars = ax.barh(top8.index, top8.values, color=BAR_COLOR, zorder=3)
for bar, val in zip(bars, top8.values):
    ax.text(val + top8.values.max() * 0.015, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", color=TEXT_COLOR, fontsize=9)

ax.set_xlabel("Feature importance (XGBoost)", color=TEXT_COLOR, fontsize=10)
ax.tick_params(colors=TEXT_COLOR, labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor(TEXT_COLOR)
    spine.set_alpha(0.4)
ax.xaxis.label.set_color(TEXT_COLOR)
ax.set_xlim(0, top8.values.max() * 1.2)
ax.xaxis.grid(True, color="#333355", linestyle="--", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout(pad=0.5)
fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved: {OUTPUT_PATH}")
print("Top-8 feature importances:")
print(top8.sort_values(ascending=False))
