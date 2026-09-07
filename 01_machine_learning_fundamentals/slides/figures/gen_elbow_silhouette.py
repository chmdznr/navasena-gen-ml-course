"""
Generate elbow_silhouette.pdf - inertia & silhouette K-means, Wholesale customers.
Run from slides/ directory: python3 figures/gen_elbow_silhouette.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "elbow_silhouette.pdf")

df = pd.read_csv(os.path.join(SCRIPT_DIR, "..", "..", "Wholesale customers data.csv"))
df.columns = df.columns.str.strip()

spend_cols = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
X_log = np.log1p(df[spend_cols])
X_scaled = StandardScaler().fit_transform(X_log)

k_range = range(2, 9)
inertias = []
silhouettes = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

best_k = list(k_range)[int(np.argmax(silhouettes))]

# ── Plot ─────────────────────────────────────────────────────────────────
BG_COLOR = "#1A1A2E"
TEXT_COLOR = "white"
LINE_COLOR = "#76B900"
MARK_COLOR = "#A3D944"
BEST_COLOR = "#EF5350"

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor(BG_COLOR)

ax0 = axes[0]
ax0.set_facecolor(BG_COLOR)
ax0.plot(list(k_range), inertias, marker="o", color=LINE_COLOR,
         markerfacecolor=MARK_COLOR, linewidth=2)
ax0.set_xlabel("K (jumlah kelompok)", color=TEXT_COLOR, fontsize=10)
ax0.set_ylabel("Inertia", color=TEXT_COLOR, fontsize=10)
ax0.set_title("Kurva inertia", color=TEXT_COLOR, fontsize=11, fontweight="bold")
ax0.tick_params(colors=TEXT_COLOR, labelsize=9)
for spine in ax0.spines.values():
    spine.set_edgecolor("#444466")
ax0.xaxis.grid(True, color="#333355", linestyle="--", linewidth=0.6, zorder=0)

ax1 = axes[1]
ax1.set_facecolor(BG_COLOR)
ax1.plot(list(k_range), silhouettes, marker="o", color=LINE_COLOR,
         markerfacecolor=MARK_COLOR, linewidth=2)
ax1.axvline(best_k, color=BEST_COLOR, linestyle="--", linewidth=1.5,
            label=f"Terbaik: K={best_k}")
ax1.set_xlabel("K (jumlah kelompok)", color=TEXT_COLOR, fontsize=10)
ax1.set_ylabel("Skor silhouette", color=TEXT_COLOR, fontsize=10)
ax1.set_title("Skor silhouette", color=TEXT_COLOR, fontsize=11, fontweight="bold")
ax1.tick_params(colors=TEXT_COLOR, labelsize=9)
for spine in ax1.spines.values():
    spine.set_edgecolor("#444466")
ax1.xaxis.grid(True, color="#333355", linestyle="--", linewidth=0.6, zorder=0)
legend = ax1.legend(loc="best", framealpha=0.25, facecolor="#2A2A4E",
                     edgecolor="#555577", labelcolor=TEXT_COLOR, fontsize=9)

plt.tight_layout(pad=0.8)
fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved: {OUTPUT_PATH}")
print(f"Inertias: {inertias}")
print(f"Silhouettes: {silhouettes}")
print(f"Best K by silhouette: {best_k}")
