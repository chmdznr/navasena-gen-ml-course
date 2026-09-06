"""
Generate svm_margin.pdf - SVC linear pada data sintetis separable, margin & support vectors.
Run from slides/ directory: python3 figures/gen_svm_margin.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "svm_margin.pdf")

X, y = make_blobs(n_samples=60, centers=2, cluster_std=1.1, random_state=42)

clf = SVC(kernel="linear", C=1000, random_state=42)
clf.fit(X, y)

BG_COLOR = "#1A1A2E"
TEXT_COLOR = "white"
POINT_COLORS = ["#EF5350", "#76B900"]
LINE_COLOR = "white"
SV_COLOR = "#A3D944"

fig, ax = plt.subplots(figsize=(6.5, 5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

for cls_val, col in enumerate(POINT_COLORS):
    mask = y == cls_val
    ax.scatter(X[mask, 0], X[mask, 1], color=col, s=35, edgecolors="white",
               linewidths=0.5, zorder=3)

# Decision boundary and margins
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300)
yy = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300)
YY, XX = np.meshgrid(yy, xx)
grid = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(grid).reshape(XX.shape)

ax.contour(XX, YY, Z, colors=LINE_COLOR, levels=[-1, 0, 1],
           linestyles=["--", "-", "--"], linewidths=[1.2, 2.2, 1.2], zorder=2)

# Circle support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=180, facecolors="none", edgecolors=SV_COLOR, linewidths=1.8, zorder=4,
           label="Support vector")

ax.set_xlabel("Feature 1", color=TEXT_COLOR, fontsize=10)
ax.set_ylabel("Feature 2", color=TEXT_COLOR, fontsize=10)
ax.tick_params(colors=TEXT_COLOR, labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#444466")
legend = ax.legend(loc="best", framealpha=0.25, facecolor="#2A2A4E",
                    edgecolor="#555577", labelcolor=TEXT_COLOR, fontsize=9)

plt.tight_layout(pad=0.5)
fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved: {OUTPUT_PATH}")
print(f"Jumlah support vectors: {len(clf.support_vectors_)}")
