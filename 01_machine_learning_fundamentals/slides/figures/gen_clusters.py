"""
Generate clusters figure for LaTeX Beamer slide deck.
Run from slides/ directory: python3 figures/gen_clusters.py
Output: figures/clusters.pdf
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Determine output path relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "clusters.pdf")

# Generate synthetic data
X, y_true = make_blobs(n_samples=400, centers=5, cluster_std=1.2, random_state=42)

# Fit KMeans
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Color map for clusters
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(5)]

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#1A1A2E")
ax.set_facecolor("#1A1A2E")

for i in range(5):
    mask = labels == i
    ax.scatter(
        X[mask, 0], X[mask, 1],
        color=colors[i],
        alpha=0.6,
        s=40,
        edgecolors="white",
        linewidths=0.4,
    )

# Centroids
ax.scatter(
    centroids[:, 0], centroids[:, 1],
    marker="X",
    s=200,
    color="red",
    edgecolors="black",
    linewidths=2,
    zorder=5,
)

# Labels
ax.set_xlabel("Fitur 1", color="white")
ax.set_ylabel("Fitur 2", color="white")

# Tick styling
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("white")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUTPUT_PATH}")
