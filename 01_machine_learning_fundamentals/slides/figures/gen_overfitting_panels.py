"""
Generate overfitting_panels.pdf for LaTeX Beamer slide deck.
Shows underfitting, optimal, and overfitting decision boundaries using KNN on make_moons.

Run from slides/ directory:
    python3 figures/gen_overfitting_panels.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "overfitting_panels.pdf")

# ── Colors ───────────────────────────────────────────────────────────────────
BG_COLOR = "#1A1A2E"
RED_TITLE = "#EF5350"
GREEN_TITLE = "#76B900"

# ── Data ─────────────────────────────────────────────────────────────────────
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Panel definitions ─────────────────────────────────────────────────────────
panels = [
    {"k": 100, "label": "Underfitting",  "title_color": RED_TITLE},
    {"k": 5,   "label": "Optimal",       "title_color": GREEN_TITLE},
    {"k": 1,   "label": "Overfitting",   "title_color": RED_TITLE},
]

# ── Decision boundary mesh ────────────────────────────────────────────────────
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.patch.set_facecolor(BG_COLOR)

# Colormap for decision regions (two-class: 0=cool, 1=warm)
region_cmap = plt.get_cmap("RdYlGn")
region_colors = ListedColormap([region_cmap(0.1), region_cmap(0.9)])

# Scatter colormap for points
point_cmap = ListedColormap(["#E53935", "#43A047"])   # red / green

for ax, panel in zip(axes, panels):
    k = panel["k"]
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc  = clf.score(X_test,  y_test)

    # Decision region
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=region_colors)

    # Test data points (generalization view)
    scatter = ax.scatter(
        X_test[:, 0], X_test[:, 1],
        c=y_test,
        cmap=point_cmap,
        edgecolors="white",
        linewidths=0.5,
        alpha=0.8,
        s=20,
        zorder=3,
    )

    # Styling
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    ax.set_title(
        panel["label"],
        color=panel["title_color"],
        fontsize=13,
        fontweight="bold",
        pad=6,
    )
    ax.set_xlabel(
        f"Train={train_acc:.3f} | Test={test_acc:.3f}",
        color="#CCCCDD",
        fontsize=9,
        labelpad=4,
    )

plt.tight_layout(pad=1.2)
plt.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", facecolor=BG_COLOR)
plt.close()

print(f"Saved: {OUTPUT_PATH}")
