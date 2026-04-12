"""
Generate decision_boundaries.pdf for LaTeX Beamer slides.
Run from the slides/ directory: python3 figures/gen_decision_boundaries.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# ── Data ────────────────────────────────────────────────────────────────────
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Models ──────────────────────────────────────────────────────────────────
models = [
    ("Decision Tree",  DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("KNN (K=5)",      KNeighborsClassifier(n_neighbors=5)),
    ("SVM Linear",     SVC(kernel="linear", random_state=42)),
    ("SVM RBF",        SVC(kernel="rbf",    random_state=42)),
]

# ── Colour palette ───────────────────────────────────────────────────────────
BG_COLOR   = "#1A1A2E"
RDYLGN     = plt.cm.get_cmap("RdYlGn")
CLASS_COLS = [RDYLGN(0.15), RDYLGN(0.85)]   # red-ish / green-ish for points

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
fig.patch.set_facecolor(BG_COLOR)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)
grid = np.c_[xx.ravel(), yy.ravel()]

for ax, (title, clf) in zip(axes, models):
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    # Decision region
    Z = clf.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.35, cmap="RdYlGn", levels=[-0.5, 0.5, 1.5])

    # Scatter points
    for cls_val, col in enumerate(CLASS_COLS):
        mask = y == cls_val
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=[col], edgecolors="white", linewidths=0.5,
            alpha=0.8, s=20, zorder=3,
        )

    # Styling
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{title}\nAcc={acc:.3f}", color="white",
                 fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

plt.tight_layout(pad=0.8)

out_path = os.path.join(os.path.dirname(__file__), "decision_boundaries.pdf")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved: {out_path}")
