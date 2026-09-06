"""
Generate knn_scaling.pdf - boundary KNN k=5 tanpa vs dengan scaling, Social_Network_Ads.
Run from slides/ directory: python3 figures/gen_knn_scaling.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "knn_scaling.pdf")

df = pd.read_csv(os.path.join(SCRIPT_DIR, "..", "..", "Social_Network_Ads.csv"))
df.columns = df.columns.str.strip()

X = df[["Age", "EstimatedSalary"]].values.astype(float)
y = df["Purchased"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

BG_COLOR = "#1A1A2E"
TEXT_COLOR = "white"
RDYLGN = plt.get_cmap("RdYlGn")
CLASS_COLS = [RDYLGN(0.15), RDYLGN(0.85)]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
fig.patch.set_facecolor(BG_COLOR)


def plot_panel(ax, Xtr, ytr, Xte, yte, title, xlabel, ylabel):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))

    Xall = np.vstack([Xtr, Xte])
    x_min, x_max = Xall[:, 0].min() - 1, Xall[:, 0].max() + 1
    y_min, y_max = Xall[:, 1].min() - 1, Xall[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.set_facecolor(BG_COLOR)
    ax.contourf(xx, yy, Z, alpha=0.35, cmap="RdYlGn", levels=[-0.5, 0.5, 1.5])
    for cls_val, col in enumerate(CLASS_COLS):
        mask = ytr == cls_val
        ax.scatter(Xtr[mask, 0], Xtr[mask, 1], c=[col], edgecolors="white",
                   linewidths=0.5, alpha=0.85, s=18, zorder=3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{title}\nAkurasi test = {acc:.3f}", color=TEXT_COLOR,
                 fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    return acc


acc_raw = plot_panel(axes[0], X_train, y_train, X_test, y_test,
                      "KNN K=5, tanpa scaling", "Usia (tahun)", "Estimasi gaji")
acc_scaled = plot_panel(axes[1], X_train_s, y_train, X_test_s, y_test,
                         "KNN K=5, dengan scaling", "Usia (scaled)", "Estimasi gaji (scaled)")

plt.tight_layout(pad=0.8)
fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved: {OUTPUT_PATH}")
print(f"Akurasi tanpa scaling: {acc_raw:.4f}, dengan scaling: {acc_scaled:.4f}")
