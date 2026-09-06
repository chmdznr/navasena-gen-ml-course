"""
Generate bias_variance.pdf - polinomial derajat 1/4/15 pada data sinus + noise.
Run from slides/ directory: python3 figures/gen_bias_variance.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "bias_variance.pdf")

rng = np.random.RandomState(42)
n = 40
X = np.sort(rng.uniform(0, 1, n))
y = np.sin(2 * np.pi * X) + rng.normal(0, 0.25, n)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

x_plot = np.linspace(0, 1, 300)

degrees = [1, 4, 15]
titles = ["Underfitting (derajat 1)", "Mengikuti pola utama (derajat 4)", "Overfitting (derajat 15)"]

BG_COLOR = "#1A1A2E"
TEXT_COLOR = "white"
TRAIN_COLOR = "#A3D944"
TEST_COLOR = "#EF5350"
LINE_COLOR = "#76B900"

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
fig.patch.set_facecolor(BG_COLOR)

for ax, deg, title in zip(axes, degrees, titles):
    model = make_pipeline(PolynomialFeatures(deg), StandardScaler(), LinearRegression())
    model.fit(X_train.reshape(-1, 1), y_train)

    train_rmse = mean_squared_error(y_train, model.predict(X_train.reshape(-1, 1))) ** 0.5
    test_rmse = mean_squared_error(y_test, model.predict(X_test.reshape(-1, 1))) ** 0.5
    y_plot = model.predict(x_plot.reshape(-1, 1))

    ax.set_facecolor(BG_COLOR)
    ax.scatter(X_train, y_train, color=TRAIN_COLOR, s=25, label="Train", zorder=3)
    ax.scatter(X_test, y_test, color=TEST_COLOR, s=25, label="Test", zorder=3)
    ax.plot(x_plot, y_plot, color=LINE_COLOR, linewidth=2)
    ax.set_ylim(-2, 2)
    ax.set_title(f"{title}\nRMSE train={train_rmse:.2f}, test={test_rmse:.2f}",
                 color=TEXT_COLOR, fontsize=10.5, fontweight="bold", pad=6)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    print(f"Derajat {deg}: RMSE train={train_rmse:.4f}, test={test_rmse:.4f}")

axes[0].legend(loc="upper right", framealpha=0.25, facecolor="#2A2A4E",
               edgecolor="#555577", labelcolor=TEXT_COLOR, fontsize=8)

plt.tight_layout(pad=0.8)
fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved: {OUTPUT_PATH}")
