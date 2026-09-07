"""
Generate regression_line.pdf for Beamer slide deck.
Run from slides/ directory: python3 figures/gen_regression_line.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# Reproducibility
np.random.seed(42)

# Synthetic data: analogi suhu vs jumlah es krim terjual (§2.4 - korelasi,
# bukan klaim sebab-akibat), suhu 15-35 C vs jumlah es krim terjual per hari
n = 80
suhu = np.random.uniform(15, 35, n)
noise = np.random.normal(0, 8, n)
es_krim_terjual = 6.0 * suhu - 40 + noise

X = suhu.reshape(-1, 1)
y = es_krim_terjual

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

x_line = np.linspace(15, 35, 300).reshape(-1, 1)
y_line = model.predict(x_line)

# --- Plot ---
BG_COLOR = "#1A1A2E"
TEXT_COLOR = "white"
POINT_COLOR = "#A3D944"
LINE_COLOR = "#76B900"

fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

ax.scatter(suhu, es_krim_terjual, color=POINT_COLOR, alpha=0.6, s=30, zorder=3)
ax.plot(x_line, y_line, color=LINE_COLOR, linewidth=2.5, zorder=4)

ax.set_xlabel("Suhu (°C)", color=TEXT_COLOR, fontsize=11)
ax.set_ylabel("Es krim terjual (buah)", color=TEXT_COLOR, fontsize=11)

ax.tick_params(colors=TEXT_COLOR, labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor(TEXT_COLOR)
    spine.set_alpha(0.4)

ax.xaxis.label.set_color(TEXT_COLOR)
ax.yaxis.label.set_color(TEXT_COLOR)

plt.tight_layout(pad=0.3)

# Save relative to this script's location (figures/)
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression_line.pdf")
fig.savefig(output_path, format="pdf", bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved: {output_path}")
