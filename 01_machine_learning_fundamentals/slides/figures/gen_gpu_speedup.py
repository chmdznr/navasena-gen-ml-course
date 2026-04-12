"""
Generate gpu_speedup.pdf - horizontal bar chart comparing CPU vs GPU training times.
Run from slides/ directory: python3 figures/gen_gpu_speedup.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Output path relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "gpu_speedup.pdf")

# Benchmark data
algorithms = [
    "Linear Regression",
    "KNN (K=5)",
    "Random Forest",
    "K-Means",
    "PCA",
    "XGBoost",
]
cpu_times = [0.15, 45.2, 52.3, 28.7, 3.4, 89.5]
gpu_times = [0.08, 3.8, 8.1, 1.2, 0.15, 12.3]
speedups = [1.9, 11.9, 6.5, 23.9, 22.7, 7.3]

# Colors
BG_COLOR = "#1A1A2E"
CPU_COLOR = "#EF5350"
GPU_COLOR = "#76B900"
SPEEDUP_COLOR = "#A3D944"
TEXT_COLOR = "white"

# Layout
n = len(algorithms)
y = np.arange(n)
bar_height = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Draw bars
bars_cpu = ax.barh(y + bar_height / 2, cpu_times, bar_height,
                   color=CPU_COLOR, label="CPU", zorder=3)
bars_gpu = ax.barh(y - bar_height / 2, gpu_times, bar_height,
                   color=GPU_COLOR, label="GPU NVIDIA", zorder=3)

# Speedup annotations at the end of each CPU bar (the longer one)
for i, (ct, speedup) in enumerate(zip(cpu_times, speedups)):
    ax.text(ct + max(cpu_times) * 0.01, y[i] + bar_height / 2,
            f"{speedup}x",
            va="center", ha="left",
            color=SPEEDUP_COLOR, fontsize=9, fontweight="bold")

# Axes styling
ax.set_yticks(y)
ax.set_yticklabels(algorithms, color=TEXT_COLOR, fontsize=10)
ax.set_xlabel("Waktu Training (detik)", color=TEXT_COLOR, fontsize=10)
ax.tick_params(axis="x", colors=TEXT_COLOR)
ax.tick_params(axis="y", colors=TEXT_COLOR)
for spine in ax.spines.values():
    spine.set_edgecolor("#444466")

ax.xaxis.label.set_color(TEXT_COLOR)
ax.set_xlim(0, max(cpu_times) * 1.18)

# Grid
ax.xaxis.grid(True, color="#333355", linestyle="--", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# Legend
legend = ax.legend(loc="lower right", framealpha=0.25,
                   facecolor="#2A2A4E", edgecolor="#555577",
                   labelcolor=TEXT_COLOR, fontsize=9)

plt.tight_layout(pad=1.2)
plt.savefig(OUTPUT_PATH, format="pdf", dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print(f"Saved: {OUTPUT_PATH}")
