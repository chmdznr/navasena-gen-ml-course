"""
Generate figures/gpu_benchmark.pdf - perbandingan CPU vs GPU (waktu relatif)
untuk Training dan Inference dalam 2 subplot bar chart, tema gelap untuk slide.
Angka ILUSTRATIF (hardcoded). Run from the directory yang berisi folder figures/:
    python3 gen_gpu_benchmark.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Output path RELATIF
OUTPUT_DIR = "figures"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "gpu_benchmark.pdf")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tema gelap
BG_COLOR = "#1A1A2E"
TEXT_COLOR = "white"
GRID_COLOR = "#333355"
PANEL_COLOR = "#2A2A4E"

# Warna bar
CPU_COLOR = "#EF5350"     # CPU (merah)
GPU_COLOR = "#76B900"     # GPU aksen hijau NVIDIA
ANNOT_COLOR = "#A3D944"   # anotasi speedup

# Data ILUSTRATIF hardcoded (waktu relatif, CPU = baseline 1.0 di tiap panel)
# Training: GPU ~15x lebih cepat; Inference: GPU ~8x lebih cepat
labels = ["CPU", "GPU"]
training_times = [15.0, 1.0]    # waktu relatif
inference_times = [8.0, 1.0]    # waktu relatif

panels = [
    ("Training", training_times),
    ("Inference", inference_times),
]

x = np.arange(len(labels))
bar_width = 0.55

fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=False)
fig.patch.set_facecolor(BG_COLOR)

for ax, (panel_title, times) in zip(axes, panels):
    ax.set_facecolor(BG_COLOR)

    colors = [CPU_COLOR, GPU_COLOR]
    bars = ax.bar(x, times, bar_width, color=colors, zorder=3)

    # Label nilai di atas tiap bar
    for rect, t in zip(bars, times):
        ax.text(rect.get_x() + rect.get_width() / 2,
                t + max(times) * 0.02,
                f"{t:.0f}x", ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=10, fontweight="bold")

    # Anotasi "~Nx lebih cepat" (CPU / GPU)
    speedup = times[0] / times[1]
    ax.annotate(
        f"~{speedup:.0f}x lebih cepat",
        xy=(1, times[1]), xytext=(0.5, max(times) * 0.6),
        ha="center", va="center",
        color=ANNOT_COLOR, fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ANNOT_COLOR, lw=1.6),
    )

    # Styling axes
    ax.set_title(panel_title, color=TEXT_COLOR, fontsize=12,
                 fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Waktu relatif", color=TEXT_COLOR, fontsize=10)
    ax.set_ylim(0, max(times) * 1.25)
    ax.tick_params(axis="x", colors=TEXT_COLOR)
    ax.tick_params(axis="y", colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    ax.yaxis.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

# Judul utama
fig.suptitle("CPU vs GPU", color=TEXT_COLOR, fontsize=15,
             fontweight="bold", y=0.99)

# Legend bersama
handles = [
    plt.Rectangle((0, 0), 1, 1, color=CPU_COLOR),
    plt.Rectangle((0, 0), 1, 1, color=GPU_COLOR),
]
fig.legend(handles, ["CPU", "GPU NVIDIA"],
           loc="lower center", ncol=2, framealpha=0.3,
           facecolor=PANEL_COLOR, edgecolor="#555577",
           labelcolor=TEXT_COLOR, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=(0, 0.04, 1, 0.95))
plt.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {OUTPUT_PATH}")
