"""
Generate figures/gpu_benchmark.pdf dari ../../gpu_results_m02.json (hasil
nb05 dijalankan di Colab T4): {"algorithms": [...], "cpu_times": [...],
"gpu_times": [...], "device_gpu": "...", "device_cpu": "...", "measured": "..."}.
Jika file belum ada, gambar placeholder gelap (belum diukur) dan exit 0.
Run dari direktori slides/: python3 figures/gen_gpu_benchmark.py
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18,
                     'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16})
import numpy as np

BG_COLOR = "#1A1A2E"
TEXT_COLOR = "white"
GRID_COLOR = "#333355"
PANEL_COLOR = "#2A2A4E"
CPU_COLOR = "#EF5350"
GPU_COLOR = "#76B900"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "gpu_benchmark.pdf")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "..", "..", "gpu_results_m02.json")

if not os.path.isfile(RESULTS_PATH):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")
    ax.text(0.5, 0.58, "Hasil benchmark GPU belum diukur", color=TEXT_COLOR,
            fontsize=30, fontweight="bold", ha="center", va="center", wrap=True)
    ax.text(0.5, 0.38, "Diisi setelah nb05 dijalankan di runtime sesi ini", color=TEXT_COLOR,
            fontsize=30, ha="center", va="center", wrap=True)
    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved placeholder: {OUTPUT_PATH}")
    raise SystemExit(0)

with open(RESULTS_PATH) as f:
    data = json.load(f)

algorithms = data["algorithms"]
cpu_times = np.array(data["cpu_times"], dtype=float)
gpu_times = np.array(data["gpu_times"], dtype=float)
device_gpu = data.get("device_gpu", "GPU")
device_cpu = data.get("device_cpu", "CPU")
measured = data.get("measured", "")

x = np.arange(len(algorithms))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

bars_cpu = ax.bar(x - width / 2, cpu_times, width, color=CPU_COLOR, label=device_cpu, zorder=3)
bars_gpu = ax.bar(x + width / 2, gpu_times, width, color=GPU_COLOR, label=device_gpu, zorder=3)

for rect, t in zip(bars_cpu, cpu_times):
    ax.text(rect.get_x() + rect.get_width() / 2, t + max(cpu_times) * 0.02,
            f"{t:.2f}s", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
for rect, t in zip(bars_gpu, gpu_times):
    ax.text(rect.get_x() + rect.get_width() / 2, t + max(cpu_times) * 0.02,
            f"{t:.2f}s", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(algorithms, color=TEXT_COLOR, fontsize=10)
ax.set_ylabel("Waktu (detik)", color=TEXT_COLOR, fontsize=11)
title = f"CPU vs GPU ({measured})" if measured else "CPU vs GPU"
ax.set_title(title, color=TEXT_COLOR, fontsize=15, fontweight="bold", pad=12)
ax.tick_params(axis="y", colors=TEXT_COLOR)
for spine in ax.spines.values():
    spine.set_edgecolor("#444466")
ax.yaxis.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
leg = ax.legend(loc="upper right", framealpha=0.3, facecolor=PANEL_COLOR,
                 edgecolor="#555577", labelcolor=TEXT_COLOR, fontsize=10)

plt.tight_layout(pad=1.0)
fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved: {OUTPUT_PATH}")
speedups = cpu_times / gpu_times
for algo, s in zip(algorithms, speedups):
    print(f"{algo}: speedup={s:.2f}x")
