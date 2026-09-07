"""
Generate gpu_speedup.pdf - horizontal bar chart CPU vs GPU, dari hasil pengukuran nb10 Colab.
Run from slides/ directory: python3 figures/gen_gpu_speedup.py

Sumber data: $SCRATCH/gpu_results.json (hasil benchmark Colab T4 nb10, dibuat manual
setelah menjalankan notebook). Format yang diharapkan:
{
  "dataset": "Covertype",
  "n_rows": 581012,
  "measured": "waktu fit",
  "device_cpu": "2 vCPU",
  "device_gpu": "Colab T4",
  "algorithms": ["XGBoost", "K-Means", ...],
  "cpu_times": [89.5, 28.7, ...],
  "gpu_times": [12.3, 1.2, ...]
}
Jika file belum ada, figur placeholder "hasil menyusul" tetap dibuat dan script exit 0
supaya tidak memblokir build.sh — angka GPU HANYA boleh berasal dari pengukuran nyata
(§2.4), bukan angka karangan.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "gpu_speedup.pdf")
# Default: gpu_results.json ditaruh sebaris dengan CSV lain (01_machine_learning_fundamentals/),
# dibuat manual setelah menjalankan nb10 di Colab. GPU_RESULTS_JSON dapat override untuk testing.
RESULTS_PATH = os.environ.get(
    "GPU_RESULTS_JSON",
    os.path.join(SCRIPT_DIR, "..", "..", "gpu_results.json"),
)

BG_COLOR = "#1A1A2E"
CPU_COLOR = "#EF5350"
GPU_COLOR = "#76B900"
SPEEDUP_COLOR = "#A3D944"
TEXT_COLOR = "white"

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

if not os.path.isfile(RESULTS_PATH):
    # Placeholder: belum ada hasil pengukuran Colab nb10.
    ax.text(0.5, 0.58, "Hasil benchmark GPU belum diukur",
            ha="center", va="center", color=TEXT_COLOR, fontsize=34,
            transform=ax.transAxes)
    ax.text(0.5, 0.40, "Diisi setelah nb10 dijalankan di Colab T4",
            ha="center", va="center", color=TEXT_COLOR, fontsize=30,
            transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    plt.tight_layout(pad=1.2)
    fig.savefig(OUTPUT_PATH, format="pdf", dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Saved placeholder: {OUTPUT_PATH} (tidak ditemukan {RESULTS_PATH})")
    raise SystemExit(0)

with open(RESULTS_PATH) as f:
    results = json.load(f)

algorithms = results["algorithms"]
cpu_times = results["cpu_times"]
gpu_times = results["gpu_times"]
speedups = [c / g if g else float("nan") for c, g in zip(cpu_times, gpu_times)]

n = len(algorithms)
y = np.arange(n)
bar_height = 0.35

bars_cpu = ax.barh(y + bar_height / 2, cpu_times, bar_height, color=CPU_COLOR,
                    label=f"CPU ({results.get('device_cpu', 'CPU')})", zorder=3)
bars_gpu = ax.barh(y - bar_height / 2, gpu_times, bar_height, color=GPU_COLOR,
                    label=f"GPU ({results.get('device_gpu', 'GPU')})", zorder=3)

for i, (ct, speedup) in enumerate(zip(cpu_times, speedups)):
    ax.text(ct * 1.08, y[i] + bar_height / 2, f"{speedup:.1f}x",
            va="center", ha="left", color=SPEEDUP_COLOR, fontsize=9, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(algorithms, color=TEXT_COLOR, fontsize=10)
ax.set_xlabel(f"Waktu {results.get("measured", "fit")} (detik, skala log)", color=TEXT_COLOR, fontsize=10)
ax.tick_params(axis="x", colors=TEXT_COLOR)
ax.tick_params(axis="y", colors=TEXT_COLOR)
for spine in ax.spines.values():
    spine.set_edgecolor("#444466")
ax.xaxis.label.set_color(TEXT_COLOR)
ax.set_xscale("log")
ax.set_xlim(min(min(cpu_times), min(gpu_times)) * 0.6, max(cpu_times) * 2.2)
ax.xaxis.grid(True, color="#333355", linestyle="--", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

caption = (f"{results.get('dataset', '')}, {results.get('n_rows', '')} baris — "
           f"{results.get('device_gpu', 'GPU')} vs {results.get('device_cpu', 'CPU')}")
ax.set_title(caption, color=TEXT_COLOR, fontsize=9, loc="left", pad=8)

legend = ax.legend(loc="lower right", framealpha=0.25, facecolor="#2A2A4E",
                    edgecolor="#555577", labelcolor=TEXT_COLOR, fontsize=9)

plt.tight_layout(pad=1.2)
fig.savefig(OUTPUT_PATH, format="pdf", dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"Saved: {OUTPUT_PATH}")
