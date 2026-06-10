"""
Generate rapids_speedup.pdf — speedup GPU vs CPU operasi NLP (Module 03 slides).
ANGKA dari benchmark_results.json hasil smoke-test Colab T4 notebook
02_nlp_on_steroids. Sebelum smoke-test: nilai ESTIMASI (jangan dipakai final).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "rapids_speedup.pdf")

# ── Data (kunci: operasi → speedup CPU/GPU) ────────────────────────────────
# >>> ESTIMASI SEMENTARA — WAJIB diganti angka benchmark_results.json (Task 7) <<<
SPEEDUP = {
    "tokenize":   8.0,
    "word count": 6.0,
    "bigram":     10.0,
    "TF-IDF + LogReg": 4.0,
}

BG_COLOR  = "#1A1A2E"
BAR_COLOR = "#76B900"  # navasena green

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

ops = list(SPEEDUP)
vals = [SPEEDUP[o] for o in ops]
bars = ax.barh(ops, vals, color=BAR_COLOR, height=0.55)
ax.bar_label(bars, fmt="%.1fx", color="white", fontsize=12, padding=4)
ax.axvline(1, color="#E0E0E0", lw=1, ls="--")
ax.text(1.05, -0.45, "CPU = 1x", color="#E0E0E0", fontsize=9)

ax.set_xlabel("Speedup vs CPU (lebih besar = lebih cepat)", color="white")
ax.tick_params(colors="white", labelsize=11)
for spine in ax.spines.values():
    spine.set_color("#444466")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUTPUT_PATH, facecolor=BG_COLOR)
print("wrote", OUTPUT_PATH)
