"""Dampak reranking: cross-encoder mendongkrak presisi konteks dibanding bi-encoder saja.
Angka ilustratif (arah & besaran selaras dengan pola nb03/nb04). Tema gelap, output PDF vektor.
Jalankan: python gen_rerank_lift.py  -> figures/rerank_lift.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, CARD, GREEN, GRAY, WHITE, ORANGE = "#1A1A2E", "#2D2D44", "#76B900", "#AAAACC", "#FFFFFF", "#FF6F00"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": WHITE, "axes.labelcolor": WHITE, "xtick.color": GRAY, "ytick.color": GRAY,
    "axes.edgecolor": GRAY, "font.size": 13, "font.family": "sans-serif",
})

metrics = ["Context\nPrecision", "Faithfulness", "Answer\nRelevancy"]
bi   = [0.61, 0.74, 0.79]   # ilustratif: bi-encoder saja (tanpa rerank)
both = [0.93, 0.91, 0.88]   # ilustratif: + cross-encoder rerank

x = np.arange(len(metrics)); w = 0.36
fig, ax = plt.subplots(figsize=(6.6, 3.6))
b1 = ax.bar(x - w/2, bi,   w, label="Bi-encoder saja",        color=GRAY,  edgecolor=BG)
b2 = ax.bar(x + w/2, both, w, label="+ Cross-encoder rerank", color=GREEN, edgecolor=BG)
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.015, f"{r.get_height():.2f}",
                ha="center", va="bottom", fontsize=10, color=WHITE)
ax.set_ylim(0, 1.08)
ax.set_ylabel("skor (0–1, makin tinggi makin baik)")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_title("Reranking mendongkrak kualitas RAG", color=WHITE, fontsize=14, pad=10)
ax.legend(facecolor=CARD, edgecolor=GRAY, labelcolor=WHITE, loc="lower right", fontsize=10)
ax.grid(axis="y", color=CARD, linewidth=0.8, alpha=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.5, 0.005, "angka ilustratif — arah selaras dengan pola nb03/nb04", ha="center", color=GRAY, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = os.path.join(os.path.dirname(__file__), "rerank_lift.pdf")
fig.savefig(out); print("wrote", out)
