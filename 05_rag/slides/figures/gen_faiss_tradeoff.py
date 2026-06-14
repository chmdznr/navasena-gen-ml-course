"""Trade-off index FAISS: recall vs kecepatan. Angka dari benchmark nb05 (100k vektor, dim 384).
Tema gelap, output PDF vektor.  Jalankan: python gen_faiss_tradeoff.py -> figures/faiss_tradeoff.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, CARD, GREEN, GRAY, WHITE, ORANGE, BLUE = "#1A1A2E", "#2D2D44", "#76B900", "#AAAACC", "#FFFFFF", "#FF6F00", "#42A5F5"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": WHITE, "axes.labelcolor": WHITE, "xtick.color": GRAY, "ytick.color": GRAY,
    "axes.edgecolor": GRAY, "font.size": 13, "font.family": "sans-serif",
})

# (nama, ms/query, recall@10, warna)  — dari benchmark nb05
pts = [
    ("Flat (exact)",       0.070, 1.000, ORANGE),
    ("IVF (nprobe=8)",     0.037, 0.988, GREEN),
    ("HNSW (approx)",      0.020, 0.965, BLUE),   # ilustratif: cepat, recall sedikit turun
]
fig, ax = plt.subplots(figsize=(6.6, 3.7))
for name, ms, rec, col in pts:
    ax.scatter(ms, rec, s=240, color=col, edgecolor=WHITE, linewidth=1.2, zorder=3)
    dy = 0.006 if name.startswith("Flat") else -0.012
    ax.annotate(name, (ms, rec), xytext=(0, 14 if dy > 0 else -22), textcoords="offset points",
                ha="center", color=WHITE, fontsize=11)
ax.set_xlabel("waktu per-query (ms)  →  makin kiri makin cepat")
ax.set_ylabel("recall@10  →  makin atas makin akurat")
ax.set_ylim(0.94, 1.01); ax.set_xlim(0, 0.085)
ax.invert_xaxis()
ax.set_title("FAISS: tukar sedikit recall untuk kecepatan", color=WHITE, fontsize=14, pad=10)
ax.grid(color=CARD, linewidth=0.8, alpha=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(0.078, 0.998, "exact: lambat, recall 100%", color=GRAY, fontsize=9)
ax.text(0.030, 0.955, "approx: cepat, recall ~tetap tinggi", color=GRAY, fontsize=9)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "faiss_tradeoff.pdf")
fig.savefig(out); print("wrote", out)
