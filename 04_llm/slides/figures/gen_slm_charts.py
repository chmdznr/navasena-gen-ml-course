"""SLM charts (nb05/06), tema gelap.
Output: figures/slm_accuracy.pdf, figures/slm_quadrant.pdf, figures/slm_gallery.pdf"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
BG, GREEN, LGREEN, WHITE, GRAY, RED = "#1A1A2E", "#76B900", "#A3D944", "#FFFFFF", "#AAAACC", "#EF5350"
HERE = os.path.dirname(__file__)
def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color(GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)

# (1) Field-accuracy head-to-head: base 0.6B, generalist 3B, specialist 0.6B
fig, ax = plt.subplots(figsize=(4.4, 3.0)); fig.patch.set_facecolor(BG); style(ax)
names = ["base 0.6B", "generalist 3B", "specialist 0.6B"]
acc   = [78, 72, 95]; colors = [GRAY, RED, GREEN]
ax.bar(names, acc, color=colors)
for i, v in enumerate(acc): ax.text(i, v+1.5, f"{v}%", ha="center", color=WHITE, fontsize=9)
ax.set_ylim(0, 105); ax.set_ylabel("Field-accuracy", color=WHITE, fontsize=9)
ax.set_title("Specialist 0.6B menang (sempit tapi dalam)", color=WHITE, fontsize=9)
plt.setp(ax.get_xticklabels(), color=WHITE, fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(HERE, "slm_accuracy.pdf"), facecolor=BG); plt.close()

# (2) VRAM x accuracy quadrant (the real SLM win = memory)
fig, ax = plt.subplots(figsize=(4.4, 3.0)); fig.patch.set_facecolor(BG); style(ax)
pts = {"base 0.6B": (0.54, 78, GRAY), "specialist 0.6B": (0.54, 95, GREEN), "generalist 3B": (1.99, 72, RED)}
for name, (x, y, c) in pts.items():
    ax.scatter(x, y, s=120, color=c); ax.annotate(name, (x, y), color=WHITE, fontsize=8, xytext=(6, 4), textcoords="offset points")
ax.set_xlabel("VRAM (GB) -- kiri lebih hemat", color=WHITE, fontsize=9)
ax.set_ylabel("Field-accuracy (%)", color=WHITE, fontsize=9)
ax.set_title("Hemat memory 3,7x + akurasi tertinggi", color=WHITE, fontsize=9)
ax.set_xlim(0, 2.3); ax.set_ylim(60, 100); plt.tight_layout()
plt.savefig(os.path.join(HERE, "slm_quadrant.pdf"), facecolor=BG); plt.close()

# (3) Task gallery: base -> specialist jump for 3 tasks
fig, ax = plt.subplots(figsize=(4.6, 3.0)); fig.patch.set_facecolor(BG); style(ax)
tasks = ["Intent", "Sentiment+Aspect", "Domain FAQ"]
base = [57, 0, 12]; spec = [100, 100, 100]
x = np.arange(len(tasks)); w = 0.38
ax.bar(x - w/2, base, w, color=GRAY, label="base 0.6B")
ax.bar(x + w/2, spec, w, color=GREEN, label="specialist 0.6B")
ax.set_xticks(x); ax.set_xticklabels(tasks, color=WHITE, fontsize=8)
ax.set_ylim(0, 112); ax.set_ylabel("Akurasi (%)", color=WHITE, fontsize=9)
ax.set_title("Satu resep QLoRA, banyak specialist 100%", color=WHITE, fontsize=9)
leg = ax.legend(facecolor=BG, edgecolor=GRAY, fontsize=8, labelcolor=WHITE)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "slm_gallery.pdf"), facecolor=BG); plt.close()
