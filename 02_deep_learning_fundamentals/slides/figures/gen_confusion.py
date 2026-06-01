"""Figure: illustrative Fashion-MNIST confusion matrix (dark theme for slides).
Output: figures/confusion.pdf  (numbers are illustrative, not from a real run).
"""
import os
import matplotlib.pyplot as plt
import numpy as np

BG = "#1A1A2E"; GREEN = "#76B900"; WHITE = "#FFFFFF"

classes = ["T-shirt", "Celana", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
# Illustrative confusion matrix: strong diagonal, Shirt<->T-shirt/Pullover/Coat confusion.
rng = np.random.default_rng(42)
cm = np.zeros((10, 10), dtype=int)
diag = [890, 970, 840, 905, 830, 950, 700, 945, 965, 940]
for i in range(10):
    cm[i, i] = diag[i]
    off = 1000 - diag[i]
    js = [j for j in range(10) if j != i]
    spread = rng.multinomial(off, np.ones(len(js)) / len(js))
    for j, s in zip(js, spread):
        cm[i, j] = s
# Emphasize Shirt(6) confusion with T-shirt(0), Pullover(2), Coat(4)
cm[6, 0] += 90; cm[6, 2] += 60; cm[6, 4] += 50; cm[6, 6] -= 200

fig, ax = plt.subplots(figsize=(8.2, 6.6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
im = ax.imshow(cm, cmap="Greens")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(classes, rotation=45, ha="right", color=WHITE, fontsize=9)
ax.set_yticklabels(classes, color=WHITE, fontsize=9)
ax.set_xlabel("Prediksi", color=WHITE, fontsize=12)
ax.set_ylabel("Aktual", color=WHITE, fontsize=12)
ax.set_title("Confusion Matrix (ilustratif)", color=WHITE, fontsize=15, fontweight="bold", pad=12)
thresh = cm.max() / 2
for i in range(10):
    for j in range(10):
        if cm[i, j] > 0:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                    color=(WHITE if cm[i, j] > thresh else "#0d0d0d"))
for spine in ax.spines.values():
    spine.set_color(WHITE)
ax.tick_params(colors=WHITE)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/confusion.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
