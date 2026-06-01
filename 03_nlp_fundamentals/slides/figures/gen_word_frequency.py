"""
Generate word_frequency.pdf — comparing word frequencies in English vs
Indonesian sample texts (illustrative example for Module 03 slides).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Output path ──────────────────────────────────────────────────────────────
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "word_frequency.pdf")

# ── Data ────────────────────────────────────────────────────────────────────
# English sample
en_words = ["the", "data", "model", "language", "is", "are", "token", "word"]
en_freq  = [12, 9, 8, 7, 6, 5, 4, 4]
# Indonesian sample
id_words = ["yang", "data", "model", "bahasa", "ini", "token", "kata", "untuk"]
id_freq  = [14, 9, 7, 6, 5, 4, 4, 3]

# ── Colors (Navasena dark theme) ────────────────────────────────────────────
BG_COLOR     = "#1A1A2E"
EN_COLOR     = "#76B900"  # navasena green
ID_COLOR     = "#42A5F5"  # complementary blue

# ── Figure ───────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
fig.patch.set_facecolor(BG_COLOR)

for ax, words, freq, color, title in [
    (ax1, en_words, en_freq, EN_COLOR, "English: top 8 words"),
    (ax2, id_words, id_freq, ID_COLOR, "Bahasa Indonesia: top 8 kata"),
]:
    # Sort by frequency
    sorted_pairs = sorted(zip(freq, words), reverse=True)
    sorted_freq = [p[0] for p in sorted_pairs]
    sorted_words = [p[1] for p in sorted_pairs]

    y_pos = np.arange(len(sorted_words))
    ax.barh(y_pos, sorted_freq, color=color, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_words, color="white", fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency", color="#AAAACC", fontsize=10)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors="#AAAACC", labelsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.2, color="#AAAACC")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

plt.tight_layout(pad=1.5)
plt.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", facecolor=BG_COLOR)
plt.close()

print(f"Saved: {OUTPUT_PATH}")
