"""Figure: RNN sine-wave prediction — actual vs predicted (dark theme for slides).
Output: figures/rnn_sine.pdf  (illustrative; predicted closely tracks actual).
"""
import os
import matplotlib.pyplot as plt
import numpy as np

BG = "#1A1A2E"; GREEN = "#76B900"; ORANGE = "#FF6F00"; WHITE = "#FFFFFF"

t = np.linspace(0, 8 * np.pi, 200)
actual = np.sin(t)
rng = np.random.default_rng(7)
# Predicted: tracks actual with small lag + noise (illustrative RNN output)
predicted = np.sin(t - 0.08) * 0.97 + rng.normal(0, 0.025, size=t.shape)

fig, ax = plt.subplots(figsize=(11, 4.3))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(t, actual, color=GREEN, linewidth=2.2, label="Aktual")
ax.plot(t, predicted, color=ORANGE, linewidth=1.6, linestyle="--", label="Prediksi RNN")
ax.set_title("Prediksi Gelombang Sinus dengan SimpleRNN", color=WHITE, fontsize=15, fontweight="bold", pad=10)
ax.set_xlabel("Waktu (timestep)", color=WHITE, fontsize=11)
ax.set_ylabel("Nilai", color=WHITE, fontsize=11)
ax.tick_params(colors=WHITE)
for spine in ax.spines.values():
    spine.set_color(WHITE)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(color=WHITE, alpha=0.12, linewidth=0.6)
leg = ax.legend(loc="upper right", fontsize=11, facecolor="#2D2D44", edgecolor=WHITE)
for txt in leg.get_texts():
    txt.set_color(WHITE)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/rnn_sine.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
