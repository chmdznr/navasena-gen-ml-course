"""
Generate figures/rnn_sine.pdf — SimpleRNN(32) dilatih 20 epoch untuk
memprediksi gelombang sinus dari window 50 timestep (sama dengan nb04). Plot aktual vs
prediksi pada data test, MAE tercetak di judul.
Run dari direktori slides/: python3 figures/gen_rnn_sine.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18,
                     'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16})
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

tf.random.set_seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)
# angka yang dikutip di slide harus sama tiap kali figur diregenerasi
tf.config.experimental.enable_op_determinism()

BG = "#1A1A2E"
GREEN = "#76B900"
ORANGE = "#FF6F00"
TEXT = "#FFFFFF"

WINDOW = 50
t = np.linspace(0, 40 * np.pi, 2000)
series = np.sin(t) + np.random.normal(0, 0.02, size=t.shape)

X, y = [], []
for i in range(len(series) - WINDOW):
    X.append(series[i:i + WINDOW])
    y.append(series[i + WINDOW])
X = np.array(X)[..., np.newaxis]
y = np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = tf.keras.Sequential([
    tf.keras.Input(shape=(WINDOW, 1)),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)

y_pred = model.predict(X_test, verbose=0).flatten()
mae = mean_absolute_error(y_test, y_pred)

t_test = np.arange(len(y_test))

fig, ax = plt.subplots(figsize=(11, 3.9))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(t_test, y_test, color=GREEN, linewidth=2.0, label="Aktual")
ax.plot(t_test, y_pred, color=ORANGE, linewidth=1.4, linestyle="--", label="Prediksi RNN")
ax.set_title(f"Prediksi Gelombang Sinus dengan SimpleRNN(32) — MAE = {mae:.4f}",
             color=TEXT, fontweight="bold", pad=10)
ax.set_xlabel("timestep (data test)", color=TEXT)
ax.set_ylabel("nilai", color=TEXT)
ax.tick_params(colors=TEXT)
for spine in ax.spines.values():
    spine.set_color(TEXT)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(color=TEXT, alpha=0.12, linewidth=0.6)
leg = ax.legend(loc="upper right", facecolor="#2D2D44", edgecolor=TEXT)
for txt in leg.get_texts():
    txt.set_color(TEXT)
fig.tight_layout()

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rnn_sine.pdf")
fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {output_path}")
print(f"MAE={mae:.4f}")
