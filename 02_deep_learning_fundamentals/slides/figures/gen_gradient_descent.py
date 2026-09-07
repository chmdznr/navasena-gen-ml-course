"""
Generate figures/gradient_descent.pdf — gradient descent 1D pada
f(w) = (w-3)^2 + 1, tiga panel learning rate (0.05 lambat, 0.5 pas,
1.05 meledak). Titik-titik langkah diberi nomor.
Run dari direktori slides/: python3 figures/gen_gradient_descent.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#1A1A2E"
TEXT = "white"
GRID = "#333355"
CURVE_COLOR = "#42A5F5"
POINT_COLOR = "#76B900"
NUM_COLOR = "#FFCA28"


def f(w):
    return (w - 3) ** 2 + 1


def grad(w):
    return 2 * (w - 3)


def run_gd(lr, w0, n_steps):
    ws = [w0]
    w = w0
    for _ in range(n_steps):
        w = w - lr * grad(w)
        ws.append(w)
        if abs(w) > 1e4:  # meledak — hentikan lebih awal
            break
    return np.array(ws)


np.random.seed(42)
w0 = -2.0

configs = [
    ("learning rate = 0.05 (terlalu kecil)", 0.05, 12),
    ("learning rate = 0.5 (pas)", 0.5, 8),
    ("learning rate = 1.05 (terlalu besar, meledak)", 1.05, 15),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
fig.patch.set_facecolor(BG)

w_curve = np.linspace(-6, 10, 300)

for ax, (title, lr, n_steps) in zip(axes, configs):
    ax.set_facecolor(BG)
    ws = run_gd(lr, w0, n_steps)
    fs = f(ws)

    # sesuaikan rentang kurva agar titik-titik yang meledak masih terlihat
    w_lo, w_hi = min(w_curve.min(), ws.min()) - 1, max(w_curve.max(), ws.max()) + 1
    wc = np.linspace(w_lo, w_hi, 400)
    ax.plot(wc, f(wc), color=CURVE_COLOR, linewidth=2.0, zorder=2)
    ax.plot(ws, fs, color=POINT_COLOR, linewidth=1.2, linestyle="--",
             marker="o", markersize=6, zorder=3)
    for i, (wi, fi) in enumerate(zip(ws, fs)):
        ax.annotate(str(i), (wi, fi), textcoords="offset points",
                    xytext=(0, 8), color=NUM_COLOR, fontsize=8, fontweight="bold",
                    ha="center")
    ax.axvline(3, color="#888899", linewidth=0.8, linestyle=":")
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold")
    ax.set_xlabel("w", color=TEXT, fontsize=10)
    ax.set_ylabel("f(w)", color=TEXT, fontsize=10)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, color=GRID, linestyle="--", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    # batasi tampilan y agar kasus meledak tetap terbaca
    y_cap = min(fs.max(), 500) * 1.15 if fs.max() > 0 else 50
    ax.set_ylim(0, max(y_cap, 20))

fig.suptitle("Gradient Descent pada f(w) = (w-3)² + 1", color=TEXT,
             fontsize=15, fontweight="bold", y=1.0)
plt.tight_layout(pad=1.0)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradient_descent.pdf")
fig.savefig(output_path, format="pdf", bbox_inches="tight", facecolor=BG)
print(f"Saved: {output_path}")
for (title, lr, n_steps) in configs:
    ws = run_gd(lr, w0, n_steps)
    print(f"lr={lr}: langkah={len(ws)-1} w_akhir={ws[-1]:.3f} f(w_akhir)={f(ws[-1]):.3f}")
