"""
Generate timeseries.pdf - deret + trend (rolling 12) & pola musiman rata-rata bulanan.
Run from slides/ directory: python3 figures/gen_timeseries.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "timeseries.pdf")

df = pd.read_csv(os.path.join(SCRIPT_DIR, "..", "..", "SeaPlaneTravel.csv"))
df.columns = df.columns.str.strip()
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").reset_index(drop=True)
passengers = df["#Passengers"]

trend = passengers.rolling(window=12, center=True).mean()

monthly_avg = passengers.groupby(df["Month"].dt.month).mean()
month_names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
               "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

# ── Plot ─────────────────────────────────────────────────────────────────
BG_COLOR = "#1A1A2E"
TEXT_COLOR = "white"
SERIES_COLOR = "#A3D944"
TREND_COLOR = "#76B900"
SEASON_COLOR = "#76B900"

fig, axes = plt.subplots(2, 1, figsize=(8, 6.5))
fig.patch.set_facecolor(BG_COLOR)

ax0 = axes[0]
ax0.set_facecolor(BG_COLOR)
ax0.plot(df["Month"], passengers, color=SERIES_COLOR, linewidth=1.2, label="Data bulanan")
ax0.plot(df["Month"], trend, color=TREND_COLOR, linewidth=2.5, label="Trend (rolling 12 bulan)")
ax0.set_ylabel("Penumpang (ribu)", color=TEXT_COLOR, fontsize=10)
ax0.set_title("Deret Waktu Penumpang & Trend", color=TEXT_COLOR, fontsize=11, fontweight="bold")
ax0.tick_params(colors=TEXT_COLOR, labelsize=8)
for spine in ax0.spines.values():
    spine.set_edgecolor("#444466")
legend0 = ax0.legend(loc="upper left", framealpha=0.25, facecolor="#2A2A4E",
                      edgecolor="#555577", labelcolor=TEXT_COLOR, fontsize=9)

ax1 = axes[1]
ax1.set_facecolor(BG_COLOR)
ax1.bar(month_names, monthly_avg.values, color=SEASON_COLOR, zorder=3)
ax1.set_xlabel("Bulan", color=TEXT_COLOR, fontsize=10)
ax1.set_ylabel("Rata-rata penumpang", color=TEXT_COLOR, fontsize=10)
ax1.set_title("Pola Musiman (rata-rata per bulan)", color=TEXT_COLOR, fontsize=11, fontweight="bold")
ax1.tick_params(colors=TEXT_COLOR, labelsize=8)
for spine in ax1.spines.values():
    spine.set_edgecolor("#444466")
ax1.yaxis.grid(True, color="#333355", linestyle="--", linewidth=0.6, zorder=0)
ax1.set_axisbelow(True)

plt.tight_layout(pad=0.8)
fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved: {OUTPUT_PATH}")
print(f"Monthly averages:\n{monthly_avg}")
