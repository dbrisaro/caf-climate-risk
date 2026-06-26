import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

BASE_DIR = Path('/home/jupyter-daniela/CAF')
shp = gpd.read_file(BASE_DIR / "data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp")
region_country = shp[["region_nam","country"]].rename(columns={"region_nam":"region_name"})

df = pd.read_parquet(BASE_DIR / "data/processed/caribbean_adm_boundaries/master_panel.parquet")
df = df.merge(region_country, on="region_name", how="left")

cross = df.groupby("country").agg(
    total_losses=("total_losses_musd","sum"),
    exposure_index=("exposure_index","mean"),
    rainfall=("rainfall","mean"),
    cyclone=("cyclone","mean"),
    drought=("drought","mean"),
).reset_index()

cross["hazard_score"] = (cross["rainfall"] + cross["cyclone"]) / 2 * 1000
cross["exposure_norm"] = cross["exposure_index"] * 100
cross["log_losses"] = np.log10(cross["total_losses"])

FONT = "Ubuntu"
COLORS = {
    "Haiti": "#C0392B",
    "Dominican Republic": "#E67E22",
    "Jamaica": "#2980B9",
    "Dominica": "#8E44AD",
    "Bahamas": "#16A085",
    "Barbados": "#27AE60",
    "Saint Lucia": "#F39C12",
    "Trinidad and Tobago": "#7F8C8D",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_alpha(0)

COUNTRY_LABELS = {
    "Dominican Republic": "Dom. Republic",
    "Trinidad and Tobago": "Trinidad",
    "Saint Lucia": "St. Lucia",
}

ax = axes[0]
ax.set_facecolor("none")

for _, row in cross.iterrows():
    c = row["country"]
    color = COLORS.get(c, "#333")
    size = row["exposure_norm"] * 80 + 100
    ax.scatter(row["hazard_score"], row["total_losses"] / 1000,
               s=size, color=color, alpha=0.85, edgecolors="white", linewidth=1.2, zorder=5)
    label = COUNTRY_LABELS.get(c, c)
    offset_x = 0.15
    offset_y = row["total_losses"] / 1000 * 0.04
    ax.text(row["hazard_score"] + offset_x, row["total_losses"] / 1000 + offset_y,
            label, fontsize=8.5, fontfamily=FONT, color="#1A1A2E", va="bottom", zorder=6)

x = cross["hazard_score"].values
y = cross["total_losses"].values / 1000
slope, intercept, r, p, _ = stats.linregress(x, y)
xfit = np.linspace(x.min(), x.max(), 100)
ax.plot(xfit, slope * xfit + intercept, "--", color="#555", lw=1.5, alpha=0.6, zorder=4)
ax.text(0.97, 0.05, f"r = {r:.2f}", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9, color="#555", fontfamily=FONT)

ax.set_xlabel("Hazard intensity (rainfall + cyclone, scaled)", fontsize=10, fontfamily=FONT, labelpad=6)
ax.set_ylabel("Total cumulative losses (billion USD)", fontsize=10, fontfamily=FONT, labelpad=6)
ax.set_title('a', loc='left', fontweight='normal', fontsize=10)

for val, label in [(0.04,"low exposure"), (0.1,"high exposure")]:
    ax.scatter([], [], s=val*100*80+100, color="#AAA", alpha=0.7, label=label, edgecolors="white")
ax.legend(title="Exposure level", fontsize=8, title_fontsize=8.5, loc="upper left",
          framealpha=0.3, edgecolor="none")

ax2 = axes[1]
ax2.set_facecolor("none")

hmed = cross["hazard_score"].median()
emed = cross["exposure_norm"].median()

ax2.axvline(hmed, color="#CCC", lw=1, ls="--", zorder=1)
ax2.axhline(emed, color="#CCC", lw=1, ls="--", zorder=1)

ax2.text(hmed * 0.3,  emed * 1.6, "Low hazard\nHigh exposure", fontsize=8, color="#AAA",
         ha="center", va="center", fontfamily=FONT)
ax2.text(hmed * 1.6,  emed * 1.6, "HIGH RISK\n(Hazard + Exposure)", fontsize=8.5, color="#C0392B",
         ha="center", va="center", fontfamily=FONT, fontweight="bold")
ax2.text(hmed * 0.3,  emed * 0.4, "Low risk", fontsize=8, color="#AAA",
         ha="center", va="center", fontfamily=FONT)
ax2.text(hmed * 1.6,  emed * 0.4, "High hazard\nLow exposure", fontsize=8, color="#AAA",
         ha="center", va="center", fontfamily=FONT)

for _, row in cross.iterrows():
    c = row["country"]
    color = COLORS.get(c, "#333")
    size = (row["total_losses"] / cross["total_losses"].max()) * 1500 + 80
    ax2.scatter(row["hazard_score"], row["exposure_norm"],
                s=size, color=color, alpha=0.85, edgecolors="white", linewidth=1.2, zorder=5)
    label = COUNTRY_LABELS.get(c, c)
    ax2.text(row["hazard_score"] + 0.1, row["exposure_norm"] + 0.15,
             label, fontsize=8.5, fontfamily=FONT, color="#1A1A2E", va="bottom", zorder=6)

ax2.set_xlabel("Hazard intensity (rainfall + cyclone, scaled)", fontsize=10, fontfamily=FONT, labelpad=6)
ax2.set_ylabel("Exposure index (mean, scaled)", fontsize=10, fontfamily=FONT, labelpad=6)
ax2.set_title('b', loc='left', fontweight='normal', fontsize=10)

plt.tight_layout(pad=2)
out = BASE_DIR / "outputs/regression/fig_regression_comparison.png"
plt.savefig(out, dpi=180, bbox_inches="tight", transparent=True)
print(f"Saved → {out}")
plt.close()
