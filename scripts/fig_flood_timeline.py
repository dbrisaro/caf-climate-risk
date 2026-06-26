import pandas as pd, geopandas as gpd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

BASE = Path('/home/jupyter-daniela/CAF')
FONT = "Ubuntu"
C_HAITI = "#C0392B"
C_DR    = "#2980B9"

flood = pd.read_parquet(BASE / "data/processed/caribbean_adm_boundaries/micro_flood/flood_events.parquet")
flood["event_date"] = pd.to_datetime(flood["event_date"])
flood["year"] = flood["event_date"].dt.year

hd = flood[flood["country"].isin(["Haiti","Dominican Republic"])].copy()
hd = hd.drop_duplicates(subset=["country","event_date","total_losses"])

shp = gpd.read_file(BASE / "data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp")
rc  = shp[["region_nam","country"]].rename(columns={"region_nam":"region_name"})
df  = pd.read_parquet(BASE / "data/processed/caribbean_adm_boundaries/master_panel.parquet")
df  = df.merge(rc, on="region_name", how="left").drop_duplicates(subset=["country","year"])
panel = df[df["country"].isin(["Haiti","Dominican Republic"])][["country","year","total_losses_musd"]]

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
fig.patch.set_alpha(0)

for idx, (ax, country, color) in enumerate(zip(axes, ["Haiti","Dominican Republic"], [C_HAITI, C_DR])):
    ax.set_facecolor("none")

    losses = panel[panel["country"]==country].copy()
    losses = losses[losses["total_losses_musd"] > 0]

    ax.bar(losses["year"], losses["total_losses_musd"],
           color=color, alpha=0.75, width=0.8, zorder=3, label="Annual losses (MUSD)")

    events = hd[hd["country"]==country].copy()
    for _, row in events.iterrows():
        y = losses[losses["year"]==row["year"]]["total_losses_musd"].values
        yval = y[0] if len(y)>0 else 2
        dtype = row["disaster_sub"] if pd.notna(row["disaster_sub"]) else row["disaster_type"]
        marker = "^" if "cyclone" in dtype.lower() or "storm" in dtype.lower() else "o"
        ax.scatter(row["year"], yval + yval*0.08, marker=marker,
                  color="white", edgecolors=color, s=60, lw=1.5, zorder=6)

    peak = losses.nlargest(2, "total_losses_musd")
    for _, row in peak.iterrows():
        ax.text(row["year"], row["total_losses_musd"]*1.05,
                f"{row['year']}\n{row['total_losses_musd']:.0f}M",
                ha="center", va="bottom", fontsize=8, fontfamily=FONT,
                color=color, fontweight="bold")

    ax.set_xlim(1994, 2026)
    ax.set_ylabel("Losses (MUSD)", fontsize=9.5, fontfamily=FONT)
    ax.set_title(f"{chr(ord('a')+idx)}  {country}", loc="left", fontweight="normal", fontsize=11, color=color)
    ax.tick_params(labelsize=9)
    ax.set_ylim(0, losses["total_losses_musd"].max()*1.25)

    ax.set_xticks(range(1995, 2026, 5))

    cyc = mpatches.Patch(color=color, alpha=0.75, label="Recorded losses")
    from matplotlib.lines import Line2D
    ev_cyc = Line2D([0],[0], marker="^", color="w", markerfacecolor="white",
                    markeredgecolor=color, markersize=7, label="Cyclone / storm")
    ev_fl  = Line2D([0],[0], marker="o", color="w", markerfacecolor="white",
                    markeredgecolor=color, markersize=7, label="Flood")
    ax.legend(handles=[cyc, ev_cyc, ev_fl], fontsize=8, loc="upper left",
              framealpha=0.3, edgecolor="none")

axes[-1].set_xlabel("Year", fontsize=10, fontfamily=FONT)
plt.tight_layout(pad=1.5)
out = BASE / "outputs/cyclone/fig_flood_timeline.png"
plt.savefig(out, dpi=180, bbox_inches="tight", transparent=True)
print(f"Saved → {out}")
plt.close()
