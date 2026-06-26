import geopandas as gpd, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

BASE_DIR = Path('/home/jupyter-daniela/CAF')
FONT = "Ubuntu"

CATEGORIES = {
    "Health":           (["hospital","clinic","pharmacy"],        "#C0392B"),
    "Education":        (["school","university"],                 "#2980B9"),
    "Emergency":        (["police","fire_station"],               "#E67E22"),
    "Infrastructure":   (["bus_station","fuel_station","airport","heliport","charging_station"], "#27AE60"),
    "Government":       (["government"],                          "#8E44AD"),
    "Religious":        (["religious"],                           "#F39C12"),
    "Residential":      (["residential"],                         "#BDC3C7"),
    "Commercial":       (["commercial","industrial"],             "#95A5A6"),
}

def cat_color(service):
    for cat, (tags, color) in CATEGORIES.items():
        if service in tags:
            return cat, color
    return "Other", "#7F8C8D"

osm  = pd.read_parquet(BASE_DIR / "data/processed/caribbean_adm_boundaries/micro_flood/osm_with_elevation.parquet")
sdi  = pd.read_parquet(BASE_DIR / "data/processed/caribbean_adm_boundaries/micro_flood/service_disruption.parquet")
cities = pd.read_parquet(BASE_DIR / "data/processed/caribbean_adm_boundaries/micro_flood/study_cities.parquet")

osm["category"], osm["color"] = zip(*osm["service"].map(cat_color))

haiti = osm[osm["country"] == "Haiti"].copy()
gdf   = gpd.GeoDataFrame(haiti, geometry=gpd.points_from_xy(haiti.lon, haiti.lat), crs="EPSG:4326")

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios":[1.5,1]})
fig.patch.set_alpha(0)

ax = axes[0]
ax.set_facecolor("#C8DDF0")

shp = gpd.read_file(BASE_DIR / "data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp")
haiti_shp = shp[shp["country"] == "Haiti"]
haiti_shp.boundary.plot(ax=ax, color="#555555", lw=0.6, zorder=2)
haiti_shp.plot(ax=ax, color="#F2EFE9", alpha=0.9, zorder=1)

for cat, (tags, color) in CATEGORIES.items():
    sub = gdf[gdf["category"] == cat]
    if len(sub) == 0: continue
    size = 6 if cat in ["Residential","Commercial"] else 18
    alpha = 0.3 if cat in ["Residential","Commercial"] else 0.85
    zorder = 3 if cat in ["Residential","Commercial"] else 5
    ax.scatter(sub.geometry.x, sub.geometry.y,
               c=color, s=size, alpha=alpha, zorder=zorder,
               edgecolors="white" if size > 10 else "none", linewidth=0.4,
               label=cat)

ax.set_xlim(-73.1, -71.8)
ax.set_ylim(18.2, 19.0)
ax.set_title("a", loc="left", fontweight="normal", fontsize=10)
ax.set_xlabel("Longitude", fontsize=9, fontfamily=FONT)
ax.set_ylabel("Latitude", fontsize=9, fontfamily=FONT)
ax.tick_params(labelsize=8)

legend_elements = [
    mpatches.Patch(color=color, label=cat, alpha=0.85)
    for cat, (tags, color) in CATEGORIES.items()
]
ax.legend(handles=legend_elements, fontsize=8, title="Service type",
          title_fontsize=8.5, loc="lower left", framealpha=0.85,
          edgecolor="none")

city_coords = {
    "Port-au-Prince": (-72.338, 18.543),
    "Croix-des-Bouquets": (-72.187, 18.616),
}
for name, (lon, lat) in city_coords.items():
    ax.text(lon, lat+0.03, name, fontsize=8, fontfamily=FONT,
            color="#1A1A2E", ha="center", fontweight="bold", zorder=10)

ax2 = axes[1]
ax2.set_facecolor("none")

critical = ["hospital","clinic","pharmacy","school","police","fire_station","government"]
osm_crit = osm[osm["service"].isin(critical)]
counts = osm_crit.groupby(["region_name","service"]).size().unstack(fill_value=0).reset_index()

name_map = {
    "Arrondissement de Croix-des-Bouquets": "Croix-des-B. (Haiti)",
    "Port-au-Prince": "Port-au-Prince (Haiti)",
    "Santo Domingo Este": "Sto. Dom. Este (DR)",
    "Villa Altagracia": "Villa Altagracia (DR)",
    "San Pedro de Macorís": "San Pedro (DR)",
    "Saint Thomas": "Saint Thomas (Barbados)",
    "North Andros": "North Andros (Bahamas)",
}
counts["city"] = counts["region_name"].map(name_map)

cat_map = {
    "hospital":"Health","clinic":"Health","pharmacy":"Health",
    "school":"Education","police":"Emergency",
    "fire_station":"Emergency","government":"Government",
}
color_map = {c: CATEGORIES[c][1] for c in ["Health","Education","Emergency","Government"]}

city_order = [name_map.get(c,c) for c in cities["region_name"].tolist()]
counts = counts.set_index("city").reindex(city_order).fillna(0)

bottom = np.zeros(len(counts))
for cat, color in color_map.items():
    tags = [t for t,c in cat_map.items() if c==cat and t in counts.columns]
    vals = counts[tags].sum(axis=1).values if tags else np.zeros(len(counts))
    bars = ax2.barh(range(len(counts)), vals, left=bottom, color=color,
                    alpha=0.85, label=cat, height=0.6)
    bottom += vals

ax2.set_yticks(range(len(counts)))
ax2.set_yticklabels(counts.index, fontsize=8.5, fontfamily=FONT)
ax2.set_xlabel("Number of critical facilities", fontsize=9, fontfamily=FONT)
ax2.set_title("b", loc="left", fontweight="normal", fontsize=10)
ax2.legend(fontsize=8, title="Category", title_fontsize=8.5,
           prop={"family":FONT},
           loc="lower right", framealpha=0.3, edgecolor="none")
ax2.tick_params(labelsize=8)

for i, total in enumerate(bottom):
    ax2.text(total+5, i, f"{int(total)}", va="center", fontsize=8.5, fontfamily=FONT)

plt.tight_layout(pad=2)
out = BASE_DIR / "outputs/exposure/fig_osm_services.png"
plt.savefig(out, dpi=180, bbox_inches="tight", transparent=True)
print(f"Saved → {out}")
plt.close()
