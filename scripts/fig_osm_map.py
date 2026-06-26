import warnings; warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

BASE_DIR  = Path('/home/jupyter-daniela/CAF')
OSM_PATH  = BASE_DIR / 'data/OSM/combined_caribbean_infrastructure.gpkg'
SHP_PATH  = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
OUT_PATH  = BASE_DIR / 'outputs/exposure/osm_map.png'

CATEGORY_STYLE = {
    '3_Mobility_Networks':       {'color': '#adb5bd', 'lw': 0.4, 'alpha': 0.5,  'zorder': 3, 'label': 'Mobility networks'},
    '4_Buildings':               {'color': '#4dabf7', 'lw': 0.0, 'alpha': 0.7,  'zorder': 4, 'label': 'Buildings'},
    '2_Basic_Social_Services':   {'color': '#f59f00', 'lw': 0.0, 'alpha': 0.9,  'zorder': 5, 'label': 'Basic social services'},
    '1_Critical_Infrastructure': {'color': '#f03e3e', 'lw': 0.0, 'alpha': 1.0,  'zorder': 6, 'label': 'Critical infrastructure'},
}

print("Loading OSM data …")
gdf = gpd.read_file(OSM_PATH, layer='combined_caribbean_infrastructure')
gdf = gdf.to_crs('EPSG:4326')
print(f"  {len(gdf):,} features loaded")

carib = gpd.read_file(SHP_PATH).to_crs('EPSG:4326')

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_facecolor('#d0e8f0')

carib.plot(ax=ax, color='#f0f0f0', edgecolor='#999999', linewidth=0.4, zorder=2)

for cat in ['3_Mobility_Networks', '4_Buildings', '2_Basic_Social_Services', '1_Critical_Infrastructure']:
    style = CATEGORY_STYLE[cat]
    subset = gdf[gdf['main_category'] == cat]
    if subset.empty:
        continue

    lines   = subset[subset.geometry.geom_type == 'LineString']
    polys   = subset[subset.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
    points  = subset[subset.geometry.geom_type == 'Point']

    if not lines.empty:
        lines.plot(ax=ax, color=style['color'], linewidth=style['lw'] or 0.4,
                   alpha=style['alpha'], zorder=style['zorder'])
    if not polys.empty:
        polys.plot(ax=ax, color=style['color'], edgecolor='none',
                   alpha=style['alpha'], zorder=style['zorder'])
    if not points.empty:
        points.plot(ax=ax, color=style['color'], markersize=3,
                    alpha=style['alpha'], zorder=style['zorder'])

ax.set_xlim(-85, -58)
ax.set_ylim(10, 28)
ax.set_xlabel('Longitude', fontsize=10)
ax.set_ylabel('Latitude', fontsize=10)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3, linewidth=0.4)

counts = gdf['main_category'].value_counts()
legend_handles = [
    mpatches.Patch(color=CATEGORY_STYLE[cat]['color'],
                   label=f"{CATEGORY_STYLE[cat]['label']}  ({counts.get(cat, 0):,})")
    for cat in CATEGORY_STYLE
]
ax.legend(handles=legend_handles, loc='lower left', fontsize=9,
          framealpha=0.95, edgecolor='#cccccc', title='OSM feature category', title_fontsize=9)

country_centroids = {
    'Dominican Republic': (-70.2, 18.9),
    'Cuba':               (-79.5, 22.0),
    'Haiti':              (-72.5, 18.9),
    'Jamaica':            (-77.3, 18.1),
    'Trinidad And Tobago':(-61.2, 10.5),
    'Saint Lucia':        (-60.98, 13.9),
    'Barbados':           (-59.55, 13.2),
    'Curaçao':            (-69.0, 12.2),
}
for country, (lon, lat) in country_centroids.items():
    ax.text(lon, lat, country, fontsize=7, color='#444444',
            ha='center', va='center', style='italic')

fig.text(0.5, 0.01, 'Source: OpenStreetMap contributors · CAF project',
         ha='center', fontsize=8, color='#888888')

fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"Saved → {OUT_PATH}")
