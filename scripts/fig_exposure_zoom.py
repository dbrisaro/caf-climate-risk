import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

BASE_DIR = Path('/home/jupyter-daniela/CAF')
EP_PATH  = BASE_DIR / 'data/processed/caribbean_adm_boundaries/exposure_panel.parquet'
SHP_PATH = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
OUT_DIR  = BASE_DIR / 'outputs/exposure'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ep  = pd.read_parquet(EP_PATH)
shp = gpd.read_file(SHP_PATH)[['region_nam', 'country', 'geometry']].rename(
        columns={'region_nam': 'region_name'})
ep  = ep.merge(shp[['region_name', 'country']], on='region_name', how='left')

delta = (ep[ep['year'] == 2030][['region_name', 'exposure_index']]
         .rename(columns={'exposure_index': 'idx_2030'})
         .merge(ep[ep['year'] == 1995][['region_name', 'exposure_index']]
                .rename(columns={'exposure_index': 'idx_1995'}),
                on='region_name'))
delta['change'] = delta['idx_2030'] - delta['idx_1995']

gdf = shp.merge(delta[['region_name', 'change']], on='region_name', how='left')
gdf = gdf.to_crs('EPSG:4326')

FOCUS  = ['Jamaica', 'Haiti', 'Dominican Republic']
XMIN, XMAX = -79.5, -67.5
YMIN, YMAX =  16.5,  21.0

BOUNDS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
N      = len(BOUNDS) - 1
cmap   = cm.get_cmap('YlOrRd', N)
norm   = mcolors.BoundaryNorm(BOUNDS, ncolors=N, clip=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor('#cfe2f3')

gdf_other = gdf[~gdf['country'].isin(FOCUS)]
if not gdf_other.empty:
    gdf_other.plot(ax=ax, color='#e0e0e0', edgecolor='#aaaaaa', linewidth=0.2)

gdf_focus = gdf[gdf['country'].isin(FOCUS)]
gdf_focus.plot(
    ax=ax,
    column='change',
    cmap=cmap,
    norm=norm,
    edgecolor='#555555',
    linewidth=0.3,
    missing_kwds={'color': '#cccccc', 'label': 'No data'},
)

for country in FOCUS:
    sub = gdf_focus[gdf_focus['country'] == country]
    if not sub.empty:
        sub.dissolve().boundary.plot(ax=ax, color='#222222', linewidth=0.8)

LABEL_ANNOTATIONS = {
    'Jamaica':            ((-78.8, 17.1),  (-77.5, 18.05)),
    'Haiti':              ((-74.5, 20.75), (-73.2, 19.4)),
    'Dominican Republic': ((-69.2, 20.75), (-70.5, 19.6)),
}
for country, (text_xy, arrow_xy) in LABEL_ANNOTATIONS.items():
    ax.annotate(
        country,
        xy=arrow_xy,
        xytext=text_xy,
        fontsize=9, ha='center', va='center',
        color='#111111',
        arrowprops=dict(arrowstyle='->', color='#333333',
                        lw=1.0, connectionstyle='arc3,rad=0.0'),
    )

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, aspect=18,
                  boundaries=BOUNDS, ticks=BOUNDS, spacing='uniform',
                  extend='max')
cb.set_label('Δ Exposure index (2030 − 1995)', fontsize=9)
cb.ax.tick_params(labelsize=8)
cb.ax.set_yticklabels([f'{b:.2f}' for b in BOUNDS])

ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
ax.set_xlabel('Longitude', fontsize=9)
ax.set_ylabel('Latitude', fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.25, linewidth=0.3)

fig.tight_layout()
out_path = OUT_DIR / 'map_change_zoom_hispaniola.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out_path}")
