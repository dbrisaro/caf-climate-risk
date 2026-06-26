#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from pathlib import Path

BASE      = Path('/home/jupyter-daniela/CAF')
RAIN_PATH = BASE / 'data/processed/caribbean_adm_boundaries/era5_output_precipitation/rainfall_aggregated_data.parquet'
SHP_PATH  = BASE / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
OUT_DIR   = BASE / 'outputs/hazards'
OUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY    = list(range(1995, 2026))
BASELINE = list(range(1995, 2010))

INSULAR     = ['Bahamas', 'Barbados', 'Dominica', 'Dominican Republic',
               'Haiti', 'Jamaica', 'Saint Lucia', 'Trinidad and Tobago']
CONTINENTAL = ['Colombia', 'Peru', 'Guyana']
ALL_COUNTRIES = INSULAR + CONTINENTAL

df  = pd.read_parquet(RAIN_PATH)
gdf = gpd.read_file(SHP_PATH)
df['country'] = df['region_name'].map(gdf.set_index('region_nam')['country'].to_dict())

n_windows = df['window'].nunique()
total_px_country = (df[df['year'].isin(BASELINE)]
                    .groupby(['country', 'year'])['n_rainfall_pixels']
                    .sum()
                    .groupby('country').mean()
                    / 0.10
                    / n_windows)

study = df[df['year'].isin(STUDY)]
p90   = (study[study['activacion'].isin(['p90', 'p95', 'p99'])]
         .groupby(['country', 'year', 'window'])['n_rainfall_pixels']
         .sum()
         .reset_index())
p90['frac'] = (p90['n_rainfall_pixels'] / p90['country'].map(total_px_country)).clip(upper=1)

heat = (p90.groupby(['country', 'year'])['frac']
           .max()
           .unstack(fill_value=0)
           .reindex(index=ALL_COUNTRIES, columns=STUDY, fill_value=0))

n_activated = (heat >= 0.10).sum(axis=0)

bounds      = [i / 10 for i in range(11)]
n_bins      = len(bounds) - 1
colors_list = [(0.97, 0.97, 0.97)] + [plt.cm.YlGnBu(i / (n_bins - 2)) for i in range(n_bins - 1)]
cmap        = ListedColormap(colors_list)
norm        = BoundaryNorm(bounds, ncolors=n_bins, clip=True)

years = list(STUDY)
x     = np.arange(len(years))

fig, (ax_bar, ax_heat) = plt.subplots(
    2, 1, figsize=(7.5, 5.2),
    gridspec_kw={'height_ratios': [1, 3.5], 'hspace': 0.08},
)

bar_colors = ['#C0392B' if n_activated[y] >= 7
              else '#E07B30' if n_activated[y] >= 4
              else '#5B9BD5'
              for y in years]
ax_bar.bar(x, n_activated.values, color=bar_colors, width=0.65, linewidth=0)
ax_bar.set_xlim(-0.5, len(years) - 0.5)
ax_bar.set_ylim(0, len(ALL_COUNTRIES) + 0.5)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([])
ax_bar.set_yticks([0, 5, 11])
ax_bar.set_yticklabels(['0', '5', '11'], fontsize=7)
ax_bar.set_ylabel('Countries\nactivated', fontsize=7, linespacing=1.3)
ax_bar.tick_params(axis='y', length=3, width=0.6)
ax_bar.tick_params(axis='x', length=0)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.spines['bottom'].set_visible(False)
ax_bar.yaxis.grid(True, linewidth=0.4, color='#EEEEEE', zorder=0)
ax_bar.set_axisbelow(True)

data = heat.values.astype(float)
im   = ax_heat.imshow(data, cmap=cmap, norm=norm, aspect='auto')

for i, country in enumerate(ALL_COUNTRIES):
    for j in range(len(years)):
        val = data[i, j]
        if norm(val) > 0:
            txt_color = 'white' if val > 0.5 else '#333333'
            ax_heat.text(j, i, f'{int(val * 100)}', ha='center', va='center',
                         fontsize=5.5, color=txt_color)

ax_heat.set_xticks(x)
ax_heat.set_xticklabels(
    [str(y) if y % 5 == 0 else '' for y in years],
    fontsize=7.5, rotation=45, ha='right'
)
ax_heat.set_yticks(range(len(ALL_COUNTRIES)))
ax_heat.set_yticklabels(ALL_COUNTRIES, fontsize=7.5)
ax_heat.tick_params(length=0)
ax_heat.axhline(len(INSULAR) - 0.5, color='#888888', linewidth=0.8, linestyle='--')
for spine in ax_heat.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(im, ax=ax_heat, orientation='horizontal',
                    fraction=0.03, pad=0.18, shrink=0.6, aspect=30,
                    ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.set_label('Max. fraction of area with extreme rainfall (≥ p90)', fontsize=7.5)
cbar.outline.set_visible(False)
cbar.ax.tick_params(size=3, width=0.6, labelsize=7)
cbar.ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

out_path = OUT_DIR / 'fig_1_6_rainfall_heatmap.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.close()
