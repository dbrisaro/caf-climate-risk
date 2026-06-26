import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path('/home/jupyter-daniela/CAF')
EP_PATH  = BASE_DIR / 'data/processed/caribbean_adm_boundaries/exposure_panel.parquet'
SHP_PATH = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
OUT_PATH = BASE_DIR / 'outputs/exposure/temporal_evolution_by_country.png'

EPOCHS = [1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]

ep  = pd.read_parquet(EP_PATH)
shp = gpd.read_file(SHP_PATH)[['region_nam', 'country']].rename(
        columns={'region_nam': 'region_name'})
ep  = ep.merge(shp, on='region_name', how='left')

def wmean(df, val_col, weight_col='pop'):
    w = df[weight_col].fillna(0)
    if w.sum() == 0:
        return df[val_col].mean()
    return np.average(df[val_col].fillna(0), weights=w)

nat = (ep.groupby(['country', 'year'])
        .apply(lambda g: wmean(g, 'exposure_index'))
        .reset_index(name='exposure_index')
        .sort_values('year'))

countries = sorted(nat['country'].dropna().unique())

cmap = plt.cm.tab10
colors    = {c: cmap(i / max(len(countries) - 1, 1)) for i, c in enumerate(countries)}
markers   = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
marker_map = {c: markers[i % len(markers)] for i, c in enumerate(countries)}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

fig, ax = plt.subplots(figsize=(10, 6))

for country in countries:
    cdf = nat[nat['country'] == country].set_index('year').reindex(EPOCHS)
    ax.plot(EPOCHS, cdf['exposure_index'],
            color=colors[country],
            marker=marker_map[country],
            markersize=6, linewidth=1.8,
            label=country)

ax.set_xticks(EPOCHS)
ax.set_xticklabels(EPOCHS, rotation=30, ha='right', fontsize=9)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Mean national exposure index', fontsize=11)
ax.legend(fontsize=9, loc='upper left', framealpha=0.95,
          title='Country', title_fontsize=9)
ax.grid(True, alpha=0.3, linewidth=0.4)

fig.text(0.5, -0.02,
         'Source: GHSL 2023, own elaboration. CAF project.',
         ha='center', fontsize=8, color='#888888')

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {OUT_PATH}")
