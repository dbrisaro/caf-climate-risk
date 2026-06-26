#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE         = Path('/home/jupyter-daniela/CAF')
DROUGHT_PATH = BASE / 'data/processed/caribbean_adm_boundaries/era5_output/droughts_aggregated_data.parquet'
SHP_PATH     = BASE / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
NE_10M       = '/home/jupyter-daniela/.local/share/cartopy/shapefiles/natural_earth/cultural/ne_10m_admin_0_countries.shp'
OUT_DIR      = BASE / 'outputs/hazards'
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR, SEASON = 2015, 'JJA'

INSULAR     = {'Dominican Republic', 'Haiti', 'Jamaica',
               'Bahamas', 'Trinidad and Tobago', 'Barbados', 'Saint Lucia', 'Dominica'}
CONTINENTAL = {'Colombia', 'Peru', 'Guyana'}

XLIM_INS = (-86.5, -58.5)
YLIM_INS = (9.0,   27.5)

XLIM_CON = (-83.5, -55.0)
YLIM_CON = (-20.0,  17.5)

CAT_ORDER  = ['No drought', 'Moderate', 'Severe', 'Extreme']
CAT_COLORS = {
    'No drought': '#F4F4F4',
    'Moderate':   '#5BA85A',
    'Severe':     '#F5C842',
    'Extreme':    '#C0392B',
}
WORLD_COLOR = '#C8C8C8'

ANNOTATIONS = {
    'Jamaica':            ('Jamaica',             -83.5, 16.5,  -77.5, 18.2),
    'Haiti':              ('Haiti',               -76.5, 25.0,  -73.4, 19.1),
    'Dominican Republic': ('Dominican\nRepublic', -67.5, 25.0,  -70.0, 19.0),
    'Bahamas':            ('Bahamas',             -77.5, 26.5,  -77.0, 24.5),
    'Trinidad and Tobago':('Trinidad\n& Tobago',  -59.5, 11.5,  -61.3, 10.6),
    'Barbados':           ('Barbados',            -59.5, 14.5,  -59.5, 13.1),
    'Saint Lucia':        ('Saint Lucia',         -59.5, 16.0,  -61.0, 14.0),
    'Dominica':           ('Dominica',            -59.5, 18.5,  -61.3, 15.4),
    'Colombia':           ('Colombia',            -81.5, 13.5,  -74.5,  7.5),
    'Peru':               ('Peru',                -81.5,-14.0,  -76.5,-10.5),
    'Guyana':             ('Guyana',              -56.2,  9.0,  -59.5,  6.0),
}

ARROW_STYLE = dict(arrowstyle='-', color='#111111', lw=0.65, shrinkA=0, shrinkB=3)

df     = pd.read_parquet(DROUGHT_PATH)
jja_yr = df[(df['window'] == SEASON) & (df['year'] == YEAR)]

piv = (jja_yr
       .pivot_table(index='region_name', columns='activacion',
                    values='n_drought_pixels', fill_value=0)
       .reindex(columns=['p1', 'p5', 'p10'], fill_value=0))

def worst_level(row):
    if row['p1']  > 0: return 'Extreme'
    if row['p5']  > 0: return 'Severe'
    if row['p10'] > 0: return 'Moderate'
    return 'No drought'

piv['category'] = piv.apply(worst_level, axis=1)

gdf = gpd.read_file(SHP_PATH).rename(columns={'region_nam': 'region_name'})
if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.set_crs('EPSG:4326') if gdf.crs is None else gdf.to_crs('EPSG:4326')
gdf = gdf.merge(piv[['category']].reset_index(), on='region_name', how='left')
gdf['category'] = gdf['category'].fillna('No drought')

gdf_ins = gdf[gdf['country'].isin(INSULAR)]
gdf_con = gdf[gdf['country'].isin(CONTINENTAL)]

world = gpd.read_file(NE_10M)
world = world[world['CONTINENT'] != 'Antarctica']

fig, (ax_ins, ax_con) = plt.subplots(
    2, 1, figsize=(6.5, 8.5),
    gridspec_kw={'height_ratios': [38, 37], 'hspace': 0.06},
)

for ax, gdf_row, xlim, ylim, panel_lbl, countries in [
    (ax_ins, gdf_ins, XLIM_INS, YLIM_INS, 'a)', INSULAR),
    (ax_con, gdf_con, XLIM_CON, YLIM_CON, 'b)', CONTINENTAL),
]:
    world.plot(ax=ax, color=WORLD_COLOR, edgecolor='#AAAAAA', linewidth=0.3)

    for cat in CAT_ORDER:
        sub = gdf_row[gdf_row['category'] == cat]
        if len(sub):
            sub.plot(ax=ax, color=CAT_COLORS[cat], linewidth=0.2, edgecolor='#555555')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.02, 0.97, panel_lbl, transform=ax.transAxes,
            fontsize=9, va='top', ha='left', color='#111111')

    for country in countries:
        label, tx, ty, ax_, ay_ = ANNOTATIONS[country]
        ax.annotate(
            label,
            xy=(ax_, ay_), xytext=(tx, ty),
            fontsize=7, ha='center', va='center',
            color='#111111', linespacing=1.3,
            arrowprops=ARROW_STYLE,
        )

legend_handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=CAT_COLORS[cat], markeredgecolor='#555555',
           markeredgewidth=0.5, markersize=8, label=cat)
    for cat in CAT_ORDER
]
fig.legend(handles=legend_handles, loc='lower center', ncol=4,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.01),
           handletextpad=0.4, columnspacing=1.2)

plt.subplots_adjust(bottom=0.07)

out_path = OUT_DIR / 'fig_1_5_drought_pixels.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.close()
