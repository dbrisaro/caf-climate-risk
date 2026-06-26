#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
import style
style.apply()

BASE    = Path('/home/jupyter-daniela/CAF')
EP_PATH = BASE / 'data/processed/caribbean_adm_boundaries/exposure_panel.parquet'
SHP     = BASE / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
OUTDIR  = BASE / 'outputs/exposure'
OUTDIR.mkdir(parents=True, exist_ok=True)

YEARS = [1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]

INSULAR = [
    'Bahamas', 'Barbados', 'Dominica', 'Dominican Republic',
    'Haiti', 'Jamaica', 'Saint Lucia', 'Trinidad and Tobago',
]
CONTINENTAL = ['Colombia', 'Guyana', 'Peru']
ALL_C = INSULAR + CONTINENTAL

COLORS = {
    'Dominican Republic':  '#27AE60',
    'Peru':                '#B7950B',
    'Colombia':            '#784212',
    'Haiti':               '#E07B30',
    'Jamaica':             '#2980B9',
    'Barbados':            '#C0392B',
    'Trinidad and Tobago': '#8E44AD',
    'Saint Lucia':         '#16A085',
    'Dominica':            '#7F8C8D',
    'Bahamas':             '#BDC3C7',
    'Guyana':              '#1A5276',
}
MARKERS = {
    'Dominican Republic': 'P', 'Peru': 'o', 'Colombia': 's',
    'Haiti': 'v', 'Jamaica': '^', 'Barbados': 'o',
    'Trinidad and Tobago': 's', 'Saint Lucia': 'D',
    'Dominica': 'X', 'Bahamas': '*', 'Guyana': '^',
}
LS = {c: '-' for c in INSULAR}
LS.update({c: '--' for c in CONTINENTAL})
SHORT = {
    'Dominican Republic':  'Dom. Republic',
    'Trinidad and Tobago': 'Trinidad & Tobago',
    'Saint Lucia':         'Saint Lucia',
    'Bahamas':             'Bahamas',   'Barbados':  'Barbados',
    'Dominica':            'Dominica',  'Haiti':     'Haiti',
    'Jamaica':             'Jamaica',   'Peru':      'Peru',
    'Colombia':            'Colombia',  'Guyana':    'Guyana',
}

ep  = pd.read_parquet(EP_PATH)
gdf = gpd.read_file(SHP)[['region_nam', 'country']].rename(
        columns={'region_nam': 'region_name'})
ep  = ep.merge(gdf, on='region_name', how='left')
ep  = ep[ep['country'].isin(ALL_C) & ep['year'].isin(YEARS)]

lo, hi = np.percentile(ep['exposure_index'].dropna(), [2, 98])
ep['pca_idx'] = np.clip((ep['exposure_index'] - lo) / (hi - lo), 0, 1)

def wmean(g):
    w = g['pop'].fillna(0).clip(lower=0)
    return np.average(g['pca_idx'].fillna(0), weights=w) if w.sum() > 0 else g['pca_idx'].mean()

nat = (ep.groupby(['country', 'year'])
         .apply(wmean)
         .reset_index(name='ei'))

base = nat[nat['year'] == 1995].set_index('country')['ei']
nat['delta'] = nat.apply(lambda r: r['ei'] - base.get(r['country'], np.nan), axis=1)

val2030 = nat[nat['year'] == 2030].set_index('country')['delta']
ordered = val2030.reindex(ALL_C).dropna().sort_values(ascending=False).index.tolist()

pivot = nat.pivot(index='country', columns='year', values='delta').round(3)
print("Δ exposure index from 1995 baseline:\n")
print(pivot.reindex(ordered).to_string())

fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor='white')

PLOT_KW = dict(markersize=4.5, linewidth=1.2,
               markeredgewidth=0.4, markeredgecolor='white', zorder=3)

for country in ordered:
    cdf = nat[nat['country'] == country].set_index('year').reindex(YEARS)
    ax.plot(YEARS, cdf['delta'],
            color=COLORS.get(country, '#999999'),
            marker=MARKERS.get(country, 'o'),
            linestyle=LS.get(country, '-'),
            **PLOT_KW)

ax.axhline(0, color='#BBBBBB', linewidth=0.6, linestyle='--', zorder=1)

ax.set_xlim(1993, 2032)
ax.set_ylim(bottom=-0.005)
ax.set_xticks(YEARS)
ax.set_xticklabels([str(y) for y in YEARS], fontsize=6.5, rotation=30, ha='right')
ax.set_ylabel('Δ Exposure index (from 1995)', fontsize=7, labelpad=4)
ax.tick_params(labelsize=6.5, length=3, width=0.5, pad=2)

ax.yaxis.grid(True, linewidth=0.35, color='#EEEEEE', zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

handles = [
    mlines.Line2D([], [],
                  color=COLORS.get(c, '#999'),
                  marker=MARKERS.get(c, 'o'),
                  linestyle=LS.get(c, '-'),
                  markersize=4, markeredgewidth=0.3, markeredgecolor='white',
                  linewidth=1.1, label=SHORT.get(c, c))
    for c in ordered
]
ax.legend(handles=handles, fontsize=6, frameon=False,
          loc='upper left', handlelength=1.8, handletextpad=0.4,
          labelspacing=0.3, ncol=2, columnspacing=1.0)

fig.tight_layout(pad=0.5)

out_path = OUTDIR / 'fig_1_10_exposure_temporal.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved → {out_path}")
