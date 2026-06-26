#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE     = Path('/home/jupyter-daniela/CAF')
HW_PATH  = BASE / 'data/processed/caribbean_adm_boundaries/heatwaves_computation/heatwaves_aggregated_data.parquet'
SHP_PATH = BASE / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
OUT_DIR  = BASE / 'outputs/hazards'
OUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY    = list(range(1995, 2026))
BASELINE = list(range(1995, 2010))

INSULAR     = ['Jamaica', 'Dominican Republic', 'Haiti',
               'Bahamas', 'Trinidad and Tobago', 'Barbados', 'Saint Lucia', 'Dominica']
CONTINENTAL = ['Colombia', 'Peru', 'Guyana']
ALL_COUNTRIES = INSULAR + CONTINENTAL

NO_DATA = {'Barbados', 'Dominica'}

COLOR_INS = '#2E86C1'
COLOR_CON = '#E07B30'

df  = pd.read_parquet(HW_PATH)
gdf = gpd.read_file(SHP_PATH)
df['country'] = df['region_name'].map(gdf.set_index('region_nam')['country'].to_dict())
df = df.rename(columns={'year_win': 'year'})

n_windows = df['window'].nunique()
total_country = (df[df['year'].isin(BASELINE)]
                 .groupby(['country', 'year'])['n_event_days'].sum()
                 .groupby('country').mean()
                 / 0.10
                 / n_windows)

study = df[df['year'].isin(STUDY)]
hw = (study[study['activacion'].isin(['p90', 'p95', 'p99'])]
      .groupby(['country', 'year', 'window'])['n_event_days'].sum()
      .reset_index())
hw['frac'] = (hw['n_event_days'] / hw['country'].map(total_country)).clip(upper=1)

heat = (hw.groupby(['country', 'year'])['frac']
          .max()
          .unstack(fill_value=0)
          .reindex(index=ALL_COUNTRIES, columns=STUDY, fill_value=0))

years = np.array(STUDY)

GRID = [
    ['Jamaica',            'Dominican Republic', 'Haiti',     'Bahamas'],
    ['Trinidad and Tobago','Barbados',           'Saint Lucia','Dominica'],
    ['Colombia',           'Peru',               'Guyana',     None],
]

fig = plt.figure(figsize=(8.5, 4.5))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.60, wspace=0.35)

for row_i, row in enumerate(GRID):
    for col_i, country in enumerate(row):
        ax = fig.add_subplot(gs[row_i, col_i])

        if country is None:
            ax.set_visible(False)
            continue

        color = COLOR_INS if country in INSULAR else COLOR_CON

        if country in NO_DATA:
            ax.set_facecolor('#F5F5F5')
            ax.text(0.5, 0.5, 'no ERA5\nsignal',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=6, color='#AAAAAA', linespacing=1.4)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            vals = heat.loc[country, STUDY].values

            ax.axhline(0.5, color='#BBBBBB', linewidth=0.6, linestyle='--', zorder=1)
            ax.fill_between(years, 0, vals, color=color, alpha=0.25, zorder=2)
            ax.plot(years, vals, color=color, linewidth=1.0, zorder=3)

            ax.set_xlim(years[0], years[-1])
            ax.set_ylim(0, 1.05)
            ax.set_xticks([2000, 2010, 2020])
            ax.set_xticklabels(['2000', '10', '20'], fontsize=5.5)
            ax.set_yticks([0, 0.5, 1.0])
            if col_i == 0:
                ax.set_yticklabels(['0%', '50%', '100%'], fontsize=5.5)
                ax.set_ylabel('Area fraction', fontsize=6, labelpad=3, color='#444444')
            else:
                ax.set_yticklabels([])
            ax.tick_params(length=2, width=0.5, pad=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.yaxis.grid(True, linewidth=0.3, color='#EEEEEE', zorder=0)
            ax.set_axisbelow(True)

        short = (country
                 .replace('Dominican Republic', 'Dominican\nRepublic')
                 .replace('Trinidad and Tobago', 'Trinidad\n& Tobago')
                 .replace('Saint Lucia', 'Saint Lucia'))
        ax.set_title(short, fontsize=6.5, pad=3, color='#222222', linespacing=1.3)

fig.subplots_adjust(left=0.08)

from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color=COLOR_INS, linewidth=1.2, label='Insular'),
    Line2D([0], [0], color=COLOR_CON, linewidth=1.2, label='Continental'),
]
fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.99, 0.01),
           fontsize=6.5, frameon=False, ncol=2,
           handlelength=1.5, handletextpad=0.4, columnspacing=1.0)

out_path = OUT_DIR / 'fig_1_8_heatwave_sparklines.png'

fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.close()
