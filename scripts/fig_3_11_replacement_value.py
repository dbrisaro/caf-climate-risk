import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.append('/home/jupyter-daniela/CAF/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

import style; style.apply()

BASE    = Path('/home/jupyter-daniela/CAF')
OSM_CSV = BASE / 'data/OSM/osm_top_ciudades_caribe_continentales.csv'
OUT_DIR = BASE / 'outputs/losses'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_COUNTRIES = [
    'Haiti', 'Dominican Republic', 'Trinidad and Tobago', 'Jamaica',
    'Barbados', 'Bahamas', 'Saint Lucia', 'Dominica',
]

SOCIAL_CATS = ['salud', 'escuelas', 'gobierno', 'emergencia']

CAT_COLORS = {
    'salud':      '#c0392b',
    'escuelas':   '#2b6cb0',
    'gobierno':   '#7d3c98',
    'emergencia': '#e67e22',
}
CAT_LABELS = {
    'salud':      'Health',
    'escuelas':   'Education',
    'gobierno':   'Government',
    'emergencia': 'Emergency services',
}

RC_MID = {
    'salud':      800 * 1200,
    'escuelas':   450 * 950,
    'gobierno':   600 * 1200,
    'emergencia': 400 * 1000,
}

df = pd.read_csv(OSM_CSV)
df = df[df['pais'].isin(ALL_COUNTRIES)].copy()
df['rv_mid'] = df['categoria'].map(RC_MID)

agg = (
    df[df.categoria.isin(SOCIAL_CATS)]
    .groupby(['pais', 'categoria'])['rv_mid']
    .sum().reset_index()
)
agg['rv_mid_m'] = agg['rv_mid'] / 1e6

pivot = agg.pivot(index='pais', columns='categoria', values='rv_mid_m').fillna(0)
pivot = pivot[[c for c in SOCIAL_CATS if c in pivot.columns]]
pivot['total'] = pivot.sum(axis=1)
pivot = pivot.sort_values('total')

fig, ax = plt.subplots(figsize=(6.5, 5.2))
fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.16)

y = np.arange(len(pivot))

for cat in SOCIAL_CATS:
    if cat not in pivot.columns:
        continue
    lefts = pivot[[c for c in SOCIAL_CATS[:SOCIAL_CATS.index(cat)]]].sum(axis=1).values
    ax.barh(y, pivot[cat].values, left=lefts, height=0.62,
            color=CAT_COLORS[cat], linewidth=0, zorder=2)

for i, (country, row) in enumerate(pivot.iterrows()):
    total = row['total']
    ax.text(total + 8, i, f'${total:,.0f}M',
            va='center', ha='left', fontsize=6.5, color='#555555')

ax.set_yticks(y)
ax.set_yticklabels(pivot.index, fontsize=7.5, color='#000000')
ax.set_xlabel('Replacement value at risk (USD million, mid scenario)', fontsize=8, color='#000000')
ax.xaxis.grid(True, color='#e8e8e8', lw=0.5, zorder=0)
ax.set_xlim(0, pivot['total'].max() * 1.18)

handles = [
    mpatches.Patch(color=CAT_COLORS[c], label=CAT_LABELS[c])
    for c in SOCIAL_CATS
]
ax.legend(handles=handles, loc='lower right', fontsize=6.5,
          frameon=False, handlelength=1.0, handletextpad=0.5,
          ncol=2, columnspacing=1.2)

for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0, colors='#000000', labelsize=7.5)
ax.set_axisbelow(True)

fig.savefig(OUT_DIR / 'fig_3_11_replacement_value.png',
            dpi=180, bbox_inches='tight', facecolor='white')
print('Saved: outputs/losses/fig_3_11_replacement_value.png')
plt.close()
