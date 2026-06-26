#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

EMDAT_PATH = Path('/home/jupyter-daniela/CAF/data/EMDAT/'
                  'public_emdat_custom_request_2025-11-21_25fc7800-ab47-4f91-905e-101ce7d788d4.xlsx')
OUT_DIR = Path('/home/jupyter-daniela/CAF/outputs/losses')
OUT_DIR.mkdir(parents=True, exist_ok=True)

CARIBBEAN = [
    'Bahamas', 'Barbados', 'Dominica', 'Dominican Republic',
    'Haiti', 'Jamaica', 'Saint Lucia', 'Trinidad and Tobago',
    'Colombia', 'Peru', 'Guyana',
]

YEARS = list(range(2000, 2026))

COLORS = {
    'Tropical cyclone': '#1A5C8A',
    'Flood (all types)': '#5FA8D3',
    'Drought':           '#E09B3D',
}

df = pd.read_excel(EMDAT_PATH, sheet_name='EM-DAT Data')
df = df[df['Country'].isin(CARIBBEAN)].copy()
df = df[df['Start Year'].between(2000, 2025)].copy()

def categorise(row):
    t, s = row['Disaster Type'], row['Disaster Subtype']
    if t == 'Storm' and s == 'Tropical cyclone': return 'Tropical cyclone'
    if t == 'Flood':                              return 'Flood (all types)'
    if t == 'Drought':                            return 'Drought'
    return None

df['cat'] = df.apply(categorise, axis=1)
df = df[df['cat'].notna()].copy()

pivot = (df.groupby(['Start Year', 'cat'])
           .size()
           .unstack(fill_value=0)
           .reindex(index=YEARS, fill_value=0))

for col in COLORS:
    if col not in pivot.columns:
        pivot[col] = 0
pivot = pivot[list(COLORS.keys())]

fig, ax = plt.subplots(figsize=(6.5, 3.5))

n_cats  = len(COLORS)
bw      = 0.16
offsets = [-0.22, 0, 0.22]
x       = np.arange(len(YEARS))

for (cat, color), offset in zip(COLORS.items(), offsets):
    ax.bar(x + offset, pivot[cat], width=bw,
           color=color, label=cat, linewidth=0)

ax.set_xticks(x)
ax.set_xticklabels([str(y) if y % 5 == 0 else '' for y in YEARS], fontsize=8)

ax.set_ylabel('Number of events', fontsize=8)
ax.tick_params(axis='y', labelsize=8, length=3, width=0.6)
ax.tick_params(axis='x', length=3, width=0.6)

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
           markersize=7, label=cat)
    for cat, color in COLORS.items()
]
ax.legend(handles=legend_handles, fontsize=8, frameon=False, loc='upper left')

import matplotlib.ticker as mticker
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

out_path = OUT_DIR / 'fig_1_3_emdat_annual.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.close()
