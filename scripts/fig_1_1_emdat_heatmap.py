#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

EMDAT_PATH = Path('/home/jupyter-daniela/CAF/data/EMDAT/'
                  'public_emdat_custom_request_2025-11-21_25fc7800-ab47-4f91-905e-101ce7d788d4.xlsx')
OUT_DIR    = Path('/home/jupyter-daniela/CAF/outputs/losses')
OUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = [
    'Bahamas', 'Barbados', 'Dominica', 'Dominican\nRepublic',
    'Haiti', 'Jamaica', 'Saint Lucia', 'Trinidad and\nTobago',
    'Colombia', 'Guyana', 'Peru',
]
COUNTRIES_EMDAT = [
    'Bahamas', 'Barbados', 'Dominica', 'Dominican Republic',
    'Haiti', 'Jamaica', 'Saint Lucia', 'Trinidad and Tobago',
    'Colombia', 'Guyana', 'Peru',
]

CATEGORIES = [
    ('Tropical\ncyclone',
     lambda r: (r['Disaster Type'] == 'Storm') &
               (r['Disaster Subtype'] == 'Tropical cyclone')),

    ('Flood\n(all types)',
     lambda r: r['Disaster Type'] == 'Flood'),

    ('Drought',
     lambda r: r['Disaster Type'] == 'Drought'),

    ('Storm\n(general)',
     lambda r: (r['Disaster Type'] == 'Storm') &
               (r['Disaster Subtype'] != 'Tropical cyclone')),

    ('Landslide\n(all types)',
     lambda r: r['Disaster Type'].isin(['Mass movement (wet)', 'Mass movement (dry)'])),

    ('Forest fire /\nWildfire',
     lambda r: r['Disaster Type'] == 'Wildfire'),
]

print("Loading EM-DAT...")
df = pd.read_excel(EMDAT_PATH, sheet_name='EM-DAT Data')
df = df[df['Country'].isin(COUNTRIES_EMDAT)].copy()
print(f"  {len(df)} events for {df['Country'].nunique()} countries")

emdat_to_display = dict(zip(COUNTRIES_EMDAT, COUNTRIES))
col_lbls = [lbl for lbl, _ in CATEGORIES]
matrix = pd.DataFrame(0, index=COUNTRIES, columns=col_lbls)

for label, filt in CATEGORIES:
    mask = filt(df)
    counts = df[mask].groupby('Country').size()
    for country, n in counts.items():
        display = emdat_to_display.get(country)
        if display:
            matrix.loc[display, label] = n

print("\nEvent counts matrix:")
print(matrix.to_string())

fig, ax = plt.subplots(figsize=(6.5, 3.5))

from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

bounds = list(range(0, 110, 10))
n_bins = len(bounds) - 1

cmap_base = plt.cm.YlOrBr
colors_list = [(1, 1, 1)] + [cmap_base(i / (n_bins - 1)) for i in range(n_bins)]
cmap = LinearSegmentedColormap.from_list('emdat_heatmap', colors_list, N=n_bins + 1)
norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)

data = matrix.values.astype(float)
im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')

col_labels = [lbl for lbl, _ in CATEGORIES]
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=8, ha='center')
ax.set_yticks(range(len(COUNTRIES)))
ax.set_yticklabels(COUNTRIES, fontsize=8)
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,
               left=True, right=False, length=3, width=0.6)

for i, country in enumerate(COUNTRIES):
    for j, (label, _) in enumerate(CATEGORIES):
        n = int(matrix.loc[country, label])
        if n > 0:
            brightness = data[i, j] / 100
            txt_color = 'white' if brightness > 0.65 else 'black'
            ax.text(j, i, str(n), ha='center', va='center',
                    fontsize=8, fontweight='normal', color=txt_color)

cbar = fig.colorbar(im, ax=ax, orientation='horizontal', location='bottom',
                    fraction=0.04, pad=0.08, shrink=1.0, aspect=40,
                    ticks=[b for b in bounds if b % 20 == 0])
cbar.set_label('Number of events', fontsize=8)
cbar.outline.set_visible(False)
cbar.ax.tick_params(size=3, width=0.6, labelsize=8)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()

out_path = OUT_DIR / 'fig_1_1_emdat_heatmap.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out_path}")
plt.close()
