#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

EMDAT_PATH = Path('/home/jupyter-daniela/CAF/data/EMDAT/'
                  'public_emdat_custom_request_2025-11-21_25fc7800-ab47-4f91-905e-101ce7d788d4.xlsx')
OUT_DIR = Path('/home/jupyter-daniela/CAF/outputs/losses')
OUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES_EMDAT = [
    'Bahamas', 'Barbados', 'Dominica', 'Dominican Republic',
    'Haiti', 'Jamaica', 'Saint Lucia', 'Trinidad and Tobago',
    'Colombia', 'Guyana', 'Peru',
]

df = pd.read_excel(EMDAT_PATH, sheet_name='EM-DAT Data')
df = df[df['Country'].isin(COUNTRIES_EMDAT)].copy()

counts = (df.groupby('Country')
            .size()
            .reindex(COUNTRIES_EMDAT)
            .sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(6.5, 4.5))

bars = ax.barh(counts.index, counts.values, color='#4C8BB5', height=0.6)

for bar, val in zip(bars, counts.values):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            str(val), va='center', ha='left', fontsize=8, fontweight='normal')

ax.set_xlabel('Number of events', fontsize=8)
ax.tick_params(axis='both', labelsize=8, length=3, width=0.6)
ax.set_xlim(0, counts.max() * 1.12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

out_path = OUT_DIR / 'fig_1_2_emdat_barplot.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.close()
