#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import style
style.apply()

BASE   = Path('/home/jupyter-daniela/CAF')
INDIR  = BASE / 'outputs/regression'
OUTDIR = BASE / 'outputs/regression'
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INDIR / 'coefficients_split.csv')

M6_VARS = ['built_s', 'built_v', 'drought', 'rainfall', 'heatwave', 'wind_kt',
           'log_pop', 'log_gdp']

LABELS = {
    'built_s':  'Built surface',
    'built_v':  'Built volume',
    'drought':  'Drought',
    'rainfall': 'Rainfall',
    'heatwave': 'Heatwave',
    'wind_kt':  'Wind speed (kt)',
    'log_pop':  'log(Population)',
    'log_gdp':  'log(GDP per capita)',
}

row   = df[df['variable'].isin(M6_VARS)].set_index('variable')
coefs = row['Caribe_coef'].astype(float)
ses   = row['Caribe_se'].astype(float)
pvals = row['Caribe_p'].astype(float)

ci_lo = coefs - 1.96 * ses
ci_hi = coefs + 1.96 * ses

order  = coefs.sort_values(ascending=True).index
labels = [LABELS[v] for v in order]
est    = coefs[order].values
lo     = ci_lo[order].values
hi     = ci_hi[order].values
pv     = pvals[order].values

SIG_COL  = '#2166AC'
MARG_COL = '#6BAED6'
NS_COL   = '#AAAAAA'

def color(p):
    if p < 0.05: return SIG_COL
    if p < 0.10: return MARG_COL
    return NS_COL

colors = [color(p) for p in pv]

fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor='white')

for i, (e, l, h, c) in enumerate(zip(est, lo, hi, colors)):
    ax.plot([l, h], [i, i], color=c, lw=1.6, solid_capstyle='round', zorder=3)
    ax.scatter(e, i, color=c, s=40, zorder=4, edgecolors='white', linewidths=0.4)

ax.axvline(0, color='#333333', lw=0.7, linestyle='--', zorder=2)

ax.set_yticks(range(len(order)))
ax.set_yticklabels(labels, fontsize=7, fontweight='normal')
ax.set_xlabel('Standardized coefficient  (dep: log(1 + Loss))',
              fontsize=7, labelpad=4, fontweight='normal')
ax.tick_params(axis='x', labelsize=6.5, length=3, width=0.5)
ax.tick_params(axis='y', length=0)

ax.xaxis.grid(True, linewidth=0.3, color='#EEEEEE', zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_linewidth(0.5)

r2_val = df[df['variable'] == 'R²']['Caribe_coef'].values
n_val  = df[df['variable'] == 'N']['Caribe_coef'].values
if len(r2_val) and len(n_val):
    ax.text(0.97, 0.03,
            f"R² = {float(r2_val[0]):.3f}   N = {int(float(n_val[0])):,}",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=6, color='#555555', fontweight='normal')

patches = [
    mpatches.Patch(color=SIG_COL,  label='p < 0.05'),
    mpatches.Patch(color=MARG_COL, label='p < 0.10'),
    mpatches.Patch(color=NS_COL,   label='n.s.'),
]
ax.legend(handles=patches, fontsize=6, frameon=False, loc='upper left',
          handlelength=1.0, handletextpad=0.4, labelspacing=0.3)

fig.tight_layout(pad=0.5)

out_path = OUTDIR / 'fig_2_1_m6_forest.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved → {out_path}")
