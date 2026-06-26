import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path('/home/jupyter-daniela/CAF')
FE_PATH  = BASE_DIR / 'outputs/regression/fixed_effects.csv'
OUT_PATH = BASE_DIR / 'outputs/regression/fixed_effects_by_country.png'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

fe = pd.read_csv(FE_PATH)

fe['alpha_se'] = fe['alpha_se'].fillna(0.0)
fe['ci95'] = 1.96 * fe['alpha_se']

fe = fe.sort_values('alpha_i', ascending=True).reset_index(drop=True)

POS_COLOR = '#2166ac'
NEG_COLOR = '#d6604d'
REF_COLOR = '#888888'

def point_color(row):
    if row['alpha_i'] == 0.0 and row['alpha_se'] == 0.0:
        return REF_COLOR
    return POS_COLOR if row['alpha_i'] >= 0 else NEG_COLOR

fe['color'] = fe.apply(point_color, axis=1)

fig, ax = plt.subplots(figsize=(7.5, 5.2))

for i, row in fe.iterrows():
    ax.plot(
        [row['alpha_i'] - row['ci95'], row['alpha_i'] + row['ci95']],
        [i, i],
        color=row['color'], lw=2.0, alpha=0.55, solid_capstyle='round',
    )
    is_ref = (row['alpha_se'] == 0.0 and row['alpha_i'] == 0.0)
    ax.scatter(
        row['alpha_i'], i,
        color=row['color'], s=60, zorder=5,
        edgecolors='white', linewidths=0.7,
        marker='D' if is_ref else 'o',
    )

x_max = (fe['alpha_i'] + fe['ci95']).max()
for i, row in fe.iterrows():
    gdp_label = f"GDP/cap: ${row['gdp_pc_usd']:,.0f}"
    x_anchor  = row['alpha_i'] + row['ci95'] + 0.04
    ax.text(
        x_anchor, i, gdp_label,
        va='center', ha='left', fontsize=8.2, color='#444444',
    )

ax.axvline(0, color='#333333', lw=0.8, linestyle='--', alpha=0.55)

ax.set_yticks(range(len(fe)))
ax.set_yticklabels(fe['country'], fontsize=9.5)
ax.set_xlabel('Fixed effect  $\\hat{\\alpha}_i$  (95% CI)', fontsize=10)

x_right = x_max + 2.1
ax.set_xlim(-2.5, x_right)
ax.tick_params(axis='x', labelsize=9)
ax.grid(axis='x', alpha=0.22, linewidth=0.4, zorder=0)

from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=POS_COLOR,
           markersize=8, label='Positive FE'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=NEG_COLOR,
           markersize=8, label='Negative FE'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor=REF_COLOR,
           markersize=8, label='Reference (Bahamas)'),
]
ax.legend(handles=handles, fontsize=8.5, loc='lower right', framealpha=0.95)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {OUT_PATH}")
