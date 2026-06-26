import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.append('/home/jupyter-daniela/CAF/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path

import style; style.apply()

OUT_DIR = Path('/home/jupyter-daniela/CAF/outputs/exposure')
OUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = [
    ('France',        8.4, 5.8),
    ('Spain',         7.2, 5.2),
    ('United States', 5.1, 4.8),
    ('Chile',         4.6, 6.3),
    ('Costa Rica',    4.1, 7.1),
]

CARIBBEAN = [
    ('Barbados',            3.13, 3.84),
    ('Trinidad and Tobago', 2.39, 2.65),
    ('Saint Lucia',         1.89, 4.39),
    ('Dominica',            1.53, 3.33),
    ('Bahamas',             1.50, 2.47),
    ('Colombia',            2.09, 2.40),
    ('Peru',                2.37, 12.23),
    ('Guyana',              1.53, 3.95),
    ('Haiti',               0.54, 2.38),
    ('Jamaica',             0.49, 1.10),
    ('Dominican Republic',  0.43, 0.58),
]

bench_df = pd.DataFrame(BENCHMARKS, columns=['country', 'health', 'education'])
carib_df  = pd.DataFrame(CARIBBEAN,  columns=['country', 'health', 'education'])

carib_sorted  = carib_df.sort_values('health', ascending=True).reset_index(drop=True)
bench_sorted  = bench_df.sort_values('health', ascending=True).reset_index(drop=True)

GAP = 1

n_carib = len(carib_sorted)
n_bench = len(bench_sorted)
n_total = n_carib + GAP + n_bench

y_carib = np.arange(n_carib)
y_bench = np.arange(n_carib + GAP, n_total)

COL_HEALTH  = '#c0392b'
COL_EDU     = '#2b6cb0'
COL_LINE_C  = '#aaaaaa'
COL_LINE_CON = '#888888'
COL_LINE_B  = '#cccccc'
DOT_SIZE    = 55
CONTINENTAL = {'Colombia', 'Peru', 'Guyana'}

fig, ax = plt.subplots(figsize=(6.5, 5.8))
fig.subplots_adjust(left=0.26, right=0.95, top=0.97, bottom=0.12)

for i, row in carib_sorted.iterrows():
    y = y_carib[i]
    lo, hi = sorted([row.health, row.education])
    is_con = row.country in CONTINENTAL
    lc = COL_LINE_CON if is_con else COL_LINE_C
    mk = 's' if is_con else 'o'
    md = 'D' if not is_con else '^'
    ax.plot([lo, hi], [y, y], color=lc, lw=1.0, zorder=1)
    ax.scatter(row.health,    y, color=COL_HEALTH, s=DOT_SIZE, zorder=3, linewidths=0,
               marker=mk)
    ax.scatter(row.education, y, color=COL_EDU,    s=DOT_SIZE, zorder=3, linewidths=0,
               marker=md)

for i, row in bench_sorted.iterrows():
    y = y_bench[i]
    lo, hi = sorted([row.health, row.education])
    ax.plot([lo, hi], [y, y], color=COL_LINE_B, lw=1.0, zorder=1)
    ax.scatter(row.health,    y, color=COL_HEALTH, s=DOT_SIZE, zorder=3,
               linewidths=0.8, edgecolors='white', alpha=0.55)
    ax.scatter(row.education, y, color=COL_EDU,    s=DOT_SIZE, zorder=3,
               linewidths=0.8, edgecolors='white', alpha=0.55, marker='D')

div_y = n_carib + GAP / 2 - 0.05
ax.axhline(div_y, color='#cccccc', lw=0.8, linestyle='--', zorder=0)

all_labels = (
    list(carib_sorted['country']) +
    [''] * GAP +
    list(bench_sorted['country'])
)
all_y = list(y_carib) + [n_carib] + list(y_bench)

ax.set_yticks(all_y)
ax.set_yticklabels(all_labels, fontsize=7.5, color='#000000')

for tick, label in zip(ax.yaxis.get_major_ticks(), all_labels):
    if label in bench_df['country'].values:
        tick.label1.set_color('#888888')

ax.set_xlabel('Facilities per 10,000 inhabitants', fontsize=8, color='#000000')
ax.set_xlim(-0.1, 13.5)
ax.xaxis.grid(True, color='#e8e8e8', lw=0.5, zorder=0)

h_dot   = mlines.Line2D([], [], color=COL_HEALTH, marker='o', markersize=5,
                         linestyle='None', label='Health facilities (insular)')
h_dot2  = mlines.Line2D([], [], color=COL_HEALTH, marker='s', markersize=5,
                         linestyle='None', label='Health facilities (continental)')
e_dot   = mlines.Line2D([], [], color=COL_EDU,    marker='D', markersize=4.5,
                         linestyle='None', label='Schools (insular)')
e_dot2  = mlines.Line2D([], [], color=COL_EDU,    marker='^', markersize=4.5,
                         linestyle='None', label='Schools (continental)')
b_patch = mpatches.Patch(color='#aaaaaa', alpha=0.6, label='International benchmarks')

ax.legend(handles=[h_dot, h_dot2, e_dot, e_dot2, b_patch],
          loc='upper right',
          fontsize=6.5, frameon=False, handlelength=1.0,
          handletextpad=0.5, ncol=1, columnspacing=1.0)

for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0, colors='#000000', labelsize=7.5)
ax.set_axisbelow(True)

fig.savefig(OUT_DIR / 'fig_3_12_benchmark_density.png',
            dpi=180, bbox_inches='tight', facecolor='white')
print('Saved: outputs/exposure/fig_3_12_benchmark_density.png')
plt.close()
