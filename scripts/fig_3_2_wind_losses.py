import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from pathlib import Path

BASE_DIR  = Path('/home/jupyter-daniela/CAF')
DATA_PATH = BASE_DIR / 'outputs/losses/table_5_1_wind_by_country_storm.csv'
OUT_PATH  = BASE_DIR / 'outputs/losses/fig_3_2_wind_losses.png'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

NAMED_STORMS = ['IVAN', 'DEAN', 'GILBERT', 'MELISSA',
                'ALLEN', 'MATTHEW', 'ANDREW', 'MARIA', 'DORIAN', 'GEORGES']

STORM_COLORS = {
    'IVAN':    '#1a3c6d',
    'DEAN':    '#2b6cb0',
    'GILBERT': '#63a0d4',
    'MELISSA': '#b8d4ea',
    'ALLEN':   '#c0392b',
    'MATTHEW': '#e67e22',
    'ANDREW':  '#f5cba7',
    'MARIA':   '#7d3c98',
    'DORIAN':  '#1e8449',
    'GEORGES': '#8b6914',
    'Other':   '#d5d5d5',
}
STORM_YEARS = {
    'IVAN': 2004, 'DEAN': 2007, 'GILBERT': 1988, 'MELISSA': 2025,
    'ALLEN': 1980, 'MATTHEW': 2016, 'ANDREW': 1992,
    'MARIA': 2017, 'DORIAN': 2019, 'GEORGES': 1998,
}
STORM_ORDER = NAMED_STORMS + ['Other']

X_BREAK = 2.4e9
X_RIGHT = 16.2e9
BAR_H   = 0.60

df = pd.read_csv(DATA_PATH)
df = df[df['country'] != 'Trinidad and Tobago'].copy()
df['storm_name'] = df['storm_name'].str.upper().str.strip()
df['storm_key']  = df['storm_name'].apply(
    lambda n: n if n in NAMED_STORMS else 'Other')

country_totals = df.groupby('country')['loss_mid'].sum().sort_values(ascending=False)
country_ranges = df.groupby('country')[['loss_low', 'loss_high']].sum()
country_order  = country_totals.index.tolist()
n_c            = len(country_order)
Y              = {c: i for i, c in enumerate(reversed(country_order))}

def make_pivot(col):
    piv = df.groupby(['country', 'storm_key'])[col].sum().unstack(fill_value=0)
    for s in STORM_ORDER:
        if s not in piv.columns:
            piv[s] = 0.0
    return piv

pivot      = make_pivot('loss_mid')
pivot_low  = make_pivot('loss_low')
pivot_high = make_pivot('loss_high')

fig = plt.figure(figsize=(6.5, 4.0))
gs  = fig.add_gridspec(
    1, 2,
    width_ratios=[1, 1],
    left=0.20, right=0.95, top=0.96, bottom=0.24,
    wspace=0.02,
)
ax_l = fig.add_subplot(gs[0])
ax_r = fig.add_subplot(gs[1], sharey=ax_l)

ax_l.set_xlim(0,       X_BREAK)
ax_r.set_xlim(X_BREAK, X_RIGHT)

legend_items = {}

for country in country_order:
    y     = Y[country]
    x_acc = 0.0
    for storm in STORM_ORDER:
        val = pivot.loc[country, storm] if country in pivot.index else 0.0
        if val < 1e4:
            x_acc += val
            continue
        color = STORM_COLORS[storm]
        for ax in (ax_l, ax_r):
            ax.barh(y, val, left=x_acc, height=BAR_H,
                    color=color, linewidth=0, zorder=2)
        if storm not in legend_items:
            yr  = STORM_YEARS.get(storm, '')
            lbl = (f'{storm.capitalize()} {yr}' if yr else storm.capitalize())
            legend_items[storm] = mpatches.Patch(color=color, label=lbl)
        x_acc += val

for country in country_order:
    y  = Y[country]
    lo = country_ranges.loc[country, 'loss_low']
    hi = country_ranges.loc[country, 'loss_high']
    for ax in (ax_l, ax_r):
        ax.plot([lo, hi], [y, y],
                color='#111111', lw=0.6, solid_capstyle='butt', zorder=5)
        for xv in (lo, hi):
            ax.plot([xv, xv], [y - 0.08, y + 0.08],
                    color='#111111', lw=0.6, zorder=5)

def fmt_left(x, _):
    v = x / 1e9
    if v == 0:
        return '$0B'
    return f'${v:.1f}B' if (v * 10) % 10 != 0 else f'${int(v)}B'

ax_l.set_xticks([0, 0.5e9, 1e9, 1.5e9, 2e9])
ax_r.set_xticks([4e9, 8e9, 12e9, 16e9])
ax_l.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_left))
ax_r.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${int(x/1e9)}B'))

short = {'Dominican Republic': 'Dom. Republic'}
ax_l.set_yticks(list(Y.values()))
ax_l.set_yticklabels([short.get(c, c) for c in reversed(country_order)],
                     fontsize=7.5, fontweight='normal', color='#000000')
ax_r.tick_params(which='both', left=False, labelleft=False)

for country in country_order:
    y     = Y[country]
    total = country_totals[country]
    hi    = country_ranges.loc[country, 'loss_high']
    lbl   = f'${total/1e9:.1f}B' if total >= 1e9 else f'${total/1e9:.2f}B'

    if hi <= X_BREAK:
        ax_l.text(hi + 0.05e9, y, lbl,
                  va='center', ha='left', fontsize=6.5, color='#555555')
    elif hi <= X_RIGHT:
        ax_r.text(hi + 0.08e9, y, lbl,
                  va='center', ha='left', fontsize=6.5, color='#555555')
    else:
        ax_r.text(X_RIGHT + 0.25e9, y, lbl,
                  va='center', ha='left', fontsize=7, color='#222222',
                  clip_on=False)

d = 0.018
for ax, side in ((ax_l, 'right'), (ax_r, 'left')):
    xf = 1.0 if side == 'right' else 0.0
    for yf in (0.03, 0.97):
        ax.plot([xf - d, xf + d], [yf - d * 1.5, yf + d * 1.5],
                transform=ax.transAxes, color='#888888',
                lw=1.0, clip_on=False, zorder=10)

for ax in (ax_l, ax_r):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, labelsize=7.5, colors='#000000')
    ax.set_ylim(-0.5, n_c - 0.5)
    ax.xaxis.grid(True, color='#e8e8e8', lw=0.5, zorder=0)
    ax.set_axisbelow(True)

visible = [s for s in STORM_ORDER
           if s in legend_items
           and s in pivot.columns
           and pivot[s].sum() > 1e6]
handles = [legend_items[s] for s in visible]

range_handle = mlines.Line2D([], [], color='#111111', lw=1.2,
                              marker='|', markersize=5, markeredgewidth=1.2,
                              label='Low–high range')
handles.append(range_handle)

leg = fig.legend(
    handles=handles,
    loc='lower center',
    bbox_to_anchor=(0.575, 0.0),
    bbox_transform=fig.transFigure,
    fontsize=6.5,
    ncol=6,
    frameon=False,
    handlelength=1.0,
    handletextpad=0.4,
    columnspacing=0.9,
)

fig.text(0.575, 0.17, 'Cumulative losses, mid-scenario (USD)',
         ha='center', va='top', fontsize=7.5, color='#000000')

fig.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved → {OUT_PATH}')
plt.close()
