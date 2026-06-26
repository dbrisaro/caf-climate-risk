import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

BASE_DIR  = Path('/home/jupyter-daniela/CAF')
DATA_PATH = BASE_DIR / 'outputs/losses/table_5_1_wind_by_country_storm.csv'
OUT_PATH  = BASE_DIR / 'outputs/losses/fig_3_3_top_events.png'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df  = pd.read_csv(DATA_PATH)
agg = df.groupby(['storm_name', 'year'])[['loss_low', 'loss_mid', 'loss_high']].sum().reset_index()
cats = df.groupby(['storm_name', 'year'])['ss_category'].first().reset_index()
agg  = agg.merge(cats, on=['storm_name', 'year'])

top5 = (agg.nlargest(5, 'loss_mid')
           .sort_values('loss_mid', ascending=True)
           .reset_index(drop=True))

Y = np.arange(len(top5))

CAT_COLORS = {
    'Cat 5': '#8b1a1a',
    'Cat 4': '#c0392b',
    'Cat 3': '#e67e22',
    'Cat 2': '#e8b84b',
    'Cat 1': '#4a9e6b',
}

fig, ax = plt.subplots(figsize=(6.5, 3.6))
fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.14)

legend_cats = {}

for i, row in top5.iterrows():
    cat   = row['ss_category']
    color = CAT_COLORS.get(cat, '#8b1a1a')
    lo    = row['loss_low']  / 1e9
    mid   = row['loss_mid']  / 1e9
    hi    = row['loss_high'] / 1e9

    ax.plot([lo, hi], [i, i], color=color, lw=0.8, alpha=0.6,
            solid_capstyle='butt', zorder=2)
    for xv in (lo, hi):
        ax.plot([xv, xv], [i - 0.07, i + 0.07], color=color, lw=0.8,
                alpha=0.6, zorder=2)
    ax.plot(mid, i, 'o', color=color, markersize=5, zorder=3)

    lbl = f'${mid:.1f}B  [${lo:.1f}–${hi:.1f}B]'
    ax.text(hi + 0.18, i, lbl, va='center', ha='left',
            fontsize=6.5, color='#888888')

    if cat not in legend_cats:
        legend_cats[cat] = mpatches.Patch(color=color, label=cat)

ylabels = [f"{r['storm_name'].capitalize()}\n{int(r['year'])}"
           for _, r in top5.iterrows()]
ax.set_yticks(Y)
ax.set_yticklabels(ylabels, fontsize=8, va='center', fontweight='normal',
                   color='#000000', linespacing=1.3)

ax.set_xlim(0, 10)
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
ax.set_xlabel('Estimated wind loss (USD billions)', fontsize=8, color='#000000')

for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0, labelsize=8, colors='#000000')
ax.set_ylim(-0.5, len(top5) - 0.5)
ax.xaxis.grid(True, color='#e8e8e8', lw=0.5, zorder=0)
ax.set_axisbelow(True)

handles = list(legend_cats.values())
ax.legend(handles=handles, title='Saffir-Simpson', title_fontsize=7.5,
          loc='lower right', fontsize=7.5, frameon=False,
          handlelength=1.0, handletextpad=0.5,
          labelcolor='#000000')

fig.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved → {OUT_PATH}')
plt.close()
