import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE_DIR = Path('/home/jupyter-daniela/CAF')
T52_PATH = BASE_DIR / 'outputs/losses/table_5_2_wind_cumulative_by_country.csv'
CAT_PATH = BASE_DIR / 'outputs/losses/historical_loss_catalog.csv'
OUT_FIG  = BASE_DIR / 'outputs/losses'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

t52 = pd.read_csv(T52_PATH)
cat = pd.read_csv(CAT_PATH)

t52 = t52[t52['total_loss_mid'] > 1e3].copy()
t52 = t52.sort_values('total_loss_mid', ascending=True).reset_index(drop=True)

t52['mid_B']  = t52['total_loss_mid']  / 1e9
t52['low_B']  = t52['total_loss_low']  / 1e9
t52['high_B'] = t52['total_loss_high'] / 1e9
t52['err_lo'] = t52['mid_B'] - t52['low_B']
t52['err_hi'] = t52['high_B'] - t52['mid_B']

BAR_COLOR = '#2166ac'
ERR_COLOR = '#555555'

fig, ax = plt.subplots(figsize=(8, 5))

y = np.arange(len(t52))
ax.barh(y, t52['high_B'] - t52['low_B'], left=t52['low_B'],
        color=BAR_COLOR, alpha=0.18, height=0.55)
ax.barh(y, t52['mid_B'], color=BAR_COLOR, alpha=0.88,
        height=0.55, edgecolor='white', linewidth=0.4)
ax.errorbar(t52['mid_B'], y,
            xerr=[t52['err_lo'], t52['err_hi']],
            fmt='none', color=ERR_COLOR, capsize=3.5,
            lw=1.1, capthick=1.1)

for i, row in t52.iterrows():
    ax.text(row['high_B'] + 0.45, i,
            f"${row['mid_B']:.1f}B", va='center',
            fontsize=8.5, color='#222222')

ax.set_yticks(y)
ax.set_yticklabels(t52['country'], fontsize=9.5)
ax.set_xlabel('Cumulative wind loss — mid scenario (USD billions)', fontsize=10)
ax.set_xlim(0, t52['high_B'].max() * 1.35)
ax.tick_params(axis='x', labelsize=9)
ax.grid(axis='x', alpha=0.25, linewidth=0.4, zorder=0)

patch_mid  = mpatches.Patch(color=BAR_COLOR, alpha=0.88, label='Mid scenario')
patch_band = mpatches.Patch(color=BAR_COLOR, alpha=0.25, label='Low–high range')
ax.legend(handles=[patch_mid, patch_band], fontsize=8.5,
          loc='lower right', frameon=False)

fig.tight_layout()
out1 = OUT_FIG / 'wind_losses_by_country.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out1.name}")

SS_COLORS = {-1:'#adb5bd', 0:'#adb5bd', 1:'#4dabf7',
              2:'#74c476', 3:'#fd8d3c', 4:'#e6550d', 5:'#a50f15'}

top5 = (cat[cat['loss_mid'] > 0]
        .nlargest(5, 'loss_mid')
        .reset_index(drop=True))
top5['low_B']  = top5['loss_low']  / 1e9
top5['mid_B']  = top5['loss_mid']  / 1e9
top5['high_B'] = top5['loss_high'] / 1e9
top5['label']  = (top5['name'].str.title()
                  + '\n' + top5['year'].astype(int).astype(str))
top5['color']  = top5['category'].fillna(-1).astype(int).map(SS_COLORS)
top5 = top5.sort_values('mid_B', ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 5))

y = np.arange(len(top5))

for i, row in top5.iterrows():
    ax.plot([row['low_B'], row['high_B']], [i, i],
            color=row['color'], lw=5, alpha=0.28,
            solid_capstyle='round')
    ax.plot([row['low_B'], row['high_B']], [i, i],
            color=row['color'], lw=1.2, alpha=0.7)
    ax.scatter(row['low_B'], i, color=row['color'], s=40,
               marker='|', linewidths=1.5, zorder=5)
    ax.scatter(row['high_B'], i, color=row['color'], s=40,
               marker='|', linewidths=1.5, zorder=5)
    ax.scatter(row['mid_B'], i, color=row['color'], s=90,
               zorder=6, edgecolors='white', linewidths=0.8)

    ax.text(row['high_B'] + 0.08, i,
            f"${row['mid_B']:.1f}B  [${row['low_B']:.1f}–${row['high_B']:.1f}B]",
            va='center', fontsize=8.2, color='#222222')

ax.set_yticks(y)
ax.set_yticklabels(top5['label'], fontsize=9.5)
ax.set_xlabel('Estimated wind loss (USD billions)', fontsize=10)
ax.set_xlim(0, top5['high_B'].max() * 1.72)
ax.tick_params(axis='x', labelsize=9)
ax.grid(axis='x', alpha=0.25, linewidth=0.4, zorder=0)
ax.axvline(0, color='#cccccc', lw=0.6)

ss_cats = sorted(top5['category'].dropna().astype(int).unique())
handles = [mpatches.Patch(color=SS_COLORS[c],
           label=f'Cat {c}' if c > 0 else 'TD', alpha=0.9)
           for c in ss_cats]
ax.legend(handles=handles, title='Saffir-Simpson',
          title_fontsize=8.5, fontsize=8.5,
          loc='lower right', frameon=False)

fig.tight_layout()
out2 = OUT_FIG / 'wind_top_events.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out2.name}")
