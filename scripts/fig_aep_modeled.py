import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

BASE_DIR = Path('/home/jupyter-daniela/CAF')
OUT_DIR  = BASE_DIR / 'outputs/losses'
OUT_DIR.mkdir(parents=True, exist_ok=True)

T = 45

cat = pd.read_csv(BASE_DIR / 'outputs/losses/historical_loss_catalog.csv')

cat = cat[cat['loss_mid'] > 0].copy()
cat = cat.sort_values('loss_mid', ascending=False).reset_index(drop=True)
cat['rank']   = cat.index + 1
cat['rp']     = T / cat['rank']
cat['aep_pct']= cat['rank'] / T * 100

aal_mid  = cat['loss_mid'].sum()  / T
aal_low  = cat['loss_low'].sum()  / T
aal_high = cat['loss_high'].sum() / T

csv_out = BASE_DIR / 'outputs/losses/aep_modeled.csv'
cat[['name','year','category','rank','rp','aep_pct',
     'loss_low','loss_mid','loss_high']].to_csv(csv_out, index=False)
print(f'CSV → {csv_out}')

fig, ax = plt.subplots(figsize=(6.5, 4.2))
fig.subplots_adjust(left=0.13, right=0.95, top=0.96, bottom=0.13)

xs_mid  = cat['loss_mid'].values  / 1e6
xs_low  = cat['loss_low'].values  / 1e6
xs_high = cat['loss_high'].values / 1e6
ys      = cat['aep_pct'].values

color = '#1a3c6d'

ax.fill_betweenx(ys, xs_low, xs_high,
                 alpha=0.15, color=color, zorder=1)
ax.plot(xs_mid, ys, color=color, lw=1.6, zorder=3)
ax.scatter(xs_mid, ys, color=color, s=18, zorder=4)

ax.axvline(aal_mid / 1e6, color=color, lw=0.8, ls='--', alpha=0.7)
ax.text(aal_mid / 1e6 * 1.03, ys.max() * 0.95,
        f'AAL ${aal_mid/1e6:.0f}M/yr',
        fontsize=6.5, color=color, va='top')

top5  = cat.head(5).sort_values('loss_mid').reset_index(drop=True)
xmax  = cat['loss_mid'].max() / 1e6
x_txt = np.linspace(xmax * 0.15, xmax * 0.88, len(top5))
y_txt = np.linspace(34, 16, len(top5))

for i, row in top5.iterrows():
    ax.annotate(
        f"{row['name'].capitalize()} {int(row['year'])}",
        xy=(row['loss_mid']/1e6, row['aep_pct']),
        xytext=(x_txt[i], y_txt[i]),
        fontsize=6.5, color='#333333',
        ha='center', va='bottom',
        arrowprops=dict(arrowstyle='-', color='#bbbbbb', lw=0.5,
                        shrinkA=0, shrinkB=2)
    )

ax.set_xlim(0, cat['loss_mid'].max() / 1e6 * 1.15)
ax.set_ylim(0, 100)
ax.set_xlabel('Total wind loss, critical infrastructure (USD millions)', fontsize=8)
ax.set_ylabel('Annual exceedance probability (%)', fontsize=8)

ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{int(y)}%'))

for sp in ax.spines.values():
    sp.set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')
ax.tick_params(length=3, labelsize=7.5, colors='#000000')
ax.grid(True, which='major', color='#e8e8e8', lw=0.5)
ax.set_axisbelow(True)

fig.savefig(OUT_DIR / 'fig_aep_modeled.png', dpi=180, bbox_inches='tight', facecolor='white')
print(f'Figure → {OUT_DIR}/fig_aep_modeled.png')
plt.close()
