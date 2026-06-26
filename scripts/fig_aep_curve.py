import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

BASE_DIR = Path('/home/jupyter-daniela/CAF')
OUT_DIR  = BASE_DIR / 'outputs/losses'

cat = pd.read_csv(BASE_DIR / 'outputs/losses/historical_loss_catalog.csv')

YEAR_MIN, YEAR_MAX = 1980, 2025
N_YEARS = YEAR_MAX - YEAR_MIN + 1

losses_mid  = cat['loss_mid'].values
losses_low  = cat['loss_low'].values
losses_high = cat['loss_high'].values

L_max  = losses_high.max()
L_grid = np.linspace(0, L_max * 1.05, 2000)

def aep_curve(losses):
    years = cat['year'].values
    aep = np.zeros(len(L_grid))
    for i, L in enumerate(L_grid):
        years_exceeding = np.unique(years[losses > L])
        aep[i] = len(years_exceeding) / N_YEARS
    return aep

aep_mid  = aep_curve(losses_mid)
aep_low  = aep_curve(losses_low)
aep_high = aep_curve(losses_high)

fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(L_grid / 1e6, aep_mid, aep_high, alpha=0.25, color='steelblue', label='Low–High range')
ax.fill_between(L_grid / 1e6, aep_low, aep_mid,  alpha=0.25, color='steelblue')
ax.plot(L_grid / 1e6, aep_mid,  color='steelblue', lw=2.5, label='Mid scenario')
ax.plot(L_grid / 1e6, aep_low,  color='steelblue', lw=1,   linestyle='--', alpha=0.6, label='Low scenario')
ax.plot(L_grid / 1e6, aep_high, color='steelblue', lw=1,   linestyle='--', alpha=0.6, label='High scenario')

nonzero = cat[cat['loss_mid'] > 0].copy()
nonzero_sorted = np.sort(nonzero['loss_mid'].values)[::-1]
aep_events = np.array([
    len(np.unique(cat['year'].values[losses_mid > L])) / N_YEARS
    for L in nonzero_sorted
])
ax.scatter(nonzero_sorted / 1e6, aep_events,
           color='steelblue', s=30, zorder=5, alpha=0.8, label='Individual storms')

top = nonzero.nlargest(5, 'loss_mid')
offsets = [(5, 28), (5, 28), (-5, 28), (5, 50), (-55, 32)]
for i, (_, row) in enumerate(top.iterrows()):
    L = row['loss_mid']
    aep_val = len(np.unique(cat['year'].values[losses_mid > L])) / N_YEARS
    label = f"{row['name']} ({int(row['year'])})"
    dx, dy = offsets[i]
    ax.annotate(label,
                xy=(L / 1e6, aep_val),
                xytext=(dx, dy), textcoords='offset points',
                fontsize=9, color='#1a1a1a', va='bottom',
                arrowprops=dict(arrowstyle='-', color='#777777', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9, lw=0.7))

ax.set_xlabel('Loss (USD millions)', fontsize=11)
ax.set_ylabel('Annual exceedance probability', fontsize=11)

ax.set_xlim(left=0)
ax.set_ylim(0, 1)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.legend(fontsize=9, framealpha=0.9)

fig.tight_layout()
out_path = OUT_DIR / 'aep_curve.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved → {out_path}")

aep_total = pd.DataFrame({
    'return_period_yr': np.where(aep_mid > 0, 1 / aep_mid, np.inf),
    'aep_pct':          aep_mid * 100,
    'loss_low_usd':     L_grid * (aep_low / aep_mid.clip(1e-9)),
    'loss_mid_usd':     L_grid,
    'loss_high_usd':    L_grid * (aep_high / aep_mid.clip(1e-9)),
})
aep_total = aep_total[aep_total['aep_pct'] > 0].drop_duplicates('aep_pct')
aep_total.to_csv(OUT_DIR / 'aep_caribbean_total.csv', index=False)
print(f"Saved → {OUT_DIR / 'aep_caribbean_total.csv'}")

country_path = BASE_DIR / 'outputs/losses/loss_by_country_storm.csv'
if country_path.exists():
    cdf = pd.read_csv(country_path)
    cdf['year'] = cdf['storm_id'].str[:4].astype(int)
    n_years_cs = cdf['year'].nunique()
    rows = []
    for country, grp in cdf.groupby('country'):
        for scenario, col in [('low', 'loss_low'), ('mid', 'loss_mid'), ('high', 'loss_high')]:
            losses_c = grp[col].values
            L_max_c  = losses_c.max() * 1.05 if losses_c.max() > 0 else 1
            L_grid_c = np.linspace(0, L_max_c, 90)
            for L in L_grid_c:
                years_exc = np.unique(grp.loc[grp[col] > L, 'year'].values)
                aep_val   = len(years_exc) / n_years_cs if n_years_cs > 0 else 0
                rp        = 1 / aep_val if aep_val > 0 else np.inf
                rows.append({'country': country, 'scenario': scenario,
                             'return_period_yr': rp, 'aep_pct': aep_val * 100,
                             'loss_usd': L})
    aep_country = pd.DataFrame(rows)
    aep_country.to_csv(OUT_DIR / 'aep_by_country.csv', index=False)
    print(f"Saved → {OUT_DIR / 'aep_by_country.csv'}")
