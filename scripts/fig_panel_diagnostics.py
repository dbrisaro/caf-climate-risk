import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import logging
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

BASE_DIR  = Path('/home/jupyter-daniela/CAF')
DATA_PATH = BASE_DIR / 'data/processed/caribbean_adm_boundaries/master_panel.parquet'
OUT_DIR   = BASE_DIR / 'outputs/regression'
OUT_DIR.mkdir(parents=True, exist_ok=True)

HAZARDS = ['drought', 'rainfall', 'heatwave', 'cyclone']
COLORS  = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']

if not DATA_PATH.exists():
    raise FileNotFoundError(f"master_panel.parquet not found: {DATA_PATH}")

print("=" * 62)
print("inspect_master_panel — Panel diagnostics")
print("=" * 62)

df = pd.read_parquet(DATA_PATH)
print(f"Shape:   {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Years:   {sorted(df['year'].unique())}")
print(f"Windows: {df['window'].unique().tolist()}")
print(f"Regions: {df['region_name'].nunique()}")
print()

coverage = pd.DataFrame({
    'non_zero_%': [(df[h] > 0).mean() * 100 for h in HAZARDS],
    'max':        [df[h].max() for h in HAZARDS],
    'mean_nonzero': [df.loc[df[h] > 0, h].mean() for h in HAZARDS],
}, index=HAZARDS)
print("Hazard coverage:")
print(coverage.round(3))
print()

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for i, (ax, h, c) in enumerate(zip(axes.flat, HAZARDS, COLORS)):
    vals = df.loc[df[h] > 0, h]
    ax.hist(vals, bins=40, color=c, edgecolor='white', linewidth=0.5)
    ax.set_title(f"{chr(ord('a')+i)}  {h.capitalize()}  (non-zero, n={len(vals):,})", loc='left', fontweight='normal', fontsize=10)
    ax.set_xlabel('value')
plt.tight_layout()
out = OUT_DIR / 'fig_panel_hazard_distributions.png'
fig.savefig(out, dpi=150)
plt.close(fig)
log.info("Saved: %s", out)

exp = df[df['window'] == 'DJF'][['region_name', 'year', 'exposure_index']].copy()
trend = exp.groupby('year')['exposure_index'].agg(['mean', 'median', 'std']).reset_index()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(trend['year'], trend['mean'],   label='mean',   color='steelblue')
ax.plot(trend['year'], trend['median'], label='median', color='darkorange', linestyle='--')
ax.fill_between(trend['year'],
                trend['mean'] - trend['std'],
                trend['mean'] + trend['std'],
                alpha=0.2, color='steelblue')
ax.set_xlabel('Year')
ax.set_ylabel('Exposure index')
ax.legend(fontsize=9)
plt.tight_layout()
out = OUT_DIR / 'fig_panel_exposure_trend.png'
fig.savefig(out, dpi=150)
plt.close(fig)
log.info("Saved: %s", out)

haz_trend = df.groupby(['year', 'window'])[HAZARDS].mean().reset_index()
colors_w  = {'DJF': '#1f77b4', 'MAM': '#2ca02c', 'JJA': '#ff7f0e', 'SON': '#d62728'}

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for i, (ax, h) in enumerate(zip(axes.flat, HAZARDS)):
    for win, grp in haz_trend.groupby('window'):
        ax.plot(grp['year'], grp[h], label=win, color=colors_w[win])
    ax.set_title(f"{chr(ord('a')+i)}  {h.capitalize()}", loc='left', fontweight='normal', fontsize=10)
    ax.set_xlabel('Year')
    ax.legend(fontsize=7)
plt.tight_layout()
out = OUT_DIR / 'fig_panel_hazard_trends.png'
fig.savefig(out, dpi=150)
plt.close(fig)
log.info("Saved: %s", out)

exp_latest = exp[exp['year'] == exp['year'].max()].sort_values('exposure_index', ascending=False).head(25)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(exp_latest['region_name'][::-1], exp_latest['exposure_index'][::-1], color='steelblue')
ax.set_xlabel('Exposure index')
ax.tick_params(axis='y', labelsize=8)
plt.tight_layout()
out = OUT_DIR / 'fig_panel_top_regions.png'
fig.savefig(out, dpi=150)
plt.close(fig)
log.info("Saved: %s", out)

cols = ['exposure_index', 'pop', 'built_s', 'built_v', 'dem'] + HAZARDS
corr = df[cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(cols, fontsize=9)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7)
plt.tight_layout()
out = OUT_DIR / 'fig_panel_correlation_matrix.png'
fig.savefig(out, dpi=150)
plt.close(fig)
log.info("Saved: %s", out)

print()
print("Done.  Outputs saved to outputs/regression/")
