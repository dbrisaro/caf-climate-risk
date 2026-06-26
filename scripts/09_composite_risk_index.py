import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

BASE_DIR      = Path('/home/jupyter-daniela/CAF')
EXPOSURE_PATH = BASE_DIR / 'data/processed/caribbean_adm_boundaries/exposure_panel.parquet'
MASTER_PATH   = BASE_DIR / 'data/processed/caribbean_adm_boundaries/master_panel.parquet'
GDP_PATH      = BASE_DIR / 'data/pib per capita.xltx'
OUT_DIR       = BASE_DIR / 'outputs/vulnerability'
OUT_DIR.mkdir(parents=True, exist_ok=True)

HAZARDS  = ['cyclone', 'rainfall', 'drought', 'heatwave']
FEATURES = ['pop', 'built_s', 'built_v', 'dem', 'gdp_pc',
            'cyclone', 'rainfall', 'drought', 'heatwave']

print("=" * 62)
print("Step 09 — Composite risk index (PCA)")
print("=" * 62)

exp = pd.read_parquet(EXPOSURE_PATH).rename(columns={'region_name': 'country'})

mp  = pd.read_parquet(MASTER_PATH)
haz = (mp.groupby(['region_name', 'year'])[HAZARDS]
         .sum()
         .reset_index()
         .rename(columns={'region_name': 'country'}))

gdp      = pd.read_excel(GDP_PATH)
gdp      = gdp.rename(columns={'St. Lucia': 'Saint Lucia'})
gdp_long = (gdp.melt(id_vars='Year', var_name='country', value_name='gdp_pc')
               .rename(columns={'Year': 'year'}))

df = exp.merge(haz, on=['country', 'year'], how='inner')
df = df.merge(gdp_long, on=['country', 'year'], how='inner')
log.info("Panel: %d rows | %d countries | years %s",
         len(df), df['country'].nunique(), sorted(df['year'].unique()))

X  = np.log1p(df[FEATURES].values)
Xn = StandardScaler().fit_transform(X)

pca_full = PCA(random_state=42)
pca_full.fit(Xn)
log.info("Variance explained — PC1: %.1f%%  PC1+PC2: %.1f%%",
         pca_full.explained_variance_ratio_[0] * 100,
         pca_full.explained_variance_ratio_[:2].sum() * 100)

pca = PCA(n_components=1, random_state=42)
pc1 = pca.fit_transform(Xn)[:, 0]
df['risk_index'] = (pc1 - pc1.min()) / (pc1.max() - pc1.min())

log.info("PC1 loadings:")
for f, l in zip(FEATURES, pca.components_[0]):
    log.info("  %-12s  %+.3f", f, l)

countries = sorted(df['country'].unique())
colors    = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
             '#9467bd', '#8c564b', '#e377c2', '#17becf']

fig, ax = plt.subplots(figsize=(11, 6))
lines = []
for i, country in enumerate(countries):
    sub  = df[df['country'] == country].sort_values('year')
    line, = ax.plot(sub['year'], sub['risk_index'],
                    marker='o', markersize=3, linewidth=1,
                    color=colors[i % len(colors)], label=country)
    lines.append(line)

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Composite risk index [0–1]', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6)
ax.set_xlim(df['year'].min() - 1, df['year'].max() + 1)
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
fig.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, 1.06),
           ncol=4, fontsize=8.5, frameon=False)
fig.suptitle('Composite climate risk index by country\n'
             '(PCA: urban exposure + GDP pc + hazards)',
             fontsize=12, y=1.13)
plt.tight_layout()
out1 = OUT_DIR / 'composite_risk_index_by_country.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
log.info("Saved: %s", out1)

loadings    = pd.Series(pca.components_[0], index=FEATURES).sort_values()
colors_load = ['#d62728' if v > 0 else '#1f77b4' for v in loadings]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(loadings.index, loadings.values, color=colors_load, alpha=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Loading (PC1)', fontsize=10)
ax.set_title('Variable contributions to composite risk index (PC1)', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
for bar, val in zip(bars, loadings.values):
    ax.text(val + 0.005 * np.sign(val), bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}', va='center', fontsize=8)
plt.tight_layout()
out2 = OUT_DIR / 'composite_risk_index_loadings.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
log.info("Saved: %s", out2)

print()
print("Done.  Outputs:")
print(f"  {out1}")
print(f"  {out2}")
