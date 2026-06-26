import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

BASE_DIR = Path('/home/jupyter-daniela/CAF')
EP_PATH  = BASE_DIR / 'data/processed/caribbean_adm_boundaries/exposure_panel.parquet'
SHP_PATH = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
OUT_DIR  = BASE_DIR / 'outputs/exposure'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS   = [1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]
VAR_COLS = ['pop', 'built_s', 'built_v']
VAR_LABELS = {'pop': 'Population', 'built_s': 'Built surface', 'built_v': 'Built volume'}
VAR_COLORS = {'pop': '#4e79a7', 'built_s': '#f28e2b', 'built_v': '#59a14f'}

COUNTRY_ORDER = [
    'Dominican Republic', 'Haiti', 'Jamaica', 'Cuba',
    'Bahamas', 'Barbados', 'Dominica', 'Saint Lucia',
    'Trinidad and Tobago',
    'Colombia', 'Peru', 'Guyana',
]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

print("=" * 60)
print("fig_exposure.py — Exposure chapter figures")
print("=" * 60)

ep  = pd.read_parquet(EP_PATH)
shp = gpd.read_file(SHP_PATH)[['region_nam', 'country', 'geometry']].rename(
        columns={'region_nam': 'region_name'})

ep  = ep.merge(shp[['region_name', 'country']], on='region_name', how='left')
gdf_base = shp.copy()

countries = sorted(ep['country'].dropna().unique())
print(f"  {len(ep):,} obs · {ep['region_name'].nunique()} regions · "
      f"{len(countries)} countries · {ep['year'].nunique()} epochs")

scaler = StandardScaler()
X_all  = scaler.fit_transform(ep[VAR_COLS].values)
pca    = PCA(n_components=1)
pca.fit(X_all)
loadings = pca.components_[0]
print(f"\n  PC1 loadings — {dict(zip(VAR_COLS, loadings.round(4)))}")
print(f"  Explained variance: {pca.explained_variance_ratio_[0]:.3f}")

ep[['z_pop', 'z_built_s', 'z_built_v']] = X_all

for j, v in enumerate(VAR_COLS):
    ep[f'contrib_{v}'] = loadings[j] * ep[f'z_{v}']

ep['contrib_abs_sum'] = (ep[[f'contrib_{v}' for v in VAR_COLS]].abs().sum(axis=1)
                         .replace(0, np.nan))
for v in VAR_COLS:
    ep[f'share_{v}'] = ep[f'contrib_{v}'].abs() / ep['contrib_abs_sum']

def wmean(df, val_col, weight_col='pop'):
    w = df[weight_col].fillna(0)
    if w.sum() == 0:
        return df[val_col].mean()
    return np.average(df[val_col].fillna(0), weights=w)

def gini(arr):
    arr = np.sort(arr[~np.isnan(arr)])
    n = len(arr)
    if n < 2 or arr.sum() == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    return (2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n

rows = []
for country in countries:
    cdf = ep[ep['country'] == country]
    idx_1995 = wmean(cdf[cdf['year'] == 1995], 'exposure_index')
    idx_2030 = wmean(cdf[cdf['year'] == 2030], 'exposure_index')
    cagr = ((idx_2030 / idx_1995) ** (1 / 35) - 1) * 100 if idx_1995 > 0 else np.nan

    for yr in EPOCHS:
        ydf = cdf[cdf['year'] == yr]
        rows.append({
            'country':       country,
            'year':          yr,
            'mean_index':    wmean(ydf, 'exposure_index'),
            'gini':          gini(ydf['exposure_index'].values),
            'cagr_to_2030':  cagr,
            'share_pop':     wmean(ydf, 'share_pop'),
            'share_built_s': wmean(ydf, 'share_built_s'),
            'share_built_v': wmean(ydf, 'share_built_v'),
        })

df_summary = pd.DataFrame(rows).round(6)
df_summary.to_csv(OUT_DIR / 'exposure_summary_stats.csv', index=False)
print("\n  Saved: exposure_summary_stats.csv")

print("\n[Fig 1] Choropleth 1995→2030 …")

delta = (ep[ep['year'] == 2030][['region_name', 'exposure_index']]
         .rename(columns={'exposure_index': 'idx_2030'})
         .merge(ep[ep['year'] == 1995][['region_name', 'exposure_index']]
                .rename(columns={'exposure_index': 'idx_1995'}),
                on='region_name'))
delta['change'] = delta['idx_2030'] - delta['idx_1995']

gdf_map = gdf_base.merge(delta[['region_name', 'change']], on='region_name', how='left')
gdf_map = gdf_map.to_crs('EPSG:4326')

vmax = gdf_map['change'].abs().quantile(0.97)
vmin = -vmax
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = plt.cm.RdBu_r

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_facecolor('#d8eaf5')
gdf_map.plot(ax=ax, column='change', cmap=cmap, norm=norm,
             edgecolor='#555555', linewidth=0.25, missing_kwds={'color': '#cccccc'})

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, aspect=20)
cb.set_label('Δ Exposure index (2030 − 1995)', fontsize=9)
cb.ax.tick_params(labelsize=8)

ax.set_xlim(-85, -55)
ax.set_ylim(-20, 30)
ax.set_xlabel('Longitude', fontsize=9)
ax.set_ylabel('Latitude', fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3, linewidth=0.3)
fig.text(0.5, 0.01, 'Source: GHSL 2023, own elaboration · CAF project',
         ha='center', fontsize=7, color='#888888')
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(OUT_DIR / 'map_change_1995_2030.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: map_change_1995_2030.png")

print("\n[Fig 2] PC1 decomposition barplot …")

decomp_rows = []
for country in countries:
    for yr in EPOCHS:
        ydf = ep[(ep['country'] == country) & (ep['year'] == yr)]
        decomp_rows.append({
            'country':       country,
            'year':          yr,
            'share_pop':     wmean(ydf, 'share_pop'),
            'share_built_s': wmean(ydf, 'share_built_s'),
            'share_built_v': wmean(ydf, 'share_built_v'),
        })
decomp = pd.DataFrame(decomp_rows)

n_countries = len(countries)
ncols = 4
nrows = int(np.ceil(n_countries / ncols))

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 3.5, nrows * 3.2),
                         sharey=True)
axes = axes.ravel()

x = np.arange(len(EPOCHS))
width = 0.75

for i, country in enumerate(countries):
    ax = axes[i]
    cdf = decomp[decomp['country'] == country].set_index('year').reindex(EPOCHS)

    bottoms = np.zeros(len(EPOCHS))
    for v in VAR_COLS:
        vals = cdf[f'share_{v}'].fillna(0).values * 100
        ax.bar(x, vals, width, bottom=bottoms,
               color=VAR_COLORS[v], label=VAR_LABELS[v], alpha=0.9)
        bottoms += vals

    ax.set_title(f"{chr(ord('a')+i)}  {country}", loc='left', fontweight='normal', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y)[2:] for y in EPOCHS], fontsize=7, rotation=0)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.tick_params(axis='y', labelsize=7)
    ax.set_xlabel("Epoch", fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

for r in range(nrows):
    axes[r * ncols].set_ylabel('Share of PC1 score (%)', fontsize=8)

legend_handles = [mpatches.Patch(color=VAR_COLORS[v], label=VAR_LABELS[v])
                  for v in VAR_COLS]
fig.legend(handles=legend_handles, loc='lower right',
           bbox_to_anchor=(0.98, 0.02), fontsize=9,
           framealpha=0.95, title='Variable', title_fontsize=9)

fig.tight_layout()
fig.savefig(OUT_DIR / 'decomposition_by_country.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: decomposition_by_country.png")

print("\n[Fig 3] Scatter level vs CAGR …")

nat_rows = []
for country in countries:
    for yr in [1995, 2030]:
        ydf = ep[(ep['country'] == country) & (ep['year'] == yr)]
        nat_rows.append({'country': country, 'year': yr,
                         'exposure_index': wmean(ydf, 'exposure_index')})
nat   = pd.DataFrame(nat_rows)
idx95 = nat[nat['year'] == 1995].set_index('country')['exposure_index']
idx30 = nat[nat['year'] == 2030].set_index('country')['exposure_index']
cagr  = ((idx30 / idx95) ** (1 / 35) - 1) * 100

scatter_df = pd.DataFrame({
    'country':   idx95.index,
    'idx_1995':  idx95.values,
    'cagr':      cagr.values,
}).dropna()

LABEL_OFFSETS = {
    'Dominican Republic':    ( 0.002, -0.25),
    'Haiti':                 ( 0.001,  0.08),
    'Jamaica':               ( 0.001, -0.25),
    'Bahamas':               ( 0.001,  0.08),
    'Barbados':              ( 0.001,  0.08),
    'Dominica':              (-0.006,  0.08),
    'Saint Lucia':           ( 0.001, -0.25),
    'Trinidad and Tobago':   ( 0.001,  0.08),
    'Colombia':              ( 0.002,  0.05),
    'Peru':                  ( 0.002, -0.25),
    'Guyana':                ( 0.001,  0.08),
}

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.scatter(scatter_df['idx_1995'], scatter_df['cagr'],
           color='#377eb8', s=80, zorder=5, edgecolors='white', linewidths=0.7)

for _, row in scatter_df.iterrows():
    dx, dy = LABEL_OFFSETS.get(row['country'], (0.001, 0.05))
    ax.annotate(row['country'],
                xy=(row['idx_1995'], row['cagr']),
                xytext=(row['idx_1995'] + dx, row['cagr'] + dy),
                fontsize=8, color='#222222',
                arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.6))

x_arr = scatter_df['idx_1995'].values
y_arr = scatter_df['cagr'].values
if len(x_arr) > 2:
    m, b = np.polyfit(x_arr, y_arr, 1)
    x_line = np.linspace(x_arr.min() * 0.9, x_arr.max() * 1.05, 100)
    ax.plot(x_line, m * x_line + b, color='#e41a1c', lw=1.4,
            linestyle='--', alpha=0.8, label=f'OLS trend  (β={m:.1f})')
    ax.legend(fontsize=9)

ax.set_xlabel('Mean national exposure index in 1995', fontsize=10)
ax.set_ylabel('CAGR 1995–2030 (%/year)', fontsize=10)
ax.grid(True, alpha=0.3, linewidth=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / 'scatter_level_vs_growth.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: scatter_level_vs_growth.png")

print("\n[Fig 4] Gini lineplot …")

gini_rows = []
for country in countries:
    for yr in EPOCHS:
        vals = ep[(ep['country'] == country) & (ep['year'] == yr)]['exposure_index'].values
        gini_rows.append({'country': country, 'year': yr, 'gini': gini(vals)})
df_gini = pd.DataFrame(gini_rows)

cmap_tab = plt.cm.tab10
country_colors = {c: cmap_tab(i / max(len(countries) - 1, 1))
                  for i, c in enumerate(countries)}

linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
ls_map = {c: linestyles[i % len(linestyles)] for i, c in enumerate(countries)}

fig, ax = plt.subplots(figsize=(10, 5.5))

for country in countries:
    cdf = df_gini[df_gini['country'] == country].sort_values('year')
    ax.plot(cdf['year'], cdf['gini'],
            color=country_colors[country],
            linestyle=ls_map[country],
            marker='o', markersize=5, lw=1.8,
            label=country)

ax.set_xlabel('Epoch', fontsize=10)
ax.set_ylabel('Gini coefficient of exposure index', fontsize=10)
ax.set_xticks(EPOCHS)
ax.set_xticklabels(EPOCHS, rotation=30, ha='right', fontsize=8)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, linewidth=0.4)
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1),
          framealpha=0.95, title='Country', title_fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / 'gini_by_country.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: gini_by_country.png")

print("\n[Fig 5] Temporal evolution by country …")

cmap_tab5  = plt.cm.tab10
color_map5 = {c: cmap_tab5(i / max(len(countries) - 1, 1)) for i, c in enumerate(countries)}
ls_map5    = {c: ['-', '--', '-.', ':'][i % 4] for i, c in enumerate(countries)}

fig, ax = plt.subplots(figsize=(13, 5.5))

for country in countries:
    cdf = df_summary[df_summary['country'] == country].set_index('year').reindex(EPOCHS)
    ax.plot(EPOCHS, cdf['mean_index'],
            color=color_map5[country],
            linestyle=ls_map5[country],
            marker='o', markersize=4, linewidth=1.8,
            label=country)

ax.set_xticks(EPOCHS)
ax.set_xticklabels(EPOCHS, rotation=30, ha='right', fontsize=9)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Mean national exposure index', fontsize=11)
ax.legend(fontsize=9, loc='upper left', frameon=False)
ax.grid(True, alpha=0.3, linewidth=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / 'temporal_evolution_by_country.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: temporal_evolution_by_country.png")

print()
print("Summary stats (1995 & 2030, national means):")
print(df_summary[df_summary['year'].isin([1995, 2030])]
      [['country', 'year', 'mean_index', 'gini', 'cagr_to_2030']]
      .to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print(f"\nAll figures → {OUT_DIR}/")
