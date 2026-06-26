import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path

BASE_DIR = Path('/home/jupyter-daniela/CAF')
SHP_PATH = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
T52_PATH = BASE_DIR / 'outputs/losses/table_5_2_wind_cumulative_by_country.csv'
CAT_PATH = BASE_DIR / 'outputs/losses/historical_loss_catalog.csv'
OUT_FIG  = BASE_DIR / 'outputs/losses'
OUT_LOSS = BASE_DIR / 'outputs/losses'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

print("=" * 60)
print("fig_wind_losses.py")
print("=" * 60)

t52 = pd.read_csv(T52_PATH)
cat = pd.read_csv(CAT_PATH)

SS_COLORS = {
    'TD':    '#adb5bd',
    'Cat 1': '#4dabf7',
    'Cat 2': '#74c476',
    'Cat 3': '#fd8d3c',
    'Cat 4': '#e6550d',
    'Cat 5': '#a50f15',
}
SS_LABEL_MAP = {-1: 'TD', 0: 'TD', 1: 'Cat 1', 2: 'Cat 2',
                 3: 'Cat 3', 4: 'Cat 4', 5: 'Cat 5'}
cat['ss_cat'] = cat['category'].map(SS_LABEL_MAP).fillna('TD')

print("\n[Fig 1] Map — cumulative wind losses by country …")

gdf = gpd.read_file(SHP_PATH).to_crs('EPSG:4326')
gdf_country = gdf.dissolve(by='country').reset_index()[['country', 'geometry']]
gdf_country = gdf_country.merge(t52[['country', 'total_loss_mid', 'pct_regional']],
                                 on='country', how='left')
gdf_country['loss_M'] = gdf_country['total_loss_mid'] / 1e6

gdf_country['loss_M'] = gdf_country['loss_M'].fillna(0)

has_loss = gdf_country['loss_M'] > 0
vmax     = gdf_country.loc[has_loss, 'loss_M'].max()
norm     = mcolors.LogNorm(vmin=0.01, vmax=vmax)
cmap     = plt.cm.YlOrRd

fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_facecolor('#cfe2f3')

if (~has_loss).any():
    gdf_country[~has_loss].plot(ax=ax, color='#e8e8e8',
                                edgecolor='#888888', linewidth=0.5)
if has_loss.any():
    gdf_country[has_loss].plot(ax=ax, column='loss_M', cmap=cmap, norm=norm,
                               edgecolor='#555555', linewidth=0.5)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.52, pad=0.02, aspect=18)
cb.set_label('Cumulative wind loss — mid scenario (USD millions)', fontsize=9)
cb.ax.tick_params(labelsize=8)

label_coords = {
    'Cuba':               (-79.5, 22.0),
    'Saint Lucia':        (-60.98, 13.9),
    'Jamaica':            (-77.3, 18.1),
    'Dominican Republic': (-70.2, 18.9),
    'Barbados':           (-59.55, 13.2),
    'Haiti':              (-72.6, 19.0),
}
for _, row in gdf_country[has_loss].iterrows():
    coord = label_coords.get(row['country'])
    if coord:
        ax.annotate(
            f"{row['country']}\n${row['loss_M']:.0f}M ({row['pct_regional']:.1f}%)",
            xy=coord, fontsize=7.5, ha='center', va='center',
            color='#111111', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='none', alpha=0.75))

ax.set_xlim(-85, -58)
ax.set_ylim(10, 28)
ax.set_xlabel('Longitude', fontsize=9)
ax.set_ylabel('Latitude', fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.25, linewidth=0.3)
fig.text(0.5, 0.01, 'Source: IBTrACS v04r01, OSM, own elaboration. CAF project.',
         ha='center', fontsize=7.5, color='#666666')
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(OUT_FIG / 'map_wind_losses_by_country.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: map_wind_losses_by_country.png")

print("\n[Fig 2] Top 10 storms barplot …")

top10 = (cat[cat['loss_mid'] > 0]
         .nlargest(10, 'loss_mid')
         .reset_index(drop=True))
top10['label']    = top10['name'].str.title() + '\n(' + top10['year'].astype(str) + ')'
top10['loss_M']   = top10['loss_mid']   / 1e6
top10['low_M']    = top10['loss_low']   / 1e6
top10['high_M']   = top10['loss_high']  / 1e6
top10['bar_color'] = top10['ss_cat'].map(SS_COLORS)

top10 = top10.sort_values('loss_M', ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 6))
y = np.arange(len(top10))

ax.barh(y, top10['high_M'] - top10['low_M'],
        left=top10['low_M'],
        color=top10['bar_color'], alpha=0.25, height=0.55)
bars = ax.barh(y, top10['loss_M'], color=top10['bar_color'],
               alpha=0.9, height=0.55, edgecolor='white', linewidth=0.4)

for i, row in top10.iterrows():
    ax.text(row['loss_M'] + 0.4, i,
            f"${row['loss_M']:.1f}M", va='center', fontsize=8.5, color='#222222')

ax.set_yticks(y)
ax.set_yticklabels(top10['label'], fontsize=9)
ax.set_xlabel('Wind loss — mid scenario (USD millions)', fontsize=10)
ax.set_xlim(0, top10['high_M'].max() * 1.18)
ax.grid(axis='x', alpha=0.3, linewidth=0.4, zorder=0)

legend_cats = sorted(set(top10['ss_cat'].tolist()), key=lambda x: list(SS_COLORS).index(x))
handles = [mpatches.Patch(color=SS_COLORS[c], label=c, alpha=0.9) for c in legend_cats]
ax.legend(handles=handles, title='Saffir-Simpson', title_fontsize=8.5,
          fontsize=8.5, loc='lower right', framealpha=0.95)

fig.text(0.5, -0.01, 'Source: IBTrACS v04r01, OSM, own elaboration. CAF project.',
         ha='center', fontsize=7.5, color='#666666')
fig.tight_layout()
fig.savefig(OUT_FIG / 'barplot_top10_storms.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: barplot_top10_storms.png")

print("\n[Table] LaTeX cumulative losses table …")

def fmt_usd(x):
    if x >= 1e6:
        return f'\\${x/1e6:.1f}M'
    if x >= 1e3:
        return f'\\${x/1e3:.0f}k'
    return f'\\${x:.0f}'

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Cumulative wind losses 1980--2025 by country (mid scenario). '
    r'Present-day OSM infrastructure exposure. '
    r'Losses represent what each historical storm would cost given today\'s '
    r'infrastructure stock (retrospective re-analysis).}',
    r'\label{tab:wind_losses_country}',
    r'\small',
    r'\begin{tabular}{lrrrrl}',
    r'\toprule',
    r'Country & Storms & Loss low & Loss mid & Loss high & \% regional \\',
    r'\midrule',
]
for _, row in t52.iterrows():
    lines.append(
        f"{row['country']} & {int(row['n_storms'])} & "
        f"{fmt_usd(row['total_loss_low'])} & "
        f"{fmt_usd(row['total_loss_mid'])} & "
        f"{fmt_usd(row['total_loss_high'])} & "
        f"{row['pct_regional']:.1f}\\% \\\\"
    )
lines += [
    r'\midrule',
    f"Total & --- & "
    f"{fmt_usd(t52['total_loss_low'].sum())} & "
    f"{fmt_usd(t52['total_loss_mid'].sum())} & "
    f"{fmt_usd(t52['total_loss_high'].sum())} & "
    r"100.0\% \\",
    r'\bottomrule',
    r'\multicolumn{6}{l}{\footnotesize Low/mid/high: reconstruction cost scenarios. '
    r'Cyclone intensity: CLIMADA H08. Vulnerability: physrisk sigmoid (Sealy \& Strobl 2017).} \\',
    r'\end{tabular}',
    r'\end{table}',
]
with open(OUT_LOSS / 'table_5_2_wind_cumulative_by_country.tex', 'w') as f:
    f.write('\n'.join(lines))
print("  Saved: table_5_2_wind_cumulative_by_country.tex")

print(f"\nDone.")
print(f"  AEP curve already at: {OUT_FIG}/aep_curve.png")
