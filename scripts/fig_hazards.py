import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path

BASE_DIR  = Path('/home/jupyter-daniela/CAF')
SHP_PATH  = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
PROC      = BASE_DIR / 'data/processed/caribbean_adm_boundaries'
DR_AGG   = PROC / 'era5_output/droughts_aggregated_data.parquet'
RR_AGG   = PROC / 'era5_output_precipitation/rainfall_aggregated_data.parquet'
HW_AGG   = PROC / 'heatwaves_computation/heatwaves_aggregated_data.parquet'
OUT_DIR  = BASE_DIR / 'outputs/hazards'
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.spines.left':  True,
    'axes.spines.bottom': True,
})
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

YEARS_PLOT = list(range(2010, 2026))
shp = gpd.read_file(SHP_PATH)[['region_nam', 'country']].rename(
        columns={'region_nam': 'region_name'})
COUNTRIES = sorted(shp['country'].unique())

print("=" * 60)
print("fig_hazards.py — Section 2.2 hazard figures (bar charts)")
print("=" * 60)

def faceted_bar(frac_df, year_col, val_col, bar_color, ref_line,
                y_label, suptitle, footnote, out_path):
    ncols = 4
    nrows = int(np.ceil(len(COUNTRIES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.2),
                             sharey=True, sharex=False)
    axes = axes.ravel()
    x    = np.arange(len(YEARS_PLOT))

    y_max_all = frac_df[val_col].max()
    y_ceil    = min(max(y_max_all * 1.18, ref_line * 1.5), 1.0)

    for i, country in enumerate(COUNTRIES):
        ax  = axes[i]
        cdf = (frac_df[frac_df['country'] == country]
               .set_index(year_col)[val_col]
               .reindex(YEARS_PLOT)
               .fillna(0))

        ax.bar(x, cdf.values, color=bar_color, alpha=0.85, width=0.72,
               edgecolor='white', linewidth=0.4)
        ax.axhline(ref_line, color='#888888', lw=0.9, linestyle='--', zorder=3)

        ax.set_title(f"{chr(ord('a')+i)}  {country}", loc='left', fontweight='normal', fontsize=9, pad=4)
        ax.set_xticks(x[::3])
        ax.set_xticklabels(
            [str(y) for y in YEARS_PLOT[::3]],
            fontsize=7, rotation=45, ha='right')
        ax.set_xlim(-0.6, len(YEARS_PLOT) - 0.4)
        ax.set_ylim(0, y_ceil)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='y', alpha=0.25, linewidth=0.4, zorder=0)

    for j in range(len(COUNTRIES), nrows * ncols):
        axes[j].set_visible(False)

    for r in range(nrows):
        axes[r * ncols].set_ylabel(y_label, fontsize=8)

    ref_patch = mpatches.Patch(color='#888888', label=f'Expected ({ref_line:.0%})',
                               alpha=0.6)
    bar_patch  = mpatches.Patch(color=bar_color, alpha=0.85, label='Observed')
    fig.legend(handles=[bar_patch, ref_patch],
               loc='lower right', bbox_to_anchor=(0.99, 0.01),
               fontsize=8, framealpha=0.95)

    fig.text(0.5, -0.01, footnote, ha='center', fontsize=7.5, color='#666666')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

print("\n  Computing drought fractions from aggregated data …")
dr_agg = pd.read_parquet(DR_AGG)
dr_agg = dr_agg.merge(shp, on='region_name', how='left')

total_ryw = (dr_agg[dr_agg['activacion'] == 'p10']
             .groupby(['country', 'year', 'window'])
             .size().reset_index(name='n_obs'))
total_country_yr = total_ryw.groupby(['country', 'year'])['n_obs'].sum().reset_index()

dr_fracs = {}
for q_label, act in [('p10', 'p10'), ('p5', 'p5'), ('p1', 'p1')]:
    hit = (dr_agg[(dr_agg['activacion'] == act) & (dr_agg['n_drought_pixels'] > 0)]
           .groupby(['country', 'year'])
           .size().reset_index(name='hit'))
    frac = total_country_yr.merge(hit, on=['country', 'year'], how='left').fillna({'hit': 0})
    frac['frac'] = frac['hit'] / frac['n_obs'].replace(0, np.nan)
    dr_fracs[q_label] = frac
    print(f"    {q_label} done")

print("\n[Fig 1] Drought fraction by country …")
faceted_bar(
    frac_df   = dr_fracs['p10'][dr_fracs['p10']['year'].between(2010, 2025)],
    year_col  = 'year',
    val_col   = 'frac',
    bar_color = '#8B5E3C',
    ref_line  = 0.10,
    y_label   = 'Fraction of sub-regions with drought event',
    suptitle  = ('Extreme drought: fraction of sub-regions affected (2010–2025)\n'
                 'ERA5 SWVL1 below decile 10 (region × season)'),
    footnote  = 'Source: ERA5 reanalysis, own elaboration. CAF project.',
    out_path  = OUT_DIR / 'drought_fraction_by_country.png',
)

print("\n[Fig 2] Drought severity — Dominican Republic (stacked bars) …")

DR_COUNTRY = 'Dominican Republic'
frac_p10 = (dr_fracs['p10'][dr_fracs['p10']['country'] == DR_COUNTRY]
            .set_index('year')['frac'].reindex(YEARS_PLOT).fillna(0))
frac_p5  = (dr_fracs['p5'][dr_fracs['p5']['country'] == DR_COUNTRY]
            .set_index('year')['frac'].reindex(YEARS_PLOT).fillna(0))
frac_p1  = (dr_fracs['p1'][dr_fracs['p1']['country'] == DR_COUNTRY]
            .set_index('year')['frac'].reindex(YEARS_PLOT).fillna(0))

seg_p1     = frac_p1.values
seg_p5_p1  = (frac_p5 - frac_p1).clip(lower=0).values
seg_p10_p5 = (frac_p10 - frac_p5).clip(lower=0).values

x = np.arange(len(YEARS_PLOT))
COLORS_SEV = ['#5C1A00', '#A0522D', '#D4956A']

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x, seg_p1,    color=COLORS_SEV[0], alpha=0.9, width=0.72,
       label='Below p1 (extreme)',  edgecolor='white', linewidth=0.3)
ax.bar(x, seg_p5_p1, bottom=seg_p1,
       color=COLORS_SEV[1], alpha=0.9, width=0.72,
       label='p1 – p5 (severe)',    edgecolor='white', linewidth=0.3)
ax.bar(x, seg_p10_p5, bottom=seg_p1 + seg_p5_p1,
       color=COLORS_SEV[2], alpha=0.9, width=0.72,
       label='p5 – p10 (moderate)', edgecolor='white', linewidth=0.3)
ax.axhline(0.10, color='#888888', lw=0.9, linestyle='--', alpha=0.8, label='p10 expected (10%)')
ax.set_xticks(x)
ax.set_xticklabels(YEARS_PLOT, fontsize=8, rotation=45, ha='right')
ax.set_xlim(-0.6, len(YEARS_PLOT) - 0.4)
ax.set_ylim(0, max(frac_p10.max() * 1.2, 0.15))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.set_ylabel('Fraction of sub-regions affected', fontsize=10)
ax.legend(fontsize=8.5, loc='upper left', framealpha=0.95)
ax.grid(axis='y', alpha=0.25, linewidth=0.4, zorder=0)
fig.text(0.5, -0.03, 'Source: ERA5 reanalysis, own elaboration. CAF project.',
         ha='center', fontsize=7.5, color='#666666')
fig.tight_layout()
fig.savefig(OUT_DIR / 'drought_severity_dominican_republic.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: drought_severity_dominican_republic.png")

print("\n[Fig 3] Rainfall fraction by country …")
rr_agg = pd.read_parquet(RR_AGG)
rr_agg = rr_agg.merge(shp, on='region_name', how='left')

total_ryw_rr = (rr_agg[rr_agg['activacion'] == 'p90']
                .groupby(['country', 'year', 'window'])
                .size().reset_index(name='n_obs'))
total_country_yr_rr = total_ryw_rr.groupby(['country', 'year'])['n_obs'].sum().reset_index()

hit_rr = (rr_agg[(rr_agg['activacion'] == 'p90') & (rr_agg['n_rainfall_pixels'] > 0)]
          .groupby(['country', 'year'])
          .size().reset_index(name='hit'))
rr_frac = total_country_yr_rr.merge(hit_rr, on=['country', 'year'], how='left').fillna({'hit': 0})
rr_frac['frac'] = rr_frac['hit'] / rr_frac['n_obs'].replace(0, np.nan)

rr_frac_plot = rr_frac[rr_frac['year'].between(2010, 2025)].copy()

rr = rr_frac_plot
rr['year'] = rr['year'].astype(int)

faceted_bar(
    frac_df   = rr,
    year_col  = 'year',
    val_col   = 'frac',
    bar_color = '#1565C0',
    ref_line  = 0.10,
    y_label   = 'Fraction of sub-regions with extreme rainfall',
    suptitle  = ('Extreme rainfall: fraction of sub-regions affected (2010–2025)\n'
                 'ERA5 total precipitation above decile 90 (region × season)'),
    footnote  = 'Source: ERA5 reanalysis, own elaboration. CAF project.',
    out_path  = OUT_DIR / 'rainfall_fraction_by_country.png',
)

print("\n[Fig 4] Heatwave fraction by country …")
hwa = pd.read_parquet(HW_AGG)
hwa = hwa.merge(shp, on='region_name', how='left')

hw_frac = (hwa[(hwa['activacion'] == 'p90') &
               (hwa['year_win'].between(2010, 2025))]
           .groupby(['country', 'year_win'])
           .apply(lambda g: (g['n_event_days'] > 0).mean())
           .reset_index(name='frac')
           .rename(columns={'year_win': 'year'}))

faceted_bar(
    frac_df   = hw_frac,
    year_col  = 'year',
    val_col   = 'frac',
    bar_color = '#C62828',
    ref_line  = 0.10,
    y_label   = 'Fraction of sub-regions affected',
    suptitle  = ('Heatwave: fraction of national area affected (2010–2025)\n'
                 'Sub-regions with ≥1 day above p90 tmax (pixel × season baseline)'),
    footnote  = 'Source: ERA5 reanalysis, own elaboration. CAF project.',
    out_path  = OUT_DIR / 'heatwave_fraction_by_country.png',
)

print("\n[Fig 5] Heatwave spatial map 2023 (p99) …")
hw23 = (hwa[(hwa['activacion'] == 'p99') & (hwa['year_win'] == 2023)]
        .groupby('region_name')['n_event_days']
        .mean()
        .reset_index())

gdf_shp = gpd.read_file(SHP_PATH).rename(columns={'region_nam': 'region_name'})
gdf_map = gdf_shp.merge(hw23, on='region_name', how='left').to_crs('EPSG:4326')
gdf_map['n_event_days'] = gdf_map['n_event_days'].fillna(0)

has_events = gdf_map['n_event_days'] > 0
vmax = float(gdf_map.loc[has_events, 'n_event_days'].quantile(0.97)) if has_events.any() else 1
vmax = max(vmax, 1)
norm_hw = mcolors.Normalize(vmin=0, vmax=vmax)
cmap_hw = plt.cm.YlOrRd

fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_facecolor('#cfe2f3')

gdf_map[~has_events].plot(ax=ax, color='#efefef', edgecolor='#aaaaaa', linewidth=0.2)
gdf_map[has_events].plot(ax=ax, column='n_event_days',
                         cmap=cmap_hw, norm=norm_hw,
                         edgecolor='#666666', linewidth=0.2)

sm = plt.cm.ScalarMappable(cmap=cmap_hw, norm=norm_hw)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.52, pad=0.02, aspect=18)
cb.set_label('Mean event-days above p99 tmax', fontsize=9)
cb.ax.tick_params(labelsize=8)

n_active = int(has_events.sum())
n_total  = len(gdf_shp)
ax.set_xlim(-85, -55)
ax.set_ylim(-20, 30)
ax.set_xlabel('Longitude', fontsize=9)
ax.set_ylabel('Latitude', fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.25, linewidth=0.3)
fig.text(0.5, 0.01, 'Source: ERA5 reanalysis, own elaboration. CAF project.',
         ha='center', fontsize=7.5, color='#666666')
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(OUT_DIR / 'heatwave_map_2023.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: heatwave_map_2023.png")

print(f"\nAll 5 figures → {OUT_DIR}/")
