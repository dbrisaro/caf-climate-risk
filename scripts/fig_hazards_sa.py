import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path

import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

BASE_DIR = Path('/home/jupyter-daniela/CAF')
PROC     = BASE_DIR / 'data/processed/caribbean_adm_boundaries'
SHP_PATH = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
OUT_DIR  = BASE_DIR / 'outputs/hazards'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SA_DR_RAW = PROC / 'era5_output/_sa_drought_raw.parquet'
SA_RR_RAW = PROC / 'era5_output_precipitation/_sa_rainfall_raw.parquet'
SA_HW_RAW = PROC / 'heatwaves_computation/_sa_heatwave_raw.parquet'

SA_COUNTRIES  = ['Colombia', 'Peru', 'Guyana']
YEARS_PLOT    = list(range(2010, 2026))

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

shp = gpd.read_file(SHP_PATH)[['region_nam', 'country']].rename(
        columns={'region_nam': 'region_name'})
shp_sa = shp[shp['country'].isin(SA_COUNTRIES)].copy()

print("=" * 60)
print("fig_hazards_sa.py — SA hazard figures")
print("=" * 60)

def faceted_bar_sa(frac_df, year_col, val_col, bar_color, ref_line,
                   y_label, footnote, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    x = np.arange(len(YEARS_PLOT))

    for i, country in enumerate(SA_COUNTRIES):
        ax  = axes[i]
        cdf = (frac_df[frac_df['country'] == country]
               .set_index(year_col)[val_col]
               .reindex(YEARS_PLOT).fillna(0))

        ax.bar(x, cdf.values, color=bar_color, alpha=0.85, width=0.72,
               edgecolor='white', linewidth=0.4)
        ax.axhline(ref_line, color='#888888', lw=0.9, linestyle='--', zorder=3)

        ax.set_title(f"{chr(ord('a')+i)}  {country}", loc='left',
                     fontweight='normal', fontsize=10, pad=4)
        ax.set_xticks(x[::3])
        ax.set_xticklabels([str(y) for y in YEARS_PLOT[::3]],
                           fontsize=8, rotation=45, ha='right')
        ax.set_xlim(-0.6, len(YEARS_PLOT) - 0.4)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', alpha=0.25, linewidth=0.4, zorder=0)
        if i == 0:
            ax.set_ylabel(y_label, fontsize=9)

    ref_patch = mpatches.Patch(color='#888888', label=f'Expected ({ref_line:.0%})', alpha=0.6)
    bar_patch  = mpatches.Patch(color=bar_color, alpha=0.85, label='Observed')
    fig.legend(handles=[bar_patch, ref_patch], loc='lower right',
               bbox_to_anchor=(0.99, 0.01), fontsize=8, framealpha=0.95)
    fig.text(0.5, -0.02, footnote, ha='center', fontsize=7.5, color='#666666')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

def pixel_fractions(raw_path, q_val, flag_fn, label):
    df = pd.read_parquet(raw_path)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df = df.merge(shp_sa, on='region_name', how='left')

    thr = (df.groupby(['pixel_id', 'window'])['value']
             .quantile(q_val).reset_index()
             .rename(columns={'value': 'thresh'}))
    df2 = df.merge(thr, on=['pixel_id', 'window'], how='left')
    df2['flag'] = flag_fn(df2['value'], df2['thresh'])

    frac = (df2[df2['year'].between(YEARS_PLOT[0], YEARS_PLOT[-1])]
               .groupby(['country', 'year'])
               .agg(n=('flag', 'count'), hit=('flag', 'sum'))
               .reset_index())
    frac['frac'] = frac['hit'] / frac['n']
    print(f"    {label} fractions computed ({len(frac)} country-year rows)")
    return frac

print("\n[Fig 1] Drought fraction — SA …")
dr_frac = pixel_fractions(
    SA_DR_RAW, q_val=0.10,
    flag_fn=lambda v, t: v < t,
    label='drought p10'
)
faceted_bar_sa(
    frac_df   = dr_frac,
    year_col  = 'year',
    val_col   = 'frac',
    bar_color = '#8B5E3C',
    ref_line  = 0.10,
    y_label   = 'Fraction of pixels below p10',
    footnote  = 'Source: ERA5 reanalysis (soil water content), own elaboration. CAF project.',
    out_path  = OUT_DIR / 'drought_fraction_sa.png',
)

print("\n[Fig 2] Rainfall fraction — SA …")
rr_frac = pixel_fractions(
    SA_RR_RAW, q_val=0.90,
    flag_fn=lambda v, t: v > t,
    label='rainfall p90'
)
faceted_bar_sa(
    frac_df   = rr_frac,
    year_col  = 'year',
    val_col   = 'frac',
    bar_color = '#2166ac',
    ref_line  = 0.10,
    y_label   = 'Fraction of pixels above p90',
    footnote  = 'Source: ERA5 reanalysis (total precipitation), own elaboration. CAF project.',
    out_path  = OUT_DIR / 'rainfall_fraction_sa.png',
)

print("\n[Fig 3] Heatwave fraction — SA …")
hw = pd.read_parquet(SA_HW_RAW)
hw['date'] = pd.to_datetime(hw['date'])
hw['year'] = hw['date'].dt.year
if 'country' not in hw.columns:
    hw = hw.merge(shp_sa.drop_duplicates('region_name'), on='region_name', how='left')
else:
    hw = hw[hw['country'].isin(SA_COUNTRIES)]

hw_frac = (hw[(hw['activacion'] == 'p90') & hw['year'].between(YEARS_PLOT[0], YEARS_PLOT[-1])]
           .groupby(['country', 'year', 'region_name'])
           .agg(event_days=('activacion', 'count'))
           .reset_index())
hw_frac['has_event'] = hw_frac['event_days'] > 0

total_regions = shp_sa.drop_duplicates(['country', 'region_name']).groupby('country')['region_name'].nunique().reset_index(name='n_regions')
hw_country = (hw_frac.groupby(['country', 'year'])['has_event'].sum()
              .reset_index(name='regions_hit'))
hw_country = hw_country.merge(total_regions, on='country', how='left')
hw_country['frac'] = hw_country['regions_hit'] / hw_country['n_regions']

faceted_bar_sa(
    frac_df   = hw_country,
    year_col  = 'year',
    val_col   = 'frac',
    bar_color = '#d6604d',
    ref_line  = 0.10,
    y_label   = 'Fraction of sub-regions with ≥1 heatwave day',
    footnote  = 'Source: ERA5 reanalysis (2m max temperature), own elaboration. CAF project.',
    out_path  = OUT_DIR / 'heatwave_fraction_sa.png',
)

print("\n[Fig 4] Heatwave map 2023 — SA …")
hw23 = (hw[(hw['activacion'] == 'p90') & (hw['year'] == 2023)]
        .groupby('region_name')['date']
        .count().reset_index(name='event_days'))

gdf_shp = gpd.read_file(SHP_PATH).rename(columns={'region_nam': 'region_name'})
gdf_sa   = gdf_shp[gdf_shp['country'].isin(SA_COUNTRIES)].copy()
gdf_map  = gdf_sa.merge(hw23, on='region_name', how='left').to_crs('EPSG:4326')
gdf_map['event_days'] = gdf_map['event_days'].fillna(0)

vmax = max(gdf_map['event_days'].quantile(0.95), 1)
cmap = plt.cm.YlOrRd
norm = mcolors.Normalize(vmin=0, vmax=vmax)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor('#d8eaf5')
gdf_map.plot(ax=ax, column='event_days', cmap=cmap, norm=norm,
             edgecolor='#555555', linewidth=0.3,
             missing_kwds={'color': '#cccccc', 'label': 'No data'})

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.02, aspect=20)
cb.set_label('Heatwave event-days (p90 threshold, 2023)', fontsize=9)
cb.ax.tick_params(labelsize=8)

ax.set_xlim(-82, -55)
ax.set_ylim(-20, 14)
ax.set_xlabel('Longitude', fontsize=9)
ax.set_ylabel('Latitude', fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3, linewidth=0.3)
ax.set_title('Heatwave intensity by sub-region — Colombia, Peru, Guyana (2023)',
             fontsize=11, pad=10)
fig.text(0.5, 0.01, 'Source: ERA5 reanalysis, own elaboration. CAF project.',
         ha='center', fontsize=7.5, color='#888888')
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(OUT_DIR / 'heatwave_map_sa_2023.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: heatwave_map_sa_2023.png")

print("\n=== fig_hazards_sa.py complete ===")
