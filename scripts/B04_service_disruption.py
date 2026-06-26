#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from rasterio.sample import sample_gen

MICRO_DIR    = Path('/home/jupyter-daniela/CAF/data/processed/caribbean_adm_boundaries/micro_flood')
MERIT_DIRS   = [
    Path('/home/jupyter-daniela/CAF/data/MERIT/dem_tif_n00w090'),
    Path('/home/jupyter-daniela/CAF/data/MERIT/dem_tif_n00w060'),
    Path('/home/jupyter-daniela/CAF/data/MERIT/dem_tif_n30w090'),
    Path('/home/jupyter-daniela/CAF/data/MERIT/dem_tif_n30w060'),
]
DISPLAYS_DIR = Path('/home/jupyter-daniela/CAF/outputs/cyclone')
DISPLAYS_DIR.mkdir(parents=True, exist_ok=True)

FLOOD_THRESHOLDS = [1, 2, 5, 10]

SERVICE_WEIGHTS = {
    'hospital':      5,
    'clinic':        4,
    'fire_station':  4,
    'police':        3,
    'school':        3,
    'pharmacy':      2,
    'university':    2,
    'social_facility': 2,
    'government':    2,
    'bus_station':   1,
    'train_station': 1,
    'fuel_station':  1,
    'airport':       3,
    'heliport':      3,
    'commercial':    1,
    'industrial':    1,
    'religious':     1,
    'residential':   1,
    'charging_station': 1,
}

SEVERITY_THRESHOLDS = {
    'minor':    1,
    'moderate': 2,
    'severe':   5,
}

def find_merit_tile(lon, lat, merit_dirs):
    tile_lat = int(np.floor(lat / 5) * 5)
    tile_lon = int(np.ceil(abs(lon) / 5) * 5)
    hemi_lat = 'n' if tile_lat >= 0 else 's'
    hemi_lon = 'w' if lon < 0 else 'e'
    fname = f"{hemi_lat}{abs(tile_lat):02d}{hemi_lon}{tile_lon:03d}_dem.tif"
    for d in merit_dirs:
        p = d / fname
        if p.exists():
            return p
    return None

def sample_elevation(df, merit_dirs):
    elevations = np.full(len(df), np.nan)
    for tile_path, group_idx in _group_by_tile(df, merit_dirs):
        if tile_path is None:
            continue
        coords = list(zip(df.loc[group_idx, 'lon'], df.loc[group_idx, 'lat']))
        try:
            with rasterio.open(tile_path) as src:
                vals = [v[0] for v in src.sample(coords)]
            elevations[group_idx] = [np.nan if v <= -9000 else v for v in vals]
        except Exception as e:
            print(f"    WARNING: could not sample {tile_path.name}: {e}")
    return elevations

def _group_by_tile(df, merit_dirs):
    tile_map = {}
    for i, row in df.iterrows():
        tile = find_merit_tile(row['lon'], row['lat'], merit_dirs)
        key = str(tile)
        if key not in tile_map:
            tile_map[key] = (tile, [])
        tile_map[key][1].append(i)
    for key, (tile_path, idxs) in tile_map.items():
        yield tile_path, idxs

def classify_events(df_events):
    losses = df_events['total_losses'].dropna()
    p25 = losses.quantile(0.25)
    p75 = losses.quantile(0.75)
    def severity(x):
        if pd.isna(x) or x <= p25:
            return 'minor'
        elif x <= p75:
            return 'moderate'
        else:
            return 'severe'
    df_events = df_events.copy()
    df_events['severity'] = df_events['total_losses'].apply(severity)
    df_events['flood_depth_m'] = df_events['severity'].map(SEVERITY_THRESHOLDS)
    return df_events

def compute_city_disruption(city_name, df_city_osm, df_city_events):
    records = []
    for _, ev in df_city_events.iterrows():
        threshold = ev['flood_depth_m']
        at_risk = df_city_osm[df_city_osm['elevation_m'] <= threshold]
        total   = len(df_city_osm)
        n_risk  = len(at_risk)
        pct_risk = (n_risk / total * 100) if total > 0 else 0

        weighted_total = sum(SERVICE_WEIGHTS.get(s, 1) for s in df_city_osm['service'])
        weighted_risk  = sum(SERVICE_WEIGHTS.get(s, 1) for s in at_risk['service'])
        sdi = (weighted_risk / weighted_total * 100) if weighted_total > 0 else 0

        by_service = (
            df_city_osm.assign(at_risk=df_city_osm['elevation_m'] <= threshold)
            .groupby('service')['at_risk']
            .agg(n_total='count', n_risk='sum')
            .assign(pct_risk=lambda x: x['n_risk'] / x['n_total'] * 100)
            .reset_index()
        )

        records.append({
            'region_name':      city_name,
            'country':          ev['country'],
            'gdp_tier':         ev['gdp_tier'],
            'event_date':       ev['event_date'],
            'disaster_type':    ev['disaster_type'],
            'total_deaths':     ev['total_deaths'],
            'total_losses_musd': ev['total_losses'],
            'severity':         ev['severity'],
            'flood_depth_m':    threshold,
            'n_total_facilities': total,
            'n_at_risk':        n_risk,
            'pct_at_risk':      round(pct_risk, 2),
            'sdi':              round(sdi, 2),
        })
    return pd.DataFrame(records)

print("=" * 70)
print("STEP 1 — Load inputs")
print("=" * 70)

df_cities  = pd.read_csv(MICRO_DIR / 'study_cities.csv')
df_events  = pd.read_csv(MICRO_DIR / 'flood_events.csv')
df_osm     = pd.read_csv(MICRO_DIR / 'osm_services.csv')

print(f"  Cities:  {len(df_cities)}")
print(f"  Events:  {len(df_events)}")
print(f"  OSM pts: {len(df_osm):,}")

print("\n" + "=" * 70)
print("STEP 2 — Sample MERIT elevation for each OSM facility")
print("=" * 70)

df_osm = df_osm.dropna(subset=['lon', 'lat']).reset_index(drop=True)
df_osm['elevation_m'] = sample_elevation(df_osm, MERIT_DIRS)

n_valid = df_osm['elevation_m'].notna().sum()
print(f"  Elevation sampled: {n_valid:,} / {len(df_osm):,} points")
print(f"  Elevation range: {df_osm['elevation_m'].min():.1f} – {df_osm['elevation_m'].max():.1f} m")

elev_summary = (
    df_osm.groupby('region_name')['elevation_m']
    .agg(median_elev='median', mean_elev='mean', pct_below_2m=lambda x: (x <= 2).mean() * 100,
         pct_below_5m=lambda x: (x <= 5).mean() * 100, pct_below_10m=lambda x: (x <= 10).mean() * 100)
    .reset_index()
)
print("\n  Elevation summary by city:")
print(elev_summary.to_string(index=False))

print("\n" + "=" * 70)
print("STEP 3 — Classify events by severity")
print("=" * 70)

df_events = classify_events(df_events)
print(df_events.groupby('severity')['total_losses'].agg(['count', 'min', 'max']).to_string())

print("\n" + "=" * 70)
print("STEP 4 — Compute Service Disruption Index per city × event")
print("=" * 70)

all_disruption = []
for city in df_cities['region_name']:
    city_osm    = df_osm[df_osm['region_name'] == city].copy()
    city_events = df_events[df_events['region_name'] == city].copy()
    if city_osm.empty or city_events.empty:
        print(f"  {city}: no OSM data or events — skip")
        continue
    df_dis = compute_city_disruption(city, city_osm, city_events)
    all_disruption.append(df_dis)
    print(f"  {city}: {len(df_dis)} event records, mean SDI = {df_dis['sdi'].mean():.1f}%")

df_disruption = pd.concat(all_disruption, ignore_index=True)

print("\n" + "=" * 70)
print("STEP 5 — Save outputs")
print("=" * 70)

df_osm.to_csv(MICRO_DIR / 'osm_with_elevation.csv', index=False)
df_osm.to_parquet(MICRO_DIR / 'osm_with_elevation.parquet', index=False)

df_disruption.to_csv(MICRO_DIR / 'service_disruption.csv', index=False)
df_disruption.to_parquet(MICRO_DIR / 'service_disruption.parquet', index=False)

print(f"  osm_with_elevation:  {len(df_osm):,} rows")
print(f"  service_disruption:  {len(df_disruption):,} rows")

worst = (
    df_disruption.loc[df_disruption.groupby('region_name')['sdi'].idxmax()]
    [['region_name', 'country', 'gdp_tier', 'event_date', 'severity',
      'flood_depth_m', 'n_total_facilities', 'n_at_risk', 'pct_at_risk', 'sdi']]
    .sort_values('sdi', ascending=False)
    .reset_index(drop=True)
)
print("\n  Worst-event SDI by city:")
print(worst.to_string(index=False))

mean_sdi = (
    df_disruption.groupby(['region_name', 'country', 'gdp_tier'])['sdi']
    .mean().reset_index().rename(columns={'sdi': 'mean_sdi'})
    .sort_values('mean_sdi', ascending=False)
)
print("\n  Mean SDI by city:")
print(mean_sdi.to_string(index=False))

CRITICAL = ['hospital', 'clinic', 'fire_station', 'police', 'pharmacy']

print("\n" + "=" * 70)
print("STEP 6 — Critical-facility disruption breakdown")
print("=" * 70)

crit_records = []
for city in df_cities['region_name']:
    city_osm    = df_osm[(df_osm['region_name'] == city) & (df_osm['service'].isin(CRITICAL))].copy()
    city_events = df_events[df_events['region_name'] == city].copy()
    if city_osm.empty or city_events.empty:
        continue
    for _, ev in city_events.iterrows():
        threshold = ev['flood_depth_m']
        for svc in CRITICAL:
            svc_df = city_osm[city_osm['service'] == svc]
            n_total = len(svc_df)
            n_risk  = (svc_df['elevation_m'] <= threshold).sum()
            crit_records.append({
                'region_name': city,
                'country':     ev['country'],
                'gdp_tier':    ev['gdp_tier'],
                'event_date':  ev['event_date'],
                'severity':    ev['severity'],
                'flood_depth_m': threshold,
                'service':     svc,
                'n_total':     n_total,
                'n_at_risk':   n_risk,
                'pct_at_risk': round(n_risk / n_total * 100, 1) if n_total > 0 else np.nan,
            })

df_crit = pd.DataFrame(crit_records)
df_crit.to_csv(MICRO_DIR / 'critical_service_disruption.csv', index=False)
df_crit.to_parquet(MICRO_DIR / 'critical_service_disruption.parquet', index=False)

crit_summary = (
    df_crit[df_crit['n_total'] > 0]
    .groupby(['region_name', 'gdp_tier', 'service'])['pct_at_risk']
    .mean().reset_index()
    .pivot(index=['region_name', 'gdp_tier'], columns='service', values='pct_at_risk')
    .reset_index()
)
print("\n  Mean % at-risk by city × critical service:")
print(crit_summary.to_string(index=False))

print("\n" + "=" * 70)
print("STEP 7 — Figures")
print("=" * 70)

BLUE   = '#2166AC'
RED    = '#D6604D'
ORANGE = '#F4A582'
GREEN  = '#4DAC26'
GRAY   = '#AAAAAA'
FONT   = 'DejaVu Sans'
plt.rcParams.update({
    'font.family': FONT, 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

GDP_COLORS = {'high': BLUE, 'medium': ORANGE, 'low': RED}

fig, ax = plt.subplots(figsize=(12, 6))

cities_ordered = (
    df_disruption.groupby('region_name')['sdi'].mean()
    .sort_values(ascending=False).index.tolist()
)

city_sdi_data = [df_disruption[df_disruption['region_name'] == c]['sdi'].values for c in cities_ordered]
bp = ax.boxplot(city_sdi_data, positions=range(len(cities_ordered)), widths=0.5,
                patch_artist=True, showfliers=False,
                medianprops={'color': 'black', 'linewidth': 1.5})

for i, (city, patch) in enumerate(zip(cities_ordered, bp['boxes'])):
    tier = df_disruption[df_disruption['region_name'] == city]['gdp_tier'].iloc[0]
    patch.set_facecolor(GDP_COLORS[tier])
    patch.set_alpha(0.7)
    x_jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(city_sdi_data[i]))
    ax.scatter(i + x_jitter, city_sdi_data[i], alpha=0.45, s=12,
               color=GDP_COLORS[tier], zorder=3)

ax.set_xticks(range(len(cities_ordered)))
ax.set_xticklabels([c.replace(' ', '\n') for c in cities_ordered], fontsize=8.5)
ax.set_ylabel('Service Disruption Index (%)', fontsize=11)
ax.set_title('Service Disruption Index by Study City\n'
             '(per EMDAT event — severity-matched flood depth)',
             fontsize=12, fontweight='bold', pad=12)

legend_handles = [mpatches.Patch(color=GDP_COLORS[t], label=f'{t.capitalize()} GDP') for t in ['high', 'medium', 'low']]
ax.legend(handles=legend_handles, fontsize=9, loc='upper right')
plt.tight_layout()
fig.savefig(DISPLAYS_DIR / 'fig_sdi_by_city.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  fig_sdi_by_city.png")

fig, ax = plt.subplots(figsize=(8, 5))

tier_order = ['high', 'medium', 'low']
tier_stats = (
    df_disruption.groupby('gdp_tier')['sdi']
    .agg(mean='mean', se=lambda x: x.sem())
    .reindex(tier_order)
)

bars = ax.bar(tier_order, tier_stats['mean'],
              color=[GDP_COLORS[t] for t in tier_order],
              yerr=tier_stats['se'], capsize=5,
              edgecolor='white', width=0.5,
              error_kw={'linewidth': 1.5})

ax.set_xlabel('GDP Tier', fontsize=11)
ax.set_ylabel('Mean Service Disruption Index (%)', fontsize=11)
ax.set_title('Service Disruption Index by GDP Tier\n(mean ± SE across all events)',
             fontsize=12, fontweight='bold', pad=12)

for bar, (tier, row) in zip(bars, tier_stats.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + row['se'] + 0.5,
            f"{row['mean']:.1f}%", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(DISPLAYS_DIR / 'fig_sdi_by_gdp_tier.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  fig_sdi_by_gdp_tier.png")

fig, ax = plt.subplots(figsize=(11, 6))

city_gdp = df_disruption[['region_name', 'gdp_tier']].drop_duplicates().set_index('region_name')['gdp_tier']

for city in cities_ordered:
    elev = df_osm[df_osm['region_name'] == city]['elevation_m'].dropna().sort_values()
    if elev.empty:
        continue
    cdf = np.arange(1, len(elev) + 1) / len(elev)
    tier = city_gdp.get(city, 'medium')
    ax.plot(elev, cdf, label=city, color=GDP_COLORS[tier], linewidth=1.8, alpha=0.85)

for thresh, ls in zip(FLOOD_THRESHOLDS, [':', '--', '-.', '-']):
    ax.axvline(thresh, color='black', linewidth=0.7, linestyle=ls, alpha=0.5,
               label=f'{thresh}m threshold')

ax.set_xlim(-2, 50)
ax.set_xlabel('Elevation (m above sea level)', fontsize=11)
ax.set_ylabel('Cumulative fraction of facilities', fontsize=11)
ax.set_title('Elevation CDF of Service Facilities by City\n'
             '(vertical lines = flood depth thresholds)',
             fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
fig.savefig(DISPLAYS_DIR / 'fig_elevation_cdf.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  fig_elevation_cdf.png")

crit_pivot = crit_summary.set_index('region_name')[CRITICAL].fillna(0)
crit_pivot = crit_pivot.loc[crit_pivot.index.isin(cities_ordered)]

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(crit_pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(range(len(CRITICAL)))
ax.set_xticklabels([s.replace('_', ' ').title() for s in CRITICAL], fontsize=10)
ax.set_yticks(range(len(crit_pivot)))
ax.set_yticklabels(crit_pivot.index, fontsize=9)

for i in range(len(crit_pivot)):
    for j in range(len(CRITICAL)):
        val = crit_pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if val > 50 else 'black')

plt.colorbar(im, ax=ax, label='Mean % facilities at risk', shrink=0.8)
ax.set_title('Critical Service Disruption — Mean % at Risk\n'
             '(averaged over all EMDAT events per city)',
             fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
fig.savefig(DISPLAYS_DIR / 'fig_critical_service_heatmap.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("  fig_critical_service_heatmap.png")

print(f"\nAll outputs → {MICRO_DIR}/")
print(f"  osm_with_elevation.csv / .parquet")
print(f"  service_disruption.csv / .parquet")
print(f"  critical_service_disruption.csv / .parquet")
print(f"\nFigures → {DISPLAYS_DIR}/")
print(f"  fig_sdi_by_city.png")
print(f"  fig_sdi_by_gdp_tier.png")
print(f"  fig_elevation_cdf.png")
print(f"  fig_critical_service_heatmap.png")
