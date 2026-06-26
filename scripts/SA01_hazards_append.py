#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

from pathlib import Path
from pyproj import datadir
datadir.set_data_dir('/home/jupyter-daniela/.conda/envs/peru_environment/share/proj')

import numpy as np
import pandas as pd
import geopandas as gpd

from utils import load_shapefile, lower_tail_activation, upper_tail_activation
from hazard_utils import run_rainfall_raw, run_droughts_raw, run_heatwaves_raw

BASE_DIR   = Path('/home/jupyter-daniela/CAF')
PROCESSED  = BASE_DIR / 'data/processed/caribbean_adm_boundaries'
SHAPEFILE  = '/home/jupyter-daniela/CAF/data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
NAME_COL   = 'region_nam'

SA_COUNTRIES = {'Colombia', 'Peru', 'Guyana'}
ALL_WINDOWS  = ['SON', 'DJF', 'MAM', 'JJA']

ERA5_RAINFALL_SA  = Path('/home/jupyter-daniela/suyana/sources/era5/total_precipitation/south_america')
ERA5_DROUGHT_SA   = Path('/home/jupyter-daniela/suyana/sources/era5/swc/south_america')
ERA5_HEATWAVE_SA  = Path('/home/jupyter-daniela/suyana/sources/era5/statistics/tmax/south_america')

shp = gpd.read_file(SHAPEFILE)
sa_targets = shp[shp['country'].isin(SA_COUNTRIES)].copy()
sa_targets = sa_targets.rename(columns={NAME_COL: 'region_name'})
sa_targets = sa_targets[['region_name', 'geometry']].copy()
sa_regions = sorted(sa_targets['region_name'].tolist())
print(f"SA targets: {len(sa_targets)} regions — {sorted(SA_COUNTRIES)}\n")

SA_RAINFALL_RAW  = PROCESSED / 'era5_output_precipitation/_sa_rainfall_raw.parquet'
SA_DROUGHT_RAW   = PROCESSED / 'era5_output/_sa_drought_raw.parquet'
SA_HEATWAVE_RAW  = PROCESSED / 'heatwaves_computation/_sa_heatwave_raw.parquet'

CARIB_RAIN_AGG  = PROCESSED / 'era5_output_precipitation/rainfall_aggregated_data.parquet'
CARIB_DROUGHT_AGG = PROCESSED / 'era5_output/droughts_aggregated_data.parquet'
CARIB_HW_AGG    = PROCESSED / 'heatwaves_computation/heatwaves_aggregated_data.parquet'

def aggregate_rainfall_from_raw(raw_path, target_regions, all_windows):
    df = pd.read_parquet(raw_path)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    agg = df.groupby(['pixel_id', 'year', 'window'], as_index=False)['value'].sum()
    agg = agg.merge(df[['pixel_id', 'region_name']].drop_duplicates(), on='pixel_id', how='left')

    pct = (
        agg.groupby(['pixel_id', 'window'])['value']
        .quantile([0.90, 0.95, 0.99])
        .unstack(level=-1)
        .reset_index()
    )
    pct.columns = ['pixel_id', 'window', 'p90', 'p95', 'p99']
    agg = agg.merge(pct, on=['pixel_id', 'window'], how='left')
    agg['activacion'] = agg.apply(upper_tail_activation, axis=1)

    df_agg = (
        agg[agg['region_name'].notna() & (agg['activacion'] != 'none')]
        .groupby(['region_name', 'year', 'window', 'activacion'])
        .size()
        .reset_index(name='n_rainfall_pixels')
    )
    all_years  = range(agg['year'].min(), agg['year'].max() + 1)
    all_levels = ['p90', 'p95', 'p99']
    idx = pd.MultiIndex.from_product(
        [target_regions, all_years, all_windows, all_levels],
        names=['region_name', 'year', 'window', 'activacion']
    )
    return (df_agg.set_index(['region_name', 'year', 'window', 'activacion'])
                  .reindex(idx, fill_value=0).reset_index())

def aggregate_drought_from_raw(raw_path, target_regions, all_windows):
    df = pd.read_parquet(raw_path)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    agg = df.groupby(['pixel_id', 'year', 'window'], as_index=False)['value'].sum()
    agg = agg.merge(df[['pixel_id', 'region_name']].drop_duplicates(), on='pixel_id', how='left')

    pct = (
        agg.groupby(['pixel_id', 'window'])['value']
        .quantile([0.10, 0.05, 0.01])
        .unstack(level=-1)
        .reset_index()
    )
    pct.columns = ['pixel_id', 'window', 'p10', 'p5', 'p1']
    agg = agg.merge(pct, on=['pixel_id', 'window'], how='left')
    agg['activacion'] = agg.apply(lower_tail_activation, axis=1)

    df_agg = (
        agg[agg['region_name'].notna() & (agg['activacion'] != 'none')]
        .groupby(['region_name', 'year', 'window', 'activacion'])
        .size()
        .reset_index(name='n_drought_pixels')
    )
    all_years  = range(agg['year'].min(), agg['year'].max() + 1)
    all_levels = ['p1', 'p5', 'p10']
    idx = pd.MultiIndex.from_product(
        [target_regions, all_years, all_windows, all_levels],
        names=['region_name', 'year', 'window', 'activacion']
    )
    return (df_agg.set_index(['region_name', 'year', 'window', 'activacion'])
                  .reindex(idx, fill_value=0).reset_index())

def aggregate_heatwave_from_raw(raw_path, target_regions, all_windows):
    df = pd.read_parquet(raw_path)
    df['year']     = df['date'].dt.year
    df['year_win'] = df.apply(
        lambda r: r['year'] + 1 if (r['window'] == 'DJF' and r['date'].month == 12) else r['year'],
        axis=1
    )
    df_agg = (
        df[df['region_name'].notna()]
        .groupby(['region_name', 'year_win', 'window', 'activacion'])
        .size()
        .reset_index(name='n_event_days')
    )
    all_years  = range(df['year_win'].min(), df['year_win'].max() + 1)
    all_levels = ['p90', 'p95', 'p99']
    idx = pd.MultiIndex.from_product(
        [target_regions, all_years, all_windows, all_levels],
        names=['region_name', 'year_win', 'window', 'activacion']
    )
    return (df_agg.set_index(['region_name', 'year_win', 'window', 'activacion'])
                  .reindex(idx, fill_value=0).reset_index())

print("=== RAINFALL (SA only) ===")
if not SA_RAINFALL_RAW.exists():
    run_rainfall_raw(era5_dir=ERA5_RAINFALL_SA, gdf=sa_targets, output_path=SA_RAINFALL_RAW)
else:
    print(f"  [skip] SA rainfall raw already exists ({SA_RAINFALL_RAW.stat().st_size/1e6:.1f} MB)")

print("  Aggregating SA rainfall...")
sa_rain_agg = aggregate_rainfall_from_raw(SA_RAINFALL_RAW, sa_regions, ALL_WINDOWS)
print(f"  SA rainfall aggregated: {sa_rain_agg.shape}")

carib_rain = pd.read_parquet(CARIB_RAIN_AGG)
merged_rain = pd.concat([carib_rain, sa_rain_agg], ignore_index=True)
merged_rain.to_parquet(CARIB_RAIN_AGG, index=False)
print(f"  Saved merged rainfall → {CARIB_RAIN_AGG}  ({merged_rain['region_name'].nunique()} regions)\n")

print("=== DROUGHT (SA only) ===")
if not SA_DROUGHT_RAW.exists():
    run_droughts_raw(era5_dir=ERA5_DROUGHT_SA, gdf=sa_targets, output_path=SA_DROUGHT_RAW)
else:
    print(f"  [skip] SA drought raw already exists ({SA_DROUGHT_RAW.stat().st_size/1e6:.1f} MB)")

print("  Aggregating SA drought...")
sa_drought_agg = aggregate_drought_from_raw(SA_DROUGHT_RAW, sa_regions, ALL_WINDOWS)
print(f"  SA drought aggregated: {sa_drought_agg.shape}")

carib_drought = pd.read_parquet(CARIB_DROUGHT_AGG)
merged_drought = pd.concat([carib_drought, sa_drought_agg], ignore_index=True)
merged_drought.to_parquet(CARIB_DROUGHT_AGG, index=False)
print(f"  Saved merged drought → {CARIB_DROUGHT_AGG}  ({merged_drought['region_name'].nunique()} regions)\n")

print("=== HEATWAVE (SA only) ===")
if not SA_HEATWAVE_RAW.exists():
    run_heatwaves_raw(era5_dir=ERA5_HEATWAVE_SA, gdf=sa_targets, output_path=SA_HEATWAVE_RAW)
else:
    print(f"  [skip] SA heatwave raw already exists ({SA_HEATWAVE_RAW.stat().st_size/1e6:.1f} MB)")

print("  Aggregating SA heatwave...")
sa_hw_agg = aggregate_heatwave_from_raw(SA_HEATWAVE_RAW, sa_regions, ALL_WINDOWS)
print(f"  SA heatwave aggregated: {sa_hw_agg.shape}")

carib_hw = pd.read_parquet(CARIB_HW_AGG)
merged_hw = pd.concat([carib_hw, sa_hw_agg], ignore_index=True)
merged_hw.to_parquet(CARIB_HW_AGG, index=False)
print(f"  Saved merged heatwave → {CARIB_HW_AGG}  ({merged_hw['region_name'].nunique()} regions)\n")

print("=== ALL HAZARDS DONE — run script 03 to rebuild master panel ===")
