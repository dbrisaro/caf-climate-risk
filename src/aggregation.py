import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from utils import get_season, get_region, lower_tail_activation, upper_tail_activation

def run_cyclones(base_dir, carib_targets, target_regions, all_windows, processed_dir=None):
    if processed_dir is None:
        processed_dir = base_dir / 'data/processed'
    output_dir = processed_dir / 'ibtracs_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / 'ibtracs_raw_data.parquet'
    agg_path = output_dir / 'ibtracs_aggregated_data.parquet'

    ds = xr.open_dataset(base_dir / 'data/IBTrACS/IBTrACS.since1980.v04r01.nc')

    sids, times = ds['sid'].values, ds['time'].values
    lats, lons  = ds['lat'].values, ds['lon'].values
    storm_speed = ds['storm_speed'].values
    usa_wind    = ds['usa_wind'].values

    n_storms, n_times = times.shape
    flat_sid   = np.repeat(sids, n_times)
    flat_time  = times.ravel()
    flat_lat   = lats.ravel()
    flat_lon   = lons.ravel()
    flat_speed = storm_speed.ravel()
    flat_wind  = usa_wind.ravel()

    valid = ~pd.isnull(flat_time) & ~np.isnan(flat_lat) & ~np.isnan(flat_lon)
    flat_sid, flat_time   = flat_sid[valid],   flat_time[valid]
    flat_lat, flat_lon    = flat_lat[valid],   flat_lon[valid]
    flat_speed, flat_wind = flat_speed[valid], flat_wind[valid]

    minlon, minlat, maxlon, maxlat = carib_targets.total_bounds
    bbox = (
        (flat_lat >= minlat) & (flat_lat <= maxlat) &
        (flat_lon >= minlon) & (flat_lon <= maxlon)
    )
    flat_sid, flat_time   = flat_sid[bbox],   flat_time[bbox]
    flat_lat, flat_lon    = flat_lat[bbox],   flat_lon[bbox]
    flat_speed, flat_wind = flat_speed[bbox], flat_wind[bbox]

    decoded_sids = [
        s.decode('utf-8').strip() if isinstance(s, bytes) else str(s).strip()
        for s in flat_sid
    ]

    gdf_tracks = gpd.GeoDataFrame(
        {'sid': decoded_sids, 'time': flat_time, 'lat': flat_lat, 'lon': flat_lon,
         'storm_speed': flat_speed, 'usa_wind': flat_wind},
        geometry=gpd.points_from_xy(flat_lon, flat_lat),
        crs='EPSG:4326'
    )
    gdf_joined = gpd.sjoin(gdf_tracks, carib_targets[['region_name', 'geometry']],
                           how='inner', predicate='within')

    df = gdf_joined[['sid', 'time', 'lat', 'lon', 'storm_speed', 'usa_wind', 'region_name']].copy()
    df['time']     = pd.to_datetime(df['time'])
    df['month']    = df['time'].dt.month
    df['year']     = df['time'].dt.year
    df['window']   = df['month'].map(get_season)
    df['year_win'] = df.apply(lambda r: r['year'] + 1 if r['month'] == 12 else r['year'], axis=1)
    df.drop(columns=['month', 'year'], inplace=True)
    df.to_parquet(raw_path, index=False)

    df_per_storm = (
        df.groupby(['sid', 'region_name', 'year_win', 'window'])
        .agg(max_storm_speed=('storm_speed', 'max'), max_usa_wind=('usa_wind', 'max'))
        .reset_index()
    )
    df_agg = (
        df_per_storm.groupby(['region_name', 'year_win', 'window'])
        .agg(frequency=('sid', 'count'),
             avg_max_storm_speed=('max_storm_speed', 'mean'),
             avg_max_usa_wind=('max_usa_wind', 'mean'))
        .reset_index()
    )

    all_years = range(df['year_win'].min(), df['year_win'].max() + 1)
    complete_index = pd.MultiIndex.from_product(
        [target_regions, all_years, all_windows],
        names=['region_name', 'year_win', 'window']
    )
    df_agg = (
        df_agg.set_index(['region_name', 'year_win', 'window'])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    df_agg.to_parquet(agg_path, index=False)
    print(f"Saved {len(df_agg):,} rows to {agg_path}")
    return df_agg

def run_droughts(base_dir, target_regions, all_windows, processed_dir=None):
    if processed_dir is None:
        processed_dir = base_dir / 'data/processed'
    output_dir = processed_dir / 'era5_output'
    agg_path   = output_dir / 'droughts_aggregated_data.parquet'

    df = pd.read_parquet(output_dir / 'era5_raw_data.parquet')
    if 'country' in df.columns:
        df = df.rename(columns={'country': 'region_name'})
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    agg = df.groupby(['pixel_id', 'year', 'window'], as_index=False)['value'].sum()
    agg = agg.sort_values(['pixel_id', 'year', 'window'])
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
    complete_index = pd.MultiIndex.from_product(
        [target_regions, all_years, all_windows, all_levels],
        names=['region_name', 'year', 'window', 'activacion']
    )
    df_agg = (
        df_agg.set_index(['region_name', 'year', 'window', 'activacion'])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    df_agg.to_parquet(agg_path, index=False)
    print(f"Saved {len(df_agg):,} rows to {agg_path}")
    return df_agg

def run_heatwaves(base_dir, carib_targets, target_regions, all_windows, processed_dir=None):
    if processed_dir is None:
        processed_dir = base_dir / 'data/processed'
    output_dir = processed_dir / 'heatwaves_computation'
    raw_path   = output_dir / 'heatwave_events_raw.parquet'
    agg_path   = output_dir / 'heatwaves_aggregated_data.parquet'

    df = pd.read_parquet(raw_path)
    if 'country' in df.columns:
        df = df.rename(columns={'country': 'region_name'})

    if 'region_name' not in df.columns or df['region_name'].isna().all():
        df['region_name'] = df.apply(lambda row: get_region(row['lat'], row['lon'], carib_targets), axis=1)
        df.to_parquet(raw_path, index=False)

    df['year']     = df['date'].dt.year
    df['year_win'] = df.apply(
        lambda row: row['year'] + 1 if (row['window'] == 'DJF' and row['date'].month == 12) else row['year'],
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
    complete_index = pd.MultiIndex.from_product(
        [target_regions, all_years, all_windows, all_levels],
        names=['region_name', 'year_win', 'window', 'activacion']
    )
    df_agg = (
        df_agg.set_index(['region_name', 'year_win', 'window', 'activacion'])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    df_agg.to_parquet(agg_path, index=False)
    print(f"Saved {len(df_agg):,} rows to {agg_path}")
    return df_agg

def run_rainfall(base_dir, target_regions, all_windows, processed_dir=None):
    if processed_dir is None:
        processed_dir = base_dir / 'data/processed'
    output_dir = processed_dir / 'era5_output_precipitation'
    agg_path   = output_dir / 'rainfall_aggregated_data.parquet'

    df = pd.read_parquet(output_dir / 'era5_raw_data.parquet')
    if 'country' in df.columns:
        df = df.rename(columns={'country': 'region_name'})
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    agg = df.groupby(['pixel_id', 'year', 'window'], as_index=False)['value'].sum()
    agg = agg.sort_values(['pixel_id', 'year', 'window'])
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
    agg[['lat', 'lon']] = agg['pixel_id'].str.split('_', expand=True).astype(float)

    df_agg = (
        agg[agg['region_name'].notna() & (agg['activacion'] != 'none')]
        .groupby(['region_name', 'year', 'window', 'activacion'])
        .size()
        .reset_index(name='n_rainfall_pixels')
    )

    all_years  = range(agg['year'].min(), agg['year'].max() + 1)
    all_levels = ['p90', 'p95', 'p99']
    complete_index = pd.MultiIndex.from_product(
        [target_regions, all_years, all_windows, all_levels],
        names=['region_name', 'year', 'window', 'activacion']
    )
    df_agg = (
        df_agg.set_index(['region_name', 'year', 'window', 'activacion'])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    df_agg.to_parquet(agg_path, index=False)
    print(f"Saved {len(df_agg):,} rows to {agg_path}")
    return df_agg
