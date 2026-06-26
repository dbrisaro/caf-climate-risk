import warnings; warnings.filterwarnings("ignore")
import os
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc4
from pathlib import Path
from itertools import product

BASE_DIR     = Path('/home/jupyter-daniela/CAF')
PROCESSED    = BASE_DIR / 'data/processed/caribbean_adm_boundaries'
SHP_PATH     = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
ERA5_DIR     = Path('/home/jupyter-daniela/suyana/sources/era5/statistics/tmax/south_america')
SA_HW_RAW    = PROCESSED / 'heatwaves_computation/_sa_heatwave_raw.parquet'
CARIB_HW_AGG = PROCESSED / 'heatwaves_computation/heatwaves_aggregated_data.parquet'
PANEL_PATH   = PROCESSED / 'master_panel.parquet'

SA_COUNTRIES = ['Colombia', 'Peru', 'Guyana']
ALL_WINDOWS  = ['SON', 'DJF', 'MAM', 'JJA']
ALL_LEVELS   = ['p90', 'p95', 'p99']

print("=" * 60)
print("SA02_heatwave_only.py — SA heatwave (memory-efficient)")
print("=" * 60)

gdf = gpd.read_file(SHP_PATH)
if gdf.crs is None:
    gdf = gdf.set_crs('EPSG:4326')
elif gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs('EPSG:4326')
if 'region_nam' in gdf.columns:
    gdf = gdf.rename(columns={'region_nam': 'region_name'})

sa_gdf     = gdf[gdf['country'].isin(SA_COUNTRIES)].copy()
sa_regions = sa_gdf['region_name'].unique().tolist()
print(f"SA targets: {len(sa_regions)} unique regions — {SA_COUNTRIES}")

print("\n=== STEP 1: SA heatwave raw computation ===")

if SA_HW_RAW.exists():
    print(f"  [skip] Raw file already exists ({SA_HW_RAW.stat().st_size/1e6:.1f} MB)")
else:
    files = sorted(ERA5_DIR.glob('era5_tmax_daily_*.nc'))
    print(f"  ERA5 files: {len(files)} years")

    ds0  = nc4.Dataset(str(files[0]))
    lats = ds0.variables['latitude'][:].data.astype('float64')
    lons = ds0.variables['longitude'][:].data.astype('float64')
    tvar = next(v for v in ds0.variables
                if ds0.variables[v].ndim == 3 and
                set(ds0.variables[v].dimensions) >= {'latitude', 'longitude'})
    ds0.close()
    nlat, nlon = len(lats), len(lons)
    print(f"  Grid: {nlat}×{nlon}, tvar='{tvar}'")
    print(f"  Lat: {lats[0]:.2f}…{lats[-1]:.2f}, Lon: {lons[0]:.2f}…{lons[-1]:.2f}")

    print("  Building pixel→region mapping via spatial join …")
    lat_grid = np.repeat(lats[:, np.newaxis], nlon, axis=1).ravel()
    lon_grid = np.repeat(lons[np.newaxis, :], nlat, axis=0).ravel()
    n_pixels = len(lat_grid)

    from shapely.geometry import Point
    pts = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(lon_grid, lat_grid)],
        crs='EPSG:4326'
    )
    joined = pts.sjoin(sa_gdf[['region_name', 'country', 'geometry']],
                       how='left', predicate='within')
    joined = joined[~joined.index.duplicated(keep='first')]
    region_arr  = joined['region_name'].reindex(pts.index).values
    country_arr = joined['country'].reindex(pts.index).values

    valid_mask = pd.notna(region_arr)
    valid_idx  = np.where(valid_mask)[0]
    n_valid    = len(valid_idx)
    print(f"  Valid pixels: {n_valid:,} / {n_pixels:,}")

    lat_idx_v = (valid_idx // nlon).astype('int32')
    lon_idx_v = (valid_idx %  nlon).astype('int32')
    valid_lats   = lat_grid[valid_idx]
    valid_lons   = lon_grid[valid_idx]
    valid_reg    = region_arr[valid_idx]
    valid_cty    = country_arr[valid_idx]

    print("  Loading all ERA5 files sequentially …")
    all_dates = []
    for f in files:
        ds = nc4.Dataset(str(f))
        tv = 'valid_time' if 'valid_time' in ds.variables else 'time'
        t  = ds.variables[tv]
        dates_f = nc4.num2date(t[:].data, t.units, getattr(t, 'calendar', 'standard'))
        all_dates.extend(dates_f)
        ds.close()
    n_times = len(all_dates)
    print(f"  Total time steps: {n_times:,}")
    mem_mb = n_valid * n_times * 4 / 1e6
    print(f"  Matrix size: {n_valid:,} × {n_times:,} = {mem_mb:.0f} MB (float32)")

    data = np.full((n_valid, n_times), np.nan, dtype='float32')
    t_pos = 0
    for fi, f in enumerate(files):
        ds  = nc4.Dataset(str(f))
        tv  = 'valid_time' if 'valid_time' in ds.variables else 'time'
        nt  = len(ds.variables[tv][:])
        arr = ds.variables[tvar][:]
        ds.close()
        chunk = arr[:, lat_idx_v, lon_idx_v]
        if hasattr(chunk, 'filled'):
            chunk_d = chunk.filled(np.nan).astype('float32')
        else:
            chunk_d = chunk.astype('float32')
        chunk_d -= 273.15
        data[:, t_pos:t_pos + nt] = chunk_d.T
        t_pos += nt
        if (fi + 1) % 10 == 0 or fi == len(files) - 1:
            print(f"    Loaded year {fi+1}/{len(files)} ({data.shape[1]:.0f} time steps so far)")

    print("  Computing per-pixel percentile thresholds (p90/p95/p99) …")
    thresholds = np.nanpercentile(data, [90, 95, 99], axis=1)
    print(f"  Thresholds shape: {thresholds.shape}")
    print(f"  p90 range: {np.nanmin(thresholds[0]):.1f}–{np.nanmax(thresholds[0]):.1f}°C")

    month_arr = np.array([d.month for d in all_dates], dtype='int8')
    day_arr   = np.array([d.day   for d in all_dates], dtype='int8')
    year_arr  = np.array([d.year  for d in all_dates], dtype='int16')

    def season_mask(sm, sd, em, ed, spans_years=False):
        if spans_years:
            return ((month_arr == sm) & (day_arr >= sd)) | ((month_arr > sm) | (month_arr < em)) | \
                   ((month_arr == em) & (day_arr <= ed))
        elif sm == em:
            return (month_arr == sm) & (day_arr >= sd) & (day_arr <= ed)
        else:
            return ((month_arr == sm) & (day_arr >= sd)) | \
                   ((month_arr > sm) & (month_arr < em)) | \
                   ((month_arr == em) & (day_arr <= ed))

    season_defs = {
        'SON': dict(sm=9,  sd=1, em=11, ed=30),
        'DJF': dict(sm=12, sd=1, em=2,  ed=29, spans_years=True),
        'MAM': dict(sm=3,  sd=1, em=5,  ed=31),
        'JJA': dict(sm=6,  sd=1, em=8,  ed=31),
    }

    print("  Detecting 5-day rolling heatwave events …")
    records = []
    for win_name, wdef in season_defs.items():
        mask   = season_mask(**wdef)
        t_idx  = np.where(mask)[0]
        if len(t_idx) < 5:
            continue
        win_data  = data[:, t_idx]
        win_dates = [all_dates[i] for i in t_idx]
        n_win_t   = len(t_idx)

        for lv_idx, (level, q_perc) in enumerate([('p90', 0), ('p95', 1), ('p99', 2)]):
            q_thresh = thresholds[q_perc, :][:, np.newaxis]
            exceeds  = (win_data > q_thresh).astype('int8')

            cs = np.cumsum(exceeds, axis=1)
            cs_shifted = np.zeros_like(cs)
            cs_shifted[:, 5:] = cs[:, :-5]
            rolling = cs - cs_shifted
            event   = (rolling[:, 4:] >= 5)

            px_idx, ti_idx = np.where(event)
            for k in range(len(px_idx)):
                px = px_idx[k]
                ti = ti_idx[k] + 4
                d  = win_dates[ti]
                records.append({
                    'date':        pd.Timestamp(d.year, d.month, d.day),
                    'lat':         float(valid_lats[px]),
                    'lon':         float(valid_lons[px]),
                    'activacion':  level,
                    'window':      win_name,
                    'region_name': valid_reg[px],
                    'country':     valid_cty[px],
                })
        print(f"    {win_name}: {sum(1 for r in records if r['window']==win_name):,} raw events")

    print(f"  Total raw event records: {len(records):,}")

    df = pd.DataFrame(records)
    if not df.empty:
        act_ord = {'p90': 0, 'p95': 1, 'p99': 2}
        df['_r'] = df['activacion'].map(act_ord)
        df = (df.sort_values('_r', ascending=False)
                .drop_duplicates(subset=['date', 'lat', 'lon'], keep='first')
                .drop(columns=['_r'])
                .reset_index(drop=True))
        df['pixel_id'] = df['lat'].apply(lambda v: f'{v:.4f}') + '_' + \
                         df['lon'].apply(lambda v: f'{v:.4f}')
        print(f"  After dedup: {len(df):,} records")
    else:
        df = pd.DataFrame(columns=['date','lat','lon','activacion','window',
                                   'region_name','country','pixel_id'])

    SA_HW_RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SA_HW_RAW, index=False)
    print(f"  Saved → {SA_HW_RAW}  ({SA_HW_RAW.stat().st_size/1e6:.1f} MB)")

print("\n=== STEP 2: Aggregate to region level ===")
hw_raw = pd.read_parquet(SA_HW_RAW)
print(f"  Raw records: {len(hw_raw):,}")
print(f"  Columns: {hw_raw.columns.tolist()}")

hw_raw['date'] = pd.to_datetime(hw_raw['date'])
hw_raw['year'] = hw_raw['date'].dt.year
hw_raw['year_win'] = hw_raw.apply(
    lambda r: r['year'] + 1 if (r['window'] == 'DJF' and r['date'].month == 12) else r['year'],
    axis=1
)

agg = (hw_raw[hw_raw['region_name'].notna()]
       .groupby(['region_name', 'year_win', 'window', 'activacion'])
       .size().reset_index(name='n_event_days'))

all_years  = list(range(hw_raw['year_win'].min(), hw_raw['year_win'].max() + 1))
idx = pd.MultiIndex.from_product(
    [sa_regions, all_years, ALL_WINDOWS, ALL_LEVELS],
    names=['region_name', 'year_win', 'window', 'activacion']
)
sa_hw_agg = (agg.set_index(['region_name', 'year_win', 'window', 'activacion'])
                .reindex(idx, fill_value=0).reset_index())
print(f"  SA aggregated: {sa_hw_agg.shape}  ({sa_hw_agg['region_name'].nunique()} regions)")

print("\n=== STEP 3: Append to Caribbean aggregated file ===")
carib_hw = pd.read_parquet(CARIB_HW_AGG)
print(f"  Caribbean rows: {len(carib_hw):,}  ({carib_hw['region_name'].nunique()} regions)")
carib_hw = carib_hw[~carib_hw['region_name'].isin(sa_regions)]
merged_hw = pd.concat([carib_hw, sa_hw_agg], ignore_index=True)
merged_hw.to_parquet(CARIB_HW_AGG, index=False)
print(f"  Merged: {len(merged_hw):,} rows  ({merged_hw['region_name'].nunique()} regions)")

print("\n=== STEP 4: Update master panel ===")
panel = pd.read_parquet(PANEL_PATH)
print(f"  Panel shape before: {panel.shape}")

hw_all = pd.read_parquet(CARIB_HW_AGG)
hw_p90 = hw_all[hw_all['activacion'] == 'p90'].copy()
hw_flag = (hw_p90.groupby(['region_name', 'year_win', 'window'])['n_event_days']
           .sum().reset_index())
hw_flag['hw_flag'] = (hw_flag['n_event_days'] > 0).astype(int)
hw_panel = (hw_flag.groupby(['region_name', 'year_win'])['hw_flag']
            .mean().reset_index()
            .rename(columns={'year_win': 'year', 'hw_flag': 'heatwave_new'}))

panel = panel.merge(hw_panel, on=['region_name', 'year'], how='left')
panel['heatwave'] = panel['heatwave_new'].fillna(panel['heatwave'])
panel = panel.drop(columns=['heatwave_new'])
panel.to_parquet(PANEL_PATH, index=False)
print(f"  Panel shape after:  {panel.shape}")

print("\n  Heatwave check — SA regions with most events in 2023:")
hw23 = (hw_raw[hw_raw['activacion'] == 'p90']
        .assign(year=lambda d: d['date'].dt.year)
        .query('year == 2023')
        .groupby(['country', 'region_name'])
        .size().reset_index(name='event_days')
        .sort_values('event_days', ascending=False).head(8))
print(hw23.to_string(index=False))

print("\n=== SA02_heatwave_only.py COMPLETE ===")
