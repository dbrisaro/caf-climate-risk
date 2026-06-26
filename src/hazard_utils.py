import os
import glob
import re
import sys
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

def extract_year_era5(filename):
    name  = os.path.basename(filename)
    match = re.search(r'_(19|20)\d{2}_', name)
    if match:
        return int(match.group(0).strip('_'))
    return None

def load_era5_files(era5_data_dir, pattern):
    files = sorted(glob.glob(os.path.join(str(era5_data_dir), pattern)))
    file_years = [(f, extract_year_era5(f)) for f in files]
    file_years = sorted([(f, y) for f, y in file_years if y is not None], key=lambda x: x[1])
    if not file_years:
        sys.exit(f"No ERA5 files found matching: {pattern}")
    files = [f for f, _ in file_years]
    years = [y for _, y in file_years]
    print(f"  Years: {years[0]}–{years[-1]}  ({len(years)} files)")
    return files, years

def open_era5_dataset(era5_files, rename_var=None):
    ds = xr.open_mfdataset(era5_files, combine='by_coords', engine='netcdf4')
    if 'lat' in ds.coords and 'latitude' not in ds.coords:
        ds = ds.rename({'lat': 'latitude'})
    if 'lon' in ds.coords and 'longitude' not in ds.coords:
        ds = ds.rename({'lon': 'longitude'})
    if 'valid_time' not in ds.coords and 'time' in ds.coords:
        ds = ds.rename({'time': 'valid_time'})
    if rename_var:
        src_var, dst_var = rename_var
        if src_var in ds:
            ds[dst_var] = ds[src_var]
    return ds

def prepare_dataset(ds, max_year=2025, time_dim='valid_time'):
    ds = ds.sel({time_dim: ds[time_dim].dt.year <= max_year})
    ds = ds.assign_coords(dayofyear=ds[time_dim].dt.dayofyear)
    return ds

def crop_dataset_with_gdf(ds, gdf, buffer=0.05):
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs('EPSG:4326')
    try:
        ds = ds.rio.write_crs('EPSG:4326')
        ds_clipped = ds.rio.clip([gdf.unary_union], crs='EPSG:4326', drop=False)
        return ds_clipped
    except Exception:
        minx, miny, maxx, maxy = gdf.total_bounds
        return ds.sel(
            latitude=slice(maxy + buffer, miny - buffer),
            longitude=slice(minx - buffer, maxx + buffer)
        )

def compute_pixel_climatology(ds, start_year=1996, time_dim='valid_time'):
    ds = ds.assign_coords(dayofyear=ds[time_dim].dt.dayofyear)
    ds_clim = ds.sel({time_dim: ds[time_dim].dt.year >= start_year})
    return ds_clim.groupby('dayofyear').mean(time_dim)

def extract_window(ds, start_month, start_day, end_month, end_day,
                   window_name, time_dim='valid_time', spans_years=False):
    t = ds[time_dim]
    m, d = t.dt.month, t.dt.day
    if spans_years:
        mask = ((m == start_month) & (d >= start_day)) | ((m > start_month) | (m < end_month)) | ((m == end_month) & (d <= end_day))
    elif start_month == end_month:
        mask = (m == start_month) & (d >= start_day) & (d <= end_day)
    else:
        mask = ((m == start_month) & (d >= start_day)) | ((m > start_month) & (m < end_month)) | ((m == end_month) & (d <= end_day))
    return ds.sel({time_dim: mask}).assign_attrs(window=window_name)

def extract_annual_seasons(ds, time_dim='valid_time'):
    return {
        'SON': extract_window(ds, 9,  1, 11, 30, 'SON', time_dim),
        'DJF': extract_window(ds, 12, 1,  2, 29, 'DJF', time_dim, spans_years=True),
        'MAM': extract_window(ds, 3,  1,  5, 31, 'MAM', time_dim),
        'JJA': extract_window(ds, 6,  1,  8, 31, 'JJA', time_dim),
    }

def _run_era5_raw(era5_dir, gdf, output_path, pattern, var_name, data_source,
                  max_year=2025, rename_var=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    files, years = load_era5_files(era5_dir, pattern)

    print(f"  Computing climatology...")
    ds_all  = open_era5_dataset(files, rename_var=rename_var)
    ds_all  = prepare_dataset(ds_all, max_year=max_year)
    ds_all  = crop_dataset_with_gdf(ds_all, gdf)
    ds_clim = compute_pixel_climatology(ds_all).compute()
    lats    = ds_clim['latitude'].values
    lons    = ds_clim['longitude'].values
    ds_all.close()

    print(f"  Pre-computing pixel→region mapping ({len(lats)}×{len(lons)} grid)...")
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    flat_lats = lat_grid.ravel().astype('float64')
    flat_lons = lon_grid.ravel().astype('float64')
    pixel_ids    = np.array([f'{la:.4f}_{lo:.4f}' for la, lo in zip(flat_lats, flat_lons)])
    region_arr   = _assign_region(flat_lats, flat_lons, gdf)
    valid        = pd.notna(region_arr)
    pixel_ids_v  = pixel_ids[valid]
    regions_v    = np.array(region_arr)[valid]
    print(f"  {valid.sum():,} pixels inside regions out of {len(valid):,} total")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total  = 0
    for f, yr in zip(files, years):
        if yr > max_year:
            continue
        print(f"  {yr}...", end=' ', flush=True)
        ds_yr   = open_era5_dataset([f], rename_var=rename_var)
        ds_yr   = prepare_dataset(ds_yr, max_year=max_year)
        ds_yr   = crop_dataset_with_gdf(ds_yr, gdf)
        ds_yr   = ds_yr.assign_coords(dayofyear=ds_yr['valid_time'].dt.dayofyear)
        ds_yr   = ds_yr.chunk({'valid_time': -1})
        ds_anom = (ds_yr.groupby('dayofyear') - ds_clim).compute()
        seasons = extract_annual_seasons(ds_anom)

        parts = []
        for window_name, ds_win in seasons.items():
            arr_sum = ds_win[var_name].sum(dim='valid_time').values.ravel()
            vals    = arr_sum[valid].astype('float32')
            finite  = np.isfinite(vals)
            parts.append(pd.DataFrame({
                'date':        pd.Timestamp(f'{yr}-01-01'),
                'pixel_id':    pixel_ids_v[finite],
                'value':       vals[finite],
                'data_source': data_source,
                'window':      window_name,
                'region_name': regions_v[finite],
            }))

        df_yr = pd.concat(parts, ignore_index=True)
        print(f"  {data_source}: {len(df_yr):,} records  ({yr})")
        table = pa.Table.from_pandas(df_yr, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), table.schema)
        writer.write_table(table)
        total += len(df_yr)
        ds_yr.close()
        ds_anom.close()

    if writer:
        writer.close()
    print(f"\n  Saved {total:,} records → {output_path}")
    return pd.read_parquet(output_path)

def run_rainfall_raw(era5_dir, gdf, output_path, max_year=2025):
    print("Processing rainfall (ERA5 total precipitation)...")
    return _run_era5_raw(
        era5_dir, gdf, output_path,
        pattern='reanalysis-era5-land_total_precipitation_*.nc',
        var_name='tp', data_source='rainfall', max_year=max_year,
    )

def run_droughts_raw(era5_dir, gdf, output_path, max_year=2025):
    print("Processing droughts (ERA5 soil water)...")
    return _run_era5_raw(
        era5_dir, gdf, output_path,
        pattern='reanalysis-era5-land_volumetric_soil_water_layer_*.nc',
        var_name='soil_water_avg', data_source='drought', max_year=max_year,
        rename_var=('swvl1', 'soil_water_avg'),
    )

def run_heatwaves_raw(era5_dir, gdf, output_path, max_year=2025):
    print("Processing heatwaves (ERA5 tmax)...")
    files, _ = load_era5_files(era5_dir, 'era5_tmax_daily_*.nc')
    ds = open_era5_dataset(files)
    if 'valid_time' in ds.coords and 'time' not in ds.coords:
        ds = ds.rename({'valid_time': 'time'})

    ds = crop_dataset_with_gdf(ds, gdf)
    tmax_var = list(ds.data_vars)[0]
    ds[tmax_var] = ds[tmax_var] - 273.15

    ds = ds.chunk({'time': -1, 'latitude': 50, 'longitude': 50})
    thresholds = ds[tmax_var].quantile([0.90, 0.95, 0.99], dim='time')

    def detect_events(threshold):
        exceeds = ds[tmax_var] > thresholds.sel(quantile=threshold)
        return (exceeds.rolling(time=5, min_periods=5).sum() == 5.0)

    records = []
    for level, q in [('p90', 0.90), ('p95', 0.95), ('p99', 0.99)]:
        events  = detect_events(q)
        seasons = {
            'SON': events.sel(time=_season_mask(events['time'], 9,  1, 11, 30)),
            'DJF': events.sel(time=_season_mask(events['time'], 12, 1,  2, 29, spans_years=True)),
            'MAM': events.sel(time=_season_mask(events['time'], 3,  1,  5, 31)),
            'JJA': events.sel(time=_season_mask(events['time'], 6,  1,  8, 31)),
        }
        for window_name, ds_win in seasons.items():
            arr  = ds_win.values
            t    = ds_win.time.values
            lats = ds_win['latitude'].values
            lons = ds_win['longitude'].values
            ti, yi, xi = np.where(arr)
            for i in range(len(ti)):
                records.append({
                    'date':       pd.Timestamp(t[ti[i]]),
                    'lat':        float(lats[yi[i]]),
                    'lon':        float(lons[xi[i]]),
                    'activacion': level,
                    'window':     window_name,
                })

    df = pd.DataFrame(records)
    if not df.empty:
        df['activacion'] = pd.Categorical(df['activacion'], categories=['p90', 'p95', 'p99'], ordered=True)
        df = df.sort_values('activacion', ascending=False).drop_duplicates(subset=['date', 'lat', 'lon'], keep='first')
        df['pixel_id'] = df.apply(lambda r: f"{r['lat']:.4f}_{r['lon']:.4f}", axis=1)
        rc = _assign_region_with_attrs(df['lat'], df['lon'], gdf)
        df['region_name'] = rc['region_name'].values
        if 'country' in rc.columns:
            df['country'] = rc['country'].values

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Saved {len(df):,} event-days → {output_path}")
    return df

def _season_mask(t, start_month, start_day, end_month, end_day, spans_years=False):
    m, d = t.dt.month, t.dt.day
    if spans_years:
        return ((m == start_month) & (d >= start_day)) | ((m > start_month) | (m < end_month)) | ((m == end_month) & (d <= end_day))
    elif start_month == end_month:
        return (m == start_month) & (d >= start_day) & (d <= end_day)
    else:
        return ((m == start_month) & (d >= start_day)) | ((m > start_month) & (m < end_month)) | ((m == end_month) & (d <= end_day))

def _assign_region(lats, lons, gdf):
    return _assign_region_with_attrs(lats, lons, gdf)['region_name'].values

def _assign_region_with_attrs(lats, lons, gdf):
    import geopandas as gpd
    from shapely.geometry import Point
    pts = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(lons, lats)],
        crs='EPSG:4326'
    )
    extra_cols = ['region_name']
    if 'country' in gdf.columns:
        extra_cols.append('country')
    gdf_4326 = gdf[extra_cols + ['geometry']].copy()
    if gdf_4326.crs is None:
        gdf_4326 = gdf_4326.set_crs('EPSG:4326')
    elif gdf_4326.crs.to_epsg() != 4326:
        gdf_4326 = gdf_4326.to_crs('EPSG:4326')
    joined = pts.sjoin(gdf_4326, how='left', predicate='within')
    joined = joined[~joined.index.duplicated(keep='first')]
    result = joined[extra_cols].reindex(pts.index)
    return result
