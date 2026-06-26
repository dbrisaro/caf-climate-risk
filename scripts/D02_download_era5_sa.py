#!/usr/bin/env python3
import cdsapi
import os
import xarray as xr
from pathlib import Path

OUT_PRECIP  = Path('/home/jupyter-daniela/suyana/sources/era5/total_precipitation/south_america')
OUT_SWC     = Path('/home/jupyter-daniela/suyana/sources/era5/swc/south_america')
OUT_TMAX    = Path('/home/jupyter-daniela/suyana/sources/era5/statistics/tmax/south_america')

for d in [OUT_PRECIP, OUT_SWC, OUT_TMAX]:
    d.mkdir(parents=True, exist_ok=True)

AREA   = [13, -83, -19, -55]
AREA_LABEL = '13N_83W_19S_55W'
YEARS  = list(range(1995, 2026))
MONTHS = [f'{m:02d}' for m in range(1, 13)]
DAYS   = [f'{d:02d}' for d in range(1, 32)]

c = cdsapi.Client()

def download(dataset, request, out_path):
    if out_path.exists():
        print(f'  [skip] {out_path.name}')
        return
    print(f'  [download] {out_path.name}')
    c.retrieve(dataset, request, str(out_path))
    print(f'  [done] {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)')

print('\n=== Precipitation ===')
for year in YEARS:
    out = OUT_PRECIP / f'reanalysis-era5-land_total_precipitation_{year}_daily_{AREA_LABEL}_0.1deg.nc'
    download('reanalysis-era5-land', {
        'variable': 'total_precipitation',
        'year': str(year),
        'month': MONTHS,
        'day': DAYS,
        'time': '00:00',
        'area': AREA,
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    }, out)

print('\n=== Soil water content ===')
for year in YEARS:
    out = OUT_SWC / f'reanalysis-era5-land_volumetric_soil_water_layer_{year}_daily_{AREA_LABEL}_0.1deg.nc'
    download('reanalysis-era5-land', {
        'variable': 'volumetric_soil_water_layer_1',
        'year': str(year),
        'month': MONTHS,
        'day': DAYS,
        'time': '00:00',
        'area': AREA,
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    }, out)

print('\n=== 2m max temperature ===')
TMAX_TIMES = ['15:00', '18:00', '21:00', '23:00']
for year in YEARS:
    out = OUT_TMAX / f'era5_tmax_daily_{year}_area_{AREA_LABEL}.nc'
    if out.exists():
        print(f'  [skip] {out.name}')
        continue
    tmp = OUT_TMAX / f'_tmp_t2m_{year}.nc'
    print(f'  [download] t2m {year} (4 times)...')
    c.retrieve('reanalysis-era5-land', {
        'variable': '2m_temperature',
        'year': str(year),
        'month': MONTHS,
        'day': DAYS,
        'time': TMAX_TIMES,
        'area': AREA,
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    }, str(tmp))
    print(f'  [processing] daily max...')
    ds = xr.open_dataset(tmp)
    time_dim = 'valid_time' if 'valid_time' in ds.coords else 'time'
    tvar = list(ds.data_vars)[0]
    ds_max = ds[tvar].resample({time_dim: '1D'}).max().to_dataset(name=tvar)
    ds_max.to_netcdf(out)
    ds.close()
    tmp.unlink(missing_ok=True)
    print(f'  [done] {out.name}  ({out.stat().st_size / 1e6:.1f} MB)')

print('\nAll downloads complete.')
