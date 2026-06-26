#!/usr/bin/env python3
import os
import datetime
import calendar
import numpy as np
from pathlib import Path
import pandas as pd
import requests
import xarray as xr
import rasterio
from rasterio.windows import from_bounds
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

NORTH      = 26
SOUTH      = 6.5
EAST       = -56
WEST       = -93
START_YEAR = 1995
END_YEAR   = 2025
OUTPUT_DIR = Path('/home/jupyter-daniela/CAF/data/chirps_v3_data')
BASE_URL   = 'https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/'
MAX_WORKERS = 10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print_lock = Lock()

def log(msg):
    with print_lock:
        print(msg)

def process_day(date_obj):
    year, month, day = date_obj.year, date_obj.month, date_obj.day
    filename  = f'chirps-v3.0.rnl.{year}.{month:02d}.{day:02d}.tif'
    url       = f'{BASE_URL}{year}/{filename}'
    temp_file = str(OUTPUT_DIR / f'temp_{year}_{month}_{day}.tif')
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            if r.status_code != 200:
                return date_obj, None, None
            with open(temp_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        with rasterio.open(temp_file) as src:
            window = from_bounds(WEST, SOUTH, EAST, NORTH, src.transform)
            data   = src.read(1, window=window)
            tf     = src.window_transform(window)
        os.remove(temp_file)
        return date_obj, data, tf
    except Exception:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return date_obj, None, None

def process_year(year):
    out_file = str(OUTPUT_DIR / f'chirps-v3.0.{year}.days_p05_RGS.nc')
    if os.path.exists(out_file):
        log(f'[SKIP] {year}')
        return year, 'skipped', 0

    days      = 366 if calendar.isleap(year) else 365
    date_list = [datetime.date(year, 1, 1) + datetime.timedelta(days=x) for x in range(days)]

    daily_results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(process_day, d): d for d in date_list}
        done = 0
        for future in as_completed(futures):
            daily_results.append(future.result())
            done += 1
            if done % 20 == 0:
                log(f'[...] {year} {done}/{len(date_list)}')

    daily_results.sort(key=lambda x: x[0])
    valid = [x for x in daily_results if x[1] is not None]

    if len(valid) < 300:
        log(f'[✗] {year} too much missing ({len(valid)}/{len(date_list)})')
        return year, 'failed', 0

    ref_shape = valid[0][1].shape
    ref_tf    = valid[0][2]
    stack     = np.full((len(date_list), *ref_shape), np.nan, dtype=np.float32)
    idx_map   = {d: i for i, d in enumerate(date_list)}
    for d, arr, _ in valid:
        if arr.shape == ref_shape:
            stack[idx_map[d]] = arr

    height, width = ref_shape
    xs, _ = rasterio.transform.xy(ref_tf, [0] * width, range(width), offset='center')
    _, ys = rasterio.transform.xy(ref_tf, range(height), [0] * height, offset='center')

    ds = xr.Dataset(
        {'precip': (('time', 'latitude', 'longitude'), stack)},
        coords={'time': pd.to_datetime(date_list), 'latitude': ys, 'longitude': xs},
    )
    ds.to_netcdf(out_file, encoding={'precip': {'zlib': True, 'complevel': 5}})
    size = os.path.getsize(out_file) / 1024 ** 2
    log(f'[✓] {year} ({size:.1f} MB)')
    return year, 'success', size

if __name__ == '__main__':
    years   = list(range(START_YEAR, END_YEAR + 1))
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_year, y): y for y in years}
        for future in as_completed(futures):
            results.append(future.result())

    ok      = [r for r in results if r[1] == 'success']
    failed  = [r for r in results if r[1] == 'failed']
    skipped = [r for r in results if r[1] == 'skipped']
    print(f'\n✓ {len(ok)}  →  {sum(r[2] for r in ok):.1f} MB')
    print(f'→ skipped: {len(skipped)}')
    print(f'✗ failed:  {len(failed)}')
