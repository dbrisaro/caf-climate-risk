#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

from pathlib import Path
from pyproj import datadir
from utils import load_shapefile
from exposure_utils import run_exposure, exposure_to_dataframe

datadir.set_data_dir('/home/jupyter-daniela/.conda/envs/peru_environment/share/proj')

BASE_DIR  = Path('/home/jupyter-daniela/CAF')
GHS_DIR   = str(BASE_DIR / 'data/GHS')
MERIT_DIR = str(BASE_DIR / 'data/MERIT')

SHAPEFILE = '/home/jupyter-daniela/CAF/data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
RUN_NAME  = Path(SHAPEFILE).stem.lower()
PROCESSED = BASE_DIR / 'data/processed' / RUN_NAME
PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUT    = PROCESSED / 'exposure_panel.parquet'
NAME_COL  = 'region_nam'
NAME_MAP  = None
TARGET_REGIONS = None
YEARS  = [1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]
TILES  = [
    'R6_C11', 'R7_C11', 'R7_C12', 'R8_C12', 'R8_C13',
    'R9_C11', 'R9_C12', 'R9_C13',
    'R10_C11', 'R10_C12',
    'R11_C11', 'R11_C12',
    'R12_C11',
]
LAYERS = ['BUILT_S', 'BUILT_V', 'POP']

targets = load_shapefile(SHAPEFILE, NAME_COL, TARGET_REGIONS, NAME_MAP)
print(f"Loaded {len(targets)} regions: {targets['region_name'].tolist()}")

region_series = run_exposure(GHS_DIR, MERIT_DIR, targets, YEARS, TILES, LAYERS)

df = exposure_to_dataframe(region_series)
df.to_parquet(OUTPUT, index=False)
print(f"\nSaved {len(df):,} rows → {OUTPUT}")
print(df.to_string())
