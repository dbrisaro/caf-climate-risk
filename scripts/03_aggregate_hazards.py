#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

from pathlib import Path
from pyproj import datadir
from utils import load_shapefile
from aggregation import run_cyclones, run_droughts, run_heatwaves, run_rainfall

datadir.set_data_dir('/home/jupyter-daniela/.conda/envs/peru_environment/share/proj')

BASE_DIR = Path('/home/jupyter-daniela/CAF')

SHAPEFILE = '/home/jupyter-daniela/CAF/data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
RUN_NAME  = Path(SHAPEFILE).stem.lower()
PROCESSED = BASE_DIR / 'data/processed' / RUN_NAME
PROCESSED.mkdir(parents=True, exist_ok=True)
NAME_COL    = 'region_nam'
NAME_MAP    = None
ALL_WINDOWS = ['SON', 'DJF', 'MAM', 'JJA']

targets        = load_shapefile(SHAPEFILE, NAME_COL, None, NAME_MAP)
TARGET_REGIONS = targets['region_name'].tolist()
print(f"Loaded {len(targets)} regions\n")

run_cyclones(BASE_DIR, targets, TARGET_REGIONS, ALL_WINDOWS, processed_dir=PROCESSED)
run_droughts(BASE_DIR, TARGET_REGIONS, ALL_WINDOWS, processed_dir=PROCESSED)
run_heatwaves(BASE_DIR, targets, TARGET_REGIONS, ALL_WINDOWS, processed_dir=PROCESSED)
run_rainfall(BASE_DIR, TARGET_REGIONS, ALL_WINDOWS, processed_dir=PROCESSED)
