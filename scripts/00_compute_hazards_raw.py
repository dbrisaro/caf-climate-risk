#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import pandas as pd
from pathlib import Path
from pyproj import datadir
from utils import load_shapefile
from hazard_utils import run_rainfall_raw, run_droughts_raw, run_heatwaves_raw

datadir.set_data_dir('/home/jupyter-daniela/.conda/envs/peru_environment/share/proj')

BASE_DIR  = Path('/home/jupyter-daniela/CAF')

SHAPEFILE = '/home/jupyter-daniela/CAF/data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
RUN_NAME  = Path(SHAPEFILE).stem.lower()
PROCESSED = BASE_DIR / 'data/processed' / RUN_NAME
PROCESSED.mkdir(parents=True, exist_ok=True)
NAME_COL  = 'region_nam'

CARIB_COUNTRIES = {
    'Bahamas', 'Barbados', 'Dominica', 'Dominican Republic',
    'Haiti', 'Jamaica', 'Saint Lucia', 'Trinidad and Tobago',
}
SA_COUNTRIES = {'Colombia', 'Peru', 'Guyana'}

ERA5_RAINFALL_CARIB  = Path('/home/jupyter-daniela/suyana/sources/era5/total_precipitation/caribbean')
ERA5_DROUGHT_CARIB   = Path('/home/jupyter-daniela/suyana/sources/era5/swc/caribbean')
ERA5_HEATWAVE_CARIB  = Path('/home/jupyter-daniela/suyana/sources/era5/statistics/tmax/caribbean')

ERA5_RAINFALL_SA     = Path('/home/jupyter-daniela/suyana/sources/era5/total_precipitation/south_america')
ERA5_DROUGHT_SA      = Path('/home/jupyter-daniela/suyana/sources/era5/swc/south_america')
ERA5_HEATWAVE_SA     = Path('/home/jupyter-daniela/suyana/sources/era5/statistics/tmax/south_america')

all_targets = load_shapefile(SHAPEFILE, NAME_COL, None, None)

import geopandas as gpd
_shp = gpd.read_file(SHAPEFILE)[['region_nam', 'country']].rename(columns={'region_nam': 'region_name'})
all_targets = all_targets.merge(_shp.drop_duplicates('region_name'), on='region_name', how='left')
print(f"Loaded {len(all_targets)} regions across {all_targets['country'].nunique()} countries\n")

carib_targets = all_targets[all_targets['country'].isin(CARIB_COUNTRIES)].copy()
sa_targets    = all_targets[all_targets['country'].isin(SA_COUNTRIES)].copy()
print(f"Caribbean: {len(carib_targets)} regions | SA: {len(sa_targets)} regions\n")

def run_and_merge(run_fn, carib_dir, sa_dir, carib_gdf, sa_gdf, out_path, tmp_suffix):
    import pyarrow.parquet as pq

    tmp_carib = out_path.parent / f'_tmp_carib_{tmp_suffix}.parquet'
    tmp_sa    = out_path.parent / f'_tmp_sa_{tmp_suffix}.parquet'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not tmp_carib.exists():
        print(f"  → Caribbean regions …")
        run_fn(era5_dir=carib_dir, gdf=carib_gdf, output_path=tmp_carib)
    else:
        print(f"  → Caribbean regions … [skip, tmp exists]")

    if not tmp_sa.exists():
        print(f"  → SA regions …")
        run_fn(era5_dir=sa_dir, gdf=sa_gdf, output_path=tmp_sa)
    else:
        print(f"  → SA regions … [skip, tmp exists]")

    print(f"  Merging …")
    schema = pq.read_schema(tmp_carib)
    total  = 0
    with pq.ParquetWriter(str(out_path), schema) as writer:
        for src in [tmp_carib, tmp_sa]:
            pf = pq.ParquetFile(src)
            for batch in pf.iter_batches(batch_size=500_000):
                writer.write_batch(batch)
                total += batch.num_rows
    tmp_carib.unlink(missing_ok=True)
    tmp_sa.unlink(missing_ok=True)
    print(f"  Merged → {out_path}  ({total:,} rows)\n")

print("=== Rainfall ===")
run_and_merge(
    run_rainfall_raw,
    ERA5_RAINFALL_CARIB, ERA5_RAINFALL_SA,
    carib_targets, sa_targets,
    PROCESSED / 'era5_output_precipitation/era5_raw_data.parquet',
    'rainfall',
)

print("=== Droughts ===")
run_and_merge(
    run_droughts_raw,
    ERA5_DROUGHT_CARIB, ERA5_DROUGHT_SA,
    carib_targets, sa_targets,
    PROCESSED / 'era5_output/era5_raw_data.parquet',
    'drought',
)

print("=== Heatwaves ===")
run_and_merge(
    run_heatwaves_raw,
    ERA5_HEATWAVE_CARIB, ERA5_HEATWAVE_SA,
    carib_targets, sa_targets,
    PROCESSED / 'heatwaves_computation/heatwave_events_raw.parquet',
    'heatwave',
)
