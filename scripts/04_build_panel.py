#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import pandas as pd
import numpy as np
from pathlib import Path
from pyproj import datadir

datadir.set_data_dir('/home/jupyter-daniela/.conda/envs/peru_environment/share/proj')

RUN_NAME = 'caribbean_adm_boundaries'
BASE     = Path('/home/jupyter-daniela/CAF/data/processed') / RUN_NAME
OUTPUT   = BASE / 'master_panel.parquet'

EXPOSURE_PATH  = BASE / 'exposure_panel.parquet'
RAINFALL_PATH  = BASE / 'era5_output_precipitation/rainfall_aggregated_data.parquet'
DROUGHT_PATH   = BASE / 'era5_output/droughts_aggregated_data.parquet'
HEATWAVE_PATH  = BASE / 'heatwaves_computation/heatwaves_aggregated_data.parquet'
CYCLONE_PATH   = BASE / 'ibtracs_output/ibtracs_aggregated_data.parquet'

TARGET_REGIONS = sorted(pd.read_parquet(EXPOSURE_PATH)['region_name'].unique().tolist())
ALL_WINDOWS = ['SON', 'DJF', 'MAM', 'JJA']
ALL_YEARS   = range(1995, 2031)

panel = pd.DataFrame(
    pd.MultiIndex.from_product(
        [TARGET_REGIONS, list(ALL_YEARS), ALL_WINDOWS],
        names=['region_name', 'year', 'window']
    ).to_frame(index=False)
)

EXPOSURE_COLS = ['exposure_index', 'var_exp', 'pop', 'built_s', 'built_v', 'dem']

df_exposure = pd.read_parquet(EXPOSURE_PATH)
panel = panel.merge(df_exposure, on=['region_name', 'year'], how='left')

for col in EXPOSURE_COLS:
    panel[col] = (
        panel.groupby('region_name')[col]
        .transform(lambda s: s.interpolate(method='linear', limit_direction='both'))
    )

df_drought = (
    pd.read_parquet(DROUGHT_PATH)
    .pivot_table(index=['region_name', 'year', 'window'], columns='activacion',
                 values='n_drought_pixels', fill_value=0)
    .reset_index()
    .rename(columns={'p1': 'drought_p1', 'p5': 'drought_p5', 'p10': 'drought_p10'})
)
df_drought.columns.name = None
panel = panel.merge(df_drought, on=['region_name', 'year', 'window'], how='left')

df_rainfall = (
    pd.read_parquet(RAINFALL_PATH)
    .pivot_table(index=['region_name', 'year', 'window'], columns='activacion',
                 values='n_rainfall_pixels', fill_value=0)
    .reset_index()
    .rename(columns={'p90': 'rainfall_p90', 'p95': 'rainfall_p95', 'p99': 'rainfall_p99'})
)
df_rainfall.columns.name = None
panel = panel.merge(df_rainfall, on=['region_name', 'year', 'window'], how='left')

df_hw = (
    pd.read_parquet(HEATWAVE_PATH)
    .pivot_table(index=['region_name', 'year_win', 'window'], columns='activacion',
                 values='n_event_days', fill_value=0)
    .reset_index()
    .rename(columns={'year_win': 'year', 'p90': 'hw_p90', 'p95': 'hw_p95', 'p99': 'hw_p99'})
)
df_hw.columns.name = None
panel = panel.merge(df_hw, on=['region_name', 'year', 'window'], how='left')

df_cyc = (
    pd.read_parquet(CYCLONE_PATH)
    .rename(columns={
        'year_win':            'year',
        'frequency':           'cyclone_frequency',
        'avg_max_storm_speed': 'cyclone_avg_storm_speed',
        'avg_max_usa_wind':    'cyclone_avg_wind_kt',
    })
)
panel = panel.merge(df_cyc, on=['region_name', 'year', 'window'], how='left')

num_cols = panel.select_dtypes(include='number').columns
panel[num_cols] = panel[num_cols].fillna(0)

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

panel['drought'] = panel['drought_p10']
panel.drop(columns=['drought_p1', 'drought_p5', 'drought_p10'], inplace=True)

panel['rainfall'] = panel['rainfall_p90']
panel.drop(columns=['rainfall_p90', 'rainfall_p95', 'rainfall_p99'], inplace=True)

panel['heatwave'] = panel['hw_p90']
panel.drop(columns=['hw_p90', 'hw_p95', 'hw_p99'], inplace=True)

cyc_cols = ['cyclone_frequency', 'cyclone_avg_storm_speed', 'cyclone_avg_wind_kt']
cyc_valid = panel[cyc_cols].sum(axis=1) > 0
if cyc_valid.sum() > 1:
    X = panel.loc[cyc_valid, cyc_cols].values
    X_norm = StandardScaler().fit_transform(np.log1p(X))
    pc1 = PCA(n_components=1, random_state=42).fit_transform(X_norm)[:, 0]
    pc1_min, pc1_max = pc1.min(), pc1.max()
    pc1 = (pc1 - pc1_min) / (pc1_max - pc1_min) if pc1_max > pc1_min else pc1
    panel['cyclone'] = 0.0
    panel.loc[cyc_valid, 'cyclone'] = pc1
else:
    panel['cyclone'] = panel['cyclone_frequency']
panel.drop(columns=cyc_cols, inplace=True)

for col in ['drought', 'rainfall', 'heatwave']:
    col_min, col_max = panel[col].min(), panel[col].max()
    if col_max > col_min:
        panel[col] = (panel[col] - col_min) / (col_max - col_min)

panel = panel.sort_values(['region_name', 'year', 'window']).reset_index(drop=True)
panel.to_parquet(OUTPUT, index=False)
print(f"Master panel saved → {OUTPUT}")
print(f"Shape: {panel.shape}")
print(f"Columns: {panel.columns.tolist()}")
