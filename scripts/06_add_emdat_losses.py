#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import json
import re
import unicodedata
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
from rapidfuzz import process as rfprocess, fuzz

_emdat_dir  = Path('/home/jupyter-daniela/CAF/data/EMDAT')
_emdat_files = sorted(_emdat_dir.glob('public_emdat_*.xlsx'))
if not _emdat_files:
    raise FileNotFoundError(f'No EM-DAT xlsx in {_emdat_dir}')
EMDAT_PATH  = _emdat_files[0]
SHAPEFILE   = Path('/home/jupyter-daniela/CAF/data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp')
RUN_NAME    = SHAPEFILE.stem.lower()
BASE        = Path('/home/jupyter-daniela/CAF/data/processed') / RUN_NAME
PANEL_PATH  = BASE / 'master_panel.parquet'
OUTPUT_PATH = BASE / 'master_panel.parquet'

VSL_MAP = {
    'Haiti':                           1.1,
    'Dominican Republic':              1.1,
    'Jamaica':                         1.1,
    'Bahamas':                         7.9,
    'Dominica':                        1.1,
    'Barbados':                        7.9,
    'Trinidad and Tobago':             7.9,
    'Saint Lucia':                     1.1,
    'Colombia':                        1.1,
    'Peru':                            1.1,
    'Guyana':                          1.1,
}
VSL_DEFAULT = 1.1

CARIB_COUNTRIES = [
    'Anguilla', 'Antigua and Barbuda', 'Bahamas', 'Barbados', 'British Virgin Islands',
    'Cayman Islands', 'Cuba', 'Dominica', 'Dominican Republic', 'Grenada', 'Guadeloupe',
    'Haiti', 'Jamaica', 'Martinique', 'Montserrat', 'Netherlands Antilles', 'Puerto Rico',
    'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin (French Part)',
    'Saint Vincent and the Grenadines', 'Sint Maarten (Dutch part)',
    'Trinidad and Tobago', 'Turks and Caicos Islands', 'United States Virgin Islands',
    'Colombia', 'Peru', 'Guyana',
]

FUZZY_THRESHOLD = 75

def normalize(s):
    if not isinstance(s, str):
        return ''
    s = unicodedata.normalize('NFD', s.lower())
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn').strip()

def parse_admin_units(raw):
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        units = json.loads(raw)
        names = []
        for u in units:
            for k, v in u.items():
                if 'name' in k and isinstance(v, str):
                    names.append(v)
        return names
    except Exception:
        return []

def fuzzy_match_regions(candidate_names, region_pool, region_norms):
    matched = set()
    for name in candidate_names:
        norm = normalize(name)
        if not norm:
            continue
        result = rfprocess.extractOne(norm, region_norms, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= FUZZY_THRESHOLD:
            matched.add(region_pool[result[2]])
    return matched

def assign_event_to_regions(row, gdf, country_region_map, region_pool, region_norms):
    country = row['Country']
    country_regions = country_region_map.get(country, [])

    if pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')):
        try:
            pt = Point(float(row['Longitude']), float(row['Latitude']))
            hits = gdf[gdf.geometry.contains(pt)]['region_name'].tolist()
            if hits:
                return hits
        except Exception:
            pass

    adm_names = parse_admin_units(row.get('Admin Units'))
    if adm_names and country_regions:
        cr_norms = [normalize(r) for r in country_regions]
        matched = fuzzy_match_regions(adm_names, country_regions, cr_norms)
        if matched:
            return list(matched)

    loc = row.get('Location', '')
    if isinstance(loc, str) and loc.strip() and country_regions:
        parts = [p.strip() for p in re.split(r'[,;/]', loc) if p.strip()]
        cr_norms = [normalize(r) for r in country_regions]
        matched = fuzzy_match_regions(parts, country_regions, cr_norms)
        if matched:
            return list(matched)

    return country_regions if country_regions else []

print("Loading EMDAT...")
df_em = pd.read_excel(EMDAT_PATH, sheet_name='EM-DAT Data')
df_em = df_em[df_em['Country'].isin(CARIB_COUNTRIES)].copy()
print(f"  {len(df_em)} Caribbean events")

df_em = df_em[df_em['Start Year'] >= 1995].copy()
print(f"  {len(df_em)} events from 1995 onwards")

CLIMATE_TYPES = ['Storm', 'Flood', 'Drought']
df_em = df_em[df_em['Disaster Type'].isin(CLIMATE_TYPES)].copy()
print(f"  {len(df_em)} climatological events (Storm / Flood / Drought)")

df_em['vsl'] = df_em['Country'].map(VSL_MAP).fillna(VSL_DEFAULT)
df_em['losses_deaths_musd'] = (
    df_em['Total Deaths'].fillna(0) * df_em['vsl']
)
df_em['total_damage_musd'] = (
    df_em["Total Damage, Adjusted ('000 US$)"].fillna(0) / 1000.0
)
df_em['total_losses_musd'] = df_em['losses_deaths_musd'] + df_em['total_damage_musd']
df_em['year'] = df_em['Start Year'].astype(int)

print(f"  Loss columns computed. Events with deaths: {(df_em['losses_deaths_musd']>0).sum()}, "
      f"with damage: {(df_em['total_damage_musd']>0).sum()}")

print("\nLoading shapefile...")
gdf = gpd.read_file(SHAPEFILE)
gdf = gdf.rename(columns={'region_nam': 'region_name'})
if gdf.crs is None:
    gdf = gdf.set_crs('EPSG:4326')
elif gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs('EPSG:4326')
print(f"  {len(gdf)} regions in {gdf['country'].nunique()} countries")

country_region_map = (
    gdf.groupby('country')['region_name']
    .apply(list)
    .to_dict()
)

emdat_to_shp_country = {
    'Bahamas':              'Bahamas',
    'Barbados':             'Barbados',
    'Dominica':             'Dominica',
    'Dominican Republic':   'Dominican Republic',
    'Haiti':                'Haiti',
    'Jamaica':              'Jamaica',
    'Saint Lucia':          'Saint Lucia',
    'Trinidad and Tobago':  'Trinidad and Tobago',
    'Colombia':             'Colombia',
    'Peru':                 'Peru',
    'Guyana':               'Guyana',
}

country_region_map_emdat = {}
for emdat_c, shp_c in emdat_to_shp_country.items():
    if shp_c in country_region_map:
        country_region_map_emdat[emdat_c] = country_region_map[shp_c]

region_pool  = gdf['region_name'].tolist()
region_norms = [normalize(r) for r in region_pool]

print("\nMatching events to regions...")
records = []
matched_pip   = 0
matched_adm   = 0
matched_loc   = 0
matched_ctry  = 0
unmatched     = 0

for _, row in df_em.iterrows():
    country = row['Country']
    country_regions = country_region_map_emdat.get(country, [])

    matched_regions = None

    if pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')):
        try:
            pt = Point(float(row['Longitude']), float(row['Latitude']))
            hits = gdf[gdf.geometry.contains(pt)]['region_name'].tolist()
            if hits:
                matched_regions = hits
                matched_pip += 1
        except Exception:
            pass

    if matched_regions is None:
        adm_names = parse_admin_units(row.get('Admin Units'))
        if adm_names and country_regions:
            cr_norms = [normalize(r) for r in country_regions]
            matched = fuzzy_match_regions(adm_names, country_regions, cr_norms)
            if matched:
                matched_regions = list(matched)
                matched_adm += 1

    if matched_regions is None:
        loc = row.get('Location', '')
        if isinstance(loc, str) and loc.strip() and country_regions:
            parts = [p.strip() for p in re.split(r'[,;/]', loc) if p.strip()]
            cr_norms = [normalize(r) for r in country_regions]
            matched = fuzzy_match_regions(parts, country_regions, cr_norms)
            if matched:
                matched_regions = list(matched)
                matched_loc += 1

    if matched_regions is None:
        if country_regions:
            matched_regions = country_regions
            matched_ctry += 1
        else:
            unmatched += 1
            continue

    n = len(matched_regions)
    for rname in matched_regions:
        records.append({
            'region_name':         rname,
            'year':                row['year'],
            'losses_deaths_musd':  row['losses_deaths_musd'] / n,
            'total_damage_musd':   row['total_damage_musd']  / n,
            'total_losses_musd':   row['total_losses_musd']  / n,
        })

print(f"  Match summary: PiP={matched_pip}, AdminUnits={matched_adm}, "
      f"Location={matched_loc}, CountryFallback={matched_ctry}, Unmatched={unmatched}")

print("\nAggregating to region-year...")
df_losses = (
    pd.DataFrame(records)
    .groupby(['region_name', 'year'])[['losses_deaths_musd', 'total_damage_musd', 'total_losses_musd']]
    .sum()
    .reset_index()
)
print(f"  {len(df_losses)} region-year rows with non-zero losses")

print("\nMerging into master panel...")
panel = pd.read_parquet(PANEL_PATH)
print(f"  Panel shape before: {panel.shape}")

for col in ['losses_deaths_musd', 'total_damage_musd', 'total_losses_musd']:
    if col in panel.columns:
        panel = panel.drop(columns=[col])

panel = panel.merge(df_losses, on=['region_name', 'year'], how='left')
for col in ['losses_deaths_musd', 'total_damage_musd', 'total_losses_musd']:
    panel[col] = panel[col].fillna(0.0)

print(f"  Panel shape after:  {panel.shape}")
print(f"  Rows with any loss > 0: {(panel['total_losses_musd'] > 0).sum():,}")

_tmp = OUTPUT_PATH.with_suffix('.tmp.parquet')
panel.to_parquet(_tmp, index=False)
_tmp.rename(OUTPUT_PATH)
print(f"\nSaved → {OUTPUT_PATH}")

print("\nTop 10 region-years by total losses:")
top = (
    panel[panel['total_losses_musd'] > 0]
    [['region_name', 'year', 'losses_deaths_musd', 'total_damage_musd', 'total_losses_musd']]
    .drop_duplicates(subset=['region_name','year'])
    .sort_values('total_losses_musd', ascending=False)
    .head(10)
)
print(top.to_string(index=False))
