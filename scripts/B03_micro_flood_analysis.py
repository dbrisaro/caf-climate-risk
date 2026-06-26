#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import requests
import warnings
import time
import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
from pathlib import Path
from shapely.geometry import box, Point

warnings.filterwarnings('ignore')

OVERPASS_ENDPOINTS = [
    'https://overpass-api.de/api/interpreter',
    'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
]
OVERPASS_TIMEOUT   = 60
MAX_RETRIES        = 3

def overpass_query(query, retries=MAX_RETRIES):
    for attempt in range(retries):
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                r = requests.get(endpoint, params={'data': query},
                                 timeout=OVERPASS_TIMEOUT)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
        if attempt < retries - 1:
            time.sleep(5 * (attempt + 1))
    return None

OSM_LAYERS = {
    'critical_infrastructure': {
        'airport':          [('aeroway', 'aerodrome|terminal')],
        'heliport':         [('aeroway', 'helipad')],
        'fuel_station':     [('amenity', 'fuel')],
        'charging_station': [('amenity', 'charging_station')],
        'government':       [('amenity', 'townhall'), ('office', 'government'),
                             ('building', 'government')],
    },
    'social_services': {
        'hospital':         [('amenity', 'hospital')],
        'clinic':           [('amenity', 'clinic')],
        'pharmacy':         [('amenity', 'pharmacy')],
        'school':           [('amenity', 'school')],
        'university':       [('amenity', 'university|college')],
        'police':           [('amenity', 'police')],
        'fire_station':     [('amenity', 'fire_station')],
    },
    'mobility': {
        'bus_station':      [('amenity', 'bus_station'),
                             ('public_transport', 'station')],
        'train_station':    [('railway', 'station')],
    },
    'buildings': {
        'commercial':       [('building', 'commercial|retail|supermarket')],
        'industrial':       [('building', 'industrial|warehouse')],
        'religious':        [('amenity', 'place_of_worship'),
                             ('building', 'cathedral|church|mosque|temple|chapel')],
        'residential':      [('building', 'residential|apartments|house|dormitory')],
    },
}

ROAD_TYPES = 'motorway|trunk|primary|secondary|tertiary|residential'

def _build_union_block(tag_pairs, bbox_str):
    lines = []
    for key, val in tag_pairs:
        lines.append(f'node["{key}"~"{val}"]({bbox_str});')
        lines.append(f'way["{key}"~"{val}"]({bbox_str});')
    return ''.join(lines)

def fetch_poi_layers(bounds, layers_dict, timeout=OVERPASS_TIMEOUT):
    s, w, n, e = bounds[1], bounds[0], bounds[3], bounds[2]
    bbox_str   = f'{s},{w},{n},{e}'

    union_block = ''
    for layer_name, tag_pairs in layers_dict.items():
        union_block += _build_union_block(tag_pairs, bbox_str)

    query = f'[out:json][timeout:{timeout}];({union_block});out center qt;'
    data  = overpass_query(query)
    if data is None or not data.get('elements'):
        return gpd.GeoDataFrame()

    rows = []
    for el in data['elements']:
        lat = el['lat'] if el['type'] == 'node' else el.get('center', {}).get('lat')
        lon = el['lon'] if el['type'] == 'node' else el.get('center', {}).get('lon')
        if lat is None or lon is None:
            continue
        tags    = el.get('tags', {})
        service = _classify_element(tags, layers_dict)
        rows.append({
            'service':  service,
            'name':     tags.get('name', ''),
            'osm_id':   el['id'],
            'osm_type': el['type'],
            'geometry': Point(lon, lat),
        })
    return gpd.GeoDataFrame(rows, crs='EPSG:4326') if rows else gpd.GeoDataFrame()

def _classify_element(tags, layers_dict):
    for layer_name, tag_pairs in layers_dict.items():
        for key, val in tag_pairs:
            import re
            tv = tags.get(key, '')
            if tv and re.search(val, tv):
                return layer_name
    return 'other'

def fetch_road_network(bounds, road_types=ROAD_TYPES, timeout=OVERPASS_TIMEOUT):
    from shapely.geometry import LineString
    s, w, n, e = bounds[1], bounds[0], bounds[3], bounds[2]
    bbox_str   = f'{s},{w},{n},{e}'
    query = (f'[out:json][timeout:{timeout}];'
             f'way["highway"~"{road_types}"]({bbox_str});'
             f'out geom qt;')
    data = overpass_query(query)
    if data is None or not data.get('elements'):
        return gpd.GeoDataFrame()
    rows = []
    for el in data['elements']:
        coords = [(nd['lon'], nd['lat']) for nd in el.get('geometry', [])
                  if 'lon' in nd and 'lat' in nd]
        if len(coords) < 2:
            continue
        tags = el.get('tags', {})
        rows.append({
            'highway':  tags.get('highway', ''),
            'name':     tags.get('name', ''),
            'osm_id':   el['id'],
            'geometry': LineString(coords),
        })
    return gpd.GeoDataFrame(rows, crs='EPSG:4326') if rows else gpd.GeoDataFrame()

RUN_NAME   = 'caribbean_adm_boundaries'
BASE       = Path('/home/jupyter-daniela/CAF/data/processed') / RUN_NAME
SHAPEFILE  = Path('/home/jupyter-daniela/CAF/data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp')
EMDAT_PATH = Path('/home/jupyter-daniela/CAF/data/EMDAT/public_emdat_custom_request_2025-11-21_25fc7800-ab47-4f91-905e-101ce7d788d4.xlsx')
MICRO_DIR  = BASE / 'micro_flood'
MICRO_DIR.mkdir(parents=True, exist_ok=True)

N_CITIES_PER_TIER = 2

GDP_TIERS = {
    'high':   ['Bahamas', 'Barbados', 'Trinidad and Tobago'],
    'medium': ['Saint Lucia', 'Dominica', 'Dominican Republic', 'Jamaica'],
    'low':    ['Haiti'],
}


WB_ISO = {
    'Bahamas':'BS','Barbados':'BB','Dominica':'DM','Dominican Republic':'DO',
    'Haiti':'HT','Jamaica':'JM','Saint Lucia':'LC','Trinidad and Tobago':'TT',
}

print("=" * 70)
print("STEP 1 — Load data")
print("=" * 70)

df   = pd.read_parquet(BASE / 'master_panel.parquet')
gdf  = gpd.read_file(SHAPEFILE).rename(columns={'region_nam': 'region_name'})
if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs('EPSG:4326')

df_em = pd.read_excel(EMDAT_PATH, sheet_name='EM-DAT Data')
df_em = df_em[
    df_em['Country'].isin(list(WB_ISO.keys())) &
    df_em['Disaster Type'].isin(['Storm', 'Flood']) &
    (df_em['Start Year'] >= 1995)
].copy()
df_em['event_date'] = pd.to_datetime({
    'year':  df_em['Start Year'].astype(int),
    'month': df_em['Start Month'].fillna(6).astype(int),
    'day':   df_em['Start Day'].fillna(1).astype(int),
}, errors='coerce')
df_em['total_losses'] = (
    df_em['Total Deaths'].fillna(0) * 1.1 +
    df_em["Total Damage, Adjusted ('000 US$)"].fillna(0) / 1000
)
print(f"  EMDAT flood/storm events: {len(df_em)}")

iso_str = ';'.join(WB_ISO.values())
wb_data = requests.get(
    f'https://api.worldbank.org/v2/country/{iso_str}/indicator/'
    f'NY.GDP.PCAP.KD?date=2015:2020&format=json&per_page=200', timeout=30
).json()[1]
iso_to_country = {v: k for k, v in WB_ISO.items()}
gdp_mean = (
    pd.DataFrame([{'country': iso_to_country[r['country']['id']], 'gdp_pc': r['value']}
                  for r in wb_data if r['value'] is not None])
    .groupby('country')['gdp_pc'].mean()
)
print(f"  GDP/cap fetched for {len(gdp_mean)} countries")

print()
print("=" * 70)
print("STEP 2 — Score and rank cities by flood/rainfall exposure")
print("=" * 70)

df['hazard_flood'] = df[['rainfall', 'cyclone']].mean(axis=1)

city_scores = (
    df.merge(gdf[['region_name', 'country']], on='region_name', how='left')
    .groupby(['region_name', 'country'])
    .agg(
        mean_rainfall=('rainfall', 'mean'),
        max_rainfall=('rainfall', 'max'),
        mean_flood_hazard=('hazard_flood', 'mean'),
        total_losses=('total_losses_musd', 'sum'),
        n_loss_events=('total_losses_musd', lambda x: (x > 0).sum()),
    )
    .reset_index()
)
city_scores['gdp_pc']   = city_scores['country'].map(gdp_mean)
city_scores['gdp_tier'] = city_scores['country'].map(
    {c: tier for tier, countries in GDP_TIERS.items() for c in countries}
)

city_scores['flood_score'] = (
    (city_scores['mean_rainfall']   - city_scores['mean_rainfall'].min()) /
    (city_scores['mean_rainfall'].max() - city_scores['mean_rainfall'].min()) * 0.4 +
    (city_scores['total_losses']    - city_scores['total_losses'].min()) /
    (city_scores['total_losses'].max()  - city_scores['total_losses'].min() + 1e-9) * 0.4 +
    (city_scores['n_loss_events']   - city_scores['n_loss_events'].min()) /
    (city_scores['n_loss_events'].max() - city_scores['n_loss_events'].min() + 1e-9) * 0.2
)

OSM_DENSITY_ANCHOR = 'Santo Domingo Este'

selected = []
for tier in ['high', 'medium', 'low']:
    tier_cities = (city_scores[city_scores['gdp_tier'] == tier]
                   .sort_values('flood_score', ascending=False)
                   .head(N_CITIES_PER_TIER))
    selected.append(tier_cities)

anchor_row = city_scores[city_scores['region_name'] == OSM_DENSITY_ANCHOR].copy()
anchor_row['note'] = 'OSM-density anchor (comparable to Haiti cities)'
selected.append(anchor_row)

df_selected = pd.concat(selected).drop_duplicates('region_name').reset_index(drop=True)
df_selected['gdp_pc'] = df_selected['gdp_pc'].round(0)
df_selected['flood_score'] = df_selected['flood_score'].round(4)

print(f"\nSelected {len(df_selected)} study-case cities:")
print(df_selected[['region_name', 'country', 'gdp_tier', 'gdp_pc',
                    'flood_score', 'total_losses', 'n_loss_events']].to_string(index=False))

print()
print("=" * 70)
print("STEP 3 — Match EMDAT events to selected cities")
print("=" * 70)

event_records = []
for _, city in df_selected.iterrows():
    country_events = df_em[df_em['Country'] == city['country']].copy()

    city_geom = gdf[gdf['region_name'] == city['region_name']]['geometry'].iloc[0]
    matched = []

    pip_events = country_events[
        country_events['Latitude'].notna() & country_events['Longitude'].notna()
    ].copy()
    for _, ev in pip_events.iterrows():
        from shapely.geometry import Point
        if city_geom.contains(Point(ev['Longitude'], ev['Latitude'])):
            matched.append(ev)

    if not matched:
        from rapidfuzz import process as rfp, fuzz
        region_norm = city['region_name'].lower()
        for _, ev in country_events.iterrows():
            loc = str(ev.get('Location', ''))
            score = rfp.extractOne(region_norm, [loc.lower()], scorer=fuzz.partial_ratio)
            if score and score[1] >= 60:
                matched.append(ev)

    if not matched:
        for _, ev in country_events.iterrows():
            matched.append(ev)

    for ev in matched:
        event_records.append({
            'region_name':    city['region_name'],
            'country':        city['country'],
            'gdp_tier':       city['gdp_tier'],
            'gdp_pc':         city['gdp_pc'],
            'disaster_type':  ev['Disaster Type'],
            'disaster_sub':   ev.get('Disaster Subtype', ''),
            'event_date':     ev['event_date'],
            'start_year':     int(ev['Start Year']),
            'start_month':    int(ev['Start Month']) if pd.notna(ev['Start Month']) else None,
            'start_day':      int(ev['Start Day'])   if pd.notna(ev['Start Day'])   else None,
            'end_year':       int(ev['End Year'])     if pd.notna(ev['End Year'])    else None,
            'end_month':      int(ev['End Month'])    if pd.notna(ev['End Month'])   else None,
            'location_text':  ev.get('Location', ''),
            'total_deaths':   ev.get('Total Deaths', 0),
            'total_losses':   ev.get('total_losses', 0),
        })

df_events = pd.DataFrame(event_records).drop_duplicates(
    subset=['region_name', 'event_date', 'disaster_type']
).sort_values(['region_name', 'event_date']).reset_index(drop=True)

print(f"\nMatched {len(df_events)} EMDAT events across {df_events['region_name'].nunique()} cities")
print(df_events[['region_name', 'country', 'disaster_type', 'event_date',
                  'total_deaths', 'total_losses']].to_string(index=False))

print()
print("=" * 70)
print("STEP 4 — Download OSM service layers for each city")
print("=" * 70)

ALL_LAYERS = {k: v for cat in OSM_LAYERS.values() for k, v in cat.items()}

osm_records = []
for _, city in df_selected.iterrows():
    region    = city['region_name']
    city_geom = gdf[gdf['region_name'] == region]['geometry'].iloc[0]
    bounds    = city_geom.bounds
    print(f"\n  {region} ({city['country']}, {city['gdp_tier']} GDP)")
    print(f"  Bounds: {[round(b,4) for b in bounds]}")

    city_osm_dir = MICRO_DIR / region.replace(' ', '_').replace('/', '_')
    city_osm_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    print(f"  [POI layers]")
    gdf_poi = fetch_poi_layers(bounds, ALL_LAYERS)
    time.sleep(12)

    if not gdf_poi.empty:
        gdf_poi['region_name'] = region
        gdf_poi['country']     = city['country']
        gdf_poi['gdp_tier']    = city['gdp_tier']

        for category, layer_dict in OSM_LAYERS.items():
            for layer_name in layer_dict:
                sub = gdf_poi[gdf_poi['service'] == layer_name].reset_index(drop=True)
                n   = len(sub)
                if n == 0:
                    print(f"    [{category}] {layer_name:<22}   0")
                    continue
                out_path = city_osm_dir / f'{layer_name}.geojson'
                sub.to_file(out_path, driver='GeoJSON')
                total += n
                print(f"    [{category}] {layer_name:<22} {n:>4} → {out_path.name}")

        for _, feat in gdf_poi.iterrows():
            osm_records.append({
                'region_name': region,
                'country':     city['country'],
                'gdp_tier':    city['gdp_tier'],
                'service':     feat['service'],
                'name':        feat.get('name', ''),
                'lon':         feat['geometry'].x,
                'lat':         feat['geometry'].y,
            })
    else:
        print("    Overpass unavailable — skipping POI for this city")

    print(f"  [Road network]")
    gdf_roads = fetch_road_network(bounds)
    time.sleep(12)

    if not gdf_roads.empty:
        gdf_roads['region_name'] = region
        gdf_roads['country']     = city['country']
        roads_path = city_osm_dir / 'road_network.geojson'
        gdf_roads.to_file(roads_path, driver='GeoJSON')
        n_roads = len(gdf_roads)
        total  += n_roads
        print(f"    road segments: {n_roads} → road_network.geojson")
        hwy_counts = gdf_roads['highway'].value_counts().to_dict()
        print(f"    breakdown: { {k: v for k,v in sorted(hwy_counts.items())} }")
    else:
        print("    Road network unavailable")

    print(f"  Total features: {total}")
    df_selected.loc[df_selected['region_name'] == region, 'total_osm_features'] = total

print()
print("=" * 70)
print("STEP 5 — Save outputs")
print("=" * 70)

df_selected.to_parquet(MICRO_DIR / 'study_cities.parquet', index=False)
df_selected.to_csv(MICRO_DIR    / 'study_cities.csv',     index=False)

df_events.to_parquet(MICRO_DIR / 'flood_events.parquet', index=False)
df_events.to_csv(MICRO_DIR     / 'flood_events.csv',     index=False)

if osm_records:
    df_osm = pd.DataFrame(osm_records)
    df_osm.to_parquet(MICRO_DIR / 'osm_services.parquet', index=False)
    df_osm.to_csv(MICRO_DIR     / 'osm_services.csv',     index=False)
    print(f"  osm_services: {len(df_osm)} total features")

print(f"\nAll outputs saved to {MICRO_DIR}/")
print()
print("SUMMARY")
print("-" * 70)
print(df_selected[['region_name', 'country', 'gdp_tier', 'gdp_pc',
                    'flood_score', 'total_losses', 'total_osm_features']].to_string(index=False))
