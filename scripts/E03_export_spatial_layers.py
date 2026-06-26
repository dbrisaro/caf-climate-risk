import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from pathlib import Path
import netCDF4 as nc
from datetime import datetime, timedelta

BASE      = Path('/home/jupyter-daniela/CAF')
OUTDIR    = BASE / 'outputs/cyclone'
EXP_OUTDIR = BASE / 'outputs/exposure'
OUTDIR.mkdir(parents=True, exist_ok=True)
EXP_OUTDIR.mkdir(parents=True, exist_ok=True)

SHP_PATH = BASE / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'

print("Loading admin boundaries …")
adm = gpd.read_file(SHP_PATH).to_crs('EPSG:4326')
adm = adm.rename(columns={'region_nam': 'region_name'})

print("\nFig 1.4 — Cyclone tracks …")
IBTRACS = BASE / 'data/IBTrACS/IBTrACS.since1980.v04r01.nc'

ds       = nc.Dataset(IBTRACS)
epoch    = datetime(1858, 11, 17)
raw_time = np.ma.filled(ds.variables['time'][:].astype(float), np.nan)
raw_lat  = np.ma.filled(ds.variables['lat'][:].astype(float), np.nan)
raw_lon  = np.ma.filled(ds.variables['lon'][:].astype(float), np.nan)
raw_wind = np.ma.filled(ds.variables['usa_wind'][:].astype(float), np.nan)
raw_wmo  = np.ma.filled(ds.variables['wmo_wind'][:].astype(float), np.nan)
raw_sid  = ds.variables['sid'][:]
raw_name = ds.variables['name'][:]

def _decode(row):
    return ''.join(
        c.decode('utf-8', errors='ignore') if isinstance(c, (bytes, bytearray)) else str(c)
        for c in row
    ).strip()

sids  = [_decode(r) for r in raw_sid]
names = [_decode(r) for r in raw_name]
n_storms, n_times = raw_time.shape

raw_time[raw_time < -9e4]  = np.nan
raw_lat[raw_lat  < -999]   = np.nan
raw_lon[raw_lon  < -999]   = np.nan
raw_wind[raw_wind < -900]  = np.nan
raw_wmo[raw_wmo  < -900]   = np.nan

flat_idx  = np.repeat(np.arange(n_storms), n_times)
flat_time = raw_time.ravel()
flat_lat  = raw_lat.ravel()
flat_lon  = raw_lon.ravel()
flat_wind = np.where(~np.isnan(raw_wind.ravel()), raw_wind.ravel(), raw_wmo.ravel())

valid = ~np.isnan(flat_time) & ~np.isnan(flat_lat) & ~np.isnan(flat_lon)
flat_time = flat_time[valid]
flat_lat  = flat_lat[valid]
flat_lon  = flat_lon[valid]
flat_wind = flat_wind[valid]
flat_idx  = flat_idx[valid]

flat_dt  = np.array([epoch + timedelta(days=float(t)) for t in flat_time])
flat_sid = np.array([sids[i]  for i in flat_idx])
flat_nam = np.array([names[i] for i in flat_idx])

mask = (
    (flat_lat >= 5)  & (flat_lat <= 32) &
    (flat_lon >= -100) & (flat_lon <= -50)
)
flat_lat  = flat_lat[mask]
flat_lon  = flat_lon[mask]
flat_wind = flat_wind[mask]
flat_dt   = flat_dt[mask]
flat_sid  = flat_sid[mask]
flat_nam  = flat_nam[mask]

df_pts = pd.DataFrame({
    'sid':  flat_sid,
    'name': flat_nam,
    'dt':   flat_dt,
    'lat':  flat_lat,
    'lon':  flat_lon,
    'wind_kt': flat_wind,
})
df_pts['year'] = pd.to_datetime(df_pts['dt'].astype(str)).dt.year
print(f"  {df_pts['sid'].nunique()} Caribbean storms, "
      f"{len(df_pts):,} track points")

track_rows = []
for sid, grp in df_pts.sort_values('dt').groupby('sid'):
    coords = list(zip(grp['lon'], grp['lat']))
    if len(coords) < 2:
        continue
    row_name  = grp['name'].iloc[0]
    year      = int(grp['year'].iloc[0])
    max_wind  = round(float(np.nanmax(grp['wind_kt'])), 1) if grp['wind_kt'].notna().any() else np.nan
    if max_wind >= 137:   ss = 'Cat 5'
    elif max_wind >= 113: ss = 'Cat 4'
    elif max_wind >= 96:  ss = 'Cat 3'
    elif max_wind >= 83:  ss = 'Cat 2'
    elif max_wind >= 64:  ss = 'Cat 1'
    elif max_wind >= 34:  ss = 'Tropical storm'
    else:                 ss = 'Tropical depression'
    track_rows.append({
        'sid':          sid,
        'name':         row_name,
        'year':         year,
        'max_wind_kt':  max_wind,
        'ss_category':  ss,
        'n_obs':        len(grp),
        'geometry':     LineString(coords),
    })

gdf_tracks = gpd.GeoDataFrame(track_rows, crs='EPSG:4326')

gdf_pts = gpd.GeoDataFrame(
    df_pts[['sid', 'name', 'year', 'dt', 'lat', 'lon', 'wind_kt']].copy(),
    geometry=[Point(lon, lat) for lon, lat in zip(df_pts['lon'], df_pts['lat'])],
    crs='EPSG:4326',
)

out_tracks = OUTDIR / 'fig_1_4_cyclone_tracks.gpkg'
gdf_tracks.to_file(out_tracks, layer='tracks',        driver='GPKG')
gdf_pts.to_file(   out_tracks, layer='track_points',  driver='GPKG')
print(f"  Saved → {out_tracks}")
print(f"    layers: tracks ({len(gdf_tracks)} storms), "
      f"track_points ({len(gdf_pts):,} obs)")

print("\nFigs 1.8/1.9 — Exposure by admin region …")
ep = pd.read_parquet(BASE / 'data/processed/caribbean_adm_boundaries/exposure_panel.parquet')

lo, hi = np.percentile(ep['exposure_index'].dropna(), [2, 98])
ep['exposure_index_norm'] = np.clip((ep['exposure_index'] - lo) / (hi - lo), 0, 1).round(4)

reg2ctr = adm.set_index('region_name')['country'].to_dict()
ep['country'] = ep['region_name'].map(reg2ctr)

base95 = ep[ep['year'] == 1995].set_index('region_name')['exposure_index_norm']
ep['delta_from_1995'] = (ep['exposure_index_norm'] -
                          ep['region_name'].map(base95)).round(4)

ep_geo = adm[['region_name', 'country', 'geometry']].merge(
    ep[['region_name', 'year', 'pop', 'exposure_index',
        'exposure_index_norm', 'delta_from_1995']],
    on='region_name', how='left',
)
ep_geo = gpd.GeoDataFrame(ep_geo, geometry='geometry', crs='EPSG:4326')

out_exp = EXP_OUTDIR / 'fig_1_8_9_exposure_admin.gpkg'
ep_geo.to_file(out_exp, layer='exposure_by_region_year', driver='GPKG')
print(f"  Saved → {out_exp}")
print(f"    rows: {len(ep_geo):,}  "
      f"(regions × years, {ep_geo['year'].nunique()} epochs)")

print("\nFig 3.1 — Study city locations …")

CITIES = [
    ('Carrefour',          'Haiti',              18.5432, -72.3959, 'Fig_3_1'),
    ('La Romana',          'Dominican Republic', 18.4267, -68.9718, 'Fig_3_1'),
    ('Castries',           'Saint Lucia',        14.0101, -60.9876, 'Fig_3_1'),
    ('Bridgetown',         'Barbados',           13.0969, -59.6145, 'Fig_3_1'),
    ('Port-au-Prince',     'Haiti',              18.5392, -72.3288, 'Fig_3_4_B03'),
    ('Santo Domingo Este', 'Dominican Republic', 18.5107, -69.7906, 'Fig_3_5_B03'),
    ('North Andros',       'Bahamas',            24.7698, -78.0890, 'B03'),
    ('Saint Thomas',       'Barbados',           13.1800, -59.5884, 'B03'),
    ('Villa Altagracia',   'Dominican Republic', 18.6542, -70.2176, 'B03'),
    ('San Pedro de Macorís','Dominican Republic',18.4822, -69.2850, 'B03'),
]

gdf_cities = gpd.GeoDataFrame(
    [{
        'city':        c[0],
        'country':     c[1],
        'source_fig':  c[4],
        'geometry':    Point(c[3], c[2]),
    } for c in CITIES],
    crs='EPSG:4326',
)

out_cities = EXP_OUTDIR / 'fig_3_1_city_locations.gpkg'
gdf_cities.to_file(out_cities, layer='study_cities', driver='GPKG')
print(f"  Saved → {out_cities}  ({len(gdf_cities)} cities)")

print("\nFigs 3.4/3.5 — Flood city OSM infrastructure …")
MICRO_DIR = BASE / 'data/processed/caribbean_adm_boundaries/micro_flood'

FLOOD_CITIES = {
    'Port-au-Prince':     ('Port-au-Prince',     'Haiti'),
    'Santo_Domingo_Este': ('Santo Domingo Este', 'Dominican Republic'),
}

out_flood = OUTDIR / 'fig_3_4_5_flood_city_infra.gpkg'

all_layers_written = []
for folder_name, (city_label, country) in FLOOD_CITIES.items():
    city_dir = MICRO_DIR / folder_name
    if not city_dir.exists():
        print(f"  WARNING: {city_dir} not found, skipping")
        continue

    geojsons = sorted(city_dir.glob('*.geojson'))
    if not geojsons:
        print(f"  WARNING: no geojsons in {city_dir}, skipping")
        continue

    city_gdfs = []
    for gj in geojsons:
        try:
            g = gpd.read_file(gj).to_crs('EPSG:4326')
            g['layer_type'] = gj.stem
            g['city']       = city_label
            g['country']    = country
            city_gdfs.append(g)
        except Exception as e:
            print(f"    skip {gj.name}: {e}")

    if not city_gdfs:
        continue

    combined = pd.concat(city_gdfs, ignore_index=True)
    keep_cols = ['city', 'country', 'layer_type', 'geometry']
    for col in ['name', 'amenity', 'building', 'highway', 'osm_id']:
        if col in combined.columns:
            keep_cols.append(col)
    combined = gpd.GeoDataFrame(
        combined[[c for c in keep_cols if c in combined.columns]],
        geometry='geometry', crs='EPSG:4326'
    )

    layer_name = folder_name.lower()
    combined.to_file(out_flood, layer=layer_name, driver='GPKG')
    all_layers_written.append((layer_name, len(combined)))
    print(f"  {city_label}: {len(combined):,} features in {len(geojsons)} layers")

print(f"  Saved → {out_flood}")
for ln, cnt in all_layers_written:
    print(f"    layer '{ln}': {cnt:,} features")

print("\nDone. Summary of outputs:")
for f in sorted(OUTDIR.glob('*.gpkg')):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name:45s} {size_mb:7.1f} MB")

print("\nExisting shareable GPKGs:")
for f in [
    BASE / 'data/OSM/combined_caribbean_infrastructure.gpkg',
    BASE / 'data/OSM/caribbean_buildings_top4.gpkg',
    BASE / 'data/caribbean_adm_boundaries.gpkg',
]:
    size_mb = f.stat().st_size / 1e6
    print(f"  {str(f).replace(str(BASE)+'/', ''):55s} {size_mb:7.1f} MB")
