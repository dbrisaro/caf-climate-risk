import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from pathlib import Path
import logging
import time

from climada.hazard import TropCyclone

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

BASE_DIR  = Path('/home/jupyter-daniela/CAF')
HDF5_PATH = BASE_DIR / 'data/footprints/footprints_all_caribbean_1980_2025.hdf5'
OSM_PATH  = BASE_DIR / 'data/OSM/caribbean_buildings_top4.gpkg'
OUT_DIR   = BASE_DIR / 'outputs/losses'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCHOOL_AMENITY  = {'school', 'kindergarten', 'childcare', 'university', 'college'}
SCHOOL_BUILDING = {'school', 'university', 'college', 'kindergarten'}
GOV_AMENITY     = {'hospital', 'clinic', 'police', 'fire_station', 'courthouse',
                   'public_building', 'doctors', 'dentist', 'community_centre', 'townhall'}
GOV_BUILDING    = {'hospital', 'clinic', 'government', 'public'}

def assign_building_type(amenity, building):
    a, b = str(amenity).lower().strip(), str(building).lower().strip()
    if a in SCHOOL_AMENITY or b in SCHOOL_BUILDING:
        return 'school'
    if a in GOV_AMENITY or b in GOV_BUILDING:
        return 'government'
    return 'other'

COUNTRY_NAME_FIX = {'Bahamas (the)': 'Bahamas'}

V_T = 25.7
SIGMOID_PARAMS = {
    'other':      {'v_h': 63.0, 'v_t': V_T},
    'school':     {'v_h': 74.7, 'v_t': V_T},
    'government': {'v_h': 86.0, 'v_t': V_T},
}

AREA_M2 = {'school': 400, 'government': 600, 'other': 120}
COST_USD_M2 = {
    'low':  {'school': 700,  'government': 900,  'other': 350},
    'mid':  {'school': 950,  'government': 1200, 'other': 500},
    'high': {'school': 1200, 'government': 1500, 'other': 700},
}

YEAR_MIN, YEAR_MAX = 1980, 2025
N_YEARS = YEAR_MAX - YEAR_MIN + 1

print("=" * 62)
print("B05 — Loss estimation (full 1980–2025 catalog, 8 study countries)")
print("=" * 62)

log.info("Loading footprints from %s …", HDF5_PATH.name)
haz = TropCyclone.from_hdf5(str(HDF5_PATH))
n_events    = haz.intensity.shape[0]
n_centroids = haz.intensity.shape[1]
cent_lat    = haz.centroids.lat
cent_lon    = haz.centroids.lon
event_names = list(haz.event_name)
log.info("  Events: %d  |  Centroids: %d", n_events, n_centroids)

log.info("Loading OSM buildings from %s …", OSM_PATH.name)
gdf = gpd.read_file(OSM_PATH, layer='buildings')
gdf['country'] = gdf['country'].replace(COUNTRY_NAME_FIX)
log.info("  Buildings: %d across %d countries", len(gdf), gdf['country'].nunique())

gdf['building_type'] = gdf.apply(
    lambda r: assign_building_type(r.get('amenity', ''), r.get('building', '')), axis=1)

geom = gdf.geometry.copy()
non_point = ~geom.geom_type.isin(['Point'])
if non_point.any():
    geom[non_point] = geom[non_point].centroid
gdf['lat'] = geom.y
gdf['lon'] = geom.x

buildings = gdf[['lat', 'lon', 'building_type', 'country']].copy().reset_index(drop=True)
log.info("  Building types: %s", buildings['building_type'].value_counts().to_dict())

log.info("Building KD-tree over %d centroids …", n_centroids)
cent_xy = np.column_stack([cent_lat, cent_lon])
tree    = cKDTree(cent_xy)
bld_xy  = buildings[['lat', 'lon']].values
_, idx  = tree.query(bld_xy, k=1)

bld_types   = buildings['building_type'].values
bld_country = buildings['country'].values
bld_area    = np.array([AREA_M2[bt] for bt in bld_types])
bld_vh      = np.array([SIGMOID_PARAMS[bt]['v_h'] for bt in bld_types])
bld_vt      = np.array([SIGMOID_PARAMS[bt]['v_t'] for bt in bld_types])
bld_value   = {sc: bld_area * np.array([COST_USD_M2[sc][bt] for bt in bld_types])
               for sc in ('low', 'mid', 'high')}

type_list     = ['school', 'government', 'other']
countries     = sorted(buildings['country'].unique())
type_masks    = {bt: bld_types == bt for bt in type_list}
country_masks = {c: bld_country == c for c in countries}

log.info("Computing losses for %d storms …", n_events)
t0 = time.time()

type_storm_rows    = []
country_storm_rows = []

for ev_i, storm_id in enumerate(event_names):
    if ev_i % 50 == 0:
        log.info("  Storm %d / %d  (%.0fs elapsed)", ev_i, n_events, time.time() - t0)

    vmax_bld = np.asarray(haz.intensity[ev_i, :].todense()).ravel()[idx]
    vn  = np.maximum(vmax_bld - bld_vt, 0.0) / (bld_vh - bld_vt)
    mdr = vn**3 / (1.0 + vn**3)

    losses = {sc: mdr * bld_value[sc] for sc in ('low', 'mid', 'high')}

    for bt in type_list:
        m = type_masks[bt]
        type_storm_rows.append({
            'storm_id':      storm_id,
            'building_type': bt,
            'loss_low':      float(losses['low'][m].sum()),
            'loss_mid':      float(losses['mid'][m].sum()),
            'loss_high':     float(losses['high'][m].sum()),
        })

    for c in countries:
        m = country_masks[c]
        country_storm_rows.append({
            'country':   c,
            'storm_id':  storm_id,
            'loss_low':  float(losses['low'][m].sum()),
            'loss_mid':  float(losses['mid'][m].sum()),
            'loss_high': float(losses['high'][m].sum()),
        })

log.info("  All storms processed in %.0fs", time.time() - t0)

df_type    = pd.DataFrame(type_storm_rows)
df_country = pd.DataFrame(country_storm_rows)

type_out    = OUT_DIR / 'loss_by_type_storm.csv'
country_out = OUT_DIR / 'loss_by_country_storm.csv'
df_type.to_csv(type_out, index=False)
df_country.to_csv(country_out, index=False)
log.info("Saved: %s  (%d rows)", type_out, len(df_type))
log.info("Saved: %s  (%d rows)", country_out, len(df_country))

social_mask = df_type['building_type'].isin(['school', 'government'])
social_low  = df_type.loc[social_mask, 'loss_low'].sum()
social_mid  = df_type.loc[social_mask, 'loss_mid'].sum()
social_high = df_type.loc[social_mask, 'loss_high'].sum()
total_mid   = df_type['loss_mid'].sum()

aal_low  = social_low  / N_YEARS
aal_mid  = social_mid  / N_YEARS
aal_high = social_high / N_YEARS
social_share = 100.0 * social_mid / total_mid if total_mid > 0 else 0.0

print()
print(f"  Total storms: {n_events}   |   Catalog: {YEAR_MIN}–{YEAR_MAX} ({N_YEARS} years)")
print(f"  Buildings:    {len(buildings):,}  |   Countries: {len(countries)}")
print()
print("  Social-infrastructure AAL  (school + government buildings):")
print(f"    Low:  ${aal_low/1e6:>8,.2f}M / yr")
print(f"    Mid:  ${aal_mid/1e6:>8,.2f}M / yr")
print(f"    High: ${aal_high/1e6:>8,.2f}M / yr")
print()
print(f"  Total mid loss across all storms: ${total_mid/1e6:>10,.1f}M")
print(f"  Social share of total (mid):       {social_share:>6.1f}%")
print()
print("Done.  Outputs:")
print(f"  {type_out}")
print(f"  {country_out}")
