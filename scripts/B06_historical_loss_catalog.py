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

from climada.hazard import TCTracks, TropCyclone
from climada.hazard.centroids import Centroids

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

BASE_DIR  = Path('/home/jupyter-daniela/CAF')
OSM_PATH  = BASE_DIR / 'data/OSM/caribbean_buildings_top4.gpkg'
HDF5_PATH = BASE_DIR / 'data/footprints/footprints_all_caribbean_1980_2025.hdf5'
OUT_DIR   = BASE_DIR / 'outputs/losses'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CARIB_BOUNDS = (-100, 5, -50, 32)
GRID_RES     = 0.1
YEAR_RANGE   = (1980, 2026)

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

print("=" * 65)
print("B06 — Historical loss catalog  (IBTrACS 1980–2025, Caribbean)")
print("=" * 65)

lon_min, lat_min, lon_max, lat_max = CARIB_BOUNDS

if HDF5_PATH.exists():
    log.info("Loading cached footprints from %s …", HDF5_PATH.name)
    haz = TropCyclone.from_hdf5(str(HDF5_PATH))
    log.info("  Events: %d  |  Centroids: %d", *haz.intensity.shape)

    log.info("Loading IBTrACS metadata for storm names/years/categories …")
    tc_meta = TCTracks.from_ibtracs_netcdf(provider='usa', basin='NA', year_range=YEAR_RANGE)
    storm_meta = {}
    for t in tc_meta.data:
        sid  = t.attrs['sid']
        name = t.attrs.get('name', sid)
        cat  = int(t.attrs.get('category', -1))
        year = int(str(t.time.values[0])[:4])
        storm_meta[sid] = {'sid': sid, 'name': name, 'year': year, 'category': cat}
    log.info("  Metadata loaded for %d storms", len(storm_meta))

else:
    log.info("Loading all NA-basin IBTrACS tracks %d–%d …", *YEAR_RANGE)
    t0 = time.time()
    tc_all = TCTracks.from_ibtracs_netcdf(
        provider='usa',
        basin='NA',
        year_range=YEAR_RANGE,
    )
    log.info("  Total NA storms loaded: %d  (%.0fs)", tc_all.size, time.time() - t0)

    carib_tracks = []
    for t in tc_all.data:
        lats = t['lat'].values
        lons = t['lon'].values
        if ((lats >= lat_min) & (lats <= lat_max) &
                (lons >= lon_min) & (lons <= lon_max)).any():
            carib_tracks.append(t)

    tc_all.data = carib_tracks
    log.info("  Caribbean-filtered storms: %d", tc_all.size)

    storm_meta = {}
    for t in tc_all.data:
        sid  = t.attrs['sid']
        name = t.attrs.get('name', sid)
        cat  = int(t.attrs.get('category', -1))
        year = int(str(t.time.values[0])[:4])
        storm_meta[sid] = {'sid': sid, 'name': name, 'year': year, 'category': cat}

    log.info("Interpolating to 1-hour timestep …")
    tc_all.equal_timestep(time_step_h=1)

    log.info("Computing TropCyclone footprints for %d storms …", tc_all.size)
    t0 = time.time()
    centroids = Centroids.from_pnt_bounds(CARIB_BOUNDS, GRID_RES)
    haz = TropCyclone.from_tracks(tc_all, centroids)
    log.info("  Done in %.0fs  |  shape: %s  |  max v_max: %.1f m/s",
             time.time() - t0, haz.intensity.shape, haz.intensity.max())

    HDF5_PATH.parent.mkdir(parents=True, exist_ok=True)
    haz.write_hdf5(str(HDF5_PATH))
    log.info("  Saved → %s", HDF5_PATH)

n_events    = haz.intensity.shape[0]
n_centroids = haz.intensity.shape[1]
cent_lat    = haz.centroids.lat
cent_lon    = haz.centroids.lon
event_names = list(haz.event_name)

for sid in event_names:
    if sid not in storm_meta:
        storm_meta[sid] = {'sid': sid, 'name': sid, 'year': None, 'category': None}

log.info("Loading OSM buildings …")
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
buildings.index.name = 'fid'
buildings = buildings.reset_index()

log.info("  Building types: %s", buildings['building_type'].value_counts().to_dict())

log.info("Building KD-tree over %d centroids …", n_centroids)
cent_xy = np.column_stack([cent_lat, cent_lon])
tree    = cKDTree(cent_xy)
bld_xy  = buildings[['lat', 'lon']].values
_, idx  = tree.query(bld_xy, k=1)

bld_types   = buildings['building_type'].values
bld_area    = np.array([AREA_M2[bt] for bt in bld_types])
bld_vh      = np.array([SIGMOID_PARAMS[bt]['v_h'] for bt in bld_types])
bld_vt      = np.array([SIGMOID_PARAMS[bt]['v_t'] for bt in bld_types])
bld_cost    = {sc: np.array([COST_USD_M2[sc][bt] for bt in bld_types])
               for sc in ('low', 'mid', 'high')}
bld_value   = {sc: bld_area * bld_cost[sc] for sc in ('low', 'mid', 'high')}

log.info("  KD-tree ready.")

log.info("Computing losses for %d storms …", n_events)
t0 = time.time()

catalog_rows = []
for ev_i, sid in enumerate(event_names):
    if ev_i % 50 == 0:
        log.info("  Storm %d / %d  (%.0fs elapsed)", ev_i, n_events, time.time() - t0)

    vmax = np.asarray(haz.intensity[ev_i, :].todense()).ravel()
    vmax_bld = vmax[idx]

    vn  = np.maximum(vmax_bld - bld_vt, 0.0) / (bld_vh - bld_vt)
    mdr = vn**3 / (1.0 + vn**3)

    meta = storm_meta.get(sid, {})
    row = {
        'sid':               sid,
        'name':              meta.get('name', sid),
        'year':              meta.get('year'),
        'category':          meta.get('category'),
        'n_buildings_damaged': int((mdr > 0).sum()),
        'loss_low':          float((mdr * bld_value['low']).sum()),
        'loss_mid':          float((mdr * bld_value['mid']).sum()),
        'loss_high':         float((mdr * bld_value['high']).sum()),
    }
    catalog_rows.append(row)

log.info("  All storms processed in %.0fs", time.time() - t0)

catalog = pd.DataFrame(catalog_rows)
catalog = catalog.sort_values('loss_mid', ascending=False).reset_index(drop=True)

out_path = OUT_DIR / 'historical_loss_catalog.csv'
catalog.to_csv(out_path, index=False)
log.info("Saved → %s  (%d rows)", out_path, len(catalog))

n_nonzero = (catalog['loss_mid'] > 0).sum()
print()
print(f"  Total storms: {len(catalog)}")
print(f"  Storms with loss > 0: {n_nonzero}")
print(f"  Max loss (mid): USD {catalog['loss_mid'].max():,.0f}")
print(f"  Median loss (mid, non-zero): USD {catalog.loc[catalog['loss_mid']>0,'loss_mid'].median():,.0f}")
print()
print(f"  Top 10 costliest storms:")
print(catalog[['sid','name','year','category','loss_mid']].head(10).to_string(index=False))
print()
print(f"Done.  → {out_path}")
