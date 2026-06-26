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

from climada.hazard import TropCyclone

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

BASE_DIR  = Path('/home/jupyter-daniela/CAF')
HDF5_PATH = BASE_DIR / 'data/footprints/footprints_16_storms.hdf5'
OSM_PATH  = BASE_DIR / 'data/OSM/combined_caribbean_infrastructure.gpkg'
OUT_DIR   = BASE_DIR / 'outputs/losses'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_MAP = {
    '2_Basic_Social_Services':   'school',
    '1_Critical_Infrastructure': 'government',
    '3_Mobility_Networks':       'other',
    '4_Buildings':               'other',
}

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

def sigmoid_mdr(v_ms: float, v_h: float, v_t: float) -> float:
    if v_ms <= v_t:
        return 0.0
    vn = (v_ms - v_t) / (v_h - v_t)
    return float(vn**3 / (1.0 + vn**3))

print("=" * 62)
print("Step 07 — Loss estimation")
print("=" * 62)

for label, path in [("Footprints HDF5", HDF5_PATH),
                    ("OSM GeoPackage",  OSM_PATH)]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    log.info("  ✓ %s: %s", label, path)

log.info("Loading footprints from %s …", HDF5_PATH.name)
haz = TropCyclone.from_hdf5(str(HDF5_PATH))

n_events    = haz.intensity.shape[0]
n_centroids = haz.intensity.shape[1]
cent_lat    = haz.centroids.lat
cent_lon    = haz.centroids.lon
event_names = list(haz.event_name)

log.info("  Events: %d  |  Centroids: %d", n_events, n_centroids)
log.info("  Storm names: %s", ", ".join(event_names))

log.info("Loading OSM infrastructure from %s …", OSM_PATH.name)
gdf = gpd.read_file(OSM_PATH, layer='combined_caribbean_infrastructure')
log.info("  Raw features: %d", len(gdf))

gdf = gdf[gdf['type'] == 'building'].copy()
log.info("  Building features: %d", len(gdf))

gdf['building_type'] = gdf['main_category'].map(CATEGORY_MAP).fillna('other')

geom = gdf.geometry
non_point = ~geom.geom_type.isin(['Point'])
if non_point.any():
    log.info("  %d non-point geometries → using centroid", non_point.sum())
    geom = geom.copy()
    geom[non_point] = geom[non_point].centroid
gdf['lat'] = geom.y
gdf['lon'] = geom.x

gdf = gdf.reset_index()
fid_col = 'fid' if 'fid' in gdf.columns else gdf.columns[0]
buildings = gdf[[fid_col, 'lat', 'lon', 'building_type', 'country']].copy()
buildings = buildings.rename(columns={fid_col: 'fid'}).reset_index(drop=True)

type_counts = buildings['building_type'].value_counts()
log.info("  Building types: %s", type_counts.to_dict())

for btype, p in SIGMOID_PARAMS.items():
    log.info("  Sigmoid params  %-12s  v_h=%.1f m/s  v_t=%.1f m/s", btype, p['v_h'], p['v_t'])

log.info("Building KD-tree over %d centroids …", n_centroids)

cent_xy = np.column_stack([cent_lat, cent_lon])
tree = cKDTree(cent_xy)

bld_xy = buildings[['lat', 'lon']].values
_, idx = tree.query(bld_xy, k=1)

log.info("  Nearest-centroid lookup done.")

records = []
for ev_i, storm_id in enumerate(event_names):
    vmax = np.asarray(haz.intensity[ev_i, :].todense()).ravel()
    vmax_bld = vmax[idx]

    df_ev = buildings[['fid', 'building_type', 'country']].copy()
    df_ev['storm_id']  = storm_id
    df_ev['vmax_ms']   = vmax_bld
    records.append(df_ev)

df = pd.concat(records, ignore_index=True)
log.info("  Exposure table: %d rows (buildings × storms)", len(df))

log.info("Computing MDR (physrisk sigmoid) …")

mdr_values = np.zeros(len(df))
for btype, p in SIGMOID_PARAMS.items():
    mask = (df['building_type'] == btype).values
    v = df.loc[mask, 'vmax_ms'].values
    vn = np.maximum(v - p['v_t'], 0.0) / (p['v_h'] - p['v_t'])
    mdr_values[mask] = vn**3 / (1.0 + vn**3)

df['mdr'] = mdr_values
log.info("  MDR range: %.4f – %.4f", df['mdr'].min(), df['mdr'].max())

log.info("Computing losses …")

for scenario in ('low', 'mid', 'high'):
    df[f'loss_{scenario}'] = df.apply(
        lambda r, s=scenario: (
            r['mdr']
            * AREA_M2[r['building_type']]
            * COST_USD_M2[s][r['building_type']]
        ),
        axis=1,
    )

log.info("  Total loss (mid scenario): USD {:,.0f}".format(df['loss_mid'].sum()))

bld_out = OUT_DIR / 'loss_by_building.csv'
df.to_csv(bld_out, index=False)
log.info("  Saved: %s  (%d rows)", bld_out, len(df))

agg = (
    df.groupby(['country', 'storm_id'])[['loss_low', 'loss_mid', 'loss_high']]
    .sum()
    .reset_index()
)
agg_out = OUT_DIR / 'loss_by_country_storm.csv'
agg.to_csv(agg_out, index=False)
log.info("  Saved: %s  (%d rows)", agg_out, len(agg))

print()
print("Done.  Outputs:")
print(f"  {bld_out}")
print(f"  {agg_out}")
