import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.append('/home/jupyter-daniela/CAF/src')

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
HDF5_PATH = BASE_DIR / 'data/footprints/footprints_all_caribbean_1980_2025.hdf5'
OSM_PATH  = BASE_DIR / 'data/OSM/caribbean_buildings_top4.gpkg'
META_PATH = BASE_DIR / 'outputs/losses/historical_loss_catalog.csv'
OUT_DIR   = BASE_DIR / 'outputs/losses'

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
AREA_M2      = {'school': 400, 'government': 600, 'other': 120}
COST_USD_M2  = {
    'low':  {'school': 700,  'government': 900,  'other': 350},
    'mid':  {'school': 950,  'government': 1200, 'other': 500},
    'high': {'school': 1200, 'government': 1500, 'other': 700},
}
SS_LABEL = {-1: 'TD', 0: 'TD', 1: 'Cat 1', 2: 'Cat 2',
             3: 'Cat 3', 4: 'Cat 4', 5: 'Cat 5'}

print("=" * 60)
print("Wind loss tables 5.1 & 5.2")
print("=" * 60)

log.info("Loading cached footprints …")
haz = TropCyclone.from_hdf5(str(HDF5_PATH))
event_names = list(haz.event_name)
cent_lat    = haz.centroids.lat
cent_lon    = haz.centroids.lon
log.info("  %d events × %d centroids", *haz.intensity.shape)

meta = pd.read_csv(META_PATH)[['sid', 'name', 'year', 'category']]
meta_dict = meta.set_index('sid').to_dict('index')

log.info("Loading OSM buildings …")
gdf = gpd.read_file(OSM_PATH, layer='buildings')
gdf['country'] = gdf['country'].replace(COUNTRY_NAME_FIX)
gdf['building_type'] = gdf.apply(
    lambda r: assign_building_type(r.get('amenity', ''), r.get('building', '')), axis=1)
geom = gdf.geometry.copy()
mask = ~geom.geom_type.isin(['Point'])
if mask.any():
    geom[mask] = geom[mask].centroid
gdf['lat'] = geom.y
gdf['lon']  = geom.x
buildings = gdf[['lat', 'lon', 'building_type', 'country']].copy().reset_index(drop=True)
buildings.index.name = 'fid'
buildings = buildings.reset_index()
log.info("  %d buildings across %d countries",
         len(buildings), buildings['country'].nunique())

tree   = cKDTree(np.column_stack([cent_lat, cent_lon]))
_, idx = tree.query(buildings[['lat', 'lon']].values, k=1)

bld_types  = buildings['building_type'].values
bld_country = buildings['country'].values
bld_area   = np.array([AREA_M2[bt]               for bt in bld_types])
bld_vh     = np.array([SIGMOID_PARAMS[bt]['v_h']  for bt in bld_types])
bld_vt     = np.array([SIGMOID_PARAMS[bt]['v_t']  for bt in bld_types])
bld_value  = {sc: bld_area * np.array([COST_USD_M2[sc][bt] for bt in bld_types])
              for sc in ('low', 'mid', 'high')}

log.info("Computing losses per country per storm …")
countries = sorted(buildings['country'].unique())
rows = []

grp_keys, grp_idx = np.unique(
    np.char.add(np.char.add(bld_country.astype(str), '|'), bld_types.astype(str)),
    return_inverse=True)
cls_loss = {sc: np.zeros(len(grp_keys)) for sc in ('low', 'mid', 'high')}

for ev_i, sid in enumerate(event_names):
    vmax     = np.asarray(haz.intensity[ev_i, :].todense()).ravel()
    vmax_bld = vmax[idx]
    vn       = np.maximum(vmax_bld - bld_vt, 0.0) / (bld_vh - bld_vt)
    mdr      = vn**3 / (1.0 + vn**3)

    if mdr.sum() == 0:
        continue

    for sc in ('low', 'mid', 'high'):
        cls_loss[sc] += np.bincount(grp_idx, weights=mdr * bld_value[sc],
                                    minlength=len(grp_keys))

    m = meta_dict.get(sid, {})
    for country in countries:
        mask = bld_country == country
        loss_mid = float((mdr[mask] * bld_value['mid'][mask]).sum())
        if loss_mid == 0:
            continue
        rows.append({
            'country':      country,
            'storm_id':     sid,
            'storm_name':   m.get('name', sid),
            'year':         int(m.get('year') or 0),
            'ss_category':  SS_LABEL.get(int(m.get('category') or -1), 'TD'),
            'loss_low':     float((mdr[mask] * bld_value['low'][mask]).sum()),
            'loss_mid':     loss_mid,
            'loss_high':    float((mdr[mask] * bld_value['high'][mask]).sum()),
        })

t51 = (pd.DataFrame(rows)
       .sort_values(['year', 'country'])
       .reset_index(drop=True))

log.info("  Table 5.1: %d rows (country-storm pairs with loss > 0)", len(t51))

t52 = (t51.groupby('country')
       .agg(
           n_storms      = ('storm_id', 'nunique'),
           total_loss_low  = ('loss_low',  'sum'),
           total_loss_mid  = ('loss_mid',  'sum'),
           total_loss_high = ('loss_high', 'sum'),
       )
       .reset_index()
       .sort_values('total_loss_mid', ascending=False)
       .reset_index(drop=True))

total_regional = t52['total_loss_mid'].sum()
t52['pct_regional'] = (t52['total_loss_mid'] / total_regional * 100).round(1)

CLASS_LABEL = {'school': 'Education', 'government': 'Government', 'other': 'Housing'}
t53 = pd.DataFrame({
    'country':        [k.split('|')[0] for k in grp_keys],
    'building_type':  [k.split('|')[1] for k in grp_keys],
    'total_loss_low':  cls_loss['low'],
    'total_loss_mid':  cls_loss['mid'],
    'total_loss_high': cls_loss['high'],
})
t53['building_class'] = t53['building_type'].map(CLASS_LABEL)
t53 = (t53[['country', 'building_type', 'building_class',
            'total_loss_low', 'total_loss_mid', 'total_loss_high']]
       .sort_values(['country', 'building_type'])
       .reset_index(drop=True))

out_51 = OUT_DIR / 'table_5_1_wind_by_country_storm.csv'
out_52 = OUT_DIR / 'table_5_2_wind_cumulative_by_country.csv'
out_53 = OUT_DIR / 'table_5_3_wind_cumulative_by_country_class.csv'
t51.to_csv(out_51, index=False)
t52.to_csv(out_52, index=False)
t53.to_csv(out_53, index=False)

log.info("Saved → %s", out_51)
log.info("Saved → %s", out_52)
log.info("Saved → %s", out_53)

print()
print("Table 5.2 — Cumulative wind losses 1980–2025 by country:")
print(t52.to_string(index=False,
      float_format=lambda x: f"{x:,.0f}" if x > 1000 else f"{x:.1f}"))
print()
print(f"Regional total (mid): USD {total_regional:,.0f}")
