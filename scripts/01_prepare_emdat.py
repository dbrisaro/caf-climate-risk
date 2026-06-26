import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

BASE_DIR   = Path('/home/jupyter-daniela/CAF')
EMDAT_DIR  = BASE_DIR / 'data/EMDAT'
SHAPEFILE  = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
OUT_FILE   = BASE_DIR / 'outputs/losses/emdat_carib_1900_2024.gpkg'

CARIBBEAN_COUNTRIES = [
    'Dominican Republic', 'Jamaica', 'Trinidad and Tobago',
    'Bahamas', 'Barbados', 'Haiti', 'Dominica', 'Saint Lucia', 'Cuba',
]

TARGET_SUBGROUPS = ['Meteorological', 'Hydrological', 'Climatological']

FLOOD_TYPES     = ['Flood (General)', 'Riverine flood', 'Flash flood', 'Coastal flood']
LANDSLIDE_TYPES = ['Landslide (wet)', 'Mudslide']

print("=" * 62)
print("Step 01 — Prepare EM-DAT dataset")
print("=" * 62)

emdat_files = list(EMDAT_DIR.glob('public_emdat_*.xlsx'))
if not emdat_files:
    raise FileNotFoundError(f"No EM-DAT xlsx found in {EMDAT_DIR}")
EMDAT_PATH = emdat_files[0]
log.info("EM-DAT file: %s", EMDAT_PATH.name)

df = pd.read_excel(EMDAT_PATH)
log.info("Raw EMDAT: %d rows", len(df))

caribbean = df[df['Country'].isin(CARIBBEAN_COUNTRIES)].copy()
log.info("Caribbean events: %d", len(caribbean))

filtered = caribbean[caribbean['Disaster Subgroup'].isin(TARGET_SUBGROUPS)].copy()
log.info("After subgroup filter (%s): %d", '/'.join(TARGET_SUBGROUPS), len(filtered))

filtered.loc[filtered['Disaster Subtype'].isin(FLOOD_TYPES), 'Disaster Subtype']     = 'Flood (All Types)'
filtered.loc[filtered['Disaster Subtype'].isin(LANDSLIDE_TYPES), 'Disaster Subtype'] = 'Landslide (All Types)'

log.info("Merging geometries from %s", SHAPEFILE.name)
adm = gpd.read_file(SHAPEFILE)

country_col = next((c for c in adm.columns if c.lower() == 'country'), None)
if country_col is None:
    raise KeyError(f"No 'country' column found in shapefile. Columns: {adm.columns.tolist()}")

country_geom = adm.dissolve(by=country_col).reset_index()[[country_col, 'geometry']]
country_geom = country_geom.rename(columns={country_col: 'country_key'})
country_geom['country_key'] = country_geom['country_key'].str.strip()

filtered['country_key'] = filtered['Country'].str.strip()
geo = filtered.merge(country_geom, on='country_key', how='left')
geo = gpd.GeoDataFrame(geo, geometry='geometry', crs=adm.crs)
geo = geo.drop(columns=['country_key'])

matched = geo['geometry'].notna().sum()
log.info("Events with geometry: %d / %d", matched, len(geo))

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
geo.to_file(str(OUT_FILE), driver='GPKG')
log.info("Saved: %s  (%d features)", OUT_FILE, len(geo))

print()
print(f"Done.  Output: {OUT_FILE}")
