import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.append('/home/jupyter-daniela/CAF/src')

import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = Path('/home/jupyter-daniela/CAF')
OSM_PATH = BASE_DIR / 'data/OSM/caribbean_buildings_top4.gpkg'
OUT_DIR  = BASE_DIR / 'outputs/exposure'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRY_NAME_FIX = {'Bahamas (the)': 'Bahamas'}

EDU_AMENITY  = {'school', 'kindergarten', 'childcare', 'university', 'college', 'library'}
EDU_BUILDING = {'school', 'university', 'college', 'kindergarten'}

HEALTH_AMENITY  = {'hospital', 'clinic', 'doctors', 'dentist', 'pharmacy',
                   'health_centre', 'nursing_home', 'social_facility'}
HEALTH_BUILDING = {'hospital', 'clinic', 'health_centre'}

HOUSING_BUILDING = {'house', 'residential', 'apartments', 'detached',
                    'terrace', 'bungalow', 'semidetached_house', 'dormitory',
                    'static_caravan', 'houseboat'}

def classify(amenity, building):
    a = str(amenity).lower().strip()
    b = str(building).lower().strip()
    if a in EDU_AMENITY or b in EDU_BUILDING:
        return 'education'
    if a in HEALTH_AMENITY or b in HEALTH_BUILDING:
        return 'health'
    if b in HOUSING_BUILDING:
        return 'housing'
    return 'other'

print("=" * 65)
print("Social Infrastructure Inventory")
print("=" * 65)

log.info("Loading OSM buildings …")
gdf = gpd.read_file(OSM_PATH, layer='buildings')
gdf['country'] = gdf['country'].replace(COUNTRY_NAME_FIX)
log.info("  %d buildings, %d countries", len(gdf), gdf['country'].nunique())

gdf['category'] = gdf.apply(
    lambda r: classify(r.get('amenity', ''), r.get('building', '')), axis=1
)

cat_counts = gdf['category'].value_counts()
log.info("Category breakdown:\n%s", cat_counts.to_string())

pivot = (
    gdf.groupby(['country', 'category'])
       .size()
       .unstack(fill_value=0)
       .reindex(columns=['education', 'health', 'housing', 'other'], fill_value=0)
       .reset_index()
)

pivot['social_total'] = pivot['education'] + pivot['health'] + pivot['housing']
pivot['total_buildings'] = pivot[['education', 'health', 'housing', 'other']].sum(axis=1)
pivot['pct_education'] = (pivot['education'] / pivot['total_buildings'] * 100).round(1)
pivot['pct_health']    = (pivot['health']    / pivot['total_buildings'] * 100).round(1)
pivot['pct_housing']   = (pivot['housing']   / pivot['total_buildings'] * 100).round(1)
pivot['pct_social']    = (pivot['social_total'] / pivot['total_buildings'] * 100).round(1)

by_country = pivot[[
    'country',
    'education', 'health', 'housing', 'social_total',
    'total_buildings',
    'pct_education', 'pct_health', 'pct_housing', 'pct_social',
]].sort_values('total_buildings', ascending=False).reset_index(drop=True)

pivot_city = (
    gdf.groupby(['country', 'city', 'category'])
       .size()
       .unstack(fill_value=0)
       .reindex(columns=['education', 'health', 'housing', 'other'], fill_value=0)
       .reset_index()
)

pivot_city['social_total']    = pivot_city['education'] + pivot_city['health'] + pivot_city['housing']
pivot_city['total_buildings'] = pivot_city[['education', 'health', 'housing', 'other']].sum(axis=1)
pivot_city['pct_education']   = (pivot_city['education'] / pivot_city['total_buildings'] * 100).round(1)
pivot_city['pct_health']      = (pivot_city['health']    / pivot_city['total_buildings'] * 100).round(1)
pivot_city['pct_housing']     = (pivot_city['housing']   / pivot_city['total_buildings'] * 100).round(1)

by_city = pivot_city[[
    'country', 'city',
    'education', 'health', 'housing', 'social_total',
    'total_buildings',
    'pct_education', 'pct_health', 'pct_housing',
]].sort_values(['country', 'total_buildings'], ascending=[True, False]).reset_index(drop=True)

out_country = OUT_DIR / 'tab_social_infra_by_country.csv'
out_city    = OUT_DIR / 'tab_social_infra_by_city.csv'

by_country.to_csv(out_country, index=False)
by_city.to_csv(out_city, index=False)

log.info("Saved → %s", out_country)
log.info("Saved → %s", out_city)

print("\nTable — Social infrastructure by country:")
print(by_country.to_string(index=False))

print("\nRegional totals:")
totals = by_country[['education', 'health', 'housing', 'social_total', 'total_buildings']].sum()
print(f"  Education : {totals['education']:>7,.0f}")
print(f"  Health    : {totals['health']:>7,.0f}")
print(f"  Housing   : {totals['housing']:>7,.0f}")
print(f"  Social    : {totals['social_total']:>7,.0f}")
print(f"  Total bld : {totals['total_buildings']:>7,.0f}")
