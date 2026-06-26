import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box
from pyproj import Transformer
import requests, glob, time
from pathlib import Path

BASE    = Path('/home/jupyter-daniela/CAF')
OUT_DIR = BASE / 'outputs/exposure'
OUT_DIR.mkdir(parents=True, exist_ok=True)

GHSL_TILES = sorted(glob.glob(str(BASE / 'data/GHS/POP/2025/*.tif')))
OSM_FILE   = BASE / 'data/OSM/osm_top_ciudades_caribe_continentales.csv'
CITY_FILE  = BASE / 'data/OSM/resumen_descarga_osm_por_ciudad.csv'
PAD_KM     = 4.0
MOLLWEIDE  = 'ESRI:54009'
WGS84      = 'EPSG:4326'

UN_WPP_PCT_517 = {
    'Haiti':                0.372,
    'Dominican Republic':   0.274,
    'Trinidad and Tobago':  0.205,
    'Jamaica':              0.265,
    'Barbados':             0.196,
    'Bahamas':              0.218,
    'Saint Lucia':          0.224,
    'Dominica':             0.218,
    'Colombia':             0.263,
    'Guyana':               0.254,
    'Peru':                 0.271,
}

ISO = {
    'Haiti':                'HTI',
    'Dominican Republic':   'DOM',
    'Trinidad and Tobago':  'TTO',
    'Jamaica':              'JAM',
    'Barbados':             'BRB',
    'Bahamas':              'BHS',
    'Saint Lucia':          'LCA',
    'Dominica':             'DMA',
    'Colombia':             'COL',
    'Guyana':               'GUY',
    'Peru':                 'PER',
}

print("Pulling World Bank net enrollment rates …")

def wb_fetch(iso, indicator, retries=3):
    url = (f'https://api.worldbank.org/v2/country/{iso}/indicator/{indicator}'
           f'?format=json&mrv=5&per_page=5')
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and len(r.json()) > 1:
                for rec in (r.json()[1] or []):
                    if rec and rec.get('value') is not None:
                        return float(rec['value'])
        except Exception:
            time.sleep(1)
    return None

ner = {}
for country, iso in ISO.items():
    ner_pri = wb_fetch(iso, 'SE.PRM.NENR')
    ner_sec = wb_fetch(iso, 'SE.SEC.NENR')
    ner[country] = {'ner_primary': ner_pri, 'ner_secondary': ner_sec}
    time.sleep(0.2)
    print(f"  {country:<25}  NER_primary={ner_pri}  NER_secondary={ner_sec}")

print("\nComputing city bounding boxes …")

osm         = pd.read_csv(OSM_FILE)
transformer = Transformer.from_crs(WGS84, MOLLWEIDE, always_xy=True)

city_boxes = {}
for (city, country), grp in osm.groupby(['ciudad', 'pais']):
    lons, lats = grp['lon'].values, grp['lat'].values
    x_min, y_min = transformer.transform(lons.min(), lats.min())
    x_max, y_max = transformer.transform(lons.max(), lats.max())
    pad = PAD_KM * 1000
    city_boxes[(city, country)] = box(x_min-pad, y_min-pad, x_max+pad, y_max+pad)

print("\nExtracting GHSL POP 2025 …")

def extract_pop(geom_mollweide):
    total = 0.0
    matched = False
    for tile_path in GHSL_TILES:
        with rasterio.open(tile_path) as src:
            if not box(*src.bounds).intersects(geom_mollweide):
                continue
            matched = True
            try:
                out, _ = rio_mask(src, [geom_mollweide], crop=True,
                                  nodata=src.nodata if src.nodata else -1)
                valid = out[out > 0]
                total += float(valid.sum())
            except Exception:
                pass
    return total if matched else None

city_pop = {}
for (city, country), geom in city_boxes.items():
    pop = extract_pop(geom)
    city_pop[(city, country)] = pop
    status = f"{pop:,.0f}" if pop is not None else "NO TILE"
    print(f"  {city} ({country}): {status}")

print("\nComputing beneficiary indicators …")

city_df  = pd.read_csv(CITY_FILE)
records  = []
WHO_THRESHOLD = 10_000

for _, row in city_df.iterrows():
    city    = row['ciudad']
    country = row['pais']
    key     = (city, country)

    urban_pop  = city_pop.get(key)
    pct_517    = UN_WPP_PCT_517.get(country)
    n_ner      = ner.get(country, {})
    ner_pri    = n_ner.get('ner_primary')
    ner_sec    = n_ner.get('ner_secondary')

    n_health = int(row['salud'])
    n_edu    = int(row['escuelas'])
    n_total  = int(row['TOTAL'])

    pct_611  = (6/13) * pct_517 if pct_517 else None
    pct_1217 = (6/13) * pct_517 if pct_517 else None

    if urban_pop and pct_611 and ner_pri:
        students_primary = urban_pop * pct_611 * (ner_pri / 100)
    else:
        students_primary = None

    if urban_pop and pct_1217 and ner_sec:
        students_secondary = urban_pop * pct_1217 * (ner_sec / 100)
    else:
        students_secondary = None

    if students_primary is not None and students_secondary is not None:
        students_total = students_primary + students_secondary
    elif students_primary is not None:
        students_total = students_primary
    else:
        students_total = None

    school_age_pop = (urban_pop * pct_517) if (urban_pop and pct_517) else None

    health_catchment = (urban_pop / n_health) if (urban_pop and n_health > 0) else None
    exceeds_who = (health_catchment > WHO_THRESHOLD) if health_catchment else None

    facilities_per_10k = (n_total / urban_pop * 10_000) if (urban_pop and urban_pop > 0) else None

    records.append({
        'region':                    row['region'],
        'country':                   country,
        'city':                      city,
        'urban_pop_ghsl2025':        round(urban_pop)    if urban_pop    else None,
        'n_education_facilities':    n_edu,
        'pct_517_unwpp2024':         round(pct_517 * 100, 1) if pct_517 else None,
        'ner_primary_pct':           round(ner_pri, 1)   if ner_pri  else None,
        'ner_secondary_pct':         round(ner_sec, 1)   if ner_sec  else None,
        'school_age_pop_517':        round(school_age_pop) if school_age_pop else None,
        'students_primary_est':      round(students_primary)   if students_primary   else None,
        'students_secondary_est':    round(students_secondary) if students_secondary else None,
        'students_total_est':        round(students_total)     if students_total     else None,
        'students_per_edu_facility': round(students_total / n_edu) if (students_total and n_edu > 0) else None,
        'n_health_facilities':       n_health,
        'health_catchment_pop':      round(health_catchment) if health_catchment else None,
        'exceeds_who_10k':           exceeds_who,
        'n_total_facilities':        n_total,
        'facilities_per_10k_urban':  round(facilities_per_10k, 1) if facilities_per_10k else None,
        'notes': ('secondary NER missing — students_total is primary only'
                  if (students_primary and not students_secondary) else
                  'GHSL tile not available — population estimates missing'
                  if urban_pop is None else ''),
    })

out      = pd.DataFrame(records)
out_path = OUT_DIR / 'city_beneficiary_indicators.csv'
out.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")

print("\n=== KEY RESULTS ===")
display_cols = ['country','city','urban_pop_ghsl2025',
                'school_age_pop_517','students_total_est',
                'health_catchment_pop','exceeds_who_10k']
print(out[display_cols].to_string(index=False))

print("\n=== CARIBBEAN TOTALS ===")
carib = out[out['region']=='Caribe']
print(f"  Total urban pop (Caribbean):     {carib['urban_pop_ghsl2025'].sum():,.0f}")
print(f"  Total school-age pop 5-17:       {carib['school_age_pop_517'].sum():,.0f}")
print(f"  Total enrolled students (est):   {carib['students_total_est'].sum():,.0f}")
print(f"  Cities exceeding WHO threshold:  {carib['exceeds_who_10k'].sum():.0f}")
