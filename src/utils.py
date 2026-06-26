import geopandas as gpd
from shapely.geometry import Point

def load_shapefile(shapefile_path, name_col, target_regions=None, name_map=None):
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.rename(columns={name_col: 'region_name'})
    gdf = gdf[['region_name', 'geometry']]
    if name_map:
        gdf['region_name'] = gdf['region_name'].map(name_map).fillna(gdf['region_name'])
    gdf = gdf.dissolve(by='region_name', as_index=False)
    gdf = gdf.to_crs("EPSG:4326")
    if target_regions is not None:
        gdf = gdf[gdf['region_name'].isin(target_regions)]
    return gdf.copy()

def get_region(lat, lon, gdf):
    point = Point(lon, lat)
    for _, row in gdf.iterrows():
        if row['geometry'].contains(point):
            return row['region_name']
    return None

def get_season(month):
    if month in [9, 10, 11]:   return 'SON'
    elif month in [12, 1, 2]:  return 'DJF'
    elif month in [3, 4, 5]:   return 'MAM'
    else:                      return 'JJA'

def lower_tail_activation(row):
    if row['value'] <= row['p1']:    return 'p1'
    elif row['value'] <= row['p5']:  return 'p5'
    elif row['value'] <= row['p10']: return 'p10'
    else:                            return 'none'

def upper_tail_activation(row):
    if row['value'] >= row['p99']:   return 'p99'
    elif row['value'] >= row['p95']: return 'p95'
    elif row['value'] >= row['p90']: return 'p90'
    else:                            return 'none'
