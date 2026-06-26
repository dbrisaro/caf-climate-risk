import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.mask import mask as rasterio_mask
from rasterio.merge import merge
from rasterio.io import MemoryFile
from affine import Affine
from shapely.geometry import box, mapping
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def layer_meta(layer):
    if layer == 'SMOD':
        return 1000, 'V2'
    return 100, 'V1'

def build_tif_path(ghs_dir, layer, year, tile):
    resolution, version = layer_meta(layer)
    folder = os.path.join(ghs_dir, layer, str(year))
    filename = f"GHS_{layer}_E{year}_GLOBE_R2023A_54009_{resolution}_{version}_0_{tile}.tif"
    return os.path.join(folder, filename)

def load_raster_window(tif_path, gdf, out_shape=(1500, 1500), buffer_m=20_000):
    with rasterio.open(tif_path) as src:
        gdf_proj = gdf.to_crs(src.crs)
        minx, miny, maxx, maxy = gdf_proj.total_bounds
        w = from_bounds(
            minx - buffer_m, miny - buffer_m,
            maxx + buffer_m, maxy + buffer_m,
            transform=src.transform
        )
        arr = src.read(
            1, window=w, out_shape=out_shape,
            resampling=rasterio.enums.Resampling.average
        )
        transform = src.window_transform(w)
        nodata = src.nodata

    arr = arr.astype('float32')
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, transform

def nan_mosaic(arr_list):
    out = np.full(arr_list[0].shape, np.nan, dtype='float32')
    for a in arr_list:
        a = a.astype('float32', copy=False)
        m = np.isnan(out) & np.isfinite(a)
        out[m] = a[m]
    return out

def find_merit_tiles(country_bounds, merit_dir):
    minlon, minlat, maxlon, maxlat = country_bounds
    country_box = box(minlon, minlat, maxlon, maxlat)
    matching = []
    for tif in glob.glob(os.path.join(str(merit_dir), '**', '*_dem.tif'), recursive=True):
        try:
            with rasterio.open(tif) as src:
                b = src.bounds
                if box(b.left, b.bottom, b.right, b.top).intersects(country_box):
                    matching.append(tif)
        except Exception:
            pass
    return sorted(matching)

def compute_mean_elevation(country_geom, dem_files):
    if not dem_files:
        return np.nan
    try:
        datasets = [rasterio.open(f) for f in dem_files]
        if len(datasets) > 1:
            mosaic, out_transform = merge(datasets)
            profile = datasets[0].profile.copy()
            profile.update({'height': mosaic.shape[1], 'width': mosaic.shape[2], 'transform': out_transform})
            dem_array = mosaic[0]
            nodata = datasets[0].nodata
        else:
            dem_array = datasets[0].read(1)
            profile = datasets[0].profile.copy()
            nodata = datasets[0].nodata
        for ds in datasets:
            ds.close()

        geom = [mapping(country_geom)]
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(dem_array, 1)
                masked_data, _ = rasterio_mask(dataset, geom, crop=True, nodata=nodata)
                masked_data = masked_data[0]

        valid = masked_data[masked_data != nodata] if nodata is not None else masked_data[np.isfinite(masked_data)]
        return float(np.mean(valid)) if len(valid) > 0 else np.nan
    except Exception:
        return np.nan

def compute_exposure_pca(arrays):
    pop     = arrays['pop']
    built_s = arrays['built_s']
    built_v = arrays['built_v']

    feature_arrays = [pop, built_s, built_v]
    feature_keys   = ['pop', 'built_s', 'built_v']

    if 'dem' in arrays and arrays['dem'] is not None:
        feature_arrays.append(arrays['dem'])
        feature_keys.append('dem')

    X = np.stack(feature_arrays, axis=-1)
    mask = np.isfinite(X).all(axis=-1) & (np.nansum(X, axis=-1) > 0)

    if mask.sum() == 0:
        return None, None, None

    Xv = np.log1p(X[mask])
    Xn = StandardScaler().fit_transform(Xv)

    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(Xn)[:, 0]
    var_explained = pca.explained_variance_ratio_[0]

    if not np.isfinite(pc1).any():
        return None, None, var_explained

    exposure_arr = np.full(pop.shape, np.nan, dtype='float32')
    exposure_arr[mask] = pc1.astype('float32')

    return exposure_arr, float(np.nanmean(exposure_arr)), var_explained

def _find_relevant_tiles(ghs_dir, layers, years, tiles, gdf):
    relevant = []
    for tile in tiles:
        found = False
        for layer in layers:
            if found:
                break
            for year in years:
                tif_path = build_tif_path(ghs_dir, layer, year, tile)
                if not os.path.exists(tif_path):
                    continue
                try:
                    with rasterio.open(tif_path) as src:
                        if gdf.to_crs(src.crs).intersects(box(*src.bounds)).any():
                            relevant.append(tile)
                            found = True
                except Exception:
                    pass
                break
    return relevant

def run_exposure(ghs_dir, merit_dir, targets, years, tiles,
                 layers=('BUILT_S', 'BUILT_V', 'POP'),
                 out_shape=(1500, 1500)):
    layer_key_map = {'BUILT_S': 'built_s', 'BUILT_V': 'built_v', 'POP': 'pop'}
    region_series = {}

    for _, row in targets.iterrows():
        region = row['region_name']
        gdf    = targets[targets['region_name'] == region].copy()
        print(f"\n{region}")

        dem_files    = find_merit_tiles(row['geometry'].bounds, merit_dir)
        dem_scalar   = compute_mean_elevation(row['geometry'], dem_files)
        print(f"  DEM mean elevation: {dem_scalar:.1f} m" if not np.isnan(dem_scalar) else "  DEM: no coverage")

        region_tiles = _find_relevant_tiles(ghs_dir, layers, years, tiles, gdf)
        if not region_tiles:
            print(f"  no tiles intersect this region, skipping")
            continue

        for year in years:
            arrays  = {}
            ref_path = None

            for layer in layers:
                key       = layer_key_map[layer]
                arr_tiles = []

                for tile in region_tiles:
                    tif_path = build_tif_path(ghs_dir, layer, year, tile)
                    if not os.path.exists(tif_path):
                        continue
                    try:
                        arr, _ = load_raster_window(tif_path, gdf, out_shape=out_shape)
                        if np.isfinite(arr).sum() == 0:
                            continue
                        arr_tiles.append(arr)
                        if ref_path is None:
                            ref_path = tif_path
                    except Exception as e:
                        print(f"  [ERROR] {year} {layer} {tile}: {e}")
                        continue

                if arr_tiles:
                    arrays[key] = nan_mosaic(arr_tiles) if len(arr_tiles) > 1 else arr_tiles[0]

            if any(k not in arrays for k in ['pop', 'built_s', 'built_v']):
                print(f"  {year}: missing GHS layers, skipping")
                continue

            if not np.isnan(dem_scalar):
                arrays['dem'] = np.full(arrays['pop'].shape, dem_scalar, dtype='float32')

            _, mean_exposure, var_exp = compute_exposure_pca(arrays)

            if mean_exposure is None:
                print(f"  {year}: no valid pixels, skipping")
                continue

            n_features = 4 if 'dem' in arrays else 3
            print(f"  {year}: exposure={mean_exposure:.4f}  var_exp={var_exp:.4f}  features={n_features}")
            region_series.setdefault(region, {})[year] = {
                'exposure_index': mean_exposure,
                'var_exp':        var_exp,
                'pop':            float(np.nanmean(arrays['pop'])),
                'built_s':        float(np.nanmean(arrays['built_s'])),
                'built_v':        float(np.nanmean(arrays['built_v'])),
                'dem':            dem_scalar,
            }

    all_vals = [
        d['exposure_index']
        for ydict in region_series.values()
        for d in ydict.values()
    ]
    g_min, g_max = min(all_vals), max(all_vals)
    if g_max > g_min:
        for region in region_series:
            for year in region_series[region]:
                v = region_series[region][year]['exposure_index']
                region_series[region][year]['exposure_index'] = (v - g_min) / (g_max - g_min)

    return region_series

def exposure_to_dataframe(region_series):
    rows = [
        {'region_name': region, 'year': year, **values}
        for region, ydict in region_series.items()
        for year, values in ydict.items()
    ]
    return pd.DataFrame(rows).sort_values(['region_name', 'year']).reset_index(drop=True)
