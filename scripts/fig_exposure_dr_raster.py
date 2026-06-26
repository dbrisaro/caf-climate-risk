
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.warp import reproject, Resampling, calculate_default_transform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

BASE   = Path("/home/jupyter-daniela/CAF")
GHS    = BASE / "data/GHS"
SHP    = BASE / "data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp"
OUTDIR = BASE / "outputs/exposure"
OUTDIR.mkdir(parents=True, exist_ok=True)

YEARS  = [1995, 2030]
LAYERS = ["BUILT_S", "BUILT_V", "POP"]

print("Loading DR boundary …")
adm = gpd.read_file(SHP)
dr  = adm[adm["country"] == "Dominican Republic"].dissolve().reset_index()

def load_layer(layer: str, year: int, geom_mollweide) -> np.ndarray:
    folder = GHS / layer / str(year)
    tifs   = sorted(folder.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No TIFs in {folder}")

    srcs = []
    for p in tifs:
        src = rasterio.open(p)
        from rasterio.transform import array_bounds
        left, bottom, right, top = src.bounds
        gx1, gy1, gx2, gy2 = geom_mollweide.total_bounds
        if right < gx1 or left > gx2 or top < gy1 or bottom > gy2:
            src.close()
            continue
        srcs.append(src)

    if not srcs:
        raise RuntimeError(f"No tiles intersect DR for {layer} {year}")

    print(f"  {layer} {year}: {len(srcs)} tile(s)")
    mosaic, transform = rio_merge(srcs)
    for s in srcs:
        s.close()

    arr = mosaic[0].astype("float32")
    arr[arr < 0] = np.nan
    return arr, transform, mosaic.shape[1], mosaic.shape[2]

def clip_to_dr(arr, transform, crs, shapes):
    import rasterio.io
    profile = {
        "driver": "GTiff", "dtype": "float32", "width": arr.shape[1],
        "height": arr.shape[0], "count": 1, "crs": crs, "transform": transform,
        "nodata": np.nan,
    }
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as ds:
            ds.write(arr, 1)
        with memfile.open() as ds:
            out, out_transform = rio_mask(ds, shapes, crop=True, nodata=np.nan,
                                          filled=True, all_touched=False)
    return out[0], out_transform

print("Reprojecting DR boundary to Mollweide …")
dr_moll = dr.to_crs("ESRI:54009")
from shapely.geometry import mapping
shapes_moll = [mapping(dr_moll.geometry.iloc[0])]

data = {}
ref_transform = None
ref_crs       = None

for year in YEARS:
    for layer in LAYERS:
        arr, transform, h, w = load_layer(layer, year, dr_moll)
        folder = GHS / layer / str(year)
        tif0   = sorted(folder.glob("*.tif"))[0]
        with rasterio.open(tif0) as src:
            native_crs = src.crs

        clipped, clip_transform = clip_to_dr(arr, transform, native_crs, shapes_moll)
        data[(layer, year)] = clipped
        if ref_transform is None:
            ref_transform = clip_transform
            ref_crs       = native_crs
            ref_shape     = clipped.shape

print("\nClipped shapes:")
for k, v in data.items():
    print(f"  {k}: {v.shape}  range [{np.nanmin(v):.2f}, {np.nanmax(v):.2f}]")

print("\nAligning all arrays to reference grid …")

def align(arr, src_transform, src_crs, dst_transform, dst_crs, dst_shape):
    dst = np.full(dst_shape, np.nan, dtype="float32")
    reproject(
        source=arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst

aligned = {}
for year in YEARS:
    folder = GHS / "BUILT_S" / str(year)
    tif0   = sorted(folder.glob("*.tif"))[0]
    with rasterio.open(tif0) as src:
        native_crs = src.crs

    bs_arr, bs_tr, _, _ = load_layer("BUILT_S", year, dr_moll)
    bs_clip, bs_ct = clip_to_dr(bs_arr, bs_tr, native_crs, shapes_moll)

    for layer in LAYERS:
        arr, tr, _, _ = load_layer(layer, year, dr_moll)
        folder_l = GHS / layer / str(year)
        tif_l    = sorted(folder_l.glob("*.tif"))[0]
        with rasterio.open(tif_l) as src:
            l_crs = src.crs
        clip, ct = clip_to_dr(arr, tr, l_crs, shapes_moll)
        if clip.shape != bs_clip.shape:
            clip = align(clip, ct, l_crs, bs_ct, native_crs, bs_clip.shape)
        aligned[(layer, year)] = clip

    if year == 1995:
        ref_shape  = bs_clip.shape
        ref_ct     = bs_ct
        ref_crs    = native_crs
        moll_ct_95 = bs_ct
        moll_sh_95 = bs_clip.shape
    else:
        moll_ct_30 = bs_ct
        moll_sh_30 = bs_clip.shape

print("Alignment done.")

print("\nComputing exposure index …")

def stack_pixels(year):
    pop  = aligned[("POP",     year)].ravel()
    bs   = aligned[("BUILT_S", year)].ravel()
    bv   = aligned[("BUILT_V", year)].ravel()
    min_len = min(len(pop), len(bs), len(bv))
    pop, bs, bv = pop[:min_len], bs[:min_len], bv[:min_len]
    mask = np.isfinite(pop) & np.isfinite(bs) & np.isfinite(bv)
    X    = np.stack([pop, bs, bv], axis=1)
    return X, mask

X95, m95 = stack_pixels(1995)
X30, m30 = stack_pixels(2030)

Xv95 = np.log1p(X95[m95])
Xv30 = np.log1p(X30[m30])
Xall = np.vstack([Xv95, Xv30])

scaler = StandardScaler().fit(Xall)
pca    = PCA(n_components=1).fit(scaler.transform(Xall))
print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")

def make_index(X, mask, ref_arr):
    pc1 = pca.transform(scaler.transform(np.log1p(X[mask])))[:, 0]
    return pc1, mask

pc95_vals, _ = make_index(X95, m95, None)
pc30_vals, _ = make_index(X30, m30, None)

all_vals   = np.concatenate([pc95_vals, pc30_vals])
vmin_raw   = np.percentile(all_vals, 2)
vmax_raw   = np.percentile(all_vals, 98)

def to_grid(vals, mask, shape):
    clipped = np.clip(vals, vmin_raw, vmax_raw)
    normed  = (clipped - vmin_raw) / (vmax_raw - vmin_raw)
    grid = np.full(mask.shape, np.nan, dtype="float32")
    grid[mask] = normed
    return grid.reshape(shape)

shape95 = aligned[("POP", 1995)].shape
shape30 = aligned[("POP", 2030)].shape
min_len95 = min(aligned[("POP",1995)].size, aligned[("BUILT_S",1995)].size, aligned[("BUILT_V",1995)].size)
min_len30 = min(aligned[("POP",2030)].size, aligned[("BUILT_S",2030)].size, aligned[("BUILT_V",2030)].size)

m95_flat = m95[:min_len95]
m30_flat = m30[:min_len30]

grid95 = to_grid(pc95_vals, m95_flat, shape95)
grid30 = to_grid(pc30_vals, m30_flat, shape30)

print(f"  1995 grid: {grid95.shape}  valid px: {np.isfinite(grid95).sum():,}")
print(f"  2030 grid: {grid30.shape}  valid px: {np.isfinite(grid30).sum():,}")

print("\nReprojecting grids to WGS84 …")
from rasterio.warp import calculate_default_transform
from rasterio.transform import array_bounds
import rasterio.crs as rcrs

src_crs = rcrs.CRS.from_user_input("ESRI:54009")
dst_crs = rcrs.CRS.from_epsg(4326)

def reproject_to_wgs84(grid, moll_transform, moll_shape):
    h, w = moll_shape
    left, bottom, right, top = array_bounds(h, w, moll_transform)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, w, h,
        left=left, bottom=bottom, right=right, top=top
    )
    dst = np.full((dst_h, dst_w), np.nan, dtype="float32")
    reproject(
        source=grid,
        destination=dst,
        src_transform=moll_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    lon_min_r = dst_transform.c
    lat_max_r = dst_transform.f
    lon_max_r = lon_min_r + dst_transform.a * dst_w
    lat_min_r = lat_max_r + dst_transform.e * dst_h
    return dst, [lon_min_r, lon_max_r, lat_min_r, lat_max_r]

grid95_wgs, ext95 = reproject_to_wgs84(grid95, moll_ct_95, moll_sh_95)
grid30_wgs, ext30 = reproject_to_wgs84(grid30, moll_ct_30, moll_sh_30)

lon_min, lat_min, lon_max, lat_max = dr.total_bounds
pad = 0.05
extent_plot = [lon_min - pad, lon_max + pad, lat_min - pad, lat_max + pad]

print("Rendering figure …")
from matplotlib.colors import LinearSegmentedColormap

from matplotlib.colors import BoundaryNorm, ListedColormap
colors_ramp = ["#fecc5c", "#fd8d3c", "#e31a1c", "#800026"]
cmap_cont = LinearSegmentedColormap.from_list("exposure_cont", colors_ramp, N=256)
bounds = [i / 10 for i in range(11)]
n_bins = len(bounds) - 1
bin_colors = [(1.0, 1.0, 1.0)] + [cmap_cont(i / (n_bins - 2)) for i in range(n_bins - 1)]
cmap = ListedColormap(bin_colors)
cmap.set_bad("#c9e8f0")
norm_discrete = BoundaryNorm(bounds, ncolors=n_bins)

dr_wgs = dr.to_crs("EPSG:4326")

fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.2),
                         facecolor="white",
                         gridspec_kw={"wspace": 0.06})

panel_labels = ["a)", "b)"]
panel_years  = ["1995", "2030"]
grids_wgs    = [grid95_wgs, grid30_wgs]
extents      = [ext95, ext30]

for i, (ax, grid, label, year) in enumerate(zip(axes, grids_wgs, panel_labels, panel_years)):
    im = ax.imshow(
        grid,
        origin="upper",
        extent=extents[i],
        cmap=cmap,
        norm=norm_discrete,
        interpolation="nearest",
        aspect="equal",
    )
    dr_wgs.boundary.plot(ax=ax, linewidth=0.5, edgecolor="#1a1a1a", facecolor="none")
    ax.set_title(f"{label} {year}", loc='left', fontsize=7,
                 fontweight='normal', pad=3, color='#111111')
    ax.set_xlim(extent_plot[0], extent_plot[1])
    ax.set_ylim(extent_plot[2], extent_plot[3])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=7, length=2)
    if i == 1:
        ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_discrete)
sm.set_array([])
cb = fig.colorbar(sm, ax=axes, orientation='horizontal',
                  pad=0.08, shrink=0.6, aspect=35)
cb.ax.tick_params(labelsize=7, length=2)
cb.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
cb.set_ticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"])
cb.outline.set_visible(False)
cb.set_label("Exposure index", fontsize=7, fontweight='normal', labelpad=4)

out_path = OUTDIR / "exposure_dr_raster_1995_2030.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved → {out_path}")
