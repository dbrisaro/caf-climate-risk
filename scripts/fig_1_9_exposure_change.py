#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from shapely.geometry import box as shp_box
from pathlib import Path
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import style
style.apply()

t0 = time.time()

BASE     = Path("/home/jupyter-daniela/CAF")
GHS      = BASE / "data/GHS"
SHP      = BASE / "data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp"
OUTDIR   = BASE / "outputs/exposure"
DATA_DIR = BASE / "outputs/exposure"
OUTDIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

YEARS  = [1995, 2030]
LAYERS = ["BUILT_S", "BUILT_V", "POP"]
NODATA = {"BUILT_S": 65535.0, "BUILT_V": 4294967295.0, "POP": -200.0}

print("Loading boundaries …")
gdf      = gpd.read_file(SHP).reset_index(drop=True)
gdf_moll = gdf.to_crs("ESRI:54009")
gdf_wgs  = gdf.to_crs("EPSG:4326")
n_reg    = len(gdf)
print(f"  {n_reg} regions")

def tile_id(path):
    p = Path(path).stem.split("_")
    return f"{p[-2]}_{p[-1]}"

def find_tif(layer, year, tile):
    matches = list((GHS / layer / str(year)).glob(f"*{tile}.tif"))
    return matches[0] if matches else None

def load_tile_arrays(year, tile):
    arrays, transform = {}, None
    for layer in LAYERS:
        p = find_tif(layer, year, tile)
        if p is None:
            return None
        with rasterio.open(p) as src:
            arr = src.read(1).astype(np.float32)
            nd  = NODATA[layer]
            arr[arr == nd] = np.nan
            if layer == "POP":
                arr[arr < 0] = np.nan
            arrays[layer] = arr
            if transform is None:
                transform = src.transform

    min_h = min(a.shape[0] for a in arrays.values())
    min_w = min(a.shape[1] for a in arrays.values())
    arrays = {k: v[:min_h, :min_w] for k, v in arrays.items()}

    return arrays["POP"], arrays["BUILT_S"], arrays["BUILT_V"], transform

def valid_mask(pop, bs, bv):
    return np.isfinite(pop) & np.isfinite(bs) & np.isfinite(bv) & ((pop > 0) | (bs > 0))

print("Finding intersecting tiles …")
bbox = gdf_moll.total_bounds
all_tile_names = sorted({tile_id(p) for p in (GHS / "BUILT_S" / "1995").glob("*.tif")})

tiles = []
for tname in all_tile_names:
    p = find_tif("BUILT_S", 1995, tname)
    if p is None:
        continue
    with rasterio.open(p) as src:
        l, b, r, t = src.bounds
        if r >= bbox[0] and l <= bbox[2] and t >= bbox[1] and b <= bbox[3]:
            tiles.append(tname)
print(f"  {len(tiles)} tiles: {tiles}")

print("\nBuilding region-ID raster cache …")
id_cache = {}
for tname in tiles:
    p = find_tif("BUILT_S", 1995, tname)
    with rasterio.open(p) as src:
        tile_box = shp_box(*src.bounds)
        sub = gdf_moll[gdf_moll.geometry.intersects(tile_box)]
        if sub.empty:
            continue
        shapes = [(geom, int(idx)) for idx, geom in zip(sub.index, sub.geometry)]
        id_cache[tname] = rio_rasterize(
            shapes, out_shape=(src.height, src.width),
            transform=src.transform, fill=-1, dtype="int32",
            all_touched=True
        )
    print(f"  {tname}: {len(sub)} regions")

print(f"  elapsed: {time.time()-t0:.0f}s")

N_SAMPLE = 50_000
rng = np.random.default_rng(42)

print("\nPass 1 — sampling pixels …")
samples = []
for year in YEARS:
    for tname in tiles:
        result = load_tile_arrays(year, tname)
        if result is None:
            continue
        pop, bs, bv, _ = result
        vmask = valid_mask(pop, bs, bv)
        if not vmask.any():
            continue
        X = np.stack([pop[vmask], bs[vmask], bv[vmask]], axis=1)
        n = min(N_SAMPLE, len(X))
        idx = rng.choice(len(X), size=n, replace=False)
        samples.append(X[idx])
        print(f"  {tname} {year}: {vmask.sum():>9,} valid px  →  sampled {n:,}")

X_sample = np.vstack(samples)
print(f"\nTotal sample: {len(X_sample):,} pixels")

X_log  = np.log1p(X_sample)
scaler = StandardScaler().fit(X_log)
pca    = PCA(n_components=1).fit(scaler.transform(X_log))
print(f"PC1 variance explained: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"elapsed: {time.time()-t0:.0f}s")

if pca.components_[0, 1] < 0:
    pca.components_ *= -1
    print("(PCA sign flipped)")

print("\nPass 2 — scoring pixels (shared mask, both years) …")
sum_sc       = {y: np.zeros(n_reg, dtype=np.float64) for y in YEARS}
shared_count = np.zeros(n_reg, dtype=np.int64)

for tname in tiles:
    if tname not in id_cache:
        continue
    r95 = load_tile_arrays(1995, tname)
    r30 = load_tile_arrays(2030, tname)
    if r95 is None or r30 is None:
        continue
    pop95, bs95, bv95, _ = r95
    pop30, bs30, bv30, _ = r30

    min_h = min(pop95.shape[0], pop30.shape[0])
    min_w = min(pop95.shape[1], pop30.shape[1])
    pop95, bs95, bv95 = pop95[:min_h,:min_w], bs95[:min_h,:min_w], bv95[:min_h,:min_w]
    pop30, bs30, bv30 = pop30[:min_h,:min_w], bs30[:min_h,:min_w], bv30[:min_h,:min_w]

    vmask = valid_mask(pop95, bs95, bv95) & valid_mask(pop30, bs30, bv30)
    if not vmask.any():
        continue

    X95  = np.stack([pop95[vmask], bs95[vmask], bv95[vmask]], axis=1)
    X30  = np.stack([pop30[vmask], bs30[vmask], bv30[vmask]], axis=1)
    pc95 = pca.transform(scaler.transform(np.log1p(X95)))[:, 0]
    pc30 = pca.transform(scaler.transform(np.log1p(X30)))[:, 0]

    id_r    = id_cache[tname][:min_h, :min_w]
    flat_id = id_r[vmask]
    in_reg  = flat_id >= 0
    if not in_reg.any():
        continue

    ids = flat_id[in_reg]
    sum_sc[1995] += np.bincount(ids, weights=pc95[in_reg], minlength=n_reg)
    sum_sc[2030] += np.bincount(ids, weights=pc30[in_reg], minlength=n_reg)
    shared_count += np.bincount(ids, minlength=n_reg).astype(int)
    print(f"  {tname}: {in_reg.sum():>9,} shared px")

print(f"elapsed: {time.time()-t0:.0f}s")

print("\nComputing delta …")
cnt = shared_count.astype(np.float64)
cnt[cnt == 0] = np.nan
mean95 = sum_sc[1995] / cnt
mean30 = sum_sc[2030] / cnt

all_m   = np.concatenate([mean95[np.isfinite(mean95)], mean30[np.isfinite(mean30)]])
vmin    = np.percentile(all_m, 2)
vmax    = np.percentile(all_m, 98)
rescale = lambda x: np.clip((x - vmin) / (vmax - vmin), 0, 1)

idx95 = np.where(np.isfinite(mean95), rescale(mean95), np.nan)
idx30 = np.where(np.isfinite(mean30), rescale(mean30), np.nan)
delta = idx30 - idx95

delta = np.where(np.isnan(delta), 0.0, delta)

print(f"  Δ range : [{np.nanmin(delta):.3f}, {np.nanmax(delta):.3f}]")
print(f"  Δ mean  : {np.nanmean(delta):.3f}")

import pandas as pd
gdf_wgs = gdf_wgs.copy()
gdf_wgs["delta"]   = delta
gdf_wgs["country"] = gdf["country"].values

delta_csv = DATA_DIR / "fig_1_9_exposure_delta.csv"
gdf_wgs[["region_nam", "country", "delta"]].rename(
    columns={"region_nam": "region_name"}
).to_csv(delta_csv, index=False)
print(f"Delta CSV saved → {delta_csv}")

summary = gdf_wgs.groupby("country")["delta"].agg(["mean","min","max","count"]).round(3)
print("\nDelta by country:\n", summary.to_string())

print("\nRendering figure …")

BOUNDS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
N      = len(BOUNDS) - 1
cmap   = cm.get_cmap("YlOrRd", N)
norm   = mcolors.BoundaryNorm(BOUNDS, ncolors=N, clip=True)

_CTR = {
    "Dominican Republic": (-70.15, 18.75, 4.5, 3.0),
    "Haiti":              (-73.10, 19.05, 4.5, 3.0),
    "Jamaica":            (-77.25, 18.10, 4.5, 3.0),
    "Bahamas":            (-77.50, 24.50, 9.0, 6.0),
    "Trinidad and Tobago":(-61.30, 10.65, 3.5, 2.2),
    "Barbados":           (-59.55, 13.15, 1.8, 1.2),
    "Saint Lucia":        (-60.98, 13.88, 1.8, 1.2),
    "Dominica":           (-61.35, 15.42, 1.8, 1.2),
    "Colombia":           (-74.50,  4.50,18.0,14.0),
    "Peru":               (-76.00, -9.00,18.0,16.0),
    "Guyana":             (-59.00,  4.80,10.0, 8.0),
}
ZOOM  = {n: (cx-ls/2, cx+ls/2, cy-la/2, cy+la/2)
         for n, (cx, cy, ls, la) in _CTR.items()}

PALETTE = [
    "#2980B9","#27AE60","#8E44AD","#D35400","#16A085",
    "#C0392B","#F39C12","#1ABC9C","#2C3E50","#7F8C8D","#E91E63",
]
COUNTRY_ORDER = list(_CTR.keys())
ZCOL  = {n: PALETTE[i] for i, n in enumerate(COUNTRY_ORDER)}

SHORT = {
    "Dominican Republic":  "Dom. Rep.",
    "Haiti":               "Haiti",
    "Jamaica":             "Jamaica",
    "Bahamas":             "Bahamas",
    "Trinidad and Tobago": "Trinidad & Tobago",
    "Barbados":            "Barbados",
    "Saint Lucia":         "Saint Lucia",
    "Dominica":            "Dominica",
    "Colombia":            "Colombia",
    "Peru":                "Peru",
    "Guyana":              "Guyana",
}

country_out = gdf_wgs.dissolve(by="country")

NCOLS_INS, NROWS_INS = 4, 3
fig = plt.figure(figsize=(10.0, 10.0), facecolor="white")
outer = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.8],
                         hspace=0.08, left=0.03, right=0.97,
                         top=0.97, bottom=0.06)

ax_main = fig.add_subplot(outer[0])
gs_ins  = outer[1].subgridspec(NROWS_INS, NCOLS_INS, hspace=0.35, wspace=0.10)

ax_main.set_facecolor("#cfe2f3")
gdf_wgs.plot("delta", ax=ax_main, cmap=cmap, norm=norm,
             linewidth=0.15, edgecolor="#888888",
             missing_kwds={"color": "#d0d0d0", "edgecolor": "#888888"})
country_out.boundary.plot(ax=ax_main, linewidth=0.5, edgecolor="#222222")

for name, (lon1, lon2, lat1, lat2) in ZOOM.items():
    ax_main.add_patch(mpatches.Rectangle(
        (lon1, lat1), lon2-lon1, lat2-lat1,
        linewidth=0.8, edgecolor=ZCOL[name], facecolor="none", zorder=5))

lon_min, lat_min, lon_max, lat_max = gdf_wgs.total_bounds
ax_main.set_xlim(lon_min-0.5, lon_max+0.5)
ax_main.set_ylim(lat_min-0.5, lat_max+0.5)
ax_main.set_xlabel(""); ax_main.set_ylabel("")
ax_main.tick_params(labelsize=6, length=2)
for sp in ax_main.spines.values(): sp.set_linewidth(0.4)

for idx, name in enumerate(COUNTRY_ORDER):
    row, col = divmod(idx, NCOLS_INS)
    ax = fig.add_subplot(gs_ins[row, col])
    lon1, lon2, lat1, lat2 = ZOOM[name]
    sub = gdf_wgs[gdf_wgs["country"] == name]
    ax.set_facecolor("#cfe2f3")
    sub.plot("delta", ax=ax, cmap=cmap, norm=norm,
             linewidth=0.4, edgecolor="#444444",
             missing_kwds={"color": "#d0d0d0", "edgecolor": "#444444"})
    sub.dissolve().boundary.plot(ax=ax, linewidth=0.6, edgecolor="#222222")
    ax.set_xlim(lon1, lon2); ax.set_ylim(lat1, lat2)
    ax.set_aspect("auto")
    ax.set_title(SHORT[name], loc="left",
                 fontsize=6.0, pad=2, color="#111111", fontweight="normal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_edgecolor(ZCOL[name])

fig.add_subplot(gs_ins[NROWS_INS-1, NCOLS_INS-1]).set_visible(False)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.05, 0.025, 0.55, 0.016])
cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
                  boundaries=BOUNDS, ticks=BOUNDS, spacing="uniform", extend="max")
cb.ax.tick_params(labelsize=6, length=2)
cb.ax.set_xticklabels([f"{b:.2f}" for b in BOUNDS])
cb.outline.set_visible(False)
cb.set_label("Δ Exposure index (2030 − 1995)", fontsize=6.5, labelpad=3)

out_path = OUTDIR / "fig_1_9_exposure_change.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved → {out_path}")
print(f"Total elapsed: {time.time()-t0:.0f}s")
