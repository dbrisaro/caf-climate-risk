
import os
import sys
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROJ_DATA", "/home/jupyter-daniela/.conda/envs/peru_environment/share/proj")
os.environ.setdefault("PROJ_LIB",  "/home/jupyter-daniela/.conda/envs/peru_environment/share/proj")

sys.path.insert(0, "/home/jupyter-daniela/CAF/src")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

import style
style.apply()

BASE_DIR = Path("/home/jupyter-daniela/CAF")
SHP_PATH = BASE_DIR / "data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp"
PROC     = BASE_DIR / "data/processed/caribbean_adm_boundaries"
HW_AGG   = PROC / "heatwaves_computation/heatwaves_aggregated_data.parquet"
MP_AGG   = PROC / "master_panel.parquet"
OUT_DIR  = BASE_DIR / "outputs/hazards"
EXP_DIR  = BASE_DIR / "outputs/exposure"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR.mkdir(parents=True, exist_ok=True)

LABEL_OFFSET = {
    "Colombia":            (-7.0,  3.5),
    "Peru":                (-6.5, -2.0),
    "Guyana":              ( 5.0,  3.0),
    "Cuba":                ( 0.0,  3.5),
    "Haiti":               (-4.5,  2.0),
    "Dominican Republic":  ( 4.5,  2.0),
    "Jamaica":             ( 0.0, -3.0),
    "Belize":              (-3.5,  2.5),
    "Trinidad and Tobago": (-4.5, -1.5),
    "Bahamas":             ( 1.5,  4.0),
    "Barbados":            (-4.0,  0.5),
}

gdf_base = (
    gpd.read_file(SHP_PATH)
       .rename(columns={"region_nam": "region_name"})
       [["country", "region_name", "geometry"]]
)
assert gdf_base.crs.to_epsg() == 4326, "Expected EPSG:4326"

gdf_countries = gdf_base.dissolve(by="country").reset_index()[["country", "geometry"]]

NE_DIR = Path("/home/jupyter-daniela/.local/share/cartopy/shapefiles/natural_earth")

ne_land = gpd.read_file(NE_DIR / "physical/ne_10m_land.shp").to_crs("EPSG:4326")

ne_borders = gpd.read_file(
    NE_DIR / "cultural/ne_10m_admin_0_boundary_lines_land.shp"
).to_crs("EPSG:4326")

def _add_country_labels(ax, gdf_countries, xlim, ylim):
    xmin, xmax = xlim
    ymin, ymax = ylim
    for _, row in gdf_countries.iterrows():
        country = row["country"]
        pt = row["geometry"].representative_point()
        dx, dy = LABEL_OFFSET.get(country, (3.0, 2.0))
        lx = pt.x + dx
        ly = pt.y + dy
        if not (xmin <= pt.x <= xmax and ymin <= pt.y <= ymax):
            continue
        margin = 2.0
        if not (xmin - margin <= lx <= xmax + margin and
                ymin - margin <= ly <= ymax + margin):
            continue
        ax.annotate(
            country,
            xy=(pt.x, pt.y),
            xytext=(lx, ly),
            fontsize=7.5, color="#1a1a1a",
            ha="center", va="center",
            clip_on=True,
            arrowprops=dict(
                arrowstyle="-",
                color="#444444",
                lw=0.6,
                shrinkA=0,
                shrinkB=3,
            ),
        )

def plot_map(
    gdf_data,
    col,
    cmap,
    norm,
    title,
    cbar_label,
    out_path,
    figsize,
    xlim,
    ylim,
    label_countries,
):
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_facecolor("#a8cfe0")

    ne_land.plot(ax=ax, color="#e0e0e0", edgecolor="none")

    gdf_data.plot(
        column=col,
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor="none",
        linewidth=0,
        missing_kwds={"color": "#d8d8d8", "edgecolor": "none", "linewidth": 0},
        legend=False,
    )
    for coll in ax.collections:
        coll.set_rasterized(True)

    ne_borders.plot(ax=ax, color="#555555", linewidth=0.45)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.50, pad=0.02, aspect=22)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.tick_params(labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    ax.set_title(title, fontsize=11, fontweight="normal", pad=8)

    _add_country_labels(ax, label_countries, xlim, ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

print("Building hazard rate tables …")

DROUGHT_AGG  = PROC / "era5_output/droughts_aggregated_data.parquet"
RAINFALL_AGG = PROC / "era5_output_precipitation/rainfall_aggregated_data.parquet"

df_drought_raw = pd.read_parquet(DROUGHT_AGG)
df_rainfall_raw = pd.read_parquet(RAINFALL_AGG)
df_hw_raw      = pd.read_parquet(HW_AGG)

n_pix_d = df_drought_raw.groupby("region_name")["n_drought_pixels"].max()
n_pix_r = df_rainfall_raw.groupby("region_name")["n_rainfall_pixels"].max()
n_pixels = pd.concat([n_pix_d, n_pix_r], axis=1).max(axis=1).clip(lower=1).rename("n_pixels")

drought_p10 = (
    df_drought_raw[df_drought_raw["activacion"] == "p10"]
    .copy()
    .merge(n_pixels, on="region_name", how="left")
)
drought_p10["frac"] = drought_p10["n_drought_pixels"] / drought_p10["n_pixels"]
drought_p10 = drought_p10.rename(columns={"year": "year_val"})

rainfall_p90 = (
    df_rainfall_raw[df_rainfall_raw["activacion"] == "p90"]
    .copy()
    .merge(n_pixels, on="region_name", how="left")
)
rainfall_p90["frac"] = rainfall_p90["n_rainfall_pixels"] / rainfall_p90["n_pixels"]
rainfall_p90 = rainfall_p90.rename(columns={"year": "year_val"})

hw_p90 = (
    df_hw_raw[df_hw_raw["activacion"] == "p90"]
    .copy()
    .merge(n_pixels, on="region_name", how="left")
)
hw_p90["frac"] = hw_p90["n_event_days"] / (hw_p90["n_pixels"] * 91.0)
hw_p90 = hw_p90.rename(columns={"year_win": "year_val"})

def _compute_metrics(df, year_col="year_val"):
    mean = (
        df[(df[year_col] >= 2010) & (df[year_col] <= 2024)]
        .groupby("region_name")["frac"].mean()
    )
    baseline = (
        df[(df[year_col] >= 1995) & (df[year_col] <= 2022)]
        .groupby("region_name")["frac"].mean()
    )
    val_2023 = df[df[year_col] == 2023].groupby("region_name")["frac"].mean()
    anomaly = val_2023 - baseline
    early = (
        df[(df[year_col] >= 2000) & (df[year_col] <= 2009)]
        .groupby("region_name")["frac"].mean()
    )
    late = (
        df[(df[year_col] >= 2015) & (df[year_col] <= 2024)]
        .groupby("region_name")["frac"].mean()
    )
    trend = late - early
    return dict(mean=mean, anomaly=anomaly, trend=trend)

hw_metrics      = _compute_metrics(hw_p90)
drought_metrics = _compute_metrics(drought_p10)
rain_metrics    = _compute_metrics(rainfall_p90)

def _norm_sym(series, pct=97):
    vals = series.dropna().abs()
    vabs = float(np.percentile(vals, pct)) if len(vals) else 0.1
    vabs = max(vabs, 1e-6)
    return mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

def _norm_pos(series, pct=98):
    vals = series.dropna()
    vmax = float(np.percentile(vals[vals > 0], pct)) if (vals > 0).any() else 0.1
    return mcolors.Normalize(vmin=0, vmax=max(vmax, 1e-6))

SA_COUNTRIES = ["Colombia", "Peru", "Guyana"]
gdf_countries_sa = gdf_countries[gdf_countries["country"].isin(SA_COUNTRIES)].copy()

XLIM = (-88, -56)
YLIM = (-22, 31)
FS   = (14, 9)

CBAR_HW  = "Fraction of sub-region area in heatwave per season (p90)"
CBAR_DHW = "Δ fraction of area in heatwave per season"

print("[Map H1] heatwave_map_mean …")
s = hw_metrics["mean"].rename("val")
gdf_m = gdf_base.merge(s.reset_index(), on="region_name", how="left")
plot_map(gdf_m, "val", "YlOrRd", _norm_pos(s),
         "Heatwave exposure — mean fraction of area affected per season (2010–2024)",
         CBAR_HW, OUT_DIR / "heatwave_map_mean.png", FS, XLIM, YLIM, gdf_countries)
print("  Saved: heatwave_map_mean.png")

print("[Map H2] heatwave_map_anomaly_2023 …")
s = hw_metrics["anomaly"].rename("val")
gdf_m = gdf_base.merge(s.reset_index(), on="region_name", how="left")
plot_map(gdf_m, "val", "RdBu_r", _norm_sym(s),
         "Heatwave anomaly — 2023 vs. 1995–2022 baseline",
         CBAR_DHW, OUT_DIR / "heatwave_map_anomaly_2023.png", FS, XLIM, YLIM, gdf_countries)
print("  Saved: heatwave_map_anomaly_2023.png")

print("[Map H3] heatwave_map_trend …")
s = hw_metrics["trend"].rename("val")
gdf_m = gdf_base.merge(s.reset_index(), on="region_name", how="left")
plot_map(gdf_m, "val", "RdBu_r", _norm_sym(s),
         "Heatwave trend — change in area affected: 2000–2009 → 2015–2024",
         CBAR_DHW, OUT_DIR / "heatwave_map_trend.png", FS, XLIM, YLIM, gdf_countries)
print("  Saved: heatwave_map_trend.png")

CBAR_DR  = "Fraction of sub-region area in drought per season (below p10)"
CBAR_DDR = "Δ fraction of area in drought per season"

print("[Map D1] drought_map_mean …")
s = drought_metrics["mean"].rename("val")
gdf_m = gdf_base.merge(s.reset_index(), on="region_name", how="left")
plot_map(gdf_m, "val", "YlOrBr", _norm_pos(s),
         "Drought exposure — mean fraction of area affected per season (2010–2024)",
         CBAR_DR, OUT_DIR / "drought_map_mean.png", FS, XLIM, YLIM, gdf_countries)
print("  Saved: drought_map_mean.png")

print("[Map D2] drought_map_anomaly_2023 …")
s = drought_metrics["anomaly"].rename("val")
gdf_m = gdf_base.merge(s.reset_index(), on="region_name", how="left")
plot_map(gdf_m, "val", "RdBu_r", _norm_sym(s),
         "Drought anomaly — 2023 vs. 1995–2022 baseline",
         CBAR_DDR, OUT_DIR / "drought_map_anomaly_2023.png", FS, XLIM, YLIM, gdf_countries)
print("  Saved: drought_map_anomaly_2023.png")

print("[Map D3] drought_map_trend …")
s = drought_metrics["trend"].rename("val")
gdf_m = gdf_base.merge(s.reset_index(), on="region_name", how="left")
plot_map(gdf_m, "val", "RdBu_r", _norm_sym(s),
         "Drought trend — change in area affected: 2000–2009 → 2015–2024",
         CBAR_DDR, OUT_DIR / "drought_map_trend.png", FS, XLIM, YLIM, gdf_countries)
print("  Saved: drought_map_trend.png")

CBAR_RF  = "Fraction of sub-region area with extreme rainfall per season (above p90)"
CBAR_DRF = "Δ fraction of area with extreme rainfall per season"

print("[Map R1] rainfall_map_mean …")
s = rain_metrics["mean"].rename("val")
gdf_m = gdf_base.merge(s.reset_index(), on="region_name", how="left")
plot_map(gdf_m, "val", "Blues", _norm_pos(s),
         "Extreme rainfall exposure — mean fraction of area affected per season (2010–2024)",
         CBAR_RF, OUT_DIR / "rainfall_map_mean.png", FS, XLIM, YLIM, gdf_countries)
print("  Saved: rainfall_map_mean.png")

print("[Map R2] rainfall_map_anomaly_2023 …")
s = rain_metrics["anomaly"].rename("val")
gdf_m = gdf_base.merge(s.reset_index(), on="region_name", how="left")
plot_map(gdf_m, "val", "RdBu_r", _norm_sym(s),
         "Extreme rainfall anomaly — 2023 vs. 1995–2022 baseline",
         CBAR_DRF, OUT_DIR / "rainfall_map_anomaly_2023.png", FS, XLIM, YLIM, gdf_countries)
print("  Saved: rainfall_map_anomaly_2023.png")

print("[Map R3] rainfall_map_trend …")
s = rain_metrics["trend"].rename("val")
gdf_m = gdf_base.merge(s.reset_index(), on="region_name", how="left")
plot_map(gdf_m, "val", "RdBu_r", _norm_sym(s),
         "Extreme rainfall trend — change in area affected: 2000–2009 → 2015–2024",
         CBAR_DRF, OUT_DIR / "rainfall_map_trend.png", FS, XLIM, YLIM, gdf_countries)
print("  Saved: rainfall_map_trend.png")

print("[Map E1] exposure_change_map …")
mp = pd.read_parquet(MP_AGG)
exp_by_region_year = mp.groupby(["region_name", "year"], as_index=False)["exposure_index"].mean()
e1995 = (
    exp_by_region_year[exp_by_region_year["year"] == 1995]
    .set_index("region_name")["exposure_index"]
)
e2030 = (
    exp_by_region_year[exp_by_region_year["year"] == 2030]
    .set_index("region_name")["exposure_index"]
)
exp_diff = (e2030 - e1995).rename("val").reset_index()
gdf_exp = gdf_base.merge(exp_diff, on="region_name", how="left")
absvals = gdf_exp["val"].dropna().abs()
vabs = float(np.percentile(absvals, 97)) if len(absvals) else 0.2
norm_exp = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
plot_map(gdf_exp, "val", "RdBu_r", norm_exp,
         "Change in built-up exposure index — 1995 → 2030",
         "Δ Exposure index (2030 − 1995)",
         EXP_DIR / "exposure_change_map.png", FS, XLIM, YLIM, gdf_countries)
print("  Saved: exposure_change_map.png")

print(f"\nAll maps → {OUT_DIR}/")
