import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap
from pathlib import Path
import logging

from climada.hazard import TCTracks, TropCyclone
from climada.hazard.centroids import Centroids

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

BASE_DIR  = Path('/home/jupyter-daniela/CAF')
SHAPEFILE = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
HDF5_DIR  = BASE_DIR / 'data/footprints'
OUT_DIR   = BASE_DIR / 'outputs/cyclone/footprints'
PER_STORM_DIR = OUT_DIR / 'all'
HDF5_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_STORM_DIR.mkdir(parents=True, exist_ok=True)

YEAR_RANGE = (2021, 2026)

CARIB_BOUNDS = (-100, 5, -50, 32)
GRID_RES     = 0.1

EMDAT_SIDS = {
    '2021182N09317',
    '2022257N16312',
    '2022311N21293',
    '2023232N13300',
    '2023321N15278',
    '2024181N09320',
    '2024309N13283',
    '2024319N16282',
    '2025266N16305',
    '2025291N11319',
}

SS_BOUNDS_MS = [0, 17.5, 32.7, 42.7, 49.4, 58.1, 70.5, 300]
SS_BOUNDS_KT = [0,  34,   64,   83,   96,  113,  137,  999]
SS_COLORS    = ['#74c0fc', '#51cf66', '#ffd43b', '#ff922b', '#f03e3e', '#c92a2a', '#862e9c']
SS_LABELS    = [
    'TD (<34 kt)', 'TS (34–63 kt)', 'Cat 1 (64–82 kt)',
    'Cat 2 (83–95 kt)', 'Cat 3 (96–112 kt)', 'Cat 4 (113–136 kt)', 'Cat 5 (≥137 kt)',
]
SS_CMAP_D  = ListedColormap(SS_COLORS)
SS_NORM_MS = BoundaryNorm(SS_BOUNDS_MS, SS_CMAP_D.N)
SS_NORM_KT = BoundaryNorm(SS_BOUNDS_KT, SS_CMAP_D.N)

FOOT_CMAP = LinearSegmentedColormap.from_list(
    'ss_continuous',
    [
        (0.00, '#ffffff'),
        (0.01, '#d4eaf7'),
        (0.15, '#74c0fc'),
        (0.30, '#51cf66'),
        (0.50, '#ffd43b'),
        (0.65, '#ff922b'),
        (0.80, '#f03e3e'),
        (0.92, '#c92a2a'),
        (1.00, '#862e9c'),
    ],
)

CAT_LABEL = {-1: 'TD', 0: 'TD', 1: 'Cat1', 2: 'Cat2', 3: 'Cat3', 4: 'Cat4', 5: 'Cat5'}

BG    = 'white'
OCEAN = '#a8d5e2'
GRID  = '#dee2e6'
TEXT  = '#212529'
MUTED = '#6c757d'

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor': BG,
    'axes.edgecolor':   GRID, 'axes.labelcolor': TEXT,
    'xtick.color':      MUTED, 'ytick.color': MUTED,
    'text.color':       TEXT, 'font.family': 'DejaVu Sans', 'font.size': 9,
})

print("=" * 62)
print(f"B02 — All Caribbean cyclone footprints  {YEAR_RANGE[0]}–{YEAR_RANGE[1]-1}")
print("=" * 62)

log.info("Loading all NA basin tracks %d–%d with provider='usa' …", *YEAR_RANGE)
tc_all = TCTracks.from_ibtracs_netcdf(
    provider='usa',
    basin='NA',
    year_range=YEAR_RANGE,
)
log.info("  Total NA storms loaded: %d", tc_all.size)

lon_min, lat_min, lon_max, lat_max = CARIB_BOUNDS
carib_tracks = []
for t in tc_all.data:
    lats = t['lat'].values
    lons = t['lon'].values
    if ((lats >= lat_min) & (lats <= lat_max) &
            (lons >= lon_min) & (lons <= lon_max)).any():
        carib_tracks.append(t)

tc_all.data = carib_tracks
log.info("  Caribbean storms (≥1 point in %.0f–%.0f°N, %.0f–%.0f°W): %d",
         lat_min, lat_max, -lon_max, -lon_min, tc_all.size)

loaded_sids = {t.attrs['sid'] for t in tc_all.data}
missing_emdat = EMDAT_SIDS - loaded_sids
if missing_emdat:
    log.warning("EMDAT SID(s) absent with provider='usa' — retrying with 'official': %s",
                missing_emdat)
    tc2 = TCTracks.from_ibtracs_netcdf(
        provider='official',
        storm_id=list(missing_emdat),
        year_range=YEAR_RANGE,
    )
    for t in tc2.data:
        log.info("  Added via 'official': %s (%s)", t.attrs['sid'], t.attrs['name'])
    tc_all.data.extend(tc2.data)

log.info("Interpolating to 1-hour timestep …")
tc_all.equal_timestep(time_step_h=1)

tracks_by_sid = {t.attrs['sid']: t for t in tc_all.data}

storms_meta = {}
for t in tc_all.data:
    sid  = t.attrs['sid']
    name = t.attrs.get('name', sid)
    cat  = int(t.attrs.get('category', -1))
    year = int(str(t.time.values[0])[:4])
    storms_meta[sid] = (name, year, cat)

log.info("Final storm count: %d", len(storms_meta))

log.info("Building Caribbean grid (%.1f° res) …", GRID_RES)
centroids = Centroids.from_pnt_bounds(CARIB_BOUNDS, GRID_RES)
NLON = len(np.unique(centroids.lon))
NLAT = len(np.unique(centroids.lat))
log.info("  Grid: %d × %d", NLAT, NLON)

log.info("Computing TropCyclone footprints for %d storms (H08) …", tc_all.size)
haz = TropCyclone.from_tracks(tc_all, centroids)
log.info("  Hazard shape: %s  |  max v_max: %.1f m/s",
         haz.intensity.shape, haz.intensity.max())

event_sid_idx = {haz.event_name[i]: i for i in range(len(haz.event_name))}

hdf5_path = HDF5_DIR / f"footprints_all_caribbean_{YEAR_RANGE[0]}_{YEAR_RANGE[1]-1}.hdf5"
haz.write_hdf5(str(hdf5_path))
log.info("Saved hazard → %s", hdf5_path)

carib_gdf = gpd.read_file(SHAPEFILE).to_crs('EPSG:4326')
LAT2D = centroids.lat.reshape(NLAT, NLON)
LON2D = centroids.lon.reshape(NLAT, NLON)

def to_grid(flat):
    g = flat.reshape(NLAT, NLON).copy()
    g[g == 0] = np.nan
    return g

def ss_legend(include_track=True):
    h = [mpatches.Patch(facecolor=c, label=l, edgecolor='#aaa', linewidth=0.3)
         for c, l in zip(SS_COLORS, SS_LABELS)]
    if include_track:
        h.append(mlines.Line2D([], [], color='black', lw=1.5, label='Storm track'))
    return h

def plot_storm_footprint(sid, out_path, emdat_match=False):
    name, year, cat = storms_meta[sid]
    ev_idx = event_sid_idx.get(sid)
    if ev_idx is None:
        log.warning("SID %s absent from hazard — skipping", sid)
        return

    intensity_flat = np.asarray(haz.intensity[ev_idx, :].todense()).flatten()
    vmax_grid = to_grid(intensity_flat)
    vmax_max  = np.nanmax(vmax_grid) if not np.all(np.isnan(vmax_grid)) else 0.0

    trk = tracks_by_sid.get(sid)

    fig, ax = plt.subplots(figsize=(13, 7), facecolor=BG)
    ax.set_facecolor(OCEAN)
    carib_gdf.plot(ax=ax, color='#e8e8e8', edgecolor='#999999', linewidth=0.3, zorder=2)

    pm = ax.pcolormesh(LON2D, LAT2D, vmax_grid,
                       cmap=FOOT_CMAP, vmin=0, vmax=70,
                       shading='nearest', zorder=3, alpha=0.9)
    cbar = fig.colorbar(pm, ax=ax, fraction=0.025, pad=0.01, label='v_max (m/s)')
    cbar.ax.tick_params(labelsize=8)

    if trk is not None:
        lats    = trk['lat'].values
        lons    = trk['lon'].values
        wind_kt = trk['max_sustained_wind'].values
        for i in range(len(lats) - 1):
            col = SS_CMAP_D(SS_NORM_KT(wind_kt[i]))
            ax.plot([lons[i], lons[i + 1]], [lats[i], lats[i + 1]],
                    color=col, lw=2.0, solid_capstyle='round', zorder=6,
                    path_effects=[pe.Stroke(linewidth=3.5, foreground='white', alpha=0.5),
                                  pe.Normal()])
        if len(lats) >= 3:
            ax.annotate('', xy=(lons[-1], lats[-1]), xytext=(lons[-3], lats[-3]),
                        arrowprops=dict(arrowstyle='->', color=TEXT, lw=1.2), zorder=7)

    ax.set_xlim(-100, -50)
    ax.set_ylim(5, 32)
    ax.set_xlabel('Longitude', color=MUTED)
    ax.set_ylabel('Latitude',  color=MUTED)
    ax.grid(True, alpha=0.2, linewidth=0.3)
    ax.legend(handles=ss_legend(), loc='lower left',
              framealpha=0.92, facecolor='white', edgecolor=GRID,
              fontsize=7, title='SS Intensity', title_fontsize=7.5)

    cat_str = CAT_LABEL.get(cat, f'Cat{cat}')
    emdat_tag = '  ★ EM-DAT event' if emdat_match else ''
    ax.set_title(
        f"{name}  ·  {year}  ·  {cat_str}  —  v_max footprint{emdat_tag}\n"
        f"Peak: {vmax_max:.1f} m/s  |  SID: {sid}",
        fontsize=11, fontweight='normal', color=TEXT, pad=8,
    )
    fig.text(0.5, 0.005,
             'Source: IBTrACS v04r01 · CLIMADA TropCyclone H08  |  Grid: 0.1°',
             ha='center', color=MUTED, fontsize=7)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()

print(f"\nGenerating {len(storms_meta)} per-storm footprint maps …")
for k, (sid, (name, year, cat)) in enumerate(
        sorted(storms_meta.items(), key=lambda x: (x[1][1], x[1][0]))):
    cat_str   = CAT_LABEL.get(cat, f'Cat{cat}')
    safe_name = name.replace(' ', '_')
    fname     = PER_STORM_DIR / f"footprint_{safe_name}_{year}_{cat_str}.png"
    is_emdat  = sid in EMDAT_SIDS
    plot_storm_footprint(sid, fname, emdat_match=is_emdat)
    tag = ' ★' if is_emdat else ''
    log.info("  [%2d/%d] %s %d (%s)%s → %s",
             k + 1, len(storms_meta), name, year, cat_str, tag, fname.name)

def composite_map(title, subtitle, sid_set, out_path, highlight_sids=None):
    indices = [event_sid_idx[s] for s in sid_set if s in event_sid_idx]
    if not indices:
        log.warning("No matching events for composite — skipping %s", out_path.name)
        return

    flat = np.asarray(
        haz.intensity[indices, :].max(axis=0).todense()
    ).flatten()
    grid     = to_grid(flat)
    vmax_val = np.nanmax(grid) if not np.all(np.isnan(grid)) else 0.0

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    ax.set_facecolor(OCEAN)
    carib_gdf.plot(ax=ax, color='#e8e8e8', edgecolor='#999999', linewidth=0.3, zorder=2)

    pm = ax.pcolormesh(LON2D, LAT2D, grid,
                       cmap=FOOT_CMAP, vmin=0, vmax=70,
                       shading='nearest', zorder=3, alpha=0.9)
    cbar = fig.colorbar(pm, ax=ax, fraction=0.018, pad=0.01,
                        label='v_max composite (m/s)')
    cbar.ax.tick_params(labelsize=8)

    hl = highlight_sids or set()
    for sid in sid_set:
        trk = tracks_by_sid.get(sid)
        if trk is None:
            continue
        lats = trk['lat'].values
        lons = trk['lon'].values
        is_hl  = sid in hl
        color  = '#e63946' if is_hl else '#444444'
        lw     = 1.8       if is_hl else 0.7
        alpha  = 0.95      if is_hl else 0.35
        zorder = 7         if is_hl else 5
        ax.plot(lons, lats, color=color, lw=lw, alpha=alpha,
                zorder=zorder, solid_capstyle='round')
        if is_hl:
            name = storms_meta[sid][0]
            year = storms_meta[sid][1]
            mid  = len(lats) // 2
            ax.text(lons[mid], lats[mid], f"{name[:4]}{str(year)[2:]}",
                    fontsize=6, color=color, fontweight='bold', zorder=8,
                    path_effects=[pe.Stroke(linewidth=1.5, foreground='white'),
                                  pe.Normal()])

    ax.set_xlim(-100, -50)
    ax.set_ylim(5, 32)
    ax.set_xlabel('Longitude', color=MUTED)
    ax.set_ylabel('Latitude',  color=MUTED)
    ax.grid(True, alpha=0.2, linewidth=0.3)

    ss_patches = [mpatches.Patch(facecolor=c, label=l, edgecolor='#aaa', linewidth=0.3)
                  for c, l in zip(SS_COLORS, SS_LABELS)]
    ax.legend(handles=ss_patches, loc='lower left', framealpha=0.92,
              facecolor='white', edgecolor=GRID, fontsize=7,
              title='SS Intensity', title_fontsize=7.5)

    ax.set_title(
        f"{title}\n{subtitle}  |  Peak: {vmax_val:.1f} m/s  |  n={len(indices)} storms",
        fontsize=12, fontweight='normal', color=TEXT, pad=10,
    )
    fig.text(0.5, 0.005,
             'Source: IBTrACS v04r01 · CLIMADA TropCyclone H08  |  Grid: 0.1°  |  '
             'Red tracks = EM-DAT matched storms',
             ha='center', color=MUTED, fontsize=7)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    log.info("  → %s", out_path.name)

print("\nGenerating composite maps …")

all_sids = set(storms_meta.keys())

composite_map(
    title=f"Caribbean composite v_max  ·  ALL storms  ·  {YEAR_RANGE[0]}–{YEAR_RANGE[1]-1}",
    subtitle="Maximum wind speed at each grid point across all Caribbean storms",
    sid_set=all_sids,
    out_path=OUT_DIR / 'footprint_composite_all.png',
    highlight_sids=EMDAT_SIDS,
)

composite_map(
    title=f"Caribbean composite v_max  ·  EM-DAT matched storms  ·  {YEAR_RANGE[0]}–{YEAR_RANGE[1]-1}",
    subtitle="10 storms matched to observed disaster events (EM-DAT)",
    sid_set=EMDAT_SIDS,
    out_path=OUT_DIR / 'footprint_composite_emdat.png',
)

print("\n" + "=" * 62)
print(f"Done.")
print(f"  HDF5 : {hdf5_path}")
print(f"  PNGs : {len(list(PER_STORM_DIR.glob('*.png')))} per-storm + "
      f"{len(list(OUT_DIR.glob('footprint_composite*.png')))} composites")
print(f"  Dirs : {OUT_DIR}")
print("=" * 62)
