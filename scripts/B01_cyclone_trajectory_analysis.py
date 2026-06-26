import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import BoundaryNorm, ListedColormap
from pathlib import Path
from datetime import datetime, timedelta
from adjustText import adjust_text

BASE_DIR   = Path('/home/jupyter-daniela/CAF')
IBTRACS_NC = BASE_DIR / 'data/IBTrACS/IBTrACS.since1980.v04r01.nc'
SHAPEFILE  = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
OUT_DIR    = BASE_DIR / 'outputs/cyclone/trajectories'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_RAW = [
    ("Môle Saint-Nicolas, le Cap-Haïtien, le Trou-du-Nord", "03/04/2021", "Haiti",                 "Flood"),
    ("Bahoruco",                                             "03/07/2021", "Dominican Republic",    "Storm"),
    ("Christ Church, Saint Philip",                         "03/07/2021", "Barbados",               "Storm"),
    ("Choiseul, Soufrière",                                 "03/07/2021", "Saint Lucia",            "Storm"),
    ("Grand'Anse, Nippes, Ouest, Sud, Sud-Est",             "03/07/2021", "Haiti",                  "Storm"),
    ("Nippes, Nord, Nord-Est, Nord-Ouest",                  "30/01/2022", "Haiti",                  "Flood"),
    ("Puerto Plata",                                        "05/03/2022", "Dominican Republic",    "Flood"),
    ("le Limbé",                                            "05/03/2022", "Haiti",                  "Flood"),
    ("Duarte, Espaillat, Hato Mayor, La Altagracia, La Romana, La Vega, "
     "María Trinidad Sánchez, Monte Plata, Puerto Plata, Samaná, "
     "San José de Ocoa, San Pedro de Macorís, Sánchez Ramírez",
                                                            "19/09/2022", "Dominican Republic",    "Storm"),
    ("Gros Islet",                                         "06/11/2022", "Saint Lucia",            "Flood"),
    ("Couva-Tabaquite-Talparo, Tunapuna/Piarco",           "26/11/2022", "Trinidad and Tobago",    "Flood"),
    ("Centre, Grand'Anse, Nippes, Nord-Ouest, Ouest, Sud, Sud-Est",
                                                            "02/06/2023", "Haiti",                  "Flood"),
    ("Azua, Bahoruco, Barahona, Dajabón, Espaillat, Independencia, "
     "La Estrelleta, La Vega, Pedernales, San Juan, Santiago, Santiago Rodríguez",
                                                            "02/06/2023", "Dominican Republic",    "Flood"),
    ("Azua, Bahoruco, Barahona, Elias Pina, Independencia", "20/08/2023", "Dominican Republic",    "Storm"),
    ("Grand'Anse Department",                               "17/11/2023", "Haiti",                  "Flood"),
    ("Azua, Bahoruco, Barahona, Distrito Nacional, Duarte, Monte Plata, "
     "Peravia, San Cristobal, San José de Ocoa, San Juan de la Maguana, "
     "San Pedro de Macoris, Santo Domingo, Sánchez Ramírez",
                                                            "17/11/2023", "Dominican Republic",    "Flood"),
    ("Clarendon, Hanover, Manchester, St. Thomas, St.Elizabeth, Westmoreland",
                                                            "03/07/2024", "Jamaica",                "Storm"),
    ("Duarte, El Seibo, Espaillat, Hato Mayor, La Altagracia, Puerto Plata, Samanà",
                                                            "10/11/2024", "Dominican Republic",    "Flood"),
    ("Les Cayes, Saint-Louis-du-Sud, Torbeck",             "10/11/2024", "Haiti",                  "Flood"),
    ("Duarte, El Seibo, Espaillat, Hato Mayor, La Altagracia, Puerto Plata, Samanà",
                                                            "17/11/2024", "Dominican Republic",    "Storm"),
    ("Grand'Anse, Sud",                                     "30/11/2024", "Haiti",                  "Flood"),
    ("Grand'Anse, Nippes, Nord-Ouest",                     "20/12/2024", "Haiti",                  "Flood"),
    ("Chansolme",                                           "16/09/2025", "Haiti",                  "Flood"),
    ("Nassau",                                              "27/09/2025", "Bahamas",                "Storm"),
    ("Clarendon, Hanover, Kingston, Manchester, Saint Andrew, Saint Ann, "
     "Saint Catherine, Saint Elizabeth, Saint James, Saint Mary, "
     "Trelawny, Westmoreland",                             "24/10/2025", "Jamaica",                "Storm"),
    ("Azua, Barahona, Distrito Nacional, Elias Pina, Peravia, San Cristobal, "
     "San José de Ocoa, San Juan, San Pedro de Macoris, Santo Domingo",
                                                            "26/10/2025", "Dominican Republic",    "Storm"),
    ("Gaspar Hernández (Espaillat), Puerto Plata",         "24/02/2026", "Dominican Republic",    "Flood"),
]

COUNTRY_CENTROIDS = {
    "Haiti":               (18.97, -72.29),
    "Dominican Republic":  (18.74, -70.16),
    "Jamaica":             (18.11, -77.30),
    "Barbados":            (13.19, -59.54),
    "Saint Lucia":         (13.91, -60.98),
    "Trinidad and Tobago": (10.65, -61.52),
    "Bahamas":             (25.04, -77.36),
    "Cuba":                (21.52, -79.02),
}

SS_BOUNDS  = [0, 34, 64, 83, 96, 113, 137, 200]
SS_COLORS  = ['#adb5bd', '#74c0fc', '#ffe066', '#ff922b', '#f03e3e', '#9b1c1c', '#6741d9']
SS_LABELS  = ['TD (<34 kt)', 'TS (34–63 kt)', 'Cat 1 (64–82 kt)',
              'Cat 2 (83–95 kt)', 'Cat 3 (96–112 kt)', 'Cat 4 (113–136 kt)', 'Cat 5 (≥137 kt)']
SS_CMAP    = ListedColormap(SS_COLORS)
SS_NORM    = BoundaryNorm(SS_BOUNDS, SS_CMAP.N)

BG       = 'white'
PANEL_BG = '#f8f9fa'
GRID     = '#dee2e6'
TEXT     = '#212529'
MUTED    = '#6c757d'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': PANEL_BG,
    'axes.edgecolor':   GRID, 'axes.labelcolor': TEXT,
    'xtick.color':      MUTED, 'ytick.color': MUTED,
    'text.color':       TEXT, 'grid.color': GRID,
    'font.family':      'DejaVu Sans', 'font.size': 9,
})

print("Loading IBTrACS …")
ds = nc.Dataset(IBTRACS_NC)

epoch    = datetime(1858, 11, 17)
raw_time = np.array(ds.variables['time'][:], dtype=float)
raw_lat  = np.array(ds.variables['lat'][:],  dtype=float)
raw_lon  = np.array(ds.variables['lon'][:],  dtype=float)
raw_wind = np.array(ds.variables['usa_wind'][:], dtype=float)
raw_wmo  = np.array(ds.variables['wmo_wind'][:], dtype=float)
raw_sid  = ds.variables['sid'][:]
raw_name = ds.variables['name'][:]

n_storms, n_times = raw_time.shape

def decode_chars(row):
    parts = []
    for c in row:
        if hasattr(c, 'decode'):
            parts.append(c.decode('utf-8', errors='ignore'))
        elif isinstance(c, (bytes, bytearray)):
            parts.append(c.decode('utf-8', errors='ignore'))
        elif isinstance(c, str):
            parts.append(c)
    return ''.join(parts).strip()

sids  = [decode_chars(row) for row in raw_sid]
names = [decode_chars(row) for row in raw_name]

flat_idx   = np.repeat(np.arange(n_storms), n_times)
flat_time  = raw_time.ravel()
flat_lat   = raw_lat.ravel()
flat_lon   = raw_lon.ravel()
flat_wind  = raw_wind.ravel()
flat_wmo   = raw_wmo.ravel()

raw_time[raw_time < -9e4]  = np.nan
raw_lat[raw_lat < -999]    = np.nan
raw_lon[raw_lon < -999]    = np.nan
raw_wind[raw_wind < -900]  = np.nan
raw_wmo[raw_wmo < -900]    = np.nan
flat_time  = raw_time.ravel()
flat_lat   = raw_lat.ravel()
flat_lon   = raw_lon.ravel()
flat_wind  = raw_wind.ravel()
flat_wmo   = raw_wmo.ravel()

valid = ~np.isnan(flat_time) & ~np.isnan(flat_lat) & ~np.isnan(flat_lon)
flat_time  = flat_time[valid]
flat_lat   = flat_lat[valid]
flat_lon   = flat_lon[valid]
flat_wind  = flat_wind[valid]
flat_wmo   = flat_wmo[valid]
flat_idx   = flat_idx[valid]

flat_sid_arr  = np.array([sids[i]  for i in flat_idx])
flat_name_arr = np.array([names[i] for i in flat_idx])

flat_dt = np.array([epoch + timedelta(days=float(t)) for t in flat_time])

wind_kt = np.where(~np.isnan(flat_wind), flat_wind, flat_wmo)

carib_mask = (
    (flat_lat >= 5)  & (flat_lat <= 32) &
    (flat_lon >= -100) & (flat_lon <= -50)
)
flat_time  = flat_time[carib_mask]
flat_lat   = flat_lat[carib_mask]
flat_lon   = flat_lon[carib_mask]
wind_kt    = wind_kt[carib_mask]
flat_dt    = flat_dt[carib_mask]
flat_sid_arr  = flat_sid_arr[carib_mask]
flat_name_arr = flat_name_arr[carib_mask]

tracks_df = pd.DataFrame({
    'sid':  flat_sid_arr,
    'name': flat_name_arr,
    'dt':   flat_dt,
    'lat':  flat_lat,
    'lon':  flat_lon,
    'wind': wind_kt,
})
tracks_df['year'] = pd.to_datetime(tracks_df['dt'].astype(str)).dt.year
print(f"  {tracks_df['sid'].nunique()} Caribbean storms loaded, "
      f"{len(tracks_df):,} track points")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

SEARCH_RADIUS_KM = 1200
SEARCH_DAYS      = 7

events = []
for ciudades, fecha, pais, evento in EVENTS_RAW:
    dt = datetime.strptime(fecha, '%d/%m/%Y')
    events.append({'ciudades': ciudades, 'date': dt, 'country': pais, 'type': evento})

events_df = pd.DataFrame(events)

print("\nMatching events to IBTrACS storms …")
event_matches = []

for _, ev in events_df.iterrows():
    c_lat, c_lon = COUNTRY_CENTROIDS.get(ev['country'], (None, None))
    if c_lat is None:
        print(f"  [warn] no centroid for {ev['country']}")
        continue

    t0 = ev['date'] - timedelta(days=SEARCH_DAYS)
    t1 = ev['date'] + timedelta(days=SEARCH_DAYS)

    window = tracks_df[(tracks_df['dt'] >= t0) & (tracks_df['dt'] <= t1)].copy()
    if window.empty:
        print(f"  [no data] {ev['date'].date()} {ev['country']}")
        event_matches.append({**ev, 'sid': None, 'storm_name': None})
        continue

    window['dist_km'] = haversine_km(c_lat, c_lon, window['lat'].values, window['lon'].values)
    nearby = window[window['dist_km'] <= SEARCH_RADIUS_KM]

    if nearby.empty:
        print(f"  [no match] {ev['date'].date()} {ev['country']} "
              f"(closest: {window['dist_km'].min():.0f} km)")
        event_matches.append({**ev, 'sid': None, 'storm_name': None})
    else:
        best = (nearby.groupby(['sid', 'name'])['dist_km']
                      .min()
                      .reset_index()
                      .sort_values('dist_km')
                      .iloc[0])
        sid  = best['sid']
        sname = best['name'] if best['name'] else sid
        print(f"  [match] {ev['date'].date()} {ev['country']} → {sname} ({sid})")
        event_matches.append({**ev, 'sid': sid, 'storm_name': sname})

matches_df = pd.DataFrame(event_matches)

matched = matches_df[matches_df['sid'].notna()].copy()
unmatched = matches_df[matches_df['sid'].isna()].copy()

print(f"\n  Matched: {len(matched)} / {len(matches_df)} events")
if len(unmatched):
    print("  Unmatched events:")
    for _, r in unmatched.iterrows():
        print(f"    {r['date'].date()} — {r['country']} ({r['type']})")

carib_gdf = gpd.read_file(SHAPEFILE).to_crs('EPSG:4326')

def ss_color(wind):
    if np.isnan(wind) or wind < 0:
        return SS_COLORS[0]
    for i, bound in enumerate(SS_BOUNDS[1:]):
        if wind < bound:
            return SS_COLORS[i]
    return SS_COLORS[-1]

def intensity_label(wind):
    for i, bound in enumerate(SS_BOUNDS[1:]):
        if wind < bound:
            return SS_LABELS[i]
    return SS_LABELS[-1]

def plot_storm(sid, storm_events, out_path):
    track = tracks_df[tracks_df['sid'] == sid].sort_values('dt').copy()
    track['wind'] = track['wind'].fillna(0)

    lat_min = min(track['lat'].min(), 5) - 2
    lat_max = max(track['lat'].max(), 30) + 2
    lon_min = min(track['lon'].min(), -100) - 2
    lon_max = max(track['lon'].max(), -55)  + 2
    lat_min = max(lat_min, 3)
    lat_max = min(lat_max, 35)
    lon_min = max(lon_min, -102)
    lon_max = min(lon_max, -48)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax.set_facecolor('#a8d5e2')

    carib_gdf.plot(ax=ax, color='#d6d6d6', edgecolor='#999999', linewidth=0.15)

    affected_countries = storm_events['country'].unique()
    country_map = {
        'Haiti':               ['Haiti'],
        'Dominican Republic':  ['Dominican Republic'],
        'Jamaica':             ['Jamaica'],
        'Barbados':            ['Barbados'],
        'Saint Lucia':         ['Saint Lucia', 'Sainte-Lucie'],
        'Trinidad and Tobago': ['Trinidad and Tobago'],
        'Bahamas':             ['Bahamas', 'The Bahamas'],
        'Cuba':                ['Cuba'],
    }
    for ac in affected_countries:
        aliases = country_map.get(ac, [ac])
        mask = carib_gdf['country'].str.contains('|'.join(aliases), case=False, na=False)
        if mask.any():
            carib_gdf[mask].plot(ax=ax, color='#f4a261', edgecolor='#c0541a',
                                 linewidth=0.7, alpha=0.7)

    lats  = track['lat'].values
    lons  = track['lon'].values
    winds = track['wind'].values

    for i in range(len(lats) - 1):
        w    = winds[i]
        col  = ss_color(w)
        ax.plot([lons[i], lons[i+1]], [lats[i], lats[i+1]],
                color=col, lw=2.5, solid_capstyle='round',
                path_effects=[pe.Stroke(linewidth=4.5, foreground='white', alpha=0.5),
                               pe.Normal()])

    sc = ax.scatter(lons, lats, c=winds, cmap=SS_CMAP, norm=SS_NORM,
                    s=20, zorder=5, linewidths=0)

    texts = []

    used_dates = set()
    for _, ev in storm_events.iterrows():
        ev_dt = ev['date']
        c_lat, c_lon = COUNTRY_CENTROIDS.get(ev['country'], (None, None))
        if c_lat is None:
            continue
        key = (ev_dt.date(), ev['country'])
        if key not in used_dates:
            ax.plot(c_lon, c_lat, marker='*', color='#e63946', markersize=13,
                    zorder=8, markeredgecolor='white', markeredgewidth=0.6)
            t = ax.text(c_lon + 2.5, c_lat + 1.5,
                        f"{ev['country']}\n{ev_dt.strftime('%d %b %Y')} · {ev['type']}",
                        fontsize=7, color=TEXT, zorder=9,
                        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=GRID,
                                  alpha=0.88, lw=0.6))
            texts.append(t)
            used_dates.add(key)

    track_daily = track.groupby(track['dt'].dt.date).first().reset_index(drop=True)
    for _, row in track_daily.iterrows():
        t = ax.text(row['lon'], row['lat'],
                    pd.Timestamp(row['dt']).strftime('%d %b'),
                    fontsize=6.5, color=MUTED, zorder=7,
                    path_effects=[pe.Stroke(linewidth=2, foreground='white'), pe.Normal()])
        texts.append(t)

    adjust_text(
        texts, ax=ax,
        expand_points=(1.6, 1.6),
        expand_text=(1.4, 1.4),
        arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5),
        force_text=(0.5, 0.6),
        force_points=(0.2, 0.3),
    )

    if len(lats) >= 2:
        ax.annotate('', xy=(lons[-1], lats[-1]),
                    xytext=(lons[-2], lats[-2]),
                    arrowprops=dict(arrowstyle='->', color=TEXT, lw=1.5))

    legend_handles = [
        mpatches.Patch(facecolor=c, label=l) for c, l in zip(SS_COLORS, SS_LABELS)
    ]
    legend_handles.append(
        mlines.Line2D([], [], marker='*', color='#e63946', markersize=10,
                      linestyle='None', label='Affected area')
    )
    ax.legend(handles=legend_handles, loc='lower left',
              framealpha=0.9, facecolor='white', edgecolor=GRID,
              fontsize=7.5, title='Intensity', title_fontsize=8)

    storm_name  = track['name'].iloc[0] if track['name'].iloc[0] else sid
    date_range  = (f"{track['dt'].min().strftime('%d %b %Y')} – "
                   f"{track['dt'].max().strftime('%d %b %Y')}")
    countries_str = ', '.join(sorted(affected_countries))

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude', fontsize=9, color=MUTED)
    ax.set_ylabel('Latitude',  fontsize=9, color=MUTED)
    ax.grid(True, alpha=0.25, linewidth=0.4)

    ax.set_title(
        f"Storm {storm_name}  ·  {date_range}\n"
        f"Affected: {countries_str}",
        fontsize=11, fontweight='normal', color=TEXT, pad=10
    )
    fig.text(0.5, 0.01,
             'Source: IBTrACS v04r01 · EM-DAT event list  |  Wind speed (kts, USA agency)',
             ha='center', color=MUTED, fontsize=7.5)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  → {out_path.name}")

unique_storms = matched.groupby('sid')
for sid, group in unique_storms:
    storm_name = group['storm_name'].iloc[0]
    safe_name  = storm_name.replace(' ', '_').replace('/', '-')
    date_tag   = group['date'].min().strftime('%Y%m')
    fname      = f"storm_{date_tag}_{safe_name}_{sid[-6:]}.png"
    plot_storm(sid, group, OUT_DIR / fname)

print("\nGenerating overview map …")
matched_sids = matched['sid'].unique()
recent_tracks = tracks_df[
    tracks_df['sid'].isin(matched_sids) &
    (tracks_df['year'] >= 2021)
].copy()

fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
ax.set_facecolor('#a8d5e2')
carib_gdf.plot(ax=ax, color='#d6d6d6', edgecolor='#999999', linewidth=0.15)

OVERVIEW_COLORS = [
    '#e63946', '#f4a261', '#2a9d8f', '#e9c46a', '#6a0572',
    '#d62828', '#457b9d', '#f77f00', '#4cc9f0', '#b5179e',
]
storm_color = {sid: OVERVIEW_COLORS[i % len(OVERVIEW_COLORS)]
               for i, sid in enumerate(sorted(matched_sids))}

MELISSA_SID = '2025291N11319'

for sid, grp in recent_tracks.groupby('sid'):
    grp = grp.sort_values('dt')
    lats  = grp['lat'].values
    lons  = grp['lon'].values
    winds = grp['wind'].fillna(0).values
    sname = grp['name'].iloc[0] if grp['name'].iloc[0] else sid
    col   = storm_color[sid]

    for i in range(len(lats) - 1):
        w = winds[i]
        if sid == MELISSA_SID and w < 64:
            continue
        lw = 1.5 + w / 40
        ax.plot([lons[i], lons[i+1]], [lats[i], lats[i+1]],
                color=col, lw=lw, alpha=0.85, solid_capstyle='round')

    plot_lats = lats if sid != MELISSA_SID else lats[winds >= 64]
    plot_lons = lons if sid != MELISSA_SID else lons[winds >= 64]
    if len(plot_lats) == 0:
        continue
    mid = len(plot_lats) // 2
    ax.annotate(sname, xy=(plot_lons[mid], plot_lats[mid]),
                fontsize=6.5, color=col, fontweight='normal',
                path_effects=[pe.Stroke(linewidth=1, foreground='white'), pe.Normal()])

for _, ev in matched.drop_duplicates(['date', 'country']).iterrows():
    c_lat, c_lon = COUNTRY_CENTROIDS.get(ev['country'], (None, None))
    if c_lat is None:
        continue
    ax.plot(c_lon, c_lat, marker='*', color='#e63946', markersize=10,
            zorder=8, markeredgecolor='white', markeredgewidth=0.4)
    ax.annotate(f"{ev['country'][:3]}·{ev['date'].strftime('%y')}",
                xy=(c_lon, c_lat), xytext=(c_lon + 0.4, c_lat + 0.4),
                fontsize=5.5, color=MUTED)

ax.set_xlim(-100, -50)
ax.set_ylim(5, 33)
ax.set_xlabel('Longitude', fontsize=9, color=MUTED)
ax.set_ylabel('Latitude',  fontsize=9, color=MUTED)
ax.grid(True, alpha=0.25, linewidth=0.4)
ax.set_title('Caribbean matched storm trajectories — 2021 to 2025\n'
             'Line width proportional to wind intensity',
             fontsize=13, fontweight='normal', color=TEXT, pad=10)

fig.text(0.5, 0.01,
         'Source: IBTrACS v04r01 · EM-DAT event list  ★ = affected area from event list',
         ha='center', color=MUTED, fontsize=7.5)

plt.tight_layout(rect=[0, 0.02, 1, 1])
overview_path = OUT_DIR / 'overview_all_storms.png'
fig.savefig(overview_path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  → {overview_path.name}")

summary = matches_df[['date', 'country', 'type', 'sid', 'storm_name']].copy()
summary['date'] = summary['date'].dt.strftime('%d %b %Y')
summary['matched'] = summary['sid'].notna()
summary_path = OUT_DIR / 'event_storm_matches.csv'
summary.to_csv(summary_path, index=False)
print(f"\nMatch summary saved → {summary_path}")
print(f"\nAll outputs in {OUT_DIR}")
