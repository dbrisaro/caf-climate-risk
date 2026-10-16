import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.append('/home/jupyter-daniela/CAF/src')

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE_DIR      = Path('/home/jupyter-daniela/CAF')
DATA_PATH     = BASE_DIR / 'data/processed/caribbean_adm_boundaries/master_panel.parquet'
SHAPEFILE     = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
IBTRACS_AGG   = BASE_DIR / 'data/processed/caribbean_adm_boundaries/ibtracs_output/ibtracs_aggregated_data.parquet'
ERA5_DR_RAW   = BASE_DIR / 'data/processed/caribbean_adm_boundaries/era5_output/era5_raw_data.parquet'
ERA5_RR_RAW   = BASE_DIR / 'data/processed/caribbean_adm_boundaries/era5_output_precipitation/era5_raw_data.parquet'
ERA5_TMAX_DIR = Path('/home/jupyter-daniela/suyana/sources/era5/statistics/tmax/caribbean')
EXCESS_CACHE  = BASE_DIR / 'data/processed/caribbean_adm_boundaries/hazard_excess.parquet'
ANOMALY_CACHE = BASE_DIR / 'data/processed/caribbean_adm_boundaries/hazard_anomaly.parquet'
OUT_DIR       = BASE_DIR / 'outputs/regression'
GDP_CACHE     = BASE_DIR / 'data/processed/caribbean_adm_boundaries/wb_gdp_pc_NY.GDP.PCAP.KD_1995_2025_11c.csv'
OUT_DIR.mkdir(parents=True, exist_ok=True)

WB_ISO = {
    'Bahamas':             'BS',
    'Barbados':            'BB',
    'Colombia':            'CO',
    'Dominica':            'DM',
    'Dominican Republic':  'DO',
    'Guyana':              'GY',
    'Haiti':               'HT',
    'Jamaica':             'JM',
    'Peru':                'PE',
    'Saint Lucia':         'LC',
    'Trinidad and Tobago': 'TT',
}

print("=" * 60)
print("07b — Panel regression suite  M1–M6")
print("=" * 60)

print("\n[1] Loading data …")

df  = pd.read_parquet(DATA_PATH)
gdf = gpd.read_file(SHAPEFILE)[['region_nam', 'country']].rename(
        columns={'region_nam': 'region_name'})
# Region names shared across countries are a single merged series in master_panel,
# which is keyed by name only; they are excluded pending a rebuild of the panel
# with unique region keys.
name_ctry = gdf.groupby('region_name')['country'].nunique()
AMBIG     = sorted(name_ctry[name_ctry > 1].index)
print(f"    Excluding shared region names ({len(AMBIG)}): {', '.join(AMBIG)}")
df  = df[~df['region_name'].isin(AMBIG)]
df  = df.merge(gdf, on='region_name', how='left')

if GDP_CACHE.exists():
    print(f"    GDP per capita: {GDP_CACHE.name} (cached World Bank series)")
    df_gdp = pd.read_csv(GDP_CACHE)
else:
    print("    Fetching GDP per capita from World Bank …")
    iso_str = ';'.join(WB_ISO.values())
    wb_data = None
    for _attempt in range(3):
        try:
            wb_data = requests.get(
                f'https://api.worldbank.org/v2/country/{iso_str}/indicator/'
                f'NY.GDP.PCAP.KD?date=1995:2025&format=json&per_page=500', timeout=120
            ).json()[1]
            break
        except Exception as e:
            print(f"      Attempt {_attempt+1} failed: {e}")
    if wb_data is None:
        raise RuntimeError("World Bank API unavailable after 3 attempts")
    iso_to_country = {v: k for k, v in WB_ISO.items()}
    df_gdp = pd.DataFrame([
        {'country': iso_to_country[r['country']['id']],
         'year':    int(r['date']),
         'gdp_pc':  r['value']}
        for r in wb_data if r['value'] is not None
    ]).sort_values(['country', 'year']).reset_index(drop=True)
    df_gdp.to_csv(GDP_CACHE, index=False)
    print(f"    Saved → {GDP_CACHE}")

panel = (
    df.groupby(['region_name', 'country', 'year'])
    .agg(
        total_losses_musd = ('total_losses_musd', 'mean'),
        built_s           = ('built_s',    'mean'),
        built_v           = ('built_v',    'mean'),
        pop               = ('pop',        'mean'),
        drought_raw       = ('drought',    'mean'),
        rainfall_raw      = ('rainfall',   'mean'),
        heatwave_raw      = ('heatwave',   'mean'),
        cyclone_raw       = ('cyclone',    'mean'),
    )
    .reset_index()
    .merge(df_gdp, on=['country', 'year'], how='left')
)

ibt = pd.read_parquet(IBTRACS_AGG)
ibt_wind = (
    ibt.groupby(['region_name', 'year_win'])['avg_max_usa_wind']
    .max()
    .reset_index()
    .rename(columns={'year_win': 'year', 'avg_max_usa_wind': 'wind_kt'})
)
panel = panel.merge(ibt_wind, on=['region_name', 'year'], how='left')
panel['wind_kt'] = panel['wind_kt'].fillna(0.0)

if EXCESS_CACHE.exists():
    print("    Loading cached hazard excess variables …")
    excess = pd.read_parquet(EXCESS_CACHE)
else:
    import xarray as xr
    print("    Computing hazard excess variables (first run — caching) …")

    dr = pd.read_parquet(ERA5_DR_RAW)
    dr['date'] = pd.to_datetime(dr['date'])
    dr['year'] = dr['date'].dt.year
    pct_dr = (dr.groupby(['pixel_id', 'window'])['value']
                .quantile(0.10).reset_index()
                .rename(columns={'value': 'p10'}))
    dr = dr.merge(pct_dr, on=['pixel_id', 'window'], how='left')
    dr_ex = dr[dr['value'] < dr['p10']].copy()
    dr_ex['excess'] = dr_ex['p10'] - dr_ex['value']
    drought_excess = (dr_ex.groupby(['region_name', 'year', 'window'])['excess']
                           .mean().reset_index()
                           .rename(columns={'excess': 'drought_excess'}))
    drought_excess = (drought_excess.groupby(['region_name', 'year'])['drought_excess']
                                    .mean().reset_index())

    rr = pd.read_parquet(ERA5_RR_RAW)
    rr['date'] = pd.to_datetime(rr['date'])
    rr['year'] = rr['date'].dt.year
    pct_rr = (rr.groupby(['pixel_id', 'window'])['value']
                .quantile(0.90).reset_index()
                .rename(columns={'value': 'p90'}))
    rr = rr.merge(pct_rr, on=['pixel_id', 'window'], how='left')
    rr_ex = rr[rr['value'] > rr['p90']].copy()
    rr_ex['excess'] = rr_ex['value'] - rr_ex['p90']
    rainfall_excess = (rr_ex.groupby(['region_name', 'year', 'window'])['excess']
                            .mean().reset_index()
                            .rename(columns={'excess': 'rainfall_excess'}))
    rainfall_excess = (rainfall_excess.groupby(['region_name', 'year'])['rainfall_excess']
                                      .mean().reset_index())

    def get_season(month):
        return {12: 'DJF', 1: 'DJF', 2: 'DJF',
                3: 'MAM', 4: 'MAM', 5: 'MAM',
                6: 'JJA', 7: 'JJA', 8: 'JJA',
                9: 'SON', 10: 'SON', 11: 'SON'}[month]

    gdf_regions = gpd.read_file(SHAPEFILE)[['region_nam', 'geometry']].rename(
                    columns={'region_nam': 'region_name'})

    tmax_files = sorted(ERA5_TMAX_DIR.glob('era5_tmax_daily_*.nc'))
    ds_list = []
    for f in tmax_files:
        ds_y = xr.open_dataset(f)
        if 'valid_time' in ds_y.coords and 'time' not in ds_y.coords:
            ds_y = ds_y.rename({'valid_time': 'time'})
        ds_list.append(ds_y)
    ds = xr.concat(ds_list, dim='time')
    tmax_var = list(ds.data_vars)[0]
    ds[tmax_var] = ds[tmax_var] - 273.15

    p90 = ds[tmax_var].quantile(0.90, dim='time')
    excess_da = (ds[tmax_var] - p90).where(ds[tmax_var] > p90, other=0.0)

    df_hw = excess_da.to_dataframe(name='excess').reset_index()
    df_hw = df_hw[df_hw['excess'] > 0].copy()
    df_hw['date']   = pd.to_datetime(df_hw['time'])
    df_hw['year']   = df_hw['date'].dt.year
    df_hw['window'] = df_hw['date'].dt.month.map(get_season)

    gdf_hw = gpd.GeoDataFrame(
        df_hw,
        geometry=gpd.points_from_xy(df_hw['longitude'], df_hw['latitude']),
        crs='EPSG:4326'
    )
    gdf_hw = gpd.sjoin(gdf_hw, gdf_regions, how='inner', predicate='within')
    heatwave_excess = (gdf_hw.groupby(['region_name', 'year', 'window'])['excess']
                             .mean().reset_index()
                             .rename(columns={'excess': 'heatwave_excess'}))
    heatwave_excess = (heatwave_excess.groupby(['region_name', 'year'])['heatwave_excess']
                                      .mean().reset_index())

    excess = drought_excess.merge(rainfall_excess, on=['region_name', 'year'], how='outer')
    excess = excess.merge(heatwave_excess,          on=['region_name', 'year'], how='outer')
    excess = excess.fillna(0.0)
    excess.to_parquet(EXCESS_CACHE, index=False)
    print(f"    Cached → {EXCESS_CACHE}")

for col in ('drought_excess', 'rainfall_excess', 'heatwave_excess'):
    mn, mx = excess[col].min(), excess[col].max()
    excess[col] = (excess[col] - mn) / (mx - mn) if mx > mn else 0.0

panel = panel.merge(excess, on=['region_name', 'year'], how='left')
for col in ('drought_excess', 'rainfall_excess', 'heatwave_excess'):
    panel[col] = panel[col].fillna(0.0)

if ANOMALY_CACHE.exists():
    print("    Loading cached hazard anomaly variables …")
    anomaly = pd.read_parquet(ANOMALY_CACHE)
else:
    import xarray as xr
    print("    Computing hazard anomaly variables (first run — caching) …")

    dr = pd.read_parquet(ERA5_DR_RAW)
    dr['date'] = pd.to_datetime(dr['date'])
    dr['year'] = dr['date'].dt.year
    dr_mean = (dr.groupby(['pixel_id', 'window'])['value']
                 .mean().reset_index().rename(columns={'value': 'hist_mean'}))
    dr = dr.merge(dr_mean, on=['pixel_id', 'window'], how='left')
    dr['anom'] = dr['value'] - dr['hist_mean']
    drought_anom = (dr.groupby(['region_name', 'year', 'window'])['anom']
                     .mean().reset_index())
    drought_anom = (drought_anom.groupby(['region_name', 'year'])['anom']
                                .mean().reset_index()
                                .rename(columns={'anom': 'drought_anom'}))

    rr = pd.read_parquet(ERA5_RR_RAW)
    rr['date'] = pd.to_datetime(rr['date'])
    rr['year'] = rr['date'].dt.year
    rr_mean = (rr.groupby(['pixel_id', 'window'])['value']
                 .mean().reset_index().rename(columns={'value': 'hist_mean'}))
    rr = rr.merge(rr_mean, on=['pixel_id', 'window'], how='left')
    rr['anom'] = rr['value'] - rr['hist_mean']
    rainfall_anom = (rr.groupby(['region_name', 'year', 'window'])['anom']
                      .mean().reset_index())
    rainfall_anom = (rainfall_anom.groupby(['region_name', 'year'])['anom']
                                  .mean().reset_index()
                                  .rename(columns={'anom': 'rainfall_anom'}))

    def get_season(month):
        return {12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',
                6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'}[month]

    gdf_regions = gpd.read_file(SHAPEFILE)[['region_nam','geometry']].rename(
                    columns={'region_nam': 'region_name'})
    tmax_files = sorted(ERA5_TMAX_DIR.glob('era5_tmax_daily_*.nc'))
    ds = xr.concat([xr.open_dataset(f) for f in tmax_files], dim='time')
    if 'valid_time' in ds.coords and 'time' not in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    tmax_var = list(ds.data_vars)[0]
    ds[tmax_var] = ds[tmax_var] - 273.15

    df_t = ds[tmax_var].to_dataframe(name='tmax').reset_index()
    df_t['date']   = pd.to_datetime(df_t['time'])
    df_t['year']   = df_t['date'].dt.year
    df_t['window'] = df_t['date'].dt.month.map(get_season)
    df_t['pixel_id'] = (df_t['latitude'].round(4).astype(str) + '_' +
                        df_t['longitude'].round(4).astype(str))

    hist_mean_hw = (df_t.groupby(['pixel_id', 'window'])['tmax']
                       .mean().reset_index().rename(columns={'tmax': 'hist_mean'}))
    df_t = df_t.merge(hist_mean_hw, on=['pixel_id', 'window'], how='left')
    df_t['anom'] = df_t['tmax'] - df_t['hist_mean']

    gdf_t = gpd.GeoDataFrame(df_t,
                geometry=gpd.points_from_xy(df_t['longitude'], df_t['latitude']),
                crs='EPSG:4326')
    gdf_t = gpd.sjoin(gdf_t, gdf_regions, how='inner', predicate='within')
    heatwave_anom = (gdf_t.groupby(['region_name', 'year', 'window'])['anom']
                          .mean().reset_index())
    heatwave_anom = (heatwave_anom.groupby(['region_name', 'year'])['anom']
                                  .mean().reset_index()
                                  .rename(columns={'anom': 'heatwave_anom'}))

    anomaly = drought_anom.merge(rainfall_anom, on=['region_name','year'], how='outer')
    anomaly = anomaly.merge(heatwave_anom,      on=['region_name','year'], how='outer')
    anomaly = anomaly.fillna(0.0)
    anomaly.to_parquet(ANOMALY_CACHE, index=False)
    print(f"    Cached → {ANOMALY_CACHE}")

for col in ('drought_anom', 'rainfall_anom', 'heatwave_anom'):
    mn, mx = anomaly[col].min(), anomaly[col].max()
    anomaly[col] = (anomaly[col] - mn) / (mx - mn) if mx > mn else 0.0

panel = panel.merge(anomaly, on=['region_name', 'year'], how='left')
for col in ('drought_anom', 'rainfall_anom', 'heatwave_anom'):
    panel[col] = panel[col].fillna(0.0)

for h in ('drought', 'rainfall', 'heatwave', 'cyclone'):
    panel[h] = panel[f'{h}_raw']

panel['hazard_avg'] = panel[['drought', 'rainfall', 'heatwave', 'cyclone']].mean(axis=1)

panel['log_loss']   = np.log1p(panel['total_losses_musd'])
panel['log_pop']    = np.log1p(panel['pop'])
panel['log_gdp']    = np.log(panel['gdp_pc'].clip(lower=1))

panel['built_s_x_haz'] = panel['built_s'] * panel['hazard_avg']
panel['built_v_x_haz'] = panel['built_v'] * panel['hazard_avg']

ALL_VARS = ['log_loss', 'built_s', 'built_v', 'hazard_avg',
            'drought', 'rainfall', 'heatwave', 'cyclone', 'wind_kt',
            'drought_excess', 'rainfall_excess', 'heatwave_excess',
            'drought_anom', 'rainfall_anom', 'heatwave_anom',
            'built_s_x_haz', 'built_v_x_haz', 'log_pop', 'log_gdp']
panel = (panel[panel['total_losses_musd'] > 0]
         .dropna(subset=ALL_VARS)
         .query('year <= 2025')
         .copy())

print(f"    Panel: {len(panel):,} obs · "
      f"{panel['region_name'].nunique()} regions · "
      f"{panel['country'].nunique()} countries · "
      f"{panel['year'].nunique()} years")

panel = panel.set_index(['region_name', 'year'])

print("\n[2] Estimating models …")

countries = sorted(panel['country'].unique())
country_dummies = pd.get_dummies(panel['country'], drop_first=True, prefix='c').astype(float)
panel = pd.concat([panel, country_dummies], axis=1)
dummy_cols = country_dummies.columns.tolist()

def fit(y_col, x_cols):
    from linearmodels import PooledOLS as POLS
    data = panel[[y_col] + x_cols + dummy_cols + ['country']].dropna().copy()
    formula_X = data[x_cols + dummy_cols]
    formula_X = formula_X.assign(const=1.0)
    mod = POLS(data[y_col], formula_X)
    res = mod.fit(cov_type='clustered', cluster_entity=False,
                  clusters=data['country'].astype('category').cat.codes)
    return res, data

SPECS = {
    'M1': ['built_s', 'built_v'],
    'M2': ['hazard_avg'],
    'M3': ['built_s', 'built_v', 'hazard_avg'],
    'M4': ['built_s', 'built_v', 'hazard_avg', 'built_s_x_haz', 'built_v_x_haz'],
    'M5': ['built_s', 'built_v', 'hazard_avg', 'built_s_x_haz', 'built_v_x_haz',
           'log_pop', 'log_gdp'],
    'M6': ['built_s', 'built_v', 'drought', 'rainfall', 'heatwave', 'wind_kt',
           'log_pop', 'log_gdp'],
    'M6b': ['built_s', 'built_v',
            'drought_excess', 'rainfall_excess', 'heatwave_excess', 'wind_kt',
            'log_pop', 'log_gdp'],
    'M6c': ['built_s', 'built_v',
            'drought_anom', 'rainfall_anom', 'heatwave_anom', 'wind_kt',
            'log_pop', 'log_gdp'],
}

LABELS = {
    'built_s':        'Built surface (built\\_s)',
    'built_v':        'Built volume (built\\_v)',
    'hazard_avg':     'Hazard avg',
    'built_s_x_haz':  'built\\_s × Hazard',
    'built_v_x_haz':  'built\\_v × Hazard',
    'log_pop':        'log(Population)',
    'log_gdp':        'log(GDP per capita)',
    'drought':        'Drought',
    'rainfall':       'Rainfall',
    'heatwave':       'Heatwave',
    'cyclone':        'Cyclone (PCA)',
    'wind_kt':        'Wind speed (kt, IBTrACS)',
    'drought_excess':   'Drought excess (m³/m³)',
    'rainfall_excess':  'Rainfall excess (m/day)',
    'heatwave_excess':  'Heatwave excess (°C)',
    'drought_anom':     'Drought anomaly (m³/m³)',
    'rainfall_anom':    'Rainfall anomaly (m/day)',
    'heatwave_anom':    'Heatwave anomaly (°C)',
}

results = {}
for tag, xcols in SPECS.items():
    res, data_used = fit('log_loss', xcols)
    results[tag] = {'res': res, 'xcols': xcols, 'data': data_used}
    print(f"    [{tag}]  R²={res.rsquared:.3f}  N={int(res.nobs):,}")

print("\n[3] Building tables …")

ALL_COEFS = ['built_s', 'built_v', 'hazard_avg',
             'built_s_x_haz', 'built_v_x_haz',
             'drought', 'rainfall', 'heatwave', 'wind_kt',
             'drought_excess', 'rainfall_excess', 'heatwave_excess',
             'drought_anom', 'rainfall_anom', 'heatwave_anom',
             'log_pop', 'log_gdp']

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

def extract(res, coef):
    try:
        est = res.params[coef]
        se  = res.std_errors[coef]
        p   = res.pvalues[coef]
        return est, se, p
    except KeyError:
        return None, None, None

csv_rows = []
for coef in ALL_COEFS:
    row = {'variable': coef}
    for tag in SPECS:
        est, se, p = extract(results[tag]['res'], coef)
        if est is None:
            row[f'{tag}_coef'] = ''
            row[f'{tag}_se']   = ''
            row[f'{tag}_p']    = ''
        else:
            row[f'{tag}_coef'] = round(est, 4)
            row[f'{tag}_se']   = round(se, 4)
            row[f'{tag}_p']    = round(p, 4)
    csv_rows.append(row)

r2_row = {'variable': 'R²'}
n_row  = {'variable': 'N'}
for tag in SPECS:
    r = results[tag]['res']
    r2_row[f'{tag}_coef'] = round(r.rsquared, 3)
    n_row[f'{tag}_coef']  = int(r.nobs)
    for k in ('_se', '_p'):
        r2_row[f'{tag}{k}'] = ''
        n_row[f'{tag}{k}']  = ''
csv_rows += [r2_row, n_row]

df_csv = pd.DataFrame(csv_rows)
df_csv.to_csv(OUT_DIR / 'coefficients_M1_M6.csv', index=False)

def latex_cell(res, coef):
    est, se, p = extract(res, coef)
    if est is None:
        return r'\multicolumn{1}{c}{---}'
    s = stars(p)
    return f'${est:.4f}^{{{s}}}$ \\\\ (${se:.4f}$)'

model_tags = list(SPECS.keys())
col_spec   = 'l' + 'c' * len(model_tags)
header     = ' & '.join([''] + model_tags) + r' \\'
sep        = r'\midrule'

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Panel vulnerability regression — M1–M6}',
    r'\label{tab:regression_panel}',
    r'\footnotesize',
    r'\begin{tabular}{' + col_spec + '}',
    r'\toprule',
    header,
    r'\midrule',
]
for coef in ALL_COEFS:
    label = LABELS.get(coef, coef)
    cells = []
    for tag in model_tags:
        est, se, p = extract(results[tag]['res'], coef)
        if est is None:
            cells.append('---')
        else:
            cells.append(f'${est:.4f}^{{{stars(p)}}}$')
    lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    se_cells = []
    for tag in model_tags:
        est, se, p = extract(results[tag]['res'], coef)
        se_cells.append(f'$({se:.4f})$' if se is not None else '')
    lines.append(' & ' + ' & '.join(se_cells) + r' \\[2pt]')

lines += [
    r'\midrule',
    r'$R^2$ & ' + ' & '.join(f'{results[t]["res"].rsquared:.3f}' for t in model_tags) + r' \\',
    r'$N$ & '   + ' & '.join(str(int(results[t]["res"].nobs)) for t in model_tags) + r' \\',
    r'\midrule',
    r'\multicolumn{' + str(len(model_tags)+1) + r'}{l}{\footnotesize $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$. Clustered SEs at country level.} \\',
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{table}',
]
with open(OUT_DIR / 'results_table.tex', 'w') as f:
    f.write('\n'.join(lines))

print(f"    Saved: coefficients_M1_M6.csv")
print(f"    Saved: results_table.tex")

print("\n[4] Forest plots …")

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})

def forest_plot(tag, out_path):
    res   = results[tag]['res']
    xcols = SPECS[tag]
    coef_names = [c for c in xcols]

    ests  = [res.params[c]      for c in coef_names]
    ci_lo = [res.conf_int().loc[c, 'lower'] for c in coef_names]
    ci_hi = [res.conf_int().loc[c, 'upper'] for c in coef_names]
    pvals = [res.pvalues[c]     for c in coef_names]
    labels = [LABELS.get(c, c)  for c in coef_names]

    colors = ['#2166AC' if p < 0.05 else '#AAAAAA' for p in pvals]
    y_pos  = range(len(coef_names))

    fig, ax = plt.subplots(figsize=(8, max(4, len(coef_names) * 0.7 + 1)))
    for i, (est, lo, hi, col) in enumerate(zip(ests, ci_lo, ci_hi, colors)):
        ax.plot([lo, hi], [i, i], color=col, lw=1.5, solid_capstyle='round')
        ax.scatter(est, i, color=col, s=50, zorder=5)

    ax.axvline(0, color='black', lw=0.8, linestyle='--')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Coefficient estimate  (dep: log1p loss)', fontsize=10)
    ax.set_title(f'{tag} — coefficient plot  (95% CI, country-clustered SE)\n'
                 f'R²={res.rsquared:.3f}  N={int(res.nobs):,}', fontsize=11)

    sig_patch  = mpatches.Patch(color='#2166AC', label='p < 0.05')
    ns_patch   = mpatches.Patch(color='#AAAAAA', label='n.s.')
    ax.legend(handles=[sig_patch, ns_patch], fontsize=9, loc='lower right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path.name}")

forest_plot('M5', OUT_DIR / 'forest_plot_M5.png')
forest_plot('M6', OUT_DIR / 'forest_plot_M6.png')

print("\n[5] Stability plot …")

STABILITY_COEFS  = ['built_s', 'built_v', 'hazard_avg']
STABILITY_MODELS = ['M3', 'M4', 'M5']
MODEL_COLORS     = {'M3': '#4DAF4A', 'M4': '#984EA3', 'M5': '#377EB8'}

fig, axes = plt.subplots(1, len(STABILITY_COEFS),
                         figsize=(4 * len(STABILITY_COEFS), 4.5),
                         sharey=False)

for ax, coef in zip(axes, STABILITY_COEFS):
    for i, tag in enumerate(STABILITY_MODELS):
        est, se, p = extract(results[tag]['res'], coef)
        if est is None:
            continue
        ci = results[tag]['res'].conf_int()
        lo = ci.loc[coef, 'lower']
        hi = ci.loc[coef, 'upper']
        col = MODEL_COLORS[tag]
        ax.errorbar(i, est, yerr=[[est - lo], [hi - est]],
                    fmt='o', color=col, capsize=5, ms=7, lw=1.5,
                    label=tag)
        ax.text(i, hi + (hi - lo) * 0.05,
                stars(p), ha='center', fontsize=9, color=col)

    ax.axhline(0, color='black', lw=0.7, linestyle='--')
    ax.set_xticks(range(len(STABILITY_MODELS)))
    ax.set_xticklabels(STABILITY_MODELS)
    ax.set_title(LABELS.get(coef, coef), fontsize=10)
    ax.set_ylabel('Estimate' if coef == STABILITY_COEFS[0] else '')

plt.suptitle('Stability test — coefficients across M3 / M4 / M5\n'
             '95% CI, country-clustered SE', fontsize=11)
plt.tight_layout()
fig.savefig(OUT_DIR / 'stability_plot.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    Saved: stability_plot.png")

print(f"\nDone (unstandardized).  All outputs → {OUT_DIR}/")

print("\n[6] Standardized regression (÷ SD, no centering) …")

STD_VARS = [
    'built_s', 'built_v',
    'hazard_avg',
    'drought', 'rainfall', 'heatwave', 'wind_kt',
    'drought_excess', 'rainfall_excess', 'heatwave_excess',
    'drought_anom',   'rainfall_anom',   'heatwave_anom',
    'log_pop', 'log_gdp',
]

panel_raw = panel.reset_index()
sd_series = panel_raw[STD_VARS].std(ddof=1)

panel_std = panel_raw.copy()
for v in STD_VARS:
    panel_std[v] = panel_raw[v] / sd_series[v]

panel_std['built_s_x_haz'] = panel_std['built_s'] * panel_std['hazard_avg']
panel_std['built_v_x_haz'] = panel_std['built_v'] * panel_std['hazard_avg']

panel_std = panel_std.set_index(['region_name', 'year'])

def fit_std(y_col, x_cols):
    from linearmodels import PooledOLS as POLS
    data = panel_std[[y_col] + x_cols + dummy_cols + ['country']].dropna().copy()
    X = data[x_cols + dummy_cols].assign(const=1.0)
    mod = POLS(data[y_col], X)
    res = mod.fit(cov_type='clustered', cluster_entity=False,
                  clusters=data['country'].astype('category').cat.codes)
    return res, data

results_std = {}
for tag, xcols in SPECS.items():
    res, data_used = fit_std('log_loss', xcols)
    results_std[tag] = {'res': res, 'xcols': xcols, 'data': data_used}
    print(f"    [{tag}]  R²={res.rsquared:.3f}  N={int(res.nobs):,}")

csv_rows_std = []
for coef in ALL_COEFS:
    row = {'variable': coef, 'sd_original': round(sd_series.get(coef, float('nan')), 6)}
    for tag in SPECS:
        est, se, p = extract(results_std[tag]['res'], coef)
        if est is None:
            row[f'{tag}_coef'] = ''
            row[f'{tag}_se']   = ''
            row[f'{tag}_p']    = ''
        else:
            row[f'{tag}_coef'] = round(est, 4)
            row[f'{tag}_se']   = round(se, 4)
            row[f'{tag}_p']    = round(p, 4)
    csv_rows_std.append(row)

r2_row_s = {'variable': 'R²', 'sd_original': ''}
n_row_s  = {'variable': 'N',  'sd_original': ''}
for tag in SPECS:
    r = results_std[tag]['res']
    r2_row_s[f'{tag}_coef'] = round(r.rsquared, 3)
    n_row_s[f'{tag}_coef']  = int(r.nobs)
    for k in ('_se', '_p'):
        r2_row_s[f'{tag}{k}'] = ''
        n_row_s[f'{tag}{k}']  = ''
csv_rows_std += [r2_row_s, n_row_s]

df_std = pd.DataFrame(csv_rows_std)
df_std.to_csv(OUT_DIR / 'coefficients_standardized.csv', index=False)
print(f"    Saved: coefficients_standardized.csv")

model_tags = list(SPECS.keys())
col_spec   = 'l' + 'c' * len(model_tags)
header_s   = ' & '.join([''] + model_tags) + r' \\'

lines_s = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Panel vulnerability regression — standardized coefficients (÷ SD, no centering)}',
    r'\label{tab:regression_standardized}',
    r'\footnotesize',
    r'\begin{tabular}{' + col_spec + '}',
    r'\toprule',
    header_s,
    r'\midrule',
]
for coef in ALL_COEFS:
    label = LABELS.get(coef, coef)
    cells, se_cells = [], []
    for tag in model_tags:
        est, se, p = extract(results_std[tag]['res'], coef)
        if est is None:
            cells.append('---')
            se_cells.append('')
        else:
            cells.append(f'${est:.4f}^{{{stars(p)}}}$')
            se_cells.append(f'$({se:.4f})$')
    lines_s.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines_s.append(' & ' + ' & '.join(se_cells) + r' \\[2pt]')

lines_s += [
    r'\midrule',
    r'$R^2$ & ' + ' & '.join(f'{results_std[t]["res"].rsquared:.3f}' for t in model_tags) + r' \\',
    r'$N$ & '   + ' & '.join(str(int(results_std[t]["res"].nobs)) for t in model_tags) + r' \\',
    r'\midrule',
    r'\multicolumn{' + str(len(model_tags)+1) + r'}{l}{\footnotesize $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$. Standardized coefficients (÷ SD). Clustered SEs at country level.} \\',
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{table}',
]
with open(OUT_DIR / 'results_table_standardized.tex', 'w') as f:
    f.write('\n'.join(lines_s))
print(f"    Saved: results_table_standardized.tex")

def forest_plot_std(tag, out_path):
    res    = results_std[tag]['res']
    xcols  = SPECS[tag]
    ests   = [res.params[c]                    for c in xcols]
    ci_lo  = [res.conf_int().loc[c, 'lower']   for c in xcols]
    ci_hi  = [res.conf_int().loc[c, 'upper']   for c in xcols]
    pvals  = [res.pvalues[c]                   for c in xcols]
    labels = [LABELS.get(c, c)                 for c in xcols]
    colors = ['#2166AC' if p < 0.05 else '#AAAAAA' for p in pvals]

    fig, ax = plt.subplots(figsize=(8, max(4, len(xcols) * 0.7 + 1)))
    for i, (est, lo, hi, col) in enumerate(zip(ests, ci_lo, ci_hi, colors)):
        ax.plot([lo, hi], [i, i], color=col, lw=1.5, solid_capstyle='round')
        ax.scatter(est, i, color=col, s=50, zorder=5)
    ax.axvline(0, color='black', lw=0.8, linestyle='--')
    ax.set_yticks(list(range(len(xcols))))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Standardized coefficient  (dep: log1p loss)', fontsize=10)
    ax.set_title(f'{tag} — standardized coefficients  (95% CI, country-clustered SE)\n'
                 f'R²={res.rsquared:.3f}  N={int(res.nobs):,}', fontsize=11)
    sig_patch = mpatches.Patch(color='#2166AC', label='p < 0.05')
    ns_patch  = mpatches.Patch(color='#AAAAAA', label='n.s.')
    ax.legend(handles=[sig_patch, ns_patch], fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path.name}")

forest_plot_std('M5',  OUT_DIR / 'forest_plot_M5_std.png')
forest_plot_std('M6',  OUT_DIR / 'forest_plot_M6_std.png')
forest_plot_std('M6b', OUT_DIR / 'forest_plot_M6b_std.png')
forest_plot_std('M6c', OUT_DIR / 'forest_plot_M6c_std.png')

fig, axes = plt.subplots(1, len(STABILITY_COEFS),
                         figsize=(4 * len(STABILITY_COEFS), 4.5), sharey=False)
for ax, coef in zip(axes, STABILITY_COEFS):
    for i, tag in enumerate(STABILITY_MODELS):
        est, se, p = extract(results_std[tag]['res'], coef)
        if est is None:
            continue
        ci = results_std[tag]['res'].conf_int()
        lo = ci.loc[coef, 'lower']
        hi = ci.loc[coef, 'upper']
        col = MODEL_COLORS[tag]
        ax.errorbar(i, est, yerr=[[est - lo], [hi - est]],
                    fmt='o', color=col, capsize=5, ms=7, lw=1.5, label=tag)
        ax.text(i, hi + (hi - lo) * 0.05,
                stars(p), ha='center', fontsize=9, color=col)
    ax.axhline(0, color='black', lw=0.7, linestyle='--')
    ax.set_xticks(range(len(STABILITY_MODELS)))
    ax.set_xticklabels(STABILITY_MODELS)
    ax.set_title(LABELS.get(coef, coef), fontsize=10)
    ax.set_ylabel('Standardized estimate' if coef == STABILITY_COEFS[0] else '')

plt.suptitle('Stability test — standardized coefficients across M3 / M4 / M5\n'
             '95% CI, country-clustered SE', fontsize=11)
plt.tight_layout()
fig.savefig(OUT_DIR / 'stability_plot_std.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"    Saved: stability_plot_std.png")

print(f"\nDone.  All outputs → {OUT_DIR}/")

print("\n[8] Publication tables …")

FINAL_MODELS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']

FINAL_GROUPS = [
    ('Exposure',               ['built_s', 'built_v']),
    ('Hazard (aggregate)',     ['hazard_avg']),
    ('Interaction',            ['built_s_x_haz', 'built_v_x_haz']),
    ('Hazard (disaggregated)', ['drought', 'rainfall', 'heatwave', 'wind_kt']),
    ('Controls',               ['log_pop', 'log_gdp']),
]

FINAL_LABELS = {
    'built_s':       r'Built surface',
    'built_v':       r'Built volume',
    'hazard_avg':    r'Hazard index (avg)',
    'built_s_x_haz': r'Built surface $\times$ Hazard',
    'built_v_x_haz': r'Built volume $\times$ Hazard',
    'drought':       r'Drought',
    'rainfall':      r'Rainfall',
    'heatwave':      r'Heatwave',
    'wind_kt':       r'Wind speed (kt)',
    'log_pop':       r'$\log$(Population)',
    'log_gdp':       r'$\log$(GDP per capita)',
}

n_fm = len(FINAL_MODELS)
col_spec_f = 'l' + 'c' * n_fm

tex_f = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Panel regression — standardized coefficients ($\div$ SD, no centering). '
    r'Dependent variable: $\log(1 + \text{Loss})$. Country fixed effects via explicit dummies. '
    r'Standard errors clustered at country level.}',
    r'\label{tab:regression_final}',
    r'\small',
    r'\begin{tabular}{' + col_spec_f + '}',
    r'\toprule',
    r' & ' + ' & '.join(f'\\textbf{{{t}}}' for t in FINAL_MODELS) + r' \\',
    r'\midrule',
]

for group_label, coefs in FINAL_GROUPS:
    tex_f.append(r'\multicolumn{' + str(n_fm + 1) + r'}{l}{'
                 r'\textit{' + group_label + r'}} \\[-2pt]')
    for coef in coefs:
        label = FINAL_LABELS.get(coef, coef)
        cells, se_cells = [], []
        for tag in FINAL_MODELS:
            est, se, p = extract(results_std[tag]['res'], coef)
            if est is None:
                cells.append(r'\multicolumn{1}{c}{---}')
                se_cells.append('')
            else:
                s = stars(p)
                cells.append(f'${est:.3f}^{{{s}}}$')
                se_cells.append(f'$({se:.3f})$')
        tex_f.append(r'\quad ' + label + ' & ' + ' & '.join(cells) + r' \\')
        tex_f.append(r' & ' + ' & '.join(se_cells) + r' \\[2pt]')
    tex_f.append(r'\addlinespace[3pt]')

tex_f += [
    r'\midrule',
    r'Country FE & '   + ' & '.join([r'\checkmark'] * n_fm) + r' \\',
    r'$R^2$ & '        + ' & '.join(f'{results_std[t]["res"].rsquared:.3f}'
                                     for t in FINAL_MODELS) + r' \\',
    r'$N$ & '          + ' & '.join(str(int(results_std[t]["res"].nobs))
                                     for t in FINAL_MODELS) + r' \\',
    r'\midrule',
    r'\multicolumn{' + str(n_fm + 1) + r'}{l}{\footnotesize '
    r'$^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$. '
    r'Cyclone exposure: annual max \textit{avg\_max\_usa\_wind} (kt, IBTrACS). '
    r'All continuous regressors divided by their sample SD.} \\',
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{table}',
]

with open(OUT_DIR / 'coefficients_final.tex', 'w') as f:
    f.write('\n'.join(tex_f))
print(f"    Saved: coefficients_final.tex")

res_m6s   = results_std['M6']['res']
countries_sorted = sorted(panel_raw['country'].unique())
ref_country = countries_sorted[0]

fe_rows = []
for country in countries_sorted:
    if country == ref_country:
        alpha_i = 0.0
        alpha_se = np.nan
        alpha_p  = np.nan
    else:
        dcol     = f'c_{country}'
        alpha_i  = float(res_m6s.params.get(dcol, np.nan))
        alpha_se = float(res_m6s.std_errors.get(dcol, np.nan))
        alpha_p  = float(res_m6s.pvalues.get(dcol, np.nan))

    mask = panel_raw['country'] == country
    fe_rows.append({
        'country':       country,
        'alpha_i':       round(alpha_i, 4),
        'alpha_se':      round(alpha_se, 4) if not np.isnan(alpha_se) else '',
        'alpha_p':       round(alpha_p,  4) if not np.isnan(alpha_p)  else '',
        'gdp_pc_usd':    round(panel_raw.loc[mask, 'gdp_pc'].mean(), 0),
        'mean_loss_musd': round(panel_raw.loc[mask, 'total_losses_musd'].mean(), 4),
        'n_obs':         int(mask.sum()),
    })

df_fe = (pd.DataFrame(fe_rows)
           .sort_values('alpha_i', ascending=False)
           .reset_index(drop=True))

df_fe.to_csv(OUT_DIR / 'fixed_effects.csv', index=False)
print(f"    Saved: fixed_effects.csv")

tex_fe = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Country fixed effects from M6 (standardized). '
    r'$\hat{\alpha}_i$ is the coefficient on the country dummy relative to '
    + ref_country.replace('&', r'\&') + r' (reference category). '
    r'GDP per capita: mean 1995--2025 (constant 2015 USD, World Bank). '
    r'Mean loss: mean of total losses in MUSD across region-years with loss $>0$.}',
    r'\label{tab:fixed_effects}',
    r'\small',
    r'\begin{tabular}{lrrrr}',
    r'\toprule',
    r'Country & $\hat{\alpha}_i$ & GDP p.c.\ (USD) & Mean loss (MUSD) & $N$ \\',
    r'\midrule',
]

for _, row in df_fe.iterrows():
    s = stars(row['alpha_p']) if row['alpha_p'] != '' else ''
    alpha_str = f"${row['alpha_i']:.3f}^{{{s}}}$"
    gdp_str   = f"${row['gdp_pc_usd']:,.0f}$"
    loss_str  = f"${row['mean_loss_musd']:.3f}$"
    tex_fe.append(
        f"{row['country'].replace('&', r'&')} & {alpha_str} & "
        f"{gdp_str} & {loss_str} & {row['n_obs']:,}" + r" \\"
    )

tex_fe += [
    r'\midrule',
    r'\multicolumn{5}{l}{\footnotesize '
    r'$^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$. '
    r'SEs clustered at country level.} \\',
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{table}',
]

with open(OUT_DIR / 'fixed_effects.tex', 'w') as f:
    f.write('\n'.join(tex_fe))
print(f"    Saved: fixed_effects.tex")

print()
print(df_fe[['country', 'alpha_i', 'gdp_pc_usd', 'mean_loss_musd', 'n_obs']].to_string(index=False))

print("\n[7] Descriptive statistics …")

DESC_VARS = ['log_loss', 'built_s', 'built_v', 'hazard_avg',
             'drought', 'rainfall', 'heatwave', 'wind_kt',
             'log_pop', 'log_gdp']

DESC_LABELS = {
    'log_loss':   r'log(1 + Loss)',
    'built_s':    r'Built surface (km$^2$)',
    'built_v':    r'Built volume (km$^3$)',
    'hazard_avg': r'Hazard index (avg)',
    'drought':    r'Drought (norm.)',
    'rainfall':   r'Rainfall (norm.)',
    'heatwave':   r'Heatwave (norm.)',
    'wind_kt':    r'Wind speed (kt)',
    'log_pop':    r'log(1 + Population)',
    'log_gdp':    r'log(GDP per capita)',
}

stats_rows = []
for v in DESC_VARS:
    s = panel_raw[v].dropna()
    stats_rows.append({
        'variable': v,
        'label':    DESC_LABELS[v],
        'n':        int(s.count()),
        'mean':     s.mean(),
        'sd':       s.std(ddof=1),
        'min':      s.min(),
        'p25':      s.quantile(0.25),
        'median':   s.median(),
        'p75':      s.quantile(0.75),
        'max':      s.max(),
    })

df_desc = pd.DataFrame(stats_rows)

df_desc.drop(columns='label').round(4).to_csv(
    OUT_DIR / 'descriptive_stats.csv', index=False)
print(f"    Saved: descriptive_stats.csv")

N_OBS  = len(panel_raw)
N_REG  = panel_raw['region_name'].nunique()
N_CTRY = panel_raw['country'].nunique()

def fmt(x, decimals=3):
    if abs(x) >= 1000:
        return f'{x:,.{decimals}f}'
    return f'{x:.{decimals}f}'

caption = (
    f'Descriptive statistics for all regression variables. '
    f'$N = {N_OBS:,}$ region-year observations across '
    f'{N_REG} sub-national regions and {N_CTRY} countries.'
)

tex_lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{' + caption + '}',
    r'\label{tab:desc_stats}',
    r'\small',
    r'\begin{tabular}{lrrrrrrrr}',
    r'\toprule',
    r'Variable & $N$ & Mean & SD & Min & P25 & Median & P75 & Max \\',
    r'\midrule',
]

for _, row in df_desc.iterrows():
    d = 3
    cells = [
        row['label'],
        f"{row['n']:,}",
        fmt(row['mean'], d),
        fmt(row['sd'],   d),
        fmt(row['min'],  d),
        fmt(row['p25'],  d),
        fmt(row['median'], d),
        fmt(row['p75'],  d),
        fmt(row['max'],  d),
    ]
    tex_lines.append(' & '.join(cells) + r' \\')

tex_lines += [
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{table}',
]

with open(OUT_DIR / 'descriptive_stats.tex', 'w') as f:
    f.write('\n'.join(tex_lines))
print(f"    Saved: descriptive_stats.tex")

print()
print(df_desc[['variable','n','mean','sd','min','median','max']]
      .to_string(index=False, float_format=lambda x: f'{x:.4f}'))
