import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.append('/home/jupyter-daniela/CAF/src')

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import statsmodels.api as sm
from pathlib import Path
from linearmodels import PanelOLS

BASE_DIR    = Path('/home/jupyter-daniela/CAF')
DATA_PATH   = BASE_DIR / 'data/processed/caribbean_adm_boundaries/master_panel.parquet'
SHAPEFILE   = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
IBTRACS_AGG = BASE_DIR / 'data/processed/caribbean_adm_boundaries/ibtracs_output/ibtracs_aggregated_data.parquet'
OUT_DIR     = BASE_DIR / 'outputs/regression'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CARIB_8 = [
    'Bahamas', 'Barbados', 'Dominica', 'Dominican Republic',
    'Haiti', 'Jamaica', 'Saint Lucia', 'Trinidad and Tobago',
]
WB_ISO = {
    'Bahamas': 'BS', 'Barbados': 'BB', 'Dominica': 'DM',
    'Dominican Republic': 'DO', 'Haiti': 'HT', 'Jamaica': 'JM',
    'Saint Lucia': 'LC', 'Trinidad and Tobago': 'TT',
}
M6_PRED  = ['built_s', 'built_v', 'drought', 'rainfall', 'heatwave', 'wind_kt', 'log_pop', 'log_gdp']
BUILT    = ['built_s', 'built_v']
HAZARD   = ['drought', 'rainfall', 'heatwave', 'wind_kt']
CONTROLS = ['log_pop', 'log_gdp']
LABELS   = {
    'built_s': 'Built surface', 'built_v': 'Built volume',
    'drought': 'Drought', 'rainfall': 'Rainfall',
    'heatwave': 'Heatwave', 'wind_kt': 'Wind speed (kt)',
    'log_pop': 'log(Population)', 'log_gdp': 'log(GDP per capita)',
}

def stars(p):
    if pd.isna(p): return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

print("=" * 66)
print("07d — Caribbean M6  |  PanelOLS entity FE, country-clustered SE")
print("=" * 66)

# ── [1] Data ──────────────────────────────────────────────────────────────────
print("\n[1] Loading data …")
df = pd.read_parquet(DATA_PATH)
gdf_shp = gpd.read_file(SHAPEFILE)[['region_nam', 'country']].rename(
    columns={'region_nam': 'region_name'})
df = df.merge(gdf_shp, on='region_name', how='left')
df = df[df['country'].isin(CARIB_8)].copy()

print("    Fetching GDP per capita (World Bank) …")
iso_str = ';'.join(WB_ISO.values())
wb_data = None
for _att in range(3):
    try:
        wb_data = requests.get(
            f'https://api.worldbank.org/v2/country/{iso_str}/indicator/'
            f'NY.GDP.PCAP.KD?date=1995:2025&format=json&per_page=500',
            timeout=120
        ).json()[1]
        break
    except Exception as e:
        print(f"      Attempt {_att+1}: {e}")
if wb_data is None:
    raise RuntimeError("World Bank API unavailable")

iso_to_ctry = {v: k for k, v in WB_ISO.items()}
df_gdp = pd.DataFrame([
    {'country': iso_to_ctry[r['country']['id']], 'year': int(r['date']), 'gdp_pc': r['value']}
    for r in wb_data if r['value'] is not None
])

panel = (
    df.groupby(['region_name', 'country', 'year']).agg(
        total_losses_musd=('total_losses_musd', 'sum'),
        built_s=('built_s', 'mean'), built_v=('built_v', 'mean'),
        pop=('pop', 'mean'), drought=('drought', 'mean'),
        rainfall=('rainfall', 'mean'), heatwave=('heatwave', 'mean'),
    ).reset_index()
    .merge(df_gdp, on=['country', 'year'], how='left')
)

ibt = pd.read_parquet(IBTRACS_AGG)
ibt_wind = (
    ibt.groupby(['region_name', 'year_win'])['avg_max_usa_wind']
    .max().reset_index()
    .rename(columns={'year_win': 'year', 'avg_max_usa_wind': 'wind_kt'})
)
panel = panel.merge(ibt_wind, on=['region_name', 'year'], how='left')
panel['wind_kt'] = panel['wind_kt'].fillna(0.0)

panel['log_loss'] = np.log1p(panel['total_losses_musd'])
panel['log_pop']  = np.log1p(panel['pop'])
panel['log_gdp']  = np.log(panel['gdp_pc'].clip(lower=1))

panel = (panel[panel['total_losses_musd'] > 0]
         .dropna(subset=['log_loss'] + M6_PRED)
         .query('year <= 2025')
         .copy())

N         = len(panel)
N_REGIONS = panel['region_name'].nunique()
print(f"    Panel: {N:,} obs · {N_REGIONS} regions · {panel['country'].nunique()} countries")
for c in CARIB_8:
    nr = panel[panel['country'] == c]['region_name'].nunique()
    no = (panel['country'] == c).sum()
    print(f"      {c:<28}: {no:>5} obs  {nr:>3} regions")

# ── [2] Standardise (÷ SD) and fit ───────────────────────────────────────────
print("\n[2] Standardising predictors and estimating PanelOLS …")
sd_map = panel[M6_PRED].std(ddof=1)
panel_s = panel.copy()
for v in M6_PRED:
    panel_s[v] = panel[v] / sd_map[v]

panel_idx = panel_s.set_index(['region_name', 'year'])
y = panel_idx['log_loss']
X = panel_idx[M6_PRED]
country_codes = panel_idx['country'].astype('category').cat.codes

mod = PanelOLS(y, X, entity_effects=True)
try:
    res = mod.fit(cov_type='clustered', clusters=country_codes)
    print("    Covariance: country-clustered  ✓")
except Exception as e:
    print(f"    Country-cluster failed ({e}); falling back to entity-clustered")
    res = mod.fit(cov_type='clustered', cluster_entity=True)

within_r2 = float(res.rsquared)

eff = res.estimated_effects
if isinstance(eff, pd.DataFrame):
    eff = eff.iloc[:, 0]
y_hat_full = res.fitted_values.squeeze() + eff
overall_r2 = float(np.corrcoef(y_hat_full.values, y.values)[0, 1] ** 2)

print(f"    Within-R²  = {within_r2:.4f}")
print(f"    Overall-R² = {overall_r2:.4f}")
print(f"    N={N:,}   Regions={N_REGIONS}")

# ── [3] Coefficients ──────────────────────────────────────────────────────────
print("\n[3] Coefficients (standardised, dep = log1p loss):")
print(f"  {'Predictor':<28} {'β':>8} {'SE':>7} {'p':>7}  sig")
print("  " + "-" * 58)
coef_rows = []
for v in M6_PRED:
    b  = float(res.params[v])
    se = float(res.std_errors[v])
    p  = float(res.pvalues[v])
    lo = float(res.conf_int().loc[v, 'lower'])
    hi = float(res.conf_int().loc[v, 'upper'])
    print(f"  {LABELS[v]:<28} {b:+8.4f} {se:7.4f} {p:7.4f}  {stars(p)}")
    coef_rows.append({'variable': v, 'label': LABELS[v],
                      'coef': b, 'se': se, 'pvalue': p, 'ci_lo': lo, 'ci_hi': hi})

df_coef = pd.DataFrame(coef_rows)
meta = pd.DataFrame([
    {'variable': 'R2_within',  'label': 'Within-R²',  'coef': within_r2,      'se': np.nan, 'pvalue': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan},
    {'variable': 'R2_overall', 'label': 'Overall-R²', 'coef': overall_r2,     'se': np.nan, 'pvalue': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan},
    {'variable': 'N',          'label': 'N',           'coef': float(N),       'se': np.nan, 'pvalue': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan},
    {'variable': 'N_regions',  'label': 'N_regions',   'coef': float(N_REGIONS), 'se': np.nan, 'pvalue': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan},
])
df_coef = pd.concat([df_coef, meta], ignore_index=True)
df_coef.to_csv(OUT_DIR / 'm6_carib_coefficients.csv', index=False)
print(f"\n  → m6_carib_coefficients.csv")

# ── [4] Partial-R² decomposition ─────────────────────────────────────────────
print("\n[4] Partial-R² decomposition …")

def fit_r2(pred_list):
    mr = PanelOLS(y, panel_idx[pred_list], entity_effects=True)
    try:
        rr = mr.fit(cov_type='clustered', clusters=country_codes)
    except Exception:
        rr = mr.fit(cov_type='clustered', cluster_entity=True)
    return float(rr.rsquared)

r2_no_built    = fit_r2([v for v in M6_PRED if v not in BUILT])
r2_no_hazard   = fit_r2([v for v in M6_PRED if v not in HAZARD])
r2_no_controls = fit_r2([v for v in M6_PRED if v not in CONTROLS])

pR2_built    = within_r2 - r2_no_built
pR2_hazard   = within_r2 - r2_no_hazard
pR2_controls = within_r2 - r2_no_controls
denom        = max(pR2_built + pR2_hazard + pR2_controls, 1e-12)

decomp = pd.DataFrame([
    {'group': 'built',    'label': 'Built environment',
     'r2_excl': r2_no_built,    'partial_r2': pR2_built,    'share': pR2_built    / denom},
    {'group': 'hazard',   'label': 'Hazard',
     'r2_excl': r2_no_hazard,   'partial_r2': pR2_hazard,   'share': pR2_hazard   / denom},
    {'group': 'controls', 'label': 'Controls',
     'r2_excl': r2_no_controls, 'partial_r2': pR2_controls, 'share': pR2_controls / denom},
    {'group': 'r2_full',  'label': 'Within-R² (full M6)',
     'r2_excl': np.nan,         'partial_r2': within_r2,    'share': 1.0},
])
decomp.to_csv(OUT_DIR / 'm6_carib_decomposition.csv', index=False)

print(f"  {'Group':<18} {'R² excl.':>10} {'Partial R²':>12} {'Share':>8}")
print("  " + "-" * 52)
for _, row in decomp[decomp['group'] != 'r2_full'].iterrows():
    print(f"  {row['label']:<18} {row['r2_excl']:>10.4f} {row['partial_r2']:>12.4f} {row['share']:>7.1%}")
print(f"  {'Full M6 within-R²':<18} {within_r2:>23.4f}")
print(f"  → m6_carib_decomposition.csv")

# ── [5] Country fixed effects ─────────────────────────────────────────────────
print("\n[5] Country fixed effects …")

entity_fe = eff.groupby(level=0).mean()
region_country = panel_idx.groupby(level=0)['country'].first()
aux = pd.DataFrame({'effect': entity_fe, 'country': region_country})

other = [c for c in CARIB_8 if c != 'Bahamas']
cdums = pd.get_dummies(aux['country'], drop_first=False)
X_aux = cdums[other].astype(float)
X_aux.insert(0, 'const', 1.0)
ols_aux = sm.OLS(aux['effect'].values, X_aux.values).fit()
pnames  = ['const'] + other
p_map   = dict(zip(pnames, ols_aux.params))
se_map  = dict(zip(pnames, ols_aux.bse))
pv_map  = dict(zip(pnames, ols_aux.pvalues))

gdp_map  = panel.groupby('country')['gdp_pc'].mean()
nobs_map = panel.groupby('country').size()

fe_rows = []
for c in CARIB_8:
    if c == 'Bahamas':
        fe_rows.append({'country': c, 'fixed_effect': 0.0, 'se': np.nan,
                        'p_value': np.nan,
                        'gdp_per_capita_usd': float(gdp_map.get(c, np.nan)),
                        'n_obs': int(nobs_map.get(c, 0))})
    else:
        fe_rows.append({'country': c,
                        'fixed_effect': float(p_map.get(c, np.nan)),
                        'se':           float(se_map.get(c, np.nan)),
                        'p_value':      float(pv_map.get(c, np.nan)),
                        'gdp_per_capita_usd': float(gdp_map.get(c, np.nan)),
                        'n_obs': int(nobs_map.get(c, 0))})

df_fe = (pd.DataFrame(fe_rows)
           .sort_values('fixed_effect', ascending=False)
           .reset_index(drop=True))

print(f"  {'Country':<25} {'α_i':>8} {'SE':>7} {'p':>8}  {'n_obs':>6}")
print("  " + "-" * 62)
for _, r in df_fe.iterrows():
    se_s = f"{r['se']:.4f}" if pd.notna(r['se']) else "  (ref)"
    p_s  = f"{r['p_value']:.4f}" if pd.notna(r['p_value']) else "       —"
    print(f"  {r['country']:<25} {r['fixed_effect']:+8.4f} {se_s:>7} {p_s:>8}  {int(r['n_obs']):>6} {stars(r['p_value'])}")

df_fe.to_csv(OUT_DIR / 'm6_carib_fixed_effects.csv', index=False)
df_fe.to_csv(OUT_DIR / 'fixed_effects_clean.csv', index=False)
print(f"\n  → m6_carib_fixed_effects.csv")
print(f"  → fixed_effects_clean.csv  (consumed by fig_2_2)")

# ── [6] Table B.5 — descriptive statistics on the M6 estimation panel ─────────
print("\n[6] Descriptive statistics (Table B.5) …")

DS_ROWS = [
    ('log(1+Loss)',     'log_loss',          'log(1+MUSD)'),
    ('Total losses',   'total_losses_musd',  'MUSD'),
    ('Built surface',  'built_s',            'km²/cell'),
    ('Built volume',   'built_v',            'km³/cell ×10³'),
    ('Drought index',  'drought',            'fraction'),
    ('Rainfall index', 'rainfall',           'fraction'),
    ('Heatwave index', 'heatwave',           '[0,1]'),
    ('Wind speed',     'wind_kt',            'kt'),
    ('log(Population)','log_pop',            'log(thous.)'),
    ('log(GDP p.c.)',  'log_gdp',            'log(2015 USD)'),
    ('Population',     'pop',                'thousands'),
    ('GDP per capita', 'gdp_pc',             '2015 USD'),
]

ds_records = []
for label, col, unit in DS_ROWS:
    s = panel[col].dropna()
    ds_records.append({'variable': label, 'units': unit, 'n': len(s),
                       'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
                       'p25': s.quantile(.25), 'p50': s.median(),
                       'p75': s.quantile(.75), 'max': s.max()})
df_ds = pd.DataFrame(ds_records)
df_ds.to_csv(OUT_DIR / 'descriptive_stats_m6_carib.csv', index=False)

def _fmt(v):
    if abs(v) >= 100:   return f"{v:,.1f}"
    if abs(v) >= 1:     return f"{v:.3f}"
    if abs(v) >= 0.001: return f"{v:.4f}"
    return f"{v:.6f}"

tex = [
    r'\begin{table}[htbp]', r'\centering',
    r'\caption{Descriptive statistics — M6 estimation panel (8 Caribbean countries, '
    r'region-years with total losses $>0$, no missing predictors). '
    f'$N={N:,}$ observations, {N_REGIONS} sub-national regions, 1995--2024. '
    r'GDP per capita: World Bank NY.GDP.PCAP.KD (constant 2015 USD).}',
    r'\label{tab:desc_stats_m6}',
    r'\small',
    r'\begin{tabular}{llrrrrrrrr}',
    r'\toprule',
    r'Variable & Units & $N$ & Mean & SD & Min & p25 & Median & p75 & Max \\',
    r'\midrule',
]
for _, r in df_ds.iterrows():
    vals = [_fmt(r[c]) for c in ('mean','sd','min','p25','p50','p75','max')]
    tex.append(f"{r['variable']} & {r['units']} & {int(r['n']):,} & " + " & ".join(vals) + r" \\")
tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
with open(OUT_DIR / 'descriptive_stats_m6_carib.tex', 'w') as f:
    f.write('\n'.join(tex))

print(f"  → descriptive_stats_m6_carib.csv")
print(f"  → descriptive_stats_m6_carib.tex")

print("\n" + "=" * 66)
print("DONE")
print(f"  N={N:,}   Regions={N_REGIONS}   Within-R²={within_r2:.4f}   Overall-R²={overall_r2:.4f}")
print(f"  Partial R²  built={pR2_built:.4f} ({pR2_built/denom:.1%})  "
      f"hazard={pR2_hazard:.4f} ({pR2_hazard/denom:.1%})  "
      f"controls={pR2_controls:.4f} ({pR2_controls/denom:.1%})")
