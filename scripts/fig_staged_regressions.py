import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pyfixest as pf
from pathlib import Path
import logging
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

BASE_DIR   = Path('/home/jupyter-daniela/CAF')
PANEL_PATH = BASE_DIR / 'data/processed/caribbean_adm_boundaries/master_panel.parquet'
SHAPEFILE  = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
OUT_DIR    = BASE_DIR / 'outputs/regression'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_LAGS  = 3
HAZARDS = ['cyclone', 'rainfall', 'drought', 'heatwave']
VSL_MAP = {
    'Haiti': 1.1, 'Dominican Republic': 1.1, 'Jamaica': 1.1, 'Saint Lucia': 1.1,
    'Dominica': 1.1, 'Bahamas': 7.9, 'Barbados': 7.9, 'Trinidad and Tobago': 7.9,
}
VSL_DEFAULT = 1.1

print("=" * 62)
print("fig_staged_regressions — Staged regression analysis")
print("=" * 62)

raw = pd.read_parquet(PANEL_PATH)

haz_annual = raw.groupby(['region_name', 'year'])[HAZARDS].sum().reset_index()

static_cols = ['region_name', 'year', 'exposure_index', 'pop',
               'built_s', 'built_v', 'dem',
               'losses_deaths_musd', 'total_damage_musd', 'total_losses_musd']
static = raw[raw['window'] == 'DJF'][static_cols].copy()

gdf = gpd.read_file(str(SHAPEFILE))
country_col = next((c for c in gdf.columns if c.lower() == 'country'), None)
region_col  = next((c for c in gdf.columns if 'region' in c.lower()), None)
region_country = gdf[[region_col, country_col]].drop_duplicates().rename(
    columns={region_col: 'region_name', country_col: 'country'}
)

ann = static.merge(haz_annual, on=['region_name', 'year'], how='inner')
ann = ann.merge(region_country, on='region_name', how='left')

ann['vsl']          = ann['country'].map(VSL_MAP).fillna(VSL_DEFAULT)
ann['deaths']       = ann['losses_deaths_musd'] / ann['vsl']
ann['deaths_pc']    = ann['deaths'] / ann['pop'].clip(lower=0.001)
ann['damage_pc']    = ann['total_damage_musd'] / ann['pop'].clip(lower=0.001)
ann['losses_pc']    = ann['total_losses_musd'] / ann['pop'].clip(lower=0.001)
ann['log_deaths_pc']  = np.log1p(ann['deaths_pc'])
ann['log_damage_pc']  = np.log1p(ann['damage_pc'])
ann['log_losses_pc']  = np.log1p(ann['losses_pc'])
ann['vuln_index']     = ann['log_losses_pc']

for h in HAZARDS:
    for lag in range(N_LAGS + 1):
        ann[f'{h}_L{lag}'] = ann.groupby('region_name')[h].shift(lag)

ann['ln_pop']          = np.log1p(ann['pop'])
ann['built_density']   = ann['built_s'] / ann['pop'].clip(lower=0.001)
ann['built_intensity'] = ann['built_v'] / ann['built_s'].clip(lower=0.001)
ann['ln_dem']          = np.log1p(ann['dem'])
ann['built_s_lag1']    = ann.groupby('region_name')['built_s'].shift(1)
ann['urban_growth']    = ann['built_s'] - ann['built_s_lag1']
ann['year_num']        = ann['year'].astype(int)

reg_df = ann.dropna(subset=[f'{h}_L{N_LAGS}' for h in HAZARDS]).copy()
log.info("Regression sample: %d obs, %d regions, years %d–%d",
         len(reg_df), reg_df['region_name'].nunique(),
         reg_df['year'].min(), reg_df['year'].max())

HAZ          = ' + '.join(f'{h}_L{l}' for h in HAZARDS for l in range(N_LAGS + 1))
URBAN_TERMS  = 'ln_pop + built_density + built_intensity + ln_dem + urban_growth'
FE_TWOWAY    = 'region_name + year'

def _tidy(m):
    t = m.tidy().reset_index()
    t.columns = ['coefficient', 'estimate', 'std_error', 't_value', 'p_value', 'ci_low', 'ci_high']
    return t

def run_stage(lhs, rhs_terms, fe=FE_TWOWAY, data=None, label=''):
    if data is None:
        data = reg_df
    formula = f'{lhs} ~ {rhs_terms} | {fe}'
    try:
        m = pf.feols(formula, data=data, vcov={'CRV1': 'region_name'})
        log.info("  %s  N=%d  R²_within=%.3f", label, m._N, m._r2_within)
        return m
    except Exception as e:
        log.warning("  %s failed: %s", label, e)
        return None

log.info("Stage 1 — Hazard only")
s1_deaths = run_stage('log_deaths_pc', HAZ, label='S1 deaths')
s1_damage = run_stage('log_damage_pc', HAZ, label='S1 damage')

log.info("Stage 3 — Hazard + Urban")
s3_deaths = run_stage('log_deaths_pc', f'{HAZ} + {URBAN_TERMS}', label='S3 deaths')
s3_damage = run_stage('log_damage_pc', f'{HAZ} + {URBAN_TERMS}', label='S3 damage')

log.info("Stage 5 — Hazard + Urban + Interactions")
interact_terms = ' + '.join(
    f'{h}_L0:{v}' for h in HAZARDS for v in ['ln_pop', 'built_density', 'ln_dem']
)
s5_deaths = run_stage('log_deaths_pc', f'{HAZ} + {URBAN_TERMS} + {interact_terms}', label='S5 deaths')
s5_damage = run_stage('log_damage_pc', f'{HAZ} + {URBAN_TERMS} + {interact_terms}', label='S5 damage')

def extract_coefs(model, terms, label):
    if model is None:
        return pd.DataFrame()
    tbl = _tidy(model)
    sub = tbl[tbl['coefficient'].isin(terms)][['coefficient', 'estimate', 'std_error', 'p_value']].copy()
    sub['stage'] = label
    return sub

l0_terms   = [f'{h}_L0' for h in HAZARDS]
coef_table = pd.concat([
    extract_coefs(s1_deaths, l0_terms, 'S1: Hazard only'),
    extract_coefs(s3_deaths, l0_terms, 'S3: + Urban'),
    extract_coefs(s5_deaths, l0_terms, 'S5: + Urban×Haz'),
], ignore_index=True)

if not coef_table.empty:
    stage_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    fig, axes = plt.subplots(1, len(HAZARDS), figsize=(14, 4), sharey=False)
    for i, (ax, haz) in enumerate(zip(axes, HAZARDS)):
        sub = coef_table[coef_table['coefficient'] == f'{haz}_L0']
        x   = range(len(sub))
        ax.bar(x, sub['estimate'], color=stage_colors[:len(sub)], alpha=0.75,
               yerr=1.96 * sub['std_error'], capsize=4, error_kw={'linewidth': 1})
        ax.set_xticks(x)
        ax.set_xticklabels(sub['stage'], rotation=20, ha='right', fontsize=8)
        ax.axhline(0, color='black', lw=0.8, linestyle='--')
        ax.set_title(f"{chr(ord('a')+i)}  {haz.capitalize()}", loc='left', fontweight='normal', fontsize=10)
        ax.set_ylabel('β L0 (log deaths pc)' if ax == axes[0] else '')
    plt.tight_layout()
    out1 = OUT_DIR / 'fig_staged_coefs.png'
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved: %s", out1)

def irf_from_model(model, hazard):
    if model is None:
        return None
    tbl  = _tidy(model)
    rows = []
    for l in range(N_LAGS + 1):
        row = tbl[tbl['coefficient'] == f'{hazard}_L{l}']
        if row.empty:
            continue
        rows.append({'lag': l, 'beta': row['estimate'].values[0], 'se': row['std_error'].values[0]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df['cumsum']    = df['beta'].cumsum()
    df['cumsum_se'] = df['se'].expanding().apply(lambda x: np.sqrt((x**2).sum()))
    return df

fig, axes = plt.subplots(2, len(HAZARDS), figsize=(14, 8), sharey='row')
model_pairs = [('S1 (hazard only)', s1_deaths, '#1f77b4'),
               ('S3 (+ urban)',     s3_deaths, '#ff7f0e')]
for col, haz in enumerate(HAZARDS):
    for row_i, (label, model, color) in enumerate(model_pairs):
        ax = axes[row_i, col]
        irf = irf_from_model(model, haz)
        if irf is None:
            ax.text(0.5, 0.5, 'n/a', ha='center', va='center', transform=ax.transAxes)
            continue
        ax.bar(irf['lag'], irf['beta'], color=color, alpha=0.7)
        ax.errorbar(irf['lag'], irf['beta'], yerr=1.96 * irf['se'],
                    fmt='none', color='black', capsize=3)
        ax.axhline(0, color='black', lw=0.7)
        panel_letter = chr(ord('a') + row_i * len(HAZARDS) + col)
        ax.set_title(f"{panel_letter}  {haz.capitalize()} — {label}", loc='left', fontweight='normal', fontsize=10)
        ax.set_xlabel('Lag (years)')
        if col == 0:
            ax.set_ylabel('β (log deaths pc)')
plt.tight_layout()
out2 = OUT_DIR / 'fig_staged_irf.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
log.info("Saved: %s", out2)

reg_bin = reg_df.copy()
for h in HAZARDS:
    col     = f'{h}_L0'
    nonzero = reg_bin.loc[reg_bin[col] > 0, col]
    p50     = nonzero.median() if len(nonzero) >= 10 else 0
    reg_bin[f'{h}_low']  = ((reg_bin[col] > 0) & (reg_bin[col] <= p50)).astype(int)
    reg_bin[f'{h}_high'] = (reg_bin[col] > p50).astype(int)

bin_terms     = ' + '.join(f'{h}_{b}' for h in HAZARDS for b in ['low', 'high'])
s1_bin_deaths = run_stage('log_deaths_pc', bin_terms, data=reg_bin, label='S1-bin')
s3_bin_deaths = run_stage('log_deaths_pc', f'{bin_terms} + {URBAN_TERMS}', data=reg_bin, label='S3-bin')

if s1_bin_deaths and s3_bin_deaths:
    fig, axes = plt.subplots(1, len(HAZARDS), figsize=(14, 4))
    for i, (ax, haz) in enumerate(zip(axes, HAZARDS)):
        for model, label, alpha in [(s1_bin_deaths, 'S1', 0.5), (s3_bin_deaths, 'S3+Urban', 0.85)]:
            tbl = _tidy(model)
            for j, b in enumerate(['low', 'high']):
                row = tbl[tbl['coefficient'] == f'{haz}_{b}']
                if row.empty:
                    continue
                x_pos = j + (0.2 if 'Urban' in label else -0.2)
                ax.bar(x_pos, row['estimate'].values[0], width=0.35,
                       color=plt.cm.Blues(alpha), label=label if j == 0 else '')
                ax.errorbar(x_pos, row['estimate'].values[0],
                            yerr=1.96 * row['std_error'].values[0],
                            fmt='none', color='black', capsize=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['low', 'high'])
        ax.axhline(0, color='black', lw=0.8)
        ax.set_title(f"{chr(ord('a')+i)}  {haz.capitalize()}", loc='left', fontweight='normal', fontsize=10)
    axes[0].set_ylabel('β (log deaths pc)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=plt.cm.Blues(a)) for a in [0.5, 0.85]]
    fig.legend(handles, ['S1: Hazard bins', 'S3: + Urban'], loc='upper right', fontsize=8)
    plt.tight_layout()
    out3 = OUT_DIR / 'fig_staged_bins.png'
    fig.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved: %s", out3)

print()
print("Done.  Outputs saved to outputs/regression/")
