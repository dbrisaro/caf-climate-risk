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

BASE_DIR = Path('/home/jupyter-daniela/CAF')
PROC     = BASE_DIR / 'data/processed/caribbean_adm_boundaries'
OUT_DIR  = BASE_DIR / 'outputs/regression'
OUT_DIR.mkdir(parents=True, exist_ok=True)

HAZARDS  = ['cyclone', 'rainfall', 'drought', 'heatwave']
N_LAGS   = 3
SCENARIOS = {
    'p90 (lenient)':  dict(rain_pct='p90', drought_pct='p10', heat_pct='p90'),
    'p95 (baseline)': dict(rain_pct='p95', drought_pct='p5',  heat_pct='p95'),
    'p99 (strict)':   dict(rain_pct='p99', drought_pct='p1',  heat_pct='p99'),
}

print("=" * 62)
print("fig_robustness_percentiles — threshold sensitivity")
print("=" * 62)

df_rain  = pd.read_parquet(PROC / 'era5_output_precipitation/rainfall_aggregated_data.parquet')
df_dry   = pd.read_parquet(PROC / 'era5_output/droughts_aggregated_data.parquet')
df_heat  = pd.read_parquet(PROC / 'heatwaves_computation/heatwaves_aggregated_data.parquet')
df_cyc   = pd.read_parquet(PROC / 'ibtracs_output/ibtracs_aggregated_data.parquet')
df_exp   = pd.read_parquet(PROC / 'exposure_panel.parquet')
df_mp    = pd.read_parquet(PROC / 'master_panel.parquet')

log.info("Raw data loaded.")

def build_panel(rain_pct='p95', drought_pct='p5', heat_pct='p95'):
    merge_cols = ['region_name', 'year', 'window']

    rain = (df_rain[df_rain['activacion'] == rain_pct].copy()
            .rename(columns={'n_rainfall_pixels': 'rainfall'}))
    rain['rainfall'] = (rain['rainfall'] > 0).astype(int)

    dry = (df_dry[df_dry['activacion'] == drought_pct].copy()
           .rename(columns={'n_drought_pixels': 'drought'}))
    dry['drought'] = (dry['drought'] > 0).astype(int)

    heat = (df_heat[df_heat['activacion'] == heat_pct].copy()
            .rename(columns={'year_win': 'year', 'n_event_days': 'heatwave'}))
    heat['heatwave'] = (heat['heatwave'] > 0).astype(int)

    cyc = df_cyc.rename(columns={'year_win': 'year'}).copy()
    cyc['cyclone'] = (cyc['frequency'] > 0).astype(int)

    # Build seasonal hazard panel first (all sources have region_name, year, window)
    panel = rain[merge_cols + ['rainfall']].merge(
        dry[merge_cols + ['drought']], on=merge_cols, how='outer')
    panel = panel.merge(heat[merge_cols + ['heatwave']], on=merge_cols, how='outer')
    panel = panel.merge(cyc[merge_cols + ['cyclone']], on=merge_cols, how='outer')

    # Merge annual exposure on region_name + year (exposure_panel has no window)
    panel = panel.merge(
        df_exp[['region_name', 'year', 'pop', 'built_s', 'built_v', 'dem', 'exposure_index']],
        on=['region_name', 'year'], how='left')

    # Merge losses from master_panel DJF window (losses are identical across all windows)
    loss_cols = ['losses_deaths_musd', 'total_damage_musd', 'total_losses_musd']
    avail = [c for c in loss_cols if c in df_mp.columns]
    loss_mp = df_mp[df_mp['window'] == 'DJF'][['region_name', 'year'] + avail].copy()
    panel = panel.merge(loss_mp, on=['region_name', 'year'], how='left')

    haz_annual = panel.groupby(['region_name', 'year'])[HAZARDS].sum().reset_index()
    static = panel[panel['window'] == 'DJF'][
        ['region_name', 'year', 'pop', 'built_s', 'built_v', 'dem',
         'exposure_index', 'losses_deaths_musd', 'total_damage_musd', 'total_losses_musd']
    ].copy()
    ann = static.merge(haz_annual, on=['region_name', 'year'], how='inner')

    ann['deaths_pc']   = ann['losses_deaths_musd'] / ann['pop'].clip(lower=0.001)
    ann['log_deaths_pc'] = np.log1p(ann['deaths_pc'])
    ann['log_damage_pc'] = np.log1p(ann['total_damage_musd'] / ann['pop'].clip(lower=0.001))
    ann['ln_pop']        = np.log1p(ann['pop'])
    ann['built_density'] = ann['built_s'] / ann['pop'].clip(lower=0.001)
    ann['built_intensity'] = ann['built_v'] / ann['built_s'].clip(lower=0.001)
    ann['ln_dem']        = np.log1p(ann['dem'])
    ann['built_s_lag1']  = ann.groupby('region_name')['built_s'].shift(1)
    ann['urban_growth']  = ann['built_s'] - ann['built_s_lag1']

    for h in HAZARDS:
        for lag in range(N_LAGS + 1):
            ann[f'{h}_L{lag}'] = ann.groupby('region_name')[h].shift(lag)

    return ann.dropna(subset=[f'{h}_L{N_LAGS}' for h in HAZARDS])

def _tidy(m):
    t = m.tidy().reset_index()
    t.columns = ['coefficient', 'estimate', 'std_error', 't_value', 'p_value', 'ci_low', 'ci_high']
    return t

def run_reg(lhs, rhs, data, label):
    try:
        m = pf.feols(f'{lhs} ~ {rhs} | region_name + year',
                     data=data, vcov={'CRV1': 'region_name'})
        log.info("  %s  N=%d  R²w=%.3f", label, m._N, m._r2_within)
        return m
    except Exception as e:
        log.warning("  %s failed: %s", label, e)
        return None

HAZ         = ' + '.join(f'{h}_L{l}' for h in HAZARDS for l in range(N_LAGS + 1))
URBAN_TERMS = 'ln_pop + built_density + built_intensity + ln_dem + urban_growth'

results = {}
for scen_name, kwargs in SCENARIOS.items():
    log.info("Scenario: %s", scen_name)
    try:
        data = build_panel(**kwargs)
        results[scen_name] = {
            's1': {'log_deaths_pc': run_reg('log_deaths_pc', HAZ, data, f'{scen_name} S1 deaths'),
                   'log_damage_pc': run_reg('log_damage_pc', HAZ, data, f'{scen_name} S1 damage')},
            's3': {'log_deaths_pc': run_reg('log_deaths_pc', f'{HAZ} + {URBAN_TERMS}', data, f'{scen_name} S3 deaths'),
                   'log_damage_pc': run_reg('log_damage_pc', f'{HAZ} + {URBAN_TERMS}', data, f'{scen_name} S3 damage')},
        }
    except Exception as e:
        log.warning("Scenario %s failed: %s", scen_name, e)

if not results:
    log.error("All scenarios failed — check that raw aggregated data files exist and have expected columns.")
    raise SystemExit(1)

l0_terms  = [f'{h}_L0' for h in HAZARDS]
coef_rows = []
for scen, stages in results.items():
    for stage_key, models in stages.items():
        m = models.get('log_deaths_pc')
        if m is None:
            continue
        tbl = _tidy(m)
        for term in l0_terms:
            row = tbl[tbl['coefficient'] == term]
            if row.empty:
                continue
            coef_rows.append({'scenario': scen, 'stage': stage_key, 'hazard': term,
                               'beta': row['estimate'].values[0],
                               'se':   row['std_error'].values[0],
                               'p_value': row['p_value'].values[0]})
coef_df = pd.DataFrame(coef_rows)

colors  = {'p90 (lenient)': '#1f77b4', 'p95 (baseline)': '#ff7f0e', 'p99 (strict)': '#2ca02c'}
offsets = {'p90 (lenient)': -0.25, 'p95 (baseline)': 0, 'p99 (strict)': 0.25}

if not coef_df.empty:

    fig, axes = plt.subplots(1, len(HAZARDS), figsize=(14, 5), sharey=False)
    for i, (ax, haz) in enumerate(zip(axes, HAZARDS)):
        for stage_key, ls in [('s1', '-'), ('s3', '--')]:
            sub = coef_df[(coef_df['hazard'] == f'{haz}_L0') & (coef_df['stage'] == stage_key)]
            for scen in sub['scenario'].unique():
                row = sub[sub['scenario'] == scen]
                x   = offsets[scen]
                ax.errorbar(x, row['beta'].values[0], yerr=1.96 * row['se'].values[0],
                            fmt='o', color=colors[scen], linestyle=ls,
                            capsize=4, label=f'{scen} {stage_key}' if ax == axes[0] else '')
        ax.axhline(0, color='black', lw=0.7)
        ax.set_title(f"{chr(ord('a')+i)}  {haz.capitalize()}", loc='left', fontweight='normal', fontsize=10)
        ax.set_xticks([])
        ax.set_xlabel('Scenario (colour) / Stage (line)')
    axes[0].set_ylabel('β L0 (log deaths pc)')
    handles = [plt.Line2D([0], [0], color=c, marker='o', label=l)
               for l, c in colors.items()]
    fig.legend(handles=handles, loc='upper right', fontsize=8)
    plt.tight_layout()
    out1 = OUT_DIR / 'fig_robustness_coef_stability.png'
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved: %s", out1)

fig, axes = plt.subplots(len(HAZARDS), 1, figsize=(12, 14), sharex=True)
for i, (ax, haz) in enumerate(zip(axes, HAZARDS)):
    for scen, col in colors.items():
        m = results.get(scen, {}).get('s1', {}).get('log_deaths_pc')
        if m is None:
            continue
        tbl   = _tidy(m)
        betas, ses = [], []
        for l in range(N_LAGS + 1):
            row = tbl[tbl['coefficient'] == f'{haz}_L{l}']
            if row.empty:
                betas.append(np.nan); ses.append(np.nan)
            else:
                betas.append(row['estimate'].values[0])
                ses.append(row['std_error'].values[0])
        betas, ses = np.array(betas), np.array(ses)
        x = np.arange(N_LAGS + 1)
        ax.plot(x, betas, color=col, marker='o', label=scen)
        ax.fill_between(x, betas - 1.96 * ses, betas + 1.96 * ses, alpha=0.12, color=col)
    ax.axhline(0, color='black', lw=0.7)
    ax.set_title(f"{chr(ord('a')+i)}  {haz.capitalize()}", loc='left', fontweight='normal', fontsize=10)
    ax.set_ylabel('β (log deaths pc)')

axes[-1].set_xlabel('Lag (years)')
axes[0].legend(fontsize=8)
plt.tight_layout()
out2 = OUT_DIR / 'fig_robustness_irf_thresholds.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
log.info("Saved: %s", out2)

print()
print("Done.  Outputs saved to outputs/regression/")
