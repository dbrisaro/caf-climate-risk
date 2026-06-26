#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')

import datetime
import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO

RUN_NAME     = 'caribbean_adm_boundaries'
BASE         = Path('/home/jupyter-daniela/CAF/data/processed') / RUN_NAME
DATA_PATH    = BASE / 'master_panel.parquet'
SHAPEFILE    = Path('/home/jupyter-daniela/CAF/data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp')
DISPLAYS_DIR = Path('/home/jupyter-daniela/CAF/outputs/regression')
REPORTS_DIR  = Path('/home/jupyter-daniela/CAF/outputs/reports')
DISPLAYS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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

print("Loading data...")
print("  Fetching GDP per capita from World Bank...")
iso_str = ';'.join(WB_ISO.values())
wb_data = requests.get(
    f'https://api.worldbank.org/v2/country/{iso_str}/indicator/'
    f'NY.GDP.PCAP.KD?date=1995:2025&format=json&per_page=500', timeout=30
).json()[1]
iso_to_country = {v: k for k, v in WB_ISO.items()}
df_gdp = pd.DataFrame([
    {'country': iso_to_country[r['country']['id']], 'year': int(r['date']), 'gdp_pc': r['value']}
    for r in wb_data if r['value'] is not None
])

df  = pd.read_parquet(DATA_PATH)
gdf = gpd.read_file(SHAPEFILE)[['region_nam', 'country']].rename(columns={'region_nam': 'region_name'})
df  = df.merge(gdf, on='region_name', how='left')

panel = (
    df.groupby(['region_name', 'country', 'year'])
    .agg(
        total_losses_musd  = ('total_losses_musd',  'sum'),
        total_damage_musd  = ('total_damage_musd',  'sum'),
        losses_deaths_musd = ('losses_deaths_musd', 'sum'),
        drought            = ('drought',   'mean'),
        rainfall           = ('rainfall',  'mean'),
        heatwave           = ('heatwave',  'mean'),
        cyclone            = ('cyclone',   'mean'),
        pop                = ('pop',       'mean'),
        built_s            = ('built_s',   'mean'),
        built_v            = ('built_v',   'mean'),
        dem                = ('dem',       'mean'),
        exposure_index     = ('exposure_index', 'mean'),
    )
    .reset_index()
    .merge(df_gdp, on=['country', 'year'], how='left')
)

panel['hazard_avg']  = panel[['drought', 'rainfall', 'heatwave', 'cyclone']].mean(axis=1)
panel['log_gdp_pc']  = np.log(panel['gdp_pc'])
panel['log_pop']     = np.log1p(panel['pop'])
panel['log_losses']  = np.log1p(panel['total_losses_musd'])
panel['log_damage']  = np.log1p(panel['total_damage_musd'])
panel['log_deaths']  = np.log1p(panel['losses_deaths_musd'])

BASE_COVS = ['log_gdp_pc', 'log_pop', 'built_s', 'built_v', 'hazard_avg']
panel_loss = (panel[panel['total_losses_musd'] > 0]
              .dropna(subset=BASE_COVS + ['exposure_index'])
              .copy())
panel_dmg  = (panel[panel['total_damage_musd'] > 0]
              .dropna(subset=BASE_COVS)
              .copy())
panel_dth  = (panel[panel['losses_deaths_musd'] > 0]
              .dropna(subset=BASE_COVS)
              .copy())

print(f"  Panel (losses>0): {len(panel_loss):,} obs, "
      f"{panel_loss['region_name'].nunique()} regions, "
      f"{panel_loss['country'].nunique()} countries")


def within(df, cols, by):
    out = df.copy()
    for c in cols:
        out[c] = df[c] - df.groupby(by)[c].transform('mean')
    return out

def add_dummies(df, col):
    return pd.get_dummies(df[col], drop_first=True, prefix=col).astype(float)

def run_ols(y, X, cov='HC3'):
    X_ = sm.add_constant(X, has_constant='add')
    return sm.OLS(y, X_).fit(cov_type=cov)

def overall_r2(model, y_actual, y_fitted):
    ss_res = np.sum((y_actual - y_fitted) ** 2)
    ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
    return 1 - ss_res / ss_tot

def fe_fitted(df, model, xcols, group_col):
    params = np.asarray(model.params)
    Xmat = sm.add_constant(df[xcols].values, has_constant='add')
    y_within_hat = Xmat @ params
    group_mean_y = df.groupby(group_col)['_y_orig'].transform('mean')
    Xmat_noc = df[xcols].values
    betas = params[1:]
    group_mean_xb = pd.Series(Xmat_noc @ betas, index=df.index)
    group_mean_xb_by_group = group_mean_xb.groupby(df[group_col]).transform('mean')
    return group_mean_y + (group_mean_xb - group_mean_xb_by_group)

results = []

def record(tag, label, model, y_actual, y_fitted, n_obs, n_regions, n_countries, xnames, df_used, dep_label='log1p(total_losses)'):
    params = np.asarray(model.params)
    pvals  = np.asarray(model.pvalues)
    bse    = np.asarray(model.bse)
    ci     = np.asarray(model.conf_int())
    sig    = ['***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.' for p in pvals]
    or2    = overall_r2(model, y_actual, y_fitted)
    coef_rows = []
    for i, name in enumerate(xnames):
        coef_rows.append({
            'model': tag, 'coef_name': name, 'estimate': params[i],
            'std_error': bse[i], 'p_value': pvals[i],
            'ci_lower': ci[i, 0], 'ci_upper': ci[i, 1], 'significance': sig[i],
        })
    results.append({
        'tag': tag, 'label': label, 'dep_var': dep_label,
        'within_r2': model.rsquared, 'overall_r2': or2,
        'n_obs': int(n_obs), 'n_regions': n_regions, 'n_countries': n_countries,
        'f_stat': model.fvalue, 'f_pval': model.f_pvalue,
        'coefs': coef_rows,
    })
    print(f"  [{tag}] {label} — Within-R²={model.rsquared:.3f}, Overall-R²={or2:.3f}, N={int(n_obs):,}")

print("\nRunning models...")
p = panel_loss.copy()
xcols = ['log_pop', 'built_s', 'built_v', 'hazard_avg', 'log_gdp_pc']
X = p[xcols].values
y = p['log_losses'].values
m1 = run_ols(y, X)
xnames = ['Intercept'] + xcols
record('M1', 'Pooled OLS (no FE)', m1, y, m1.fittedvalues,
       m1.nobs, p['region_name'].nunique(), p['country'].nunique(), xnames, p)

p = panel_loss.copy()
all_cols = ['log_losses', 'log_pop', 'built_s', 'built_v', 'hazard_avg', 'log_gdp_pc']
p_w = within(p, all_cols, 'country')
X = p_w[xcols].values
y = p_w['log_losses'].values
m2 = run_ols(y, X)
params2 = np.asarray(m2.params)
means_c = p.groupby('country')[all_cols].mean()
alpha2  = (means_c['log_losses']
           - params2[1]*means_c['log_pop']
           - params2[2]*means_c['built_s']
           - params2[3]*means_c['built_v']
           - params2[4]*means_c['hazard_avg']
           - params2[5]*means_c['log_gdp_pc'])
p['alpha'] = p['country'].map(alpha2)
y_hat2 = (p['alpha'] + params2[1]*p['log_pop'] + params2[2]*p['built_s']
          + params2[3]*p['built_v'] + params2[4]*p['hazard_avg'] + params2[5]*p['log_gdp_pc'])
xnames = ['Intercept (demeaned)'] + xcols
record('M2', 'Country FE (within)', m2, p['log_losses'].values, y_hat2.values,
       m2.nobs, p['region_name'].nunique(), p['country'].nunique(), xnames, p)

p = panel_loss.copy()
country_dummies = add_dummies(p, 'country')
year_dummies    = add_dummies(p, 'year')
X = pd.concat([p[xcols], country_dummies, year_dummies], axis=1).values
y = p['log_losses'].values
m3 = run_ols(y, X)
xnames_m3 = (['Intercept'] + xcols
             + [f'country={c}' for c in country_dummies.columns]
             + [f'year={yr}' for yr in year_dummies.columns])
record('M3', 'Two-way FE (country + year)', m3, y, m3.fittedvalues,
       m3.nobs, p['region_name'].nunique(), p['country'].nunique(), xnames_m3, p)

p = panel_loss.copy()
xcols4 = ['log_pop', 'built_s', 'built_v', 'drought', 'rainfall', 'heatwave', 'cyclone', 'log_gdp_pc']
all4   = ['log_losses'] + xcols4
p_w = within(p, all4, 'country')
X = p_w[xcols4].values
y = p_w['log_losses'].values
m4 = run_ols(y, X)
xnames4 = ['Intercept (demeaned)'] + xcols4
record('M4', 'Country FE, hazards disaggregated', m4, y, m4.fittedvalues,
       m4.nobs, p['region_name'].nunique(), p['country'].nunique(), xnames4, p)

p = panel_loss.copy()
xcols5 = ['exposure_index', 'hazard_avg', 'log_gdp_pc']
all5   = ['log_losses'] + xcols5
p_w = within(p, all5, 'country')
X = p_w[xcols5].values
y = p_w['log_losses'].values
m5 = run_ols(y, X)
xnames5 = ['Intercept (demeaned)'] + xcols5
record('M5', 'Country FE, PCA exposure index', m5, y, m5.fittedvalues,
       m5.nobs, p['region_name'].nunique(), p['country'].nunique(), xnames5, p)

p = panel_dmg.copy()
p['log_damage'] = np.log1p(p['total_damage_musd'])
xcols6 = ['log_pop', 'built_s', 'built_v', 'hazard_avg', 'log_gdp_pc']
all6   = ['log_damage'] + xcols6
p_w = within(p, all6, 'country')
X = p_w[xcols6].values
y = p_w['log_damage'].values
m6 = run_ols(y, X)
xnames6 = ['Intercept (demeaned)'] + xcols6
record('M6', 'Country FE, dep=log(damage)', m6, y, m6.fittedvalues,
       m6.nobs, p['region_name'].nunique(), p['country'].nunique(), xnames6, p,
       dep_label='log1p(total_damage)')

p = panel_dth.copy()
p['log_deaths'] = np.log1p(p['losses_deaths_musd'])
xcols7 = ['log_pop', 'built_s', 'built_v', 'hazard_avg', 'log_gdp_pc']
all7   = ['log_deaths'] + xcols7
p_w = within(p, all7, 'country')
X = p_w[xcols7].values
y = p_w['log_deaths'].values
m7 = run_ols(y, X)
xnames7 = ['Intercept (demeaned)'] + xcols7
record('M7', 'Country FE, dep=log(mortality losses)', m7, y, m7.fittedvalues,
       m7.nobs, p['region_name'].nunique(), p['country'].nunique(), xnames7, p,
       dep_label='log1p(losses_deaths)')

p = panel_loss.copy()
p['hazard_x_gdp'] = p['hazard_avg'] * p['log_gdp_pc']
xcols8 = ['log_pop', 'built_s', 'built_v', 'hazard_avg', 'log_gdp_pc', 'hazard_x_gdp']
all8   = ['log_losses'] + xcols8
p_w = within(p, all8, 'country')
X = p_w[xcols8].values
y = p_w['log_losses'].values
m8 = run_ols(y, X)
xnames8 = ['Intercept (demeaned)'] + xcols8
record('M8', 'Country FE, hazard × GDP interaction', m8, y, m8.fittedvalues,
       m8.nobs, p['region_name'].nunique(), p['country'].nunique(), xnames8, p)

for tag_n, hazard in [('M9', 'drought'), ('M10', 'rainfall'), ('M11', 'heatwave'), ('M12', 'cyclone')]:
    p = panel_loss.copy()
    xcols_h = ['log_pop', 'built_s', 'built_v', hazard, 'log_gdp_pc']
    all_h   = ['log_losses'] + xcols_h
    p_w = within(p, all_h, 'country')
    X = p_w[xcols_h].values
    y = p_w['log_losses'].values
    mh = run_ols(y, X)
    xnames_h = ['Intercept (demeaned)'] + xcols_h
    record(tag_n, f'Country FE, hazard={hazard} only', mh, y, mh.fittedvalues,
           mh.nobs, p['region_name'].nunique(), p['country'].nunique(), xnames_h, p)

print("\nBuilding summary tables...")

CORE_MODELS  = ['M1', 'M2', 'M3', 'M4', 'M5', 'M8']
COMMON_COEFS = ['log_pop', 'built_s', 'built_v', 'hazard_avg', 'log_gdp_pc']

def fmt_coef(est, sig):
    stars = {'***': '***', '**': '**', '*': '*', 'n.s.': ''}.get(sig, '')
    return f'{est:.4f}{stars}'

def get_coef(res, coef_name):
    for row in res['coefs']:
        if row['coef_name'] == coef_name:
            return fmt_coef(row['estimate'], row['significance']), row['p_value']
    return '—', None

def build_coef_table(model_tags, coef_names):
    header = '| Coefficient | ' + ' | '.join(model_tags) + ' |'
    sep    = '|:---|' + '|'.join([':---:'] * len(model_tags)) + '|'
    rows   = [header, sep]
    for coef in coef_names:
        cells = []
        for tag in model_tags:
            res = next(r for r in results if r['tag'] == tag)
            val, _ = get_coef(res, coef)
            cells.append(val)
        rows.append(f'| `{coef}` | ' + ' | '.join(cells) + ' |')
    for stat_key, stat_label in [('within_r2', 'Within R²'), ('overall_r2', 'Overall R²'),
                                   ('n_obs', 'N'), ('n_regions', 'Regions'), ('n_countries', 'Countries')]:
        cells = []
        for tag in model_tags:
            res = next(r for r in results if r['tag'] == tag)
            v = res[stat_key]
            cells.append(f'{v:.3f}' if isinstance(v, float) else str(v))
        rows.append(f'| **{stat_label}** | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)

print("Generating figures...")

BLUE = '#2166AC'; RED = '#D6604D'; GRAY = '#BBBBBB'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})

fig, axes = plt.subplots(1, len(COMMON_COEFS), figsize=(16, 5), sharey=False)
model_colors = {'M1': '#E41A1C', 'M2': '#377EB8', 'M3': '#4DAF4A',
                'M4': '#984EA3', 'M5': '#FF7F00', 'M8': '#A65628'}
for ax, coef in zip(axes, COMMON_COEFS):
    y_vals, colors, labels = [], [], []
    for tag in ['M1', 'M2', 'M3', 'M5']:
        res = next(r for r in results if r['tag'] == tag)
        val, pval = get_coef(res, coef)
        ci_row = next((c for c in res['coefs'] if c['coef_name'] == coef), None)
        if ci_row is None:
            continue
        est = ci_row['estimate']
        lo  = ci_row['ci_lower']
        hi  = ci_row['ci_upper']
        color = model_colors[tag]
        y_pos = ['M1', 'M2', 'M3', 'M5'].index(tag)
        ax.barh(y_pos, est, xerr=[[est - lo], [hi - est]],
                color=color, alpha=0.85, height=0.5, capsize=3,
                error_kw={'linewidth': 1.2})
    ax.axvline(0, color='black', linewidth=0.7, linestyle='--')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['M1\nPooled', 'M2\nCntry FE', 'M3\n2-way FE', 'M5\nExp.idx'],
                       fontsize=8)
    ax.set_title(coef, fontsize=9, fontweight='bold')
    ax.set_xlabel('Estimate', fontsize=8)

plt.suptitle('Coefficient Estimates Across Specifications (M1/M2/M3/M5)\n95% CI, HC3 robust SEs',
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(DISPLAYS_DIR / 'fig_regression_coefplot.png', dpi=180, bbox_inches='tight')
plt.close(fig)

tags_all        = [r['tag'] for r in results]
within_r2       = [r['within_r2'] for r in results]
overall_r2_vals = [r['overall_r2'] for r in results]

x = np.arange(len(tags_all))
width = 0.38
fig, ax = plt.subplots(figsize=(14, 5))
bars1 = ax.bar(x - width/2, within_r2, width, label='Within R²', color=BLUE, alpha=0.85)
bars2 = ax.bar(x + width/2, overall_r2_vals, width, label='Overall R²', color=RED, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(tags_all, fontsize=9)
ax.set_ylabel('R²')
ax.set_ylim(0, max(max(within_r2), max(overall_r2_vals)) * 1.25)
ax.legend()
ax.set_title('Within vs Overall R² by Specification', fontsize=12, fontweight='bold')
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', fontsize=7, color=BLUE)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', fontsize=7, color=RED)
plt.tight_layout()
fig.savefig(DISPLAYS_DIR / 'fig_regression_r2.png', dpi=180, bbox_inches='tight')
plt.close(fig)

HAZARD_NAMES    = ['drought', 'rainfall', 'heatwave', 'cyclone']
HAZ_TAGS_SINGLE = {'drought': 'M9', 'rainfall': 'M10', 'heatwave': 'M11', 'cyclone': 'M12'}

fig, ax = plt.subplots(figsize=(9, 5))
y_pos = np.arange(len(HAZARD_NAMES))
width = 0.35
for i, haz in enumerate(HAZARD_NAMES):
    res_m4 = next(r for r in results if r['tag'] == 'M4')
    ci_m4  = next((c for c in res_m4['coefs'] if c['coef_name'] == haz), None)
    tag_s  = HAZ_TAGS_SINGLE[haz]
    res_s  = next(r for r in results if r['tag'] == tag_s)
    ci_s   = next((c for c in res_s['coefs'] if c['coef_name'] == haz), None)
    if ci_m4:
        est4   = ci_m4['estimate']
        err4   = [[est4 - ci_m4['ci_lower']], [ci_m4['ci_upper'] - est4]]
        color4 = BLUE if ci_m4['p_value'] < 0.05 else GRAY
        ax.barh(i + width/2, est4, xerr=err4, height=width, color=color4,
                alpha=0.85, capsize=3, label='M4 (joint)' if i == 0 else '_')
    if ci_s:
        ests   = ci_s['estimate']
        errs   = [[ests - ci_s['ci_lower']], [ci_s['ci_upper'] - ests]]
        colors = RED if ci_s['p_value'] < 0.05 else GRAY
        ax.barh(i - width/2, ests, xerr=errs, height=width, color=colors,
                alpha=0.85, capsize=3, label='M9–M12 (single)' if i == 0 else '_')

ax.axvline(0, color='black', linewidth=0.7, linestyle='--')
ax.set_yticks(y_pos)
ax.set_yticklabels(HAZARD_NAMES, fontsize=11)
ax.set_xlabel('Coefficient estimate (semi-elasticity on log-losses)')
ax.set_title('Hazard Coefficients: Joint (M4) vs Single-Hazard (M9–M12)\nCountry FE, HC3 SEs — Blue/Red = significant, Grey = n.s.',
             fontsize=11, fontweight='bold')
ax.legend()
plt.tight_layout()
fig.savefig(DISPLAYS_DIR / 'fig_regression_hazards.png', dpi=180, bbox_inches='tight')
plt.close(fig)

print("Writing report...")

lines = []
lines.append("# CAF Caribbean — Regression Suite")
lines.append(f"**Date:** {datetime.date.today().isoformat()}  ")
lines.append(f"**Run:** `{RUN_NAME}` — {panel_loss['country'].nunique()} countries, {panel_loss['region_name'].nunique()} ADM regions, 1995–2025  ")
lines.append(f"**Dependent variable (default):** log1p(total\\_losses\\_musd) — EMDAT climatological events  ")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Model Inventory")
lines.append("")
lines.append("| Tag | Specification | Dep. Var | FE | N |")
lines.append("|:----|:--------------|:---------|:---|--:|")
for r in results:
    fe_str = 'Country + Year' if r['tag'] == 'M3' else ('Country' if r['tag'] not in ['M1'] else 'None')
    lines.append(f"| {r['tag']} | {r['label']} | {r['dep_var']} | {fe_str} | {r['n_obs']:,} |")

lines.append("")
lines.append("---")
lines.append("")
lines.append("## Main Results: Core Specifications (M1–M5, M8)")
lines.append("")
lines.append("> \\* p<0.05 &nbsp; \\*\\* p<0.01 &nbsp; \\*\\*\\* p<0.001 &nbsp; (blank) n.s.  ")
lines.append("> HC3 heteroskedasticity-robust standard errors throughout.")
lines.append("")
lines.append(build_coef_table(['M1', 'M2', 'M3', 'M5', 'M8'],
                               COMMON_COEFS + ['hazard_x_gdp']))
lines.append("")
lines.append("---")
lines.append("")
lines.append("## M4: Country FE with Disaggregated Hazards")
lines.append("")
lines.append(build_coef_table(['M4'], ['log_pop', 'built_s', 'built_v',
                                        'drought', 'rainfall', 'heatwave', 'cyclone', 'log_gdp_pc']))
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Alternative Dependent Variables (M6, M7)")
lines.append("")
lines.append("> M6 uses rows with `total_damage_musd > 0`; M7 uses rows with `losses_deaths_musd > 0`.  ")
lines.append("> Same Country FE within estimator and HC3 SEs.")
lines.append("")
lines.append(build_coef_table(['M6', 'M7'], ['log_pop', 'built_s', 'built_v', 'hazard_avg', 'log_gdp_pc']))
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Single-Hazard Regressions (M9–M12)")
lines.append("")
lines.append("> Each model runs Country FE OLS with one hazard at a time.  ")
lines.append("> Useful for assessing whether collinearity between hazards affects M4.")
lines.append("")
lines.append("| Tag | Hazard | Estimate | SE | p-value | Sig | Within R² | Overall R² | N |")
lines.append("|:----|:-------|:--------:|:--:|:-------:|:---:|:---------:|:----------:|--:|")
for tag in ['M9', 'M10', 'M11', 'M12']:
    res = next(r for r in results if r['tag'] == tag)
    haz = {'M9': 'drought', 'M10': 'rainfall', 'M11': 'heatwave', 'M12': 'cyclone'}[tag]
    coef_row = next(c for c in res['coefs'] if c['coef_name'] == haz)
    lines.append(f"| {tag} | {haz} | {coef_row['estimate']:.4f} | {coef_row['std_error']:.4f} "
                 f"| {coef_row['p_value']:.4f} | {coef_row['significance']} "
                 f"| {res['within_r2']:.3f} | {res['overall_r2']:.3f} | {res['n_obs']:,} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Model Fit Comparison")
lines.append("")
lines.append("| Tag | Label | Within R² | Overall R² | F-stat | F p-val | N |")
lines.append("|:----|:------|:---------:|:----------:|-------:|--------:|--:|")
for r in results:
    lines.append(f"| {r['tag']} | {r['label']} | {r['within_r2']:.3f} | {r['overall_r2']:.3f} "
                 f"| {r['f_stat']:.2f} | {r['f_pval']:.2e} | {r['n_obs']:,} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Figures")
lines.append("")
lines.append("All figures saved to `outputs/regression/`:")
lines.append("")
lines.append("- `fig_regression_coefplot.png` — Coefficient estimates with 95% CIs for M1/M2/M3/M5")
lines.append("- `fig_regression_r2.png` — Within vs overall R² across all specifications")
lines.append("- `fig_regression_hazards.png` — Joint (M4) vs single-hazard (M9–M12) hazard coefficients")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Interpretation Notes")
lines.append("")
lines.append("- **M1 vs M2**: Adding country FE typically changes slope estimates substantially — ")
lines.append("  indicating meaningful between-country omitted variable bias in the pooled model.")
lines.append("- **M3 vs M2**: Two-way FE absorbs year trends. If year dummies matter, Within-R² should rise noticeably.")
lines.append("- **M4**: Disaggregating hazards tests whether averaging masks heterogeneous effects by hazard type.")
lines.append("- **M5**: Using `exposure_index` (PCA of pop/built_s/built_v/dem) as a single composite tests")
lines.append("  whether dimension reduction loses information vs keeping raw components.")
lines.append("- **M6/M7**: Comparing total losses vs damage vs mortality losses reveals whether economic")
lines.append("  vs human outcomes have different structural drivers.")
lines.append("- **M8**: The `hazard × log_gdp_pc` interaction tests the hypothesis that richer countries")
lines.append("  convert hazard exposure into lower losses (adaptive capacity gradient).")
lines.append("- **M9–M12**: Isolating each hazard removes between-hazard collinearity. If coefficients")
lines.append("  differ substantially from M4, multicollinearity is a concern.")

report_text = '\n'.join(lines)
report_path = REPORTS_DIR / '08_regression_suite.md'
with open(report_path, 'w') as f:
    f.write(report_text)

print(f"\nDone.")
print(f"  Report:  {report_path}")
print(f"  Figures: {DISPLAYS_DIR}/fig_regression_*.png")
