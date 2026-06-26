import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.append('/home/jupyter-daniela/CAF/src')

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from linearmodels import PooledOLS

BASE_DIR    = Path('/home/jupyter-daniela/CAF')
DATA_PATH   = BASE_DIR / 'data/processed/caribbean_adm_boundaries/master_panel.parquet'
SHAPEFILE   = BASE_DIR / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
IBTRACS_AGG = BASE_DIR / 'data/processed/caribbean_adm_boundaries/ibtracs_output/ibtracs_aggregated_data.parquet'
EXCESS_CACHE = BASE_DIR / 'data/processed/caribbean_adm_boundaries/hazard_excess.parquet'
OUT_DIR     = BASE_DIR / 'outputs/regression'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTINENTAL = ['Colombia', 'Peru', 'Guyana']
CARIBBEAN   = ['Bahamas', 'Barbados', 'Dominica', 'Dominican Republic',
               'Haiti', 'Jamaica', 'Saint Lucia', 'Trinidad and Tobago']

WB_ISO = {
    'Bahamas':             'BS', 'Barbados':            'BB',
    'Colombia':            'CO', 'Dominica':            'DM',
    'Dominican Republic':  'DO', 'Guyana':              'GY',
    'Haiti':               'HT', 'Jamaica':             'JM',
    'Peru':                'PE', 'Saint Lucia':         'LC',
    'Trinidad and Tobago': 'TT',
}

LABELS = {
    'built_s':   'Built surface',
    'built_v':   'Built volume',
    'drought':   'Drought',
    'rainfall':  'Rainfall',
    'heatwave':  'Heatwave',
    'wind_kt':   'Wind speed (kt)',
    'log_pop':   'log(Population)',
    'log_gdp':   'log(GDP per capita)',
}

M6_VARS = ['built_s', 'built_v', 'drought', 'rainfall', 'heatwave', 'wind_kt',
           'log_pop', 'log_gdp']

print("=" * 60)
print("07c — M6 por zona: Full / Caribe / Continental")
print("=" * 60)

print("\n[1] Cargando datos...")
df  = pd.read_parquet(DATA_PATH)
gdf = gpd.read_file(SHAPEFILE)[['region_nam', 'country']].rename(
        columns={'region_nam': 'region_name'})
df  = df.merge(gdf, on='region_name', how='left')

print("    Descargando GDP per capita...")
iso_str = ';'.join(WB_ISO.values())
wb_data = None
for attempt in range(3):
    try:
        wb_data = requests.get(
            f'https://api.worldbank.org/v2/country/{iso_str}/indicator/'
            f'NY.GDP.PCAP.KD?date=1995:2025&format=json&per_page=500', timeout=120
        ).json()[1]
        break
    except Exception as e:
        print(f"      Intento {attempt+1} fallido: {e}")
if wb_data is None:
    raise RuntimeError("World Bank API no disponible")

iso_to_country = {v: k for k, v in WB_ISO.items()}
df_gdp = pd.DataFrame([
    {'country': iso_to_country[r['country']['id']], 'year': int(r['date']), 'gdp_pc': r['value']}
    for r in wb_data if r['value'] is not None
])

panel = (
    df.groupby(['region_name', 'country', 'year'])
    .agg(
        total_losses_musd = ('total_losses_musd', 'sum'),
        built_s           = ('built_s',    'mean'),
        built_v           = ('built_v',    'mean'),
        pop               = ('pop',        'mean'),
        drought_raw       = ('drought',    'mean'),
        rainfall_raw      = ('rainfall',   'mean'),
        heatwave_raw      = ('heatwave',   'mean'),
    )
    .reset_index()
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

for h in ('drought', 'rainfall', 'heatwave'):
    panel[h] = panel[f'{h}_raw']

panel['log_loss'] = np.log1p(panel['total_losses_musd'])
panel['log_pop']  = np.log1p(panel['pop'])
panel['log_gdp']  = np.log(panel['gdp_pc'].clip(lower=1))

ALL_VARS = ['log_loss'] + M6_VARS
panel = (panel[panel['total_losses_musd'] > 0]
         .dropna(subset=ALL_VARS)
         .query('year <= 2025')
         .copy())

print(f"    Panel completo: {len(panel):,} obs · "
      f"{panel['region_name'].nunique()} regiones · "
      f"{panel['country'].nunique()} países")

def run_m6(data, label):
    d = data.copy().reset_index(drop=True)

    sd = d[M6_VARS].std(ddof=1)
    for v in M6_VARS:
        d[v] = d[v] / sd[v]

    dummies = pd.get_dummies(d['country'], drop_first=True, prefix='c').astype(float)
    d = pd.concat([d, dummies], axis=1)
    dummy_cols = dummies.columns.tolist()

    d = d.set_index(['region_name', 'year'])

    X = d[M6_VARS + dummy_cols].assign(const=1.0)
    mod = PooledOLS(d['log_loss'], X)
    res = mod.fit(cov_type='clustered', cluster_entity=False,
                  clusters=d['country'].astype('category').cat.codes)

    n_paises = data['country'].nunique()
    print(f"    [{label}]  R²={res.rsquared:.3f}  N={int(res.nobs):,}  "
          f"países={n_paises}  clusters={n_paises}")
    return res, sd

print("\n[2] Estimando modelos...")

carib_mask = panel['country'].isin(CARIBBEAN)
cont_mask  = panel['country'].isin(CONTINENTAL)

res_full,  sd_full  = run_m6(panel,               'Full')
res_carib, sd_carib = run_m6(panel[carib_mask],   'Caribe')
res_cont,  sd_cont  = run_m6(panel[cont_mask],    'Continental')

print("\n[3] Construyendo tabla comparativa...")

def stars(p):
    if p is None or np.isnan(p): return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

def extract(res, coef):
    try:
        return float(res.params[coef]), float(res.std_errors[coef]), float(res.pvalues[coef])
    except KeyError:
        return None, None, None

rows = []
for v in M6_VARS:
    row = {'variable': v}
    for tag, res in [('Full', res_full), ('Caribe', res_carib), ('Continental', res_cont)]:
        est, se, p = extract(res, v)
        row[f'{tag}_coef'] = round(est, 4) if est is not None else ''
        row[f'{tag}_se']   = round(se,  4) if se  is not None else ''
        row[f'{tag}_p']    = round(p,   4) if p   is not None else ''
    rows.append(row)

for tag, res in [('Full', res_full), ('Caribe', res_carib), ('Continental', res_cont)]:
    pass
rows.append({'variable': 'R²',
             'Full_coef': round(res_full.rsquared, 3),
             'Caribe_coef': round(res_carib.rsquared, 3),
             'Continental_coef': round(res_cont.rsquared, 3)})
rows.append({'variable': 'N',
             'Full_coef': int(res_full.nobs),
             'Caribe_coef': int(res_carib.nobs),
             'Continental_coef': int(res_cont.nobs)})

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_DIR / 'coefficients_split.csv', index=False)
print("    Guardado: coefficients_split.csv")

print()
print(f"{'Variable':<12} {'Full':>10} {'Caribe':>12} {'Continental':>14}")
print("-" * 52)
for v in M6_VARS:
    row = df_out[df_out['variable'] == v].iloc[0]
    def fmt(tag):
        c = row.get(f'{tag}_coef', '')
        p = row.get(f'{tag}_p', '')
        if c == '': return '   ---'
        s = stars(float(p)) if p != '' else ''
        return f"{float(c):+.3f}{s}"
    print(f"{LABELS[v]:<20} {fmt('Full'):>10} {fmt('Caribe'):>12} {fmt('Continental'):>14}")
print(f"{'R²':<20} {res_full.rsquared:>10.3f} {res_carib.rsquared:>12.3f} {res_cont.rsquared:>14.3f}")
print(f"{'N':<20} {int(res_full.nobs):>10,} {int(res_carib.nobs):>12,} {int(res_cont.nobs):>14,}")

print("\n[4] Forest plots...")

COLORS = {'Full': '#555555', 'Caribe': '#2166AC', 'Continental': '#D6604D'}

def make_forest(res, sd, label, color, out_path):
    ests   = [extract(res, v)[0] for v in M6_VARS]
    ci_lo  = [res.conf_int().loc[v, 'lower'] if v in res.params.index else None for v in M6_VARS]
    ci_hi  = [res.conf_int().loc[v, 'upper'] if v in res.params.index else None for v in M6_VARS]
    pvals  = [extract(res, v)[2] for v in M6_VARS]
    labels = [LABELS[v] for v in M6_VARS]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, (est, lo, hi, p) in enumerate(zip(ests, ci_lo, ci_hi, pvals)):
        if est is None:
            continue
        col = color if (p is not None and p < 0.05) else '#BBBBBB'
        ax.plot([lo, hi], [i, i], color=col, lw=1.8, solid_capstyle='round')
        ax.scatter(est, i, color=col, s=55, zorder=5)

    ax.axvline(0, color='black', lw=0.8, linestyle='--', alpha=0.6)
    ax.set_yticks(range(len(M6_VARS)))
    ax.set_yticklabels(labels, fontsize=10, fontweight='normal')
    ax.set_xlabel('Coeficiente estandarizado (VD: log pérdidas)', fontsize=9,
                  fontweight='normal')
    ax.tick_params(size=0)

    sig   = mpatches.Patch(color=color,   label='p < 0.05')
    ns    = mpatches.Patch(color='#BBBBBB', label='n.s.')
    leg = ax.legend(handles=[sig, ns], fontsize=8, loc='lower right',
                    frameon=False)

    ax.text(0.02, 0.02, f'N={int(res.nobs):,}  R²={res.rsquared:.3f}',
            transform=ax.transAxes, fontsize=8, color='#444444',
            va='bottom', fontweight='normal')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Guardado: {out_path.name}")

make_forest(res_full,  sd_full,  'Full',        COLORS['Full'],        OUT_DIR/'fig_split_full.png')
make_forest(res_carib, sd_carib, 'Caribe',      COLORS['Caribe'],      OUT_DIR/'fig_split_caribe.png')
make_forest(res_cont,  sd_cont,  'Continental', COLORS['Continental'], OUT_DIR/'fig_split_continental.png')

fig, axes = plt.subplots(1, 3, figsize=(6.5, 4.5), sharey=True)
subtitles = ['Muestra completa\n(11 países)',
             'Solo Caribe\n(8 países)',
             'Solo Continental\n(Colombia, Perú, Guyana)']

for ax, (tag, res), subtitle, color in zip(
        axes,
        [('Full', res_full), ('Caribe', res_carib), ('Continental', res_cont)],
        subtitles,
        [COLORS['Full'], COLORS['Caribe'], COLORS['Continental']]):

    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, v in enumerate(M6_VARS):
        est, se, p = extract(res, v)
        if est is None:
            continue
        lo = res.conf_int().loc[v, 'lower']
        hi = res.conf_int().loc[v, 'upper']
        col = color if (p is not None and p < 0.05) else '#CCCCCC'
        ax.plot([lo, hi], [i, i], color=col, lw=1.6, solid_capstyle='round')
        ax.scatter(est, i, color=col, s=45, zorder=5)

    ax.axvline(0, color='black', lw=0.7, linestyle='--', alpha=0.5)
    ax.tick_params(size=0)
    ax.set_xlabel('β estand.', fontsize=8, fontweight='normal')
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=8, fontweight='normal')
    ax.text(0.5, -0.08, f'N={int(res.nobs):,}  R²={res.rsquared:.3f}',
            transform=ax.transAxes, ha='center', fontsize=7.5,
            color='#555555', fontweight='normal')

axes[0].set_yticks(range(len(M6_VARS)))
axes[0].set_yticklabels([LABELS[v] for v in M6_VARS], fontsize=9,
                         fontweight='normal')

fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT_DIR / 'fig_split_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("    Guardado: fig_split_comparison.png")

print("\nListo. Outputs en outputs/regression/")
