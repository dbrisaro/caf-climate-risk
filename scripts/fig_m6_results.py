import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE_DIR = Path('/home/jupyter-daniela/CAF')
FE_PATH   = BASE_DIR / 'outputs/regression/m6_carib_fixed_effects.csv'
CF_PATH   = BASE_DIR / 'outputs/regression/m6_carib_coefficients.csv'
DEC_PATH  = BASE_DIR / 'outputs/regression/m6_carib_decomposition.csv'
OUT_FIG  = BASE_DIR / 'outputs/regression'
OUT_REG  = BASE_DIR / 'outputs/regression'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

fe  = pd.read_csv(FE_PATH)
cf  = pd.read_csv(CF_PATH)
dec = pd.read_csv(DEC_PATH).set_index('group')

print("=" * 62)
print("Fixed effects — M6 (reference: Bahamas)")
print("=" * 62)
print(f"  {'Country':<25} {'α_i':>7} {'SE':>7} {'p':>7} {'GDP/cap':>9}")
print("  " + "-" * 57)
for _, r in fe.iterrows():
    alpha = f"{r['fixed_effect']:.4f}" if pd.notna(r['fixed_effect']) else "0.0000 (ref)"
    se    = f"{r['se']:.4f}" if pd.notna(r['se']) else "—"
    p     = f"{r['p_value']:.4f}" if pd.notna(r['p_value']) else "—"
    gdp   = f"${r['gdp_per_capita_usd']:,.0f}"
    print(f"  {r['country']:<25} {alpha:>12} {se:>7} {p:>7} {gdp:>9}")

def sig(p):
    if pd.isna(p): return ''
    p = float(p)
    if p < 0.001: return '$^{***}$'
    if p < 0.01:  return '$^{**}$'
    if p < 0.05:  return '$^{*}$'
    if p < 0.10:  return '$^{\\dagger}$'
    return ''

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Country fixed effects, model M6. Reference category: Bahamas. '
    r'GDP per capita (2015 USD, World Bank). '
    r'Standard errors clustered at country level.}',
    r'\label{tab:fixed_effects_m6}',
    r'\small',
    r'\begin{tabular}{lrrrr}',
    r'\toprule',
    r'Country & $\hat{\alpha}_i$ & SE & $p$ & GDP per capita (USD) \\',
    r'\midrule',
]
for _, r in fe.iterrows():
    is_ref    = pd.isna(r['se'])
    alpha_str = f"{r['fixed_effect']:.4f}{sig(r['p_value'])}" if not is_ref else r'0.0000 \textit{(ref.)}'
    se_str    = f"{r['se']:.4f}" if not is_ref else '—'
    p_str     = f"{r['p_value']:.4f}" if not is_ref else '—'
    gdp_str   = f"\\${r['gdp_per_capita_usd']:,.0f}"
    lines.append(f"{r['country']} & {alpha_str} & {se_str} & {p_str} & {gdp_str} \\\\")

lines += [
    r'\bottomrule',
    r'\multicolumn{5}{l}{\footnotesize $^{***}p<0.001$, $^{**}p<0.01$, '
    r'$^{*}p<0.05$, $^{\dagger}p<0.10$.} \\',
    r'\end{tabular}',
    r'\end{table}',
]
with open(OUT_REG / 'fixed_effects_table.tex', 'w') as f:
    f.write('\n'.join(lines))
print("\n  Saved: fixed_effects_table.tex")

print("  (fixed_effects_clean.csv written by 07d — no rewrite needed)")

print("\n[Fig 1] M6 coefficient dot plot …")

VAR_LABELS = {
    'built_s':   'Built surface (built_s)',
    'built_v':   'Built volume (built_v)',
    'drought':   'Drought index',
    'rainfall':  'Rainfall index',
    'heatwave':  'Heatwave index',
    'wind_kt':   'Wind speed (kt)',
    'log_pop':   'log(Population)',
    'log_gdp':   'log(GDP per capita)',
}
GROUP = {
    'built_s':  'Built environment',
    'built_v':  'Built environment',
    'drought':  'Hazard',
    'rainfall': 'Hazard',
    'heatwave': 'Hazard',
    'wind_kt':  'Hazard',
    'log_pop':  'Controls',
    'log_gdp':  'Controls',
}
COLORS = {
    'Built environment': '#2166ac',
    'Hazard':            '#d6604d',
    'Controls':          '#888888',
}

m6 = (cf[cf['variable'].isin(VAR_LABELS)]
      [['variable', 'coef', 'se']]
      .dropna(subset=['coef'])
      .copy())
m6['label'] = m6['variable'].map(VAR_LABELS)
m6['group'] = m6['variable'].map(GROUP)
m6['ci95']  = 1.96 * m6['se']
m6 = m6.sort_values('coef', ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7, 5.5))

for i, row in m6.iterrows():
    color = COLORS[row['group']]
    ax.plot([row['coef'] - row['ci95'], row['coef'] + row['ci95']],
            [i, i], color=color, lw=2.2, alpha=0.6, solid_capstyle='round')
    ax.scatter(row['coef'], i, color=color, s=55, zorder=5,
               edgecolors='white', linewidths=0.6)

ax.axvline(0, color='#333333', lw=0.8, linestyle='--', alpha=0.6)

ax.set_yticks(range(len(m6)))
ax.set_yticklabels(m6['label'], fontsize=9.5)

for i, row in m6.iterrows():
    coef = row['coef']
    offset = 0.025 if coef >= 0 else -0.025
    ha = 'left' if coef >= 0 else 'right'
    ax.text(coef + offset, i, f"{coef:+.3f}", va='center',
            fontsize=8, color='#333333')

ax.set_xlabel('Standardized coefficient (β)', fontsize=10)
ax.set_xlim(m6['coef'].min() - m6['ci95'].max() - 0.15,
            m6['coef'].max() + m6['ci95'].max() + 0.20)
ax.tick_params(axis='x', labelsize=9)
ax.grid(axis='x', alpha=0.25, linewidth=0.4, zorder=0)

handles = [mpatches.Patch(color=COLORS[g], label=g)
           for g in ['Built environment', 'Hazard', 'Controls']]
ax.legend(handles=handles, fontsize=9, loc='lower right', framealpha=0.95)

fig.tight_layout()
out1 = OUT_FIG / 'm6_coefficients.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out1.name}")

print("\n[Fig 2] Within-country variance decomposition …")

r2_full          = float(dec.loc['r2_full',  'partial_r2'])
partial_built    = float(dec.loc['built',    'partial_r2'])
partial_hazard   = float(dec.loc['hazard',   'partial_r2'])
partial_controls = float(dec.loc['controls', 'partial_r2'])
shares = {
    'Built\nenvironment': float(dec.loc['built',    'share']),
    'Hazard':             float(dec.loc['hazard',   'share']),
    'Controls':           float(dec.loc['controls', 'share']),
}
colors_bar = ['#2166ac', '#d6604d', '#888888']

fig, ax = plt.subplots(figsize=(5.5, 4.5))

keys   = list(shares.keys())
vals   = [shares[k] * 100 for k in keys]
bars   = ax.bar(keys, vals, color=colors_bar, width=0.5,
                edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f'{val:.1f}%', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#222222')

ax.set_ylabel('Share of within-country R²  (%)', fontsize=10)
ax.set_ylim(0, 88)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=9)
ax.grid(axis='y', alpha=0.3, linewidth=0.4, zorder=0)

ax.text(0, vals[0] / 2, f'ΔR²={partial_built:.4f}', ha='center', va='center',
        fontsize=8, color='white', fontweight='bold')
ax.text(1, vals[1] / 2, f'ΔR²={partial_hazard:.4f}', ha='center', va='center',
        fontsize=8, color='white', fontweight='bold')

ax.annotate(
    f'Within-R² (M6) = {r2_full:.4f}\n'
    f'N=2,583  |  271 regions  |  8 countries\n'
    f'PanelOLS entity FE, country-clustered SE',
    xy=(0.97, 0.97), xycoords='axes fraction',
    ha='right', va='top', fontsize=7.5, color='#555555',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5',
              edgecolor='#cccccc', alpha=0.9)
)

fig.tight_layout()
out2 = OUT_FIG / 'variance_decomposition.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out2.name}")

print("\nDone.")
