import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.append('/home/jupyter-daniela/CAF/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import style; style.apply()

BASE_DIR = Path('/home/jupyter-daniela/CAF')
OSM_CSV  = BASE_DIR / 'data/OSM/osm_top_ciudades_caribe_continentales.csv'
LOSS_CSV = BASE_DIR / 'outputs/losses/loss_by_type_storm.csv'
OUT_DIR  = BASE_DIR / 'outputs/losses'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CARIB_COUNTRIES = [
    'Haiti', 'Dominican Republic', 'Trinidad and Tobago', 'Jamaica',
    'Barbados', 'Bahamas', 'Saint Lucia', 'Dominica',
]

CATEGORIA_LABELS = {
    'salud':      'Health',
    'escuelas':   'Education',
    'gobierno':   'Government',
    'emergencia': 'Emergency services',
    'transporte': 'Transport',
    'servicios':  'Other services',
}
SOCIAL_CATS = ['salud', 'escuelas', 'gobierno', 'emergencia']

CAT_COLORS = {
    'salud':      '#e03131',
    'escuelas':   '#1971c2',
    'gobierno':   '#2f9e44',
    'emergencia': '#e67700',
}

AREA_M2 = {
    'salud':      800,
    'escuelas':   450,
    'gobierno':   600,
    'emergencia': 400,
    'transporte': 300,
    'servicios':  200,
}
COST_MID = {
    'salud':      1200,
    'escuelas':   950,
    'gobierno':   1200,
    'emergencia': 1000,
    'transporte': 600,
    'servicios':  450,
}
COST_LOW  = {k: v * 0.67 for k, v in COST_MID.items()}
COST_HIGH = {k: v * 1.33 for k, v in COST_MID.items()}

df_osm = pd.read_csv(OSM_CSV)
df_osm = df_osm[(df_osm['region'] == 'Caribe') & df_osm['pais'].isin(CARIB_COUNTRIES)].copy()
df_osm = df_osm.rename(columns={'pais': 'country'})

for scenario, cost_map in [('low', COST_LOW), ('mid', COST_MID), ('high', COST_HIGH)]:
    df_osm[f'rv_{scenario}'] = df_osm['categoria'].map(
        lambda c: AREA_M2.get(c, 300) * cost_map.get(c, 400)
    )

by_country_cat = (
    df_osm[df_osm.categoria.isin(SOCIAL_CATS)]
    .groupby(['country', 'categoria'])[['rv_low', 'rv_mid', 'rv_high']]
    .sum().reset_index()
)
by_country_cat.to_csv(OUT_DIR / 'social_exposure_by_country_type.csv', index=False)

print("=" * 65)
print("Replacement value at risk — social infrastructure (mid, USD M)")
print("=" * 65)
total_by_cat = df_osm[df_osm.categoria.isin(SOCIAL_CATS)].groupby('categoria')[['rv_mid', 'rv_low', 'rv_high']].sum()
for cat in SOCIAL_CATS:
    n = int((df_osm.categoria == cat).sum())
    mid = total_by_cat.loc[cat, 'rv_mid'] / 1e6
    lo  = total_by_cat.loc[cat, 'rv_low']  / 1e6
    hi  = total_by_cat.loc[cat, 'rv_high'] / 1e6
    print(f"  {CATEGORIA_LABELS[cat]:<22} n={n:>5,}  ${lo:.1f}M – ${mid:.1f}M – ${hi:.1f}M")
total_social = total_by_cat['rv_mid'].sum() / 1e6
print(f"  {'TOTAL SOCIAL':<22} n={int((df_osm.categoria.isin(SOCIAL_CATS)).sum()):>5,}  ${total_social:.1f}M (mid)")

pivot = (
    by_country_cat.pivot(index='country', columns='categoria', values='rv_mid')
    .fillna(0) / 1e6
)
pivot = pivot[[c for c in SOCIAL_CATS if c in pivot.columns]]
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(pivot))
for cat in SOCIAL_CATS:
    if cat not in pivot.columns:
        continue
    ax.bar(range(len(pivot)), pivot[cat], bottom=bottom,
           label=CATEGORIA_LABELS[cat], color=CAT_COLORS[cat], alpha=0.85, zorder=3)
    bottom += pivot[cat].values

ax.set_xticks(range(len(pivot)))
ax.set_xticklabels(pivot.index, rotation=20, ha='right')
ax.set_ylabel('Replacement value at risk (USD million, mid scenario)')
ax.set_title('Social infrastructure replacement value by country\n'
             '(8 Caribbean countries, city-level OSM inventory)')
ax.legend(title='Sub-type', loc='upper right', fontsize=8.5)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

for i, country in enumerate(pivot.index):
    total = pivot.loc[country].sum()
    ax.text(i, total + 0.5, f'${total:.0f}M', ha='center', va='bottom', fontsize=7.5)

fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_q3_replacement_value.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: fig_q3_replacement_value.png")

df_loss = pd.read_csv(LOSS_CSV)

BTYPE_IS_SOCIAL = {'school': True, 'government': True, 'other': False}
df_loss['is_social'] = df_loss['building_type'].map(BTYPE_IS_SOCIAL)

by_storm = df_loss.groupby(['storm_id', 'is_social'])['loss_mid'].sum().unstack(fill_value=0)
by_storm.columns = ['other', 'social']
by_storm['total'] = by_storm['social'] + by_storm['other']
by_storm['social_pct'] = 100 * by_storm['social'] / by_storm['total'].replace(0, np.nan)
by_storm = by_storm[by_storm['total'] > 0].sort_values('total', ascending=False)

print("\n" + "=" * 65)
print("Wind losses — social vs other buildings (full 1980–2025 catalog)")
print("=" * 65)
print(f"{'Storm':<22} {'Social':>12} {'Other':>12} {'Total':>12} {'Social%':>8}")
print("-" * 65)
for storm, row in by_storm.iterrows():
    print(f"  {storm:<20} ${row.social/1e6:>8.2f}M  ${row.other/1e6:>8.2f}M"
          f"  ${row.total/1e6:>8.2f}M  {row.social_pct:>7.1f}%")

total_social_loss = df_loss[df_loss.is_social]['loss_mid'].sum()
total_all_loss    = df_loss['loss_mid'].sum()
n_storms_total = df_loss['storm_id'].nunique()
print(f"\nAll {n_storms_total} storms: social = ${total_social_loss/1e6:.2f}M  |  "
      f"total = ${total_all_loss/1e6:.2f}M  |  "
      f"social share = {100*total_social_loss/total_all_loss:.1f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

storms = by_storm.index.tolist()
x = range(len(storms))
ax1.bar(x, by_storm['other'] / 1e6, label='Other buildings', color='#adb5bd', alpha=0.85, zorder=3)
ax1.bar(x, by_storm['social'] / 1e6, bottom=by_storm['other'] / 1e6,
        label='Social infrastructure', color='#e03131', alpha=0.85, zorder=3)
ax1.set_xticks(list(x))
ax1.set_xticklabels(storms, rotation=30, ha='right', fontsize=7.5)
ax1.set_ylabel('Wind losses — mid scenario (USD million)')
ax1.set_title('Total wind losses by storm\n(social vs other buildings)')
ax1.legend(fontsize=8.5)
ax1.yaxis.grid(True, alpha=0.3, zorder=0)
ax1.set_axisbelow(True)

ax2.barh(range(len(storms)), by_storm['social_pct'], color='#e03131', alpha=0.85, zorder=3)
ax2.set_yticks(range(len(storms)))
ax2.set_yticklabels(storms, fontsize=7.5)
ax2.set_xlabel('Social infrastructure share of total losses (%)')
ax2.set_title('Social infrastructure loss share by storm')
ax2.axvline(100 * total_social_loss / total_all_loss, color='#333', lw=1.2, linestyle='--',
            label=f'Average: {100*total_social_loss/total_all_loss:.1f}%')
ax2.legend(fontsize=8)
ax2.xaxis.grid(True, alpha=0.3, zorder=0)
ax2.set_axisbelow(True)

fig.suptitle('Social infrastructure wind losses — full 1980–2025 catalog (B05)\n'
             '8 Caribbean study countries, 916,772 buildings',
             fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_q3_loss_share.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_q3_loss_share.png")
