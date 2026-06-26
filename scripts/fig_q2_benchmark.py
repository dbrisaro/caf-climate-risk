import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.append('/home/jupyter-daniela/CAF/src')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import style; style.apply()

BASE_DIR  = Path('/home/jupyter-daniela/CAF')
OSM_FILE  = BASE_DIR / 'data/OSM/resumen_descarga_osm_por_pais.csv'
OUT_DIR   = BASE_DIR / 'outputs/exposure'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CARIB_POP = {
    'Haiti':                11454,
    'Dominican Republic':   11117,
    'Trinidad and Tobago':   1408,
    'Jamaica':               2820,
    'Barbados':               281,
    'Bahamas':                393,
    'Saint Lucia':            180,
    'Dominica':                72,
}

OECD_HEALTH = {
    'France':       8.4,
    'Spain':        7.2,
    'United States':5.1,
    'Chile':        4.6,
    'Costa Rica':   4.1,
}

OECD_EDUCATION = {
    'France':        5.8,
    'Spain':         5.2,
    'United States': 4.8,
    'Chile':         6.3,
    'Costa Rica':    7.1,
}

osm = pd.read_csv(OSM_FILE)
carib = osm[osm['region'] == 'Caribe'].copy()
carib['pop_10k'] = carib['pais'].map(CARIB_POP)

carib['health_per10k']    = carib['salud']    / (carib['pop_10k'] / 10)
carib['education_per10k'] = carib['escuelas'] / (carib['pop_10k'] / 10)

carib = carib.sort_values('health_per10k', ascending=False)

print("Caribbean facilities per 10,000 inhabitants:")
print(carib[['pais', 'salud', 'escuelas', 'health_per10k', 'education_per10k']].to_string(index=False))

CARIB_COLOR = '#1971c2'
OECD_COLOR  = '#868e96'
COMP_COLOR  = '#2f9e44'

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, metric, oecd_dict, col, title, note in [
    (axes[0], 'health_per10k',    OECD_HEALTH,
     'salud',    'Health facilities per 10,000 inhabitants',
     'Hospitals, clinics, pharmacies (OSM) vs\nWHO/OECD national health system counts'),
    (axes[1], 'education_per10k', OECD_EDUCATION,
     'escuelas', 'Schools per 10,000 inhabitants',
     'Primary + secondary schools (OSM) vs\nUNESCO school density estimates'),
]:
    labels  = list(carib['pais'])
    values  = list(carib[metric])
    colors  = [CARIB_COLOR] * len(labels)

    oecd_sorted = sorted(oecd_dict.items(), key=lambda x: -x[1])
    for country, val in oecd_sorted:
        color = COMP_COLOR if country in ('Chile', 'Costa Rica') else OECD_COLOR
        ax.axvline(val, color=color, alpha=0.5, lw=1.4,
                   linestyle='--' if country in ('Chile', 'Costa Rica') else '-')
        ax.text(val + 0.05, len(labels) - 0.3, country,
                color=color, fontsize=7.5, va='top', style='italic')

    y = range(len(labels))
    ax.barh(list(y), values, color=colors, alpha=0.85, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Facilities per 10,000 inhabitants')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.text(0.98, 0.02, note, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=6.5, color='#666', style='italic')
    ax.xaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

carib_patch = mpatches.Patch(color=CARIB_COLOR, alpha=0.85, label='Caribbean countries (OSM)')
oecd_patch  = mpatches.Patch(color=OECD_COLOR,  alpha=0.7,  label='High-income reference (WHO/UNESCO)')
comp_patch  = mpatches.Patch(color=COMP_COLOR,  alpha=0.7,  label='Regional comparators (Chile, Costa Rica)')
fig.legend(handles=[carib_patch, oecd_patch, comp_patch],
           loc='lower center', ncol=3, fontsize=8.5, frameon=True,
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle('Caribbean social infrastructure density vs international benchmarks',
             fontsize=12, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_q2_benchmark_combined.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: fig_q2_benchmark_combined.png")

print("\n" + "="*72)
print(f"{'Country':<25} {'Pop (000s)':>10} {'Health fac':>12} {'per 10k':>8} {'Schools':>10} {'per 10k':>8}")
print("-"*72)
for _, row in carib.iterrows():
    print(f"  {row['pais']:<23} {CARIB_POP[row['pais']]:>10,}  "
          f"{int(row['salud']):>10,}  {row['health_per10k']:>8.2f}  "
          f"{int(row['escuelas']):>10,}  {row['education_per10k']:>8.2f}")
print("-"*72)
print("\nOECD / comparator benchmarks (health facilities per 10,000):")
for c, v in sorted(OECD_HEALTH.items(), key=lambda x: -x[1]):
    print(f"  {c:<20}: {v:.1f}")
print("\nOECD / comparator benchmarks (schools per 10,000):")
for c, v in sorted(OECD_EDUCATION.items(), key=lambda x: -x[1]):
    print(f"  {c:<20}: {v:.1f}")
