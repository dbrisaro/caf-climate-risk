import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

BASE    = Path('/home/jupyter-daniela/CAF')
OUT_DIR = BASE / 'outputs/exposure'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIS_CSV = BASE / 'data/OSM/resumen_descarga_osm_por_pais.csv'

CAT_COLORS = {
    'escuelas':   '#2b6cb0',
    'salud':      '#c0392b',
    'gobierno':   '#7d3c98',
    'emergencia': '#e67e22',
    'transporte': '#1e8449',
    'servicios':  '#bdbdbd',
}
CAT_LABELS = {
    'escuelas':   'Education',
    'salud':      'Health',
    'gobierno':   'Government',
    'emergencia': 'Emergency services',
    'transporte': 'Transport',
    'servicios':  'Other services',
}

def caf_ax(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, colors='#000000', labelsize=7.5)
    ax.set_axisbelow(True)

pais = pd.read_csv(PAIS_CSV)
pais = pais.sort_values('escuelas', ascending=False).reset_index(drop=True)

cats = ['escuelas','salud','gobierno','emergencia','transporte','servicios']

for c in cats:
    pais[f'pct_{c}'] = pais[c] / pais['TOTAL'] * 100

fig, ax = plt.subplots(figsize=(6.5, 4.8))
fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.22)

y     = range(len(pais))
ylbls = pais['pais'].tolist()

for i, row in pais.iterrows():
    x_acc = 0
    for c in cats:
        val = row[f'pct_{c}']
        ax.barh(i, val, left=x_acc, height=0.65,
                color=CAT_COLORS[c], linewidth=0, zorder=2)
        x_acc += val

ax.set_yticks(list(y))
ax.set_yticklabels(ylbls, fontsize=7.5, color='#000000')
ax.set_xlim(0, 100)
ax.set_xlabel('Share of facilities (%)', fontsize=8, color='#000000')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}%'))
ax.xaxis.grid(True, color='#e8e8e8', lw=0.5)

handles = [mpatches.Patch(color=CAT_COLORS[c], label=CAT_LABELS[c]) for c in cats]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fontsize=6.5, frameon=False, handlelength=1.0, handletextpad=0.4,
          ncol=3, columnspacing=1.0)
caf_ax(ax)

fig.savefig(OUT_DIR / 'fig_3_8_composition.png',
            dpi=180, bbox_inches='tight', facecolor='white')
print('Saved fig_3_8_composition.png')
plt.close()
