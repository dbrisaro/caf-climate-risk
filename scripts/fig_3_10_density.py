import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

BASE    = Path('/home/jupyter-daniela/CAF')
OUT_DIR = BASE / 'outputs/exposure'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITY_CSV = BASE / 'outputs/exposure/city_beneficiary_indicators.csv'

COUNTRY_COLORS = {
    'Bahamas':            '#1f78b4',
    'Barbados':           '#33a02c',
    'Dominica':           '#6a3d9a',
    'Dominican Republic': '#e31a1c',
    'Haiti':              '#ff7f00',
    'Jamaica':            '#b15928',
    'Saint Lucia':        '#a6cee3',
    'Trinidad and Tobago':'#b2df8a',
    'Colombia':           '#fb9a99',
    'Guyana':             '#fdbf6f',
    'Peru':               '#cab2d6',
}

def caf_ax(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, colors='#000000', labelsize=7.5)
    ax.set_axisbelow(True)

def pop_fmt(x, _):
    if x < 0.5:  return f'{x * 1000:.0f}k'
    elif x < 5:  return f'{x:.1f}M'
    else:        return f'{int(round(x))}M'

AP = dict(arrowstyle='-', color='#cccccc', lw=0.5, shrinkA=3, shrinkB=3)

LABELS_310 = {
    'Castries':       (  6,  5),
    'Lima':           (  6,  5),
    'Linden':         (  6,  0),
    'Trujillo':       (  6,  0),
    'Bridgetown':     (  6,  5),
    'Roseau':         (-45,  5),
    'Old Harbour':    (  6, -5),
    'Port-de-Paix':   (  6,  5),
    'Port of Spain':  (  6,  5),
    'Port-au-Prince': (  6, -6),
    'Nassau':         (  6,  5),
    'Kingston':       (  6, -5),
    'Bogota':         (-38,  6),
}

city = pd.read_csv(CITY_CSV)
city = city.dropna(subset=['facilities_per_10k_urban', 'urban_pop_ghsl2025'])

s_sqrt = city['n_total_facilities'] ** 0.5
city['_s'] = 35 + (s_sqrt - s_sqrt.min()) / (s_sqrt.max() - s_sqrt.min()) * 340

fig, ax = plt.subplots(figsize=(6.5, 5.0))
fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.26)

for country in sorted(city['country'].unique()):
    grp = city[city['country'] == country]
    ax.scatter(grp['urban_pop_ghsl2025'] / 1e6,
               grp['facilities_per_10k_urban'],
               s=grp['_s'],
               color=COUNTRY_COLORS.get(country, '#aaa'),
               alpha=0.85, linewidths=0.4, edgecolors='white',
               zorder=3, label=country)

ax.set_xscale('log')
ax.set_xlabel('Urban population', fontsize=8, color='#000000')
ax.set_ylabel('Social infrastructure facilities per 10,000 inhabitants',
              fontsize=8, color='#000000')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(pop_fmt))
ax.grid(True, color='#e8e8e8', lw=0.5, which='major', zorder=1)

for _, row in city.iterrows():
    if row['city'] in LABELS_310:
        dx, dy = LABELS_310[row['city']]
        ax.annotate(row['city'],
                    xy=(row['urban_pop_ghsl2025'] / 1e6, row['facilities_per_10k_urban']),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=5.5, color='#333333', va='center',
                    fontweight='normal', arrowprops=AP)

handles = [mpatches.Patch(color=COUNTRY_COLORS[c], label=c)
           for c in sorted(city['country'].unique())]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.14),
          fontsize=6, frameon=False, handlelength=1.0, handletextpad=0.4,
          borderpad=0, ncol=6)

caf_ax(ax)
ax.spines['left'].set_visible(True)
ax.spines['left'].set_color('#aaaaaa')
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_color('#aaaaaa')
fig.savefig(OUT_DIR / 'fig_3_10_density.png',
            dpi=180, bbox_inches='tight', facecolor='white')
print('Saved fig_3_10_density.png')
plt.close()
