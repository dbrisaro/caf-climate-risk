import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path('/home/jupyter-daniela/CAF')
OUT_DIR  = BASE_DIR / 'outputs/losses'
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_MIN, YEAR_MAX = 1980, 2025
T         = YEAR_MAX - YEAR_MIN + 1
SUBGROUPS = ['Meteorological', 'Hydrological', 'Climatological']

ISOS = {
    'HTI': ('Haiti',                1.1),
    'DOM': ('Dominican Republic',   1.1),
    'JAM': ('Jamaica',              1.1),
    'BHS': ('Bahamas',              7.9),
    'DMA': ('Dominica',             1.1),
    'BRB': ('Barbados',             7.9),
    'TTO': ('Trinidad and Tobago',  7.9),
    'LCA': ('Saint Lucia',          1.1),
    'COL': ('Colombia',             1.1),
    'PER': ('Peru',                 1.1),
    'GUY': ('Guyana',               1.1),
}

xlsx = list((BASE_DIR / 'data/EMDAT').glob('*.xlsx'))[0]
df   = pd.read_excel(xlsx)

rows = []
for iso, (name, vsl) in ISOS.items():
    sub = df[
        (df['ISO'] == iso) &
        (df['Start Year'] >= YEAR_MIN) &
        (df['Start Year'] <= YEAR_MAX) &
        (df['Disaster Subgroup'].isin(SUBGROUPS))
    ]
    avg_deaths = sub['Total Deaths'].fillna(0).sum() / T
    avg_damage = sub["Total Damage, Adjusted ('000 US$)"].fillna(0).sum() / 1000 / T
    mort_cost  = round(avg_deaths * vsl, 2)
    total      = round(mort_cost + avg_damage, 2)
    rows.append({
        'Country':                                name,
        'Average Annual Fatalities':              round(avg_deaths, 2),
        'Base VSL (Million US$)':                 vsl,
        'Annual Social Cost of Mortality (M US$)': mort_cost,
        'Average Annual Physical Damage (M US$)': round(avg_damage, 2),
        'Total Economic Impact (M US$)':          total,
    })

tab = pd.DataFrame(rows)
tab = tab.sort_values('Total Economic Impact (M US$)', ascending=False).reset_index(drop=True)

csv_path = OUT_DIR / 'tab_4_1_country_impacts.csv'
tab.to_csv(csv_path, index=False)
print(f'Saved: {csv_path}')
print(tab.to_string(index=False))

col_fmt = 'lrrrrrr'
latex = tab.to_latex(
    index=False,
    float_format='%.2f',
    column_format=col_fmt,
    caption='Average annual disaster impacts by country estimated from the EM-DAT database '
            'for 1980 to 2025, including fatalities, monetized social costs of mortality '
            'using Value of Statistical Life estimates, physical damages, and total economic '
            'impact. Values expressed in million US dollars.',
    label='tab:4_1_country_impacts',
    escape=True,
)
tex_path = OUT_DIR / 'tab_4_1_country_impacts.tex'
tex_path.write_text(latex)
print(f'Saved: {tex_path}')
