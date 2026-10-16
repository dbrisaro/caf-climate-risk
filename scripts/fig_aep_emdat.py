import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

BASE_DIR = Path('/home/jupyter-daniela/CAF')
OUT_DIR  = BASE_DIR / 'outputs/losses'
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOSS_DIR = BASE_DIR / 'outputs/losses'

YEAR_MIN, YEAR_MAX = 1980, 2025
T    = YEAR_MAX - YEAR_MIN + 1
ISOS = ['HTI','DOM','JAM','BHS','DMA','BRB','TTO','LCA','COL','PER','GUY']
VSL  = {'HTI':1.1,'DOM':1.1,'JAM':1.1,'BHS':7.9,
        'DMA':1.1,'BRB':7.9,'TTO':7.9,'LCA':1.1,
        'COL':1.1,'PER':1.1,'GUY':1.1}

SUBGROUPS = ['Meteorological', 'Hydrological', 'Climatological']

xlsx = list((BASE_DIR / 'data/EMDAT').glob('*.xlsx'))[0]
df   = pd.read_excel(xlsx)

sub = df[
    (df['ISO'].isin(ISOS)) &
    (df['Start Year'] >= YEAR_MIN) &
    (df['Start Year'] <= YEAR_MAX) &
    (df['Disaster Subgroup'].isin(SUBGROUPS))
].copy()

sub['VSL_M']        = sub['ISO'].map(VSL)
sub['damage_M']     = sub["Total Damage, Adjusted ('000 US$)"].fillna(0) / 1000
sub['deaths']       = sub['Total Deaths'].fillna(0)
sub['mortality_M']  = sub['deaths'] * sub['VSL_M']
sub['total_M']      = sub['damage_M'] + sub['mortality_M']

ev = (sub.groupby(['DisNo.', 'Start Year', 'Disaster Type'])
         .agg(damage_M   =('damage_M',    'sum'),
              mortality_M=('mortality_M', 'sum'),
              total_M    =('total_M',     'sum'),
              deaths     =('deaths',      'sum'),
              countries  =('Country', lambda x: ', '.join(sorted(x.unique()))))
         .reset_index())

ann = (ev.groupby('Start Year')
         .agg(damage_M   =('damage_M',    'sum'),
              mortality_M=('mortality_M', 'sum'),
              total_M    =('total_M',     'sum'),
              n_events   =('DisNo.',      'count'),
              top_type   =('Disaster Type', lambda x: x.value_counts().index[0]),
              countries  =('countries', lambda x: ', '.join(
                               sorted(set(c for v in x for c in v.split(', ')))))[:20])
         .reset_index())

all_years = pd.DataFrame({'Start Year': range(YEAR_MIN, YEAR_MAX + 1)})
ann = all_years.merge(ann, on='Start Year', how='left').fillna(0)

def compute_aep_annual(col, df=ann, t=T):
    d = df[df[col] > 0][['Start Year', col, 'top_type', 'countries']].copy()
    d = d.sort_values(col, ascending=False).reset_index(drop=True)
    d['rank']    = d.index + 1
    d['rp']      = t / d['rank']
    d['aep_pct'] = d['rank'] / t * 100
    return d

ev_sorted_phys = compute_aep_annual('damage_M')
ev_sorted_vsl  = compute_aep_annual('total_M')

cols_phys = ['Start Year','top_type','countries','damage_M','rank','rp','aep_pct']
cols_vsl  = ['Start Year','top_type','countries','total_M','rank','rp','aep_pct']

ev_sorted_phys[cols_phys].to_csv(LOSS_DIR / 'aep_emdat_physical.csv', index=False)
ev_sorted_vsl [cols_vsl ].to_csv(LOSS_DIR / 'aep_emdat_vsl.csv',      index=False)

aal_phys = ev['damage_M'].sum()  / T
aal_vsl  = ev['total_M'].sum()   / T

modeled  = pd.read_csv(LOSS_DIR / 'aep_modeled.csv')
aal_mod  = modeled['loss_mid'].sum() / T

summary = pd.DataFrame({
    'source':    ['Modeled B06 (wind, physical)',
                  'EM-DAT (physical damage only)',
                  'EM-DAT (physical + VSL mortality)'],
    'AAL_M_USD': [aal_mod/1e6, aal_phys, aal_vsl],
    'note': ['Wind losses to critical infrastructure only',
             'All atmospheric & climatological disasters',
             'All atmospheric & climatological disasters']
})
summary.to_csv(LOSS_DIR / 'aep_shareable_summary.csv', index=False)
print('CSVs saved.')

def style_ax(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(length=3, labelsize=7.5, colors='#000000')
    ax.grid(True, which='major', color='#e8e8e8', lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y,_: f'{int(y)}%'))
    ax.set_ylabel('Annual exceedance probability (%)', fontsize=8)

def fmt_x(ax, xmax):
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    ax.set_xlabel('Loss (USD millions)', fontsize=8)

def annotate_top(ax, ev_df, col, n=5):
    top  = ev_df.head(n).sort_values(col).reset_index(drop=True)
    xmax = ev_df[col].max()
    x_txt = np.linspace(xmax * 0.15, xmax * 0.88, len(top))
    y_txt = np.linspace(34, 16, len(top))
    for i, row in top.iterrows():
        lbl = f"{row['top_type']} {int(row['Start Year'])}\n{str(row['countries'])[:22]}"
        ax.annotate(lbl,
                    xy=(row[col], row['aep_pct']),
                    xytext=(x_txt[i], y_txt[i]),
                    fontsize=5.5, color='#333333',
                    ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-', color='#bbbbbb', lw=0.5,
                                    shrinkA=0, shrinkB=2))

fig, ax = plt.subplots(figsize=(6.5, 4.2))
fig.subplots_adjust(left=0.13, right=0.95, top=0.96, bottom=0.13)

color = '#8b1a1a'
xs = ev_sorted_phys['damage_M'].values
ys = ev_sorted_phys['aep_pct'].values

ax.plot(xs, ys, color=color, lw=1.4, zorder=3)
ax.scatter(xs, ys, color=color, s=18, zorder=4)
ax.axvline(aal_phys, color=color, lw=0.8, ls='--', alpha=0.7)
ax.text(aal_phys * 1.03, ys.max() * 0.95,
        f'AAL ${aal_phys:.0f}M/yr',
        fontsize=6.5, color=color, va='top')
annotate_top(ax, ev_sorted_phys, 'damage_M')

style_ax(ax)
fmt_x(ax, ev_sorted_phys['damage_M'].max() * 1.15)

fig.savefig(OUT_DIR / 'fig_aep_emdat_physical.png',
            dpi=180, bbox_inches='tight', facecolor='white')
print(f'Figure → fig_aep_emdat_physical.png')
plt.close()

fig, ax = plt.subplots(figsize=(6.5, 4.2))
fig.subplots_adjust(left=0.13, right=0.95, top=0.96, bottom=0.13)

color2 = '#5c3317'
xs_v = ev_sorted_vsl['total_M'].values
ys_v = ev_sorted_vsl['aep_pct'].values

ax.plot(xs_v, ys_v, color=color2, lw=1.4, zorder=3)
ax.scatter(xs_v, ys_v, color=color2, s=18, zorder=4)
ax.axvline(aal_vsl, color=color2, lw=0.8, ls='--', alpha=0.7)
ax.text(aal_vsl * 1.03, ys_v.max() * 0.95,
        f'AAL ${aal_vsl:.0f}M/yr',
        fontsize=6.5, color=color2, va='top')
annotate_top(ax, ev_sorted_vsl, 'total_M')

style_ax(ax)
fmt_x(ax, ev_sorted_vsl['total_M'].max() * 1.15)
ax.set_xlabel('Physical damage + mortality cost (USD millions)', fontsize=8)

fig.savefig(OUT_DIR / 'fig_aep_emdat_vsl.png',
            dpi=180, bbox_inches='tight', facecolor='white')
print(f'Figure → fig_aep_emdat_vsl.png')
plt.close()

print(f'\nAAL modeled:        ${aal_mod/1e6:.0f} M/yr')
print(f'AAL EM-DAT physical: ${aal_phys:.0f} M/yr')
print(f'AAL EM-DAT + VSL:    ${aal_vsl:.0f} M/yr')
