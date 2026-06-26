#!/usr/bin/env python3
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from pathlib import Path
import style
style.apply()

BASE    = Path("/home/jupyter-daniela/CAF")
EP_PATH = BASE / "data/processed/caribbean_adm_boundaries/exposure_panel.parquet"
SHP     = BASE / "data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp"
OUTDIR  = BASE / "outputs/exposure"
OUTDIR.mkdir(parents=True, exist_ok=True)

YEARS_ALL = [1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]

print("Loading data …")
ep  = pd.read_parquet(EP_PATH)
gdf = gpd.read_file(SHP).reset_index(drop=True)
shp_small = gdf[['region_nam','country']].rename(columns={'region_nam':'region_name'})
ep  = ep.merge(shp_small, on='region_name', how='left')

print("Computing additive index …")
def norm_col(series):
    log_s = np.log1p(series)
    lo, hi = np.percentile(log_s.dropna(), [2, 98])
    return np.clip((log_s - lo) / (hi - lo), 0, 1)

ep['pop_n']  = norm_col(ep['pop'])
ep['bs_n']   = norm_col(ep['built_s'])
ep['bv_n']   = norm_col(ep['built_v'])
ep['add_idx'] = (ep['pop_n'] + ep['bs_n'] + ep['bv_n']) / 3.0

lo_p, hi_p = np.percentile(ep['exposure_index'].dropna(), [2, 98])
ep['pca_idx'] = np.clip((ep['exposure_index'] - lo_p) / (hi_p - lo_p), 0, 1)

print(f"  PCA  range: [{ep['pca_idx'].min():.3f}, {ep['pca_idx'].max():.3f}]")
print(f"  Add  range: [{ep['add_idx'].min():.3f}, {ep['add_idx'].max():.3f}]")

gdf_wgs = gdf.to_crs("EPSG:4326")
gdf_wgs['country'] = gdf['country'].values

def pivot_col(col):
    piv = ep.drop_duplicates(['region_name','year']).pivot(
              index='region_name', columns='year', values=col)
    piv.columns = [f"{col}_{y}" for y in piv.columns]
    return piv

piv_pca = pivot_col('pca_idx')
piv_add = pivot_col('add_idx')
piv_all = pd.concat([piv_pca, piv_add], axis=1)

gdf_wgs = gdf_wgs.merge(
    piv_all.reset_index().rename(columns={'region_name': 'region_nam'}),
    on='region_nam', how='left'
)

gdf_wgs['delta_pca'] = (gdf_wgs['pca_idx_2030'] - gdf_wgs['pca_idx_1995']).clip(lower=0)
gdf_wgs['delta_add'] = (gdf_wgs['add_idx_2030'] - gdf_wgs['add_idx_1995']).clip(lower=0)

country_out = gdf_wgs.dissolve(by='country')

BOUNDS_D = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
N_D      = len(BOUNDS_D) - 1
cmap_d   = cm.get_cmap("YlOrRd", N_D)
norm_d   = mcolors.BoundaryNorm(BOUNDS_D, ncolors=N_D, clip=True)

cmap_pca, norm_pca, BOUNDS_PCA = cmap_d, norm_d, BOUNDS_D
cmap_add, norm_add, BOUNDS_ADD = cmap_d, norm_d, BOUNDS_D

BOUNDS_A = [i/10 for i in range(11)]
N_A      = len(BOUNDS_A) - 1
cmap_a   = cm.get_cmap("YlOrRd", N_A)
norm_a   = mcolors.BoundaryNorm(BOUNDS_A, ncolors=N_A, clip=True)

LON_SPAN, LAT_SPAN = 4.5, 3.0
_CTR  = {"Jamaica": (-77.25, 18.10), "Haiti": (-73.10, 19.05),
         "Dominican Republic": (-70.15, 18.75)}
ZOOM  = {n: (cx-LON_SPAN/2, cx+LON_SPAN/2, cy-LAT_SPAN/2, cy+LAT_SPAN/2)
         for n, (cx,cy) in _CTR.items()}
ZCOL  = {"Jamaica": "#2980B9", "Haiti": "#27AE60", "Dominican Republic": "#8E44AD"}
TAGS  = {"Jamaica": "a)", "Haiti": "b)", "Dominican Republic": "c)"}

print("Rendering compare_1_9 …")

def draw_change_panel(fig, gs, col_idx, data_col, title, cmap, norm, bounds):
    ax_main = fig.add_subplot(gs[:, col_idx*2])
    ax_ins  = [fig.add_subplot(gs[i, col_idx*2+1]) for i in range(3)]

    ax_main.set_facecolor("#cfe2f3")
    gdf_wgs.plot(data_col, ax=ax_main, cmap=cmap, norm=norm,
                 linewidth=0.15, edgecolor="#888888",
                 missing_kwds={"color":"#d0d0d0","edgecolor":"#888888"})
    country_out.boundary.plot(ax=ax_main, linewidth=0.5, edgecolor="#222222")
    for name, (lon1,lon2,lat1,lat2) in ZOOM.items():
        ax_main.add_patch(mpatches.Rectangle(
            (lon1,lat1), lon2-lon1, lat2-lat1,
            linewidth=1.0, edgecolor=ZCOL[name], facecolor="none", zorder=5))
    l,b,r,t = gdf_wgs.total_bounds
    ax_main.set_xlim(l-0.5, r+0.5); ax_main.set_ylim(b-0.5, t+0.5)
    ax_main.tick_params(labelsize=6, length=2)
    ax_main.set_title(title, fontsize=7, pad=3, loc="left")
    for sp in ax_main.spines.values(): sp.set_linewidth(0.4)

    for ax, name in zip(ax_ins, list(ZOOM.keys())):
        lon1,lon2,lat1,lat2 = ZOOM[name]
        sub = gdf_wgs[gdf_wgs['country']==name]
        ax.set_facecolor("#cfe2f3")
        sub.plot(data_col, ax=ax, cmap=cmap, norm=norm,
                 linewidth=0.4, edgecolor="#444444",
                 missing_kwds={"color":"#d0d0d0","edgecolor":"#444444"})
        sub.dissolve().boundary.plot(ax=ax, linewidth=0.6, edgecolor="#222222")
        ax.set_xlim(lon1,lon2); ax.set_ylim(lat1,lat2); ax.set_aspect("auto")
        ax.set_title(f"{TAGS[name]} {name.split()[0]}", loc="left", fontsize=6, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.8); sp.set_edgecolor(ZCOL[name])

    x0 = ax_main.get_position().x0
    x1 = ax_ins[0].get_position().x1
    cax = fig.add_axes([x0, 0.06, x1 - x0, 0.022])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal",
                       boundaries=bounds, ticks=bounds,
                       spacing="uniform", extend="max")
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.ax.set_xticklabels([f"{b:.2f}" for b in bounds], fontsize=5.5)
    cb.outline.set_visible(False)
    cb.set_label("Δ Exposure index (2030 − 1995)", fontsize=6, labelpad=3)

fig19 = plt.figure(figsize=(13.0, 5.4), facecolor="white")
gs19  = fig19.add_gridspec(3, 4, width_ratios=[2.2, 1, 2.2, 1],
                           hspace=0.12, wspace=0.18,
                           left=0.02, right=0.98, top=0.93, bottom=0.14)
draw_change_panel(fig19, gs19, 0, 'delta_pca', 'PCA index  —  Δ (2030 − 1995)',
                  cmap_pca, norm_pca, BOUNDS_PCA)
draw_change_panel(fig19, gs19, 1, 'delta_add', 'Additive index  —  Δ (2030 − 1995)',
                  cmap_add, norm_add, BOUNDS_ADD)

fig19.savefig(OUTDIR/"compare_1_9_methods.png", dpi=200,
              bbox_inches="tight", facecolor="white")
plt.close()
print("  → compare_1_9_methods.png")

print("Rendering compare_1_8 …")

dr = gdf_wgs[gdf_wgs['country']=="Dominican Republic"].copy()

fig18, axes = plt.subplots(2, 2, figsize=(9.0, 5.2), facecolor="white",
                           gridspec_kw={'hspace':0.06,'wspace':0.06})
configs18 = [
    ('pca_idx_1995', 'PCA — 1995',      axes[0,0]),
    ('pca_idx_2030', 'PCA — 2030',      axes[0,1]),
    ('add_idx_1995', 'Additive — 1995', axes[1,0]),
    ('add_idx_2030', 'Additive — 2030', axes[1,1]),
]
for col, title, ax in configs18:
    ax.set_facecolor("#cfe2f3")
    dr.plot(col, ax=ax, cmap=cmap_a, norm=norm_a,
            linewidth=0.2, edgecolor="#888",
            missing_kwds={"color":"#d0d0d0","edgecolor":"#888"})
    dr.dissolve().boundary.plot(ax=ax, linewidth=0.5, edgecolor="#222")
    l,b,r,t = dr.total_bounds
    ax.set_xlim(l-0.1,r+0.1); ax.set_ylim(b-0.1,t+0.1)
    ax.set_title(title, fontsize=7, pad=3, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.4)

sm2 = plt.cm.ScalarMappable(cmap=cmap_a, norm=norm_a); sm2.set_array([])
cb2 = fig18.colorbar(sm2, ax=axes, orientation="horizontal",
                     pad=0.04, shrink=0.6, aspect=35, ticks=[0,.2,.4,.6,.8,1])
cb2.ax.tick_params(labelsize=6); cb2.outline.set_visible(False)
cb2.set_label("Exposure index", fontsize=6.5)

fig18.savefig(OUTDIR/"compare_1_8_methods.png", dpi=200,
              bbox_inches="tight", facecolor="white")
plt.close()
print("  → compare_1_8_methods.png")

print("Rendering compare_1_10 …")

INSULAR     = ['Bahamas','Barbados','Dominica','Dominican Republic',
               'Haiti','Jamaica','Saint Lucia','Trinidad and Tobago']
CONTINENTAL = ['Colombia','Guyana','Peru']
ALL_C       = INSULAR + CONTINENTAL

COLORS = {'Barbados':'#C0392B','Trinidad and Tobago':'#8E44AD','Jamaica':'#2980B9',
          'Saint Lucia':'#16A085','Haiti':'#E07B30','Dominican Republic':'#27AE60',
          'Dominica':'#7F8C8D','Bahamas':'#BDC3C7',
          'Peru':'#B7950B','Colombia':'#784212','Guyana':'#1A5276'}
MARKERS = {'Barbados':'o','Trinidad and Tobago':'s','Jamaica':'^','Saint Lucia':'D',
           'Haiti':'v','Dominican Republic':'P','Dominica':'X','Bahamas':'*',
           'Peru':'o','Colombia':'s','Guyana':'^'}
LS = {c:'-' for c in INSULAR}; LS.update({c:'--' for c in CONTINENTAL})
SHORT = {'Dominican Republic':'Dom. Rep.','Trinidad and Tobago':'T&T',
         'Saint Lucia':'St. Lucia','Bahamas':'Bahamas','Barbados':'Barbados',
         'Dominica':'Dominica','Haiti':'Haiti','Jamaica':'Jamaica',
         'Peru':'Peru','Colombia':'Colombia','Guyana':'Guyana'}

def wmean(g, val_col):
    w = g['pop'].fillna(0).clip(lower=0)
    return np.average(g[val_col].fillna(0), weights=w) if w.sum() > 0 else g[val_col].mean()

def country_temporal(val_col):
    rows = []
    for country, grp in ep.groupby('country'):
        for year, sub in grp.groupby('year'):
            rows.append({'country': country, 'year': year,
                         'ei': wmean(sub, val_col)})
    return pd.DataFrame(rows)

nat_pca = country_temporal('pca_idx')
nat_add = country_temporal('add_idx')

def order_2030(nat):
    v = nat[nat['year']==2030].set_index('country')['ei']
    return v.reindex(ALL_C).dropna().sort_values(ascending=False).index.tolist()

ordered_pca = order_2030(nat_pca)
ordered_add = order_2030(nat_add)

PLOT_KW = dict(markersize=4, linewidth=1.1,
               markeredgewidth=0.4, markeredgecolor="white", zorder=3)

def make_ax(ax, nat, ordered, title):
    base = nat[nat['year']==1995].set_index('country')['ei']
    for c in ordered:
        cdf    = nat[nat['country']==c].set_index('year').reindex(YEARS_ALL)
        delta  = cdf['ei'] - base.get(c, np.nan)
        ax.plot(YEARS_ALL, delta,
                color=COLORS.get(c,'#999'), marker=MARKERS.get(c,'o'),
                linestyle=LS.get(c,'-'), **PLOT_KW)

    ax.axhline(0, color='#AAAAAA', linewidth=0.6, linestyle='--', zorder=1)
    ax.set_title(title, fontsize=7, pad=3, loc='left')
    ax.set_xticks(YEARS_ALL)
    ax.set_xticklabels([str(y) for y in YEARS_ALL], fontsize=6,
                        rotation=30, ha='right')
    ax.tick_params(labelsize=6, length=2, width=0.4)
    ax.yaxis.grid(True, linewidth=0.3, color='#EEEEEE', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.4); ax.spines['bottom'].set_linewidth(0.4)
    ax.set_xlim(1993, 2032)

fig110, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 3.8), facecolor="white")
make_ax(ax_l, nat_pca, ordered_pca, 'PCA index')
make_ax(ax_r, nat_add, ordered_add, 'Additive index')
ax_l.set_ylabel("Δ Exposure index (from 1995)", fontsize=7)

handles = [mlines.Line2D([],[],
           color=COLORS.get(c,'#999'), marker=MARKERS.get(c,'o'),
           linestyle=LS.get(c,'-'), markersize=3.5, linewidth=1.0,
           markeredgewidth=0.3, markeredgecolor='white', label=SHORT.get(c,c))
           for c in ordered_add if c != 'Peru' or
           not (nat_add[nat_add['country']!='Peru']['ei'].dropna().max() * 1.5 <
                nat_add[nat_add['country']=='Peru']['ei'].dropna().mean())]
ax_r.legend(handles=handles, fontsize=5.5, frameon=False, loc='upper left',
            ncol=2, handlelength=1.5, handletextpad=0.4,
            labelspacing=0.3, columnspacing=0.8)

fig110.tight_layout(pad=0.6)
fig110.savefig(OUTDIR/"compare_1_10_methods.png", dpi=200,
               bbox_inches="tight", facecolor="white")
plt.close()
print("  → compare_1_10_methods.png")
print("\nDone.")
