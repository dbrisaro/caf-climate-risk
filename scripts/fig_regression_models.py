import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from pathlib import Path
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

BASE_DIR = Path('/home/jupyter-daniela/CAF')
FONT = "Ubuntu"
C_M1 = "#1B4F8A"; C_M2 = "#1A7A5E"; C_M3 = "#C05B1A"; C_NS = "#CCCCCC"

def tobit_loglik(params, y, X):
    beta = params[:-1]; sigma = np.exp(params[-1])
    xb   = X @ beta;    unc   = y > 0
    ll   = 0.0
    if unc.any():
        ll += stats.norm.logpdf((y[unc]-xb[unc])/sigma).sum() - np.log(sigma)*unc.sum()
    if (~unc).any():
        ll += stats.norm.logcdf(-xb[~unc]/sigma).sum()
    return -ll

def tobit_fit(y, X, names):
    n, k = X.shape
    mask = y > 0
    b0   = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
    s0   = np.log(max(np.std(y[mask] - X[mask]@b0), 1e-4))
    res  = optimize.minimize(tobit_loglik, np.append(b0,s0), args=(y,X),
                             method='L-BFGS-B', options={'maxiter':8000,'ftol':1e-12})
    from scipy.optimize import approx_fprime
    eps = 1e-5
    H = np.zeros((k+1, k+1))
    g0 = approx_fprime(res.x, lambda p: tobit_loglik(p,y,X), eps)
    for j in range(k+1):
        dx      = np.zeros(k+1); dx[j] = eps
        gj      = approx_fprime(res.x+dx, lambda p: tobit_loglik(p,y,X), eps)
        H[:,j]  = (gj - g0) / eps
    H = (H + H.T) / 2
    try:    cov = np.linalg.inv(H)
    except: cov = np.eye(k+1)*np.nan
    se    = np.sqrt(np.abs(np.diag(cov)))[:-1]
    betas = res.x[:-1]
    tstat = betas / np.where(se>0, se, 1e-12)
    pvals = 2 * stats.norm.sf(np.abs(tstat))
    return dict(zip(names, zip(betas, se, pvals)))

shp = gpd.read_file(BASE_DIR / "data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp")
rc  = shp[["region_nam","country"]].rename(columns={"region_nam":"region_name"})
df  = pd.read_parquet(BASE_DIR / "data/processed/caribbean_adm_boundaries/master_panel.parquet")
df  = df.merge(rc, on="region_name", how="left").drop_duplicates(subset=["region_name","year"])

cy = df.groupby(["country","year"]).agg(
    total_losses=("total_losses_musd","mean"),
    exposure=("exposure_index","mean"),
    rainfall=("rainfall","mean"),
    cyclone=("cyclone","mean"),
).reset_index()
cy["hazard"]     = (cy["rainfall"]+cy["cyclone"])/2
cy["log_losses"] = np.log(cy["total_losses"]+1)
cy["hazard_sc"]  = cy["hazard"]*1000
cy["exposure_sc"]= cy["exposure"]*100

base = "Trinidad and Tobago"
dummies    = pd.get_dummies(cy["country"], drop_first=False, dtype=float).drop(columns=[base])
dummy_cols = list(dummies.columns)
cy = pd.concat([cy.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
y  = cy["log_losses"].values.astype(float)

X1 = np.column_stack([np.ones(len(y)), cy["hazard_sc"].values])
r1 = tobit_fit(y, X1, ["const","hazard"])

X2 = np.column_stack([np.ones(len(y)), cy["hazard_sc"].values, cy["exposure_sc"].values])
r2 = tobit_fit(y, X2, ["const","hazard","exposure"])

X3 = np.column_stack([np.ones(len(y))] + [cy[c].values for c in ["hazard_sc","exposure_sc"]+dummy_cols])
r3 = tobit_fit(y, X3, ["const","hazard","exposure"]+dummy_cols)

for tag, res in [("M1",r1),("M2",r2),("M3",r3)]:
    print(f"\n{tag}:")
    for k,(b,s,p) in res.items():
        if k=="const": continue
        star = "***" if p<0.01 else "**" if p<0.05 else "†" if p<0.1 else ""
        print(f"  {k:30s} β={b:+.3f}  SE={s:.3f}  p={p:.3f} {star}")

SHORT = {"Dominican Republic":"Dom. Republic","Trinidad and Tobago":"Trinidad (base)","Saint Lucia":"St. Lucia"}

fig, axes = plt.subplots(1,3, figsize=(17,6),
    gridspec_kw={"width_ratios":[1,1.2,1.8]})
fig.patch.set_alpha(0)

def dot_panel(ax, entries, color, title, xlim=None):
    ax.set_facecolor("none")
    for i,(lbl,b,s,p) in enumerate(entries):
        c = color if p < 0.1 else C_NS
        ax.errorbar(b, i, xerr=[[max(b-b+s,0) if False else s],[s]],
                    fmt="o", color=c, ecolor=c, capsize=5, markersize=7, lw=2, zorder=4)
        ax.plot([b-1.96*s, b+1.96*s],[i,i], color=c, lw=1.5, zorder=3)
        star = "***" if p<0.01 else "**" if p<0.05 else "†" if p<0.1 else "n.s."
        ax.text(b+1.96*s+0.02, i, f"β={b:+.3f} {star}", va="center",
                fontsize=8.5, fontfamily=FONT, color="#1A1A2E")
    ax.axvline(0, color="#888", lw=1, ls="--", alpha=0.7)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e[0] for e in entries], fontsize=9.5, fontfamily=FONT)
    ax.set_xlabel("Tobit coefficient (β)  ·  DV: log(losses+1)", fontsize=9, fontfamily=FONT)
    if xlim: ax.set_xlim(xlim)

dot_panel(axes[0],
    [("Hazard intensity", *r1["hazard"])],
    C_M1, "M1: Hazard only\n(Tobit, N=288)")
axes[0].set_title('a', loc='left', fontweight='normal', fontsize=10)

dot_panel(axes[1],
    [("Hazard intensity", *r2["hazard"]),
     ("Exposure index",   *r2["exposure"])],
    C_M2, "M2: Hazard + Exposure\n(Tobit, N=288)")
axes[1].set_title('b', loc='left', fontweight='normal', fontsize=10)

fe_entries = []
for c in ["hazard","exposure"]+dummy_cols:
    b,s,p = r3[c]
    lbl = {"hazard":"Hazard intensity","exposure":"Exposure index"}.get(c, SHORT.get(c,c))
    fe_entries.append((lbl, b, s, p))

top = fe_entries[:2]
countries_fe = sorted(fe_entries[2:], key=lambda x: x[1])
all_entries = top + countries_fe

dot_panel(axes[2], all_entries, C_M3,
    "M3: Hazard + Exposure + Country FE\n(Tobit, N=288)",
    xlim=(min(b-1.96*s for _,b,s,_ in all_entries)-1.5,
          max(b+1.96*s for _,b,s,_ in all_entries)+3.5))
axes[2].set_title('c', loc='left', fontweight='normal', fontsize=10)

plt.tight_layout(pad=2.5)
out = BASE_DIR / "outputs/regression/fig_regression_models.png"
plt.savefig(out, dpi=180, bbox_inches="tight", transparent=True)
print(f"\nSaved → {out}")
plt.close()
