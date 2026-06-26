#!/usr/bin/env python3
import sys; sys.path.append('/home/jupyter-daniela/CAF/src')
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import style
style.apply()

BASE    = Path('/home/jupyter-daniela/CAF')
OUTDIR  = BASE / 'outputs/regression'
OUTDIR.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(BASE / 'outputs/regression/fixed_effects_clean.csv')

fe_rows = []
for _, row in raw.iterrows():
    alpha  = float(row['fixed_effect'])
    se     = row['se']
    is_ref = np.isnan(se) if isinstance(se, float) else False
    lo = hi = alpha
    if not is_ref:
        se    = float(se)
        lo    = alpha - 1.96 * se
        hi    = alpha + 1.96 * se
    fe_rows.append({'country': row['country'], 'alpha': alpha,
                    'lo': lo, 'hi': hi, 'is_ref': is_ref,
                    'gdp_pc': float(row['gdp_per_capita_usd'])})

fe = pd.DataFrame(fe_rows).sort_values('alpha', ascending=True).reset_index(drop=True)

SIG_COL = '#2166AC'
REF_COL = '#888888'

fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor='white')
for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_linewidth(0.5)

for i, row in fe.iterrows():
    col = REF_COL if row['is_ref'] else SIG_COL
    if not row['is_ref']:
        ax.plot([row['lo'], row['hi']], [i, i],
                color=col, lw=1.5, solid_capstyle='round', zorder=3)
    ax.scatter(row['alpha'], i, color=col, s=40, zorder=4,
               edgecolors='white', linewidths=0.4,
               marker='D' if row['is_ref'] else 'o')

    x_ann = max(row['hi'], row['alpha']) + 0.03
    gdp_str = f"${row['gdp_pc']:,.0f}" if not np.isnan(row['gdp_pc']) else ''
    ax.text(x_ann, i, gdp_str, va='center', ha='left',
            fontsize=6.5, color='#555555', fontweight='normal')

ax.axvline(0, color='#333333', lw=0.7, linestyle='--', zorder=2)
ax.set_yticks(range(len(fe)))
ax.set_yticklabels(fe['country'], fontsize=8, fontweight='normal')
ax.set_xlabel('Country fixed effect  (ref: Bahamas)', fontsize=7,
              labelpad=4, fontweight='normal')
ax.tick_params(axis='x', labelsize=6.5, length=3, width=0.5)
ax.tick_params(axis='y', length=0)
ax.xaxis.grid(True, linewidth=0.3, color='#EEEEEE', zorder=0)
ax.set_axisbelow(True)

ref_patch = mpatches.Patch(color=REF_COL, label='Reference (Bahamas)')
sig_patch = mpatches.Patch(color=SIG_COL, label='Other countries (95% CI)')
ax.legend(handles=[ref_patch, sig_patch], fontsize=6, frameon=False,
          loc='lower right', handlelength=1.0, handletextpad=0.4)

fig.tight_layout(pad=0.5)
out = OUTDIR / 'fig_2_2_fixed_effects.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved → {out}")
