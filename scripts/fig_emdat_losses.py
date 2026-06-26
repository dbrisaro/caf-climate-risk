import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import logging
import sys
sys.path.append('/home/jupyter-daniela/CAF/src')
import style
style.apply()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

BASE_DIR  = Path('/home/jupyter-daniela/CAF')
GPKG_PATH = BASE_DIR / 'outputs/losses/emdat_carib_1900_2024.gpkg'
OUT_DIR   = BASE_DIR / 'outputs/losses'
OUT_DIR.mkdir(parents=True, exist_ok=True)

VSL_MAP = {
    'Haiti': 1.1, 'Dominican Republic': 1.1, 'Jamaica': 1.1,
    'Bahamas': 7.9, 'Dominica': 1.1, 'Barbados': 7.9,
    'Trinidad and Tobago': 7.9, 'Saint Lucia': 1.1,
}
VSL_DEFAULT = 1.1

if not GPKG_PATH.exists():
    raise FileNotFoundError(
        f"EM-DAT GeoPackage not found: {GPKG_PATH}\n"
        "  → Run prepare_emdat.py first."
    )

print("=" * 62)
print("fig_emdat_losses — EM-DAT Caribbean loss analysis")
print("=" * 62)

df = gpd.read_file(str(GPKG_PATH))
log.info("Loaded %d events from %s", len(df), GPKG_PATH.name)

df['vsl'] = df['Country'].map(VSL_MAP).fillna(VSL_DEFAULT)
df['total_damage_musd']  = df["Total Damage, Adjusted ('000 US$)"].fillna(0) / 1000
df['deaths_loss_musd']   = df['Total Deaths'].fillna(0) * df['vsl']
df['total_losses_musd']  = df['total_damage_musd'] + df['deaths_loss_musd']

log.info("Total losses: USD %.1f billion across %d events",
         df['total_losses_musd'].sum() / 1000, len(df))

losses = df['total_losses_musd'].dropna().sort_values(ascending=False).reset_index(drop=True)
n      = len(losses)
aep    = (np.arange(1, n + 1)) / (n + 1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(losses, aep, color='#1971c2', lw=1.2, label='Empirical AEP')

for pct, label in [(0.50, 'p50'), (0.90, 'p90'), (0.99, 'p99')]:
    val = losses.quantile(1 - pct)
    ax.axhline(pct, color='grey', lw=0.6, linestyle='--', alpha=0.7)
    ax.axvline(val, color='grey', lw=0.6, linestyle='--', alpha=0.7)
    ax.text(val + losses.max() * 0.01, pct + 0.01, f'{label}: {val:.0f} M', fontsize=7.5)

ax.set_xlabel('Total losses (USD million)', fontsize=10)
ax.set_ylabel('Annual Exceedance Probability', fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
out1 = OUT_DIR / 'fig_emdat_aep_curve.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
log.info("Saved: %s", out1)

country_summary = (
    df.groupby('Country')['total_losses_musd']
    .sum()
    .sort_values()
)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(country_summary.index, country_summary.values, color='#1971c2', alpha=0.8)
ax.set_xlabel('Total losses 1900–2024 (USD million)', fontsize=10)
for bar, val in zip(bars, country_summary.values):
    ax.text(val + country_summary.max() * 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:,.0f}', va='center', fontsize=8)
plt.tight_layout()
out2 = OUT_DIR / 'fig_emdat_loss_by_country.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
log.info("Saved: %s", out2)

print()
print("Done.  Outputs:")
print(f"  {out1}")
print(f"  {out2}")
