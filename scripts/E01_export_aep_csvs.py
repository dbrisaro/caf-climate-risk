import pandas as pd
from pathlib import Path

BASE = Path('/home/jupyter-daniela/CAF/outputs/losses')

mod = pd.read_csv(BASE / 'aep_modeled.csv')
mod_out = pd.DataFrame({
    'storm_name':       mod['name'].str.capitalize(),
    'year':             mod['year'].astype(int),
    'ss_category':      mod['category'],
    'return_period_yr': mod['rp'].round(1),
    'aep_pct':          mod['aep_pct'].round(2),
    'loss_low_USD_M':   (mod['loss_low']  / 1e6).round(1),
    'loss_mid_USD_M':   (mod['loss_mid']  / 1e6).round(1),
    'loss_high_USD_M':  (mod['loss_high'] / 1e6).round(1),
})
mod_out.to_csv(BASE / 'share_aep_modeled.csv', index=False)
print(f'share_aep_modeled.csv  — {len(mod_out)} rows')

phys = pd.read_csv(BASE / 'aep_emdat_physical.csv')
phys_out = pd.DataFrame({
    'year':                    phys['Start Year'].astype(int),
    'dominant_disaster_type':  phys['top_type'],
    'countries_affected':      phys['countries'],
    'return_period_yr':        phys['rp'].round(1),
    'aep_pct':                 phys['aep_pct'].round(2),
    'total_physical_damage_USD_M': phys['damage_M'].round(1),
})
phys_out.to_csv(BASE / 'share_aep_emdat_physical.csv', index=False)
print(f'share_aep_emdat_physical.csv  — {len(phys_out)} rows')

vsl = pd.read_csv(BASE / 'aep_emdat_vsl.csv')
vsl_out = pd.DataFrame({
    'year':                      vsl['Start Year'].astype(int),
    'dominant_disaster_type':    vsl['top_type'],
    'countries_affected':        vsl['countries'],
    'return_period_yr':          vsl['rp'].round(1),
    'aep_pct':                   vsl['aep_pct'].round(2),
    'total_economic_impact_USD_M': vsl['total_M'].round(1),
})
vsl_out.to_csv(BASE / 'share_aep_emdat_vsl.csv', index=False)
print(f'share_aep_emdat_vsl.csv  — {len(vsl_out)} rows')
