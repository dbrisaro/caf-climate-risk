import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

BASE = Path('/home/jupyter-daniela/CAF')
OUT  = BASE / 'outputs/reports/report_data_all_countries.xlsx'

EMDAT_PATH = list((BASE / 'data/EMDAT').glob('*.xlsx'))[0]

T         = 45
SUBGROUPS = ['Meteorological', 'Hydrological', 'Climatological']
STUDY     = list(range(1995, 2026))
BASELINE  = list(range(1995, 2010))
YEARS_EP  = [1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]

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

COUNTRIES_EMDAT = [
    'Bahamas', 'Barbados', 'Dominica', 'Dominican Republic',
    'Haiti', 'Jamaica', 'Saint Lucia', 'Trinidad and Tobago',
    'Colombia', 'Guyana', 'Peru',
]

SHP_PATH = BASE / 'data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp'
gdf_shp  = gpd.read_file(SHP_PATH)[['region_nam', 'country']].rename(
               columns={'region_nam': 'region_name'})
reg2ctr  = gdf_shp.set_index('region_name')['country'].to_dict()

print("Building sheets …")
writer = pd.ExcelWriter(OUT, engine='xlsxwriter')

print("  Loading EM-DAT …")
emdat_raw = pd.read_excel(EMDAT_PATH, sheet_name='EM-DAT Data')
emdat_ctr = emdat_raw[emdat_raw['Country'].isin(COUNTRIES_EMDAT)].copy()

print("  Fig_1_1_EmDAT_Types …")
CAT_FILTERS = [
    ('Tropical cyclone',
     lambda d: (d['Disaster Type'] == 'Storm') & (d['Disaster Subtype'] == 'Tropical cyclone')),
    ('Flood (all types)',
     lambda d: d['Disaster Type'] == 'Flood'),
    ('Drought',
     lambda d: d['Disaster Type'] == 'Drought'),
    ('Storm (general)',
     lambda d: (d['Disaster Type'] == 'Storm') & (d['Disaster Subtype'] != 'Tropical cyclone')),
    ('Landslide',
     lambda d: d['Disaster Type'].isin(['Mass movement (wet)', 'Mass movement (dry)'])),
    ('Wildfire',
     lambda d: d['Disaster Type'] == 'Wildfire'),
]
rows11 = []
for cat_label, filt in CAT_FILTERS:
    counts = emdat_ctr[filt(emdat_ctr)].groupby('Country').size()
    for country in COUNTRIES_EMDAT:
        rows11.append({
            'country': country,
            'disaster_category': cat_label,
            'n_events': int(counts.get(country, 0)),
        })
(pd.DataFrame(rows11)
   .sort_values(['country', 'disaster_category'])
   .to_excel(writer, sheet_name='Fig_1_1_EmDAT_Types', index=False))
del rows11, counts

print("  Fig_1_2_EmDAT_Events …")
fig12 = (emdat_ctr.groupby('Country')
                  .size()
                  .reindex(COUNTRIES_EMDAT)
                  .reset_index(name='total_events')
                  .rename(columns={'Country': 'country'})
                  .sort_values('total_events', ascending=False))
fig12.to_excel(writer, sheet_name='Fig_1_2_EmDAT_Events', index=False)
del fig12

print("  Fig_1_3_EmDAT_Annual …")
df13 = emdat_ctr[emdat_ctr['Start Year'].between(2000, 2025)].copy()

def _cat13(row):
    t, s = row['Disaster Type'], row['Disaster Subtype']
    if t == 'Storm' and s == 'Tropical cyclone': return 'Tropical cyclone'
    if t == 'Flood':  return 'Flood (all types)'
    if t == 'Drought': return 'Drought'
    return None

df13['category'] = df13.apply(_cat13, axis=1)
df13 = df13[df13['category'].notna()]
fig13 = (df13.groupby(['Start Year', 'category'])
             .size()
             .unstack(fill_value=0)
             .reindex(index=range(2000, 2026), fill_value=0)
             .reset_index()
             .rename(columns={'Start Year': 'year'}))
for col in ['Tropical cyclone', 'Flood (all types)', 'Drought']:
    if col not in fig13.columns:
        fig13[col] = 0
fig13.to_excel(writer, sheet_name='Fig_1_3_EmDAT_Annual', index=False)
del df13, fig13

print("  Fig_1_5_Drought …")
df_dr  = pd.read_parquet(BASE / 'data/processed/caribbean_adm_boundaries/era5_output/droughts_aggregated_data.parquet')
jja15  = df_dr[(df_dr['window'] == 'JJA') & (df_dr['year'] == 2015)]
piv    = (jja15.pivot_table(index='region_name', columns='activacion',
                             values='n_drought_pixels', fill_value=0)
               .reindex(columns=['p1', 'p5', 'p10'], fill_value=0))

def worst_level(row):
    if row['p1']  > 0: return 'Extreme'
    if row['p5']  > 0: return 'Severe'
    if row['p10'] > 0: return 'Moderate'
    return 'No drought'

piv['drought_category'] = piv.apply(worst_level, axis=1)
fig15 = (piv.reset_index()
            .assign(country=lambda d: d['region_name'].map(reg2ctr))
            [['country', 'region_name', 'p10', 'p5', 'p1', 'drought_category']]
            .rename(columns={'p10': 'n_pixels_moderate', 'p5': 'n_pixels_severe', 'p1': 'n_pixels_extreme'})
            .sort_values(['country', 'region_name']))
fig15.to_excel(writer, sheet_name='Fig_1_5_Drought', index=False)
del df_dr, jja15, piv, fig15

print("  Fig_1_6_Rainfall …")
df_rr = pd.read_parquet(
    BASE / 'data/processed/caribbean_adm_boundaries/era5_output_precipitation/rainfall_aggregated_data.parquet')
df_rr['country'] = df_rr['region_name'].map(reg2ctr)
n_regions = df_rr.groupby('country')['region_name'].nunique().rename('n_regions')
ext = df_rr[(df_rr['activacion'] == 'p90') & (df_rr['year'].isin(STUDY))]
above = (ext[ext['n_rainfall_pixels'] > 0]
             .groupby(['country', 'year'])['region_name'].nunique()
             .rename('n_regions_above').reset_index())
fig16 = (above.merge(n_regions, on='country')
              .assign(pct_territory_p90=lambda d: (d['n_regions_above'] / d['n_regions'] * 100).round(2))
              [['country', 'year', 'pct_territory_p90']]
              .sort_values(['country', 'year']))
fig16.to_excel(writer, sheet_name='Fig_1_6_Rainfall', index=False)
del df_rr, ext, above, fig16

print("  Fig_1_7_Heatwave …")
df_hw = pd.read_parquet(
    BASE / 'data/processed/caribbean_adm_boundaries/heatwaves_computation/heatwaves_aggregated_data.parquet')
df_hw['country'] = df_hw['region_name'].map(reg2ctr)
df_hw = df_hw.rename(columns={'year_win': 'year'})
total_ctr = (df_hw[df_hw['year'].isin(BASELINE)]
                 .groupby(['country', 'year'])['n_event_days'].sum()
                 .groupby('country').mean() / 0.10)
hw = (df_hw[df_hw['year'].isin(STUDY) & df_hw['activacion'].isin(['p90', 'p95', 'p99'])]
          .groupby(['country', 'year', 'window'])['n_event_days'].sum()
          .reset_index())
hw['frac'] = (hw['n_event_days'] / hw['country'].map(total_ctr)).clip(upper=1)
fig17 = (hw.groupby(['country', 'year'])['frac']
            .max().reset_index()
            .rename(columns={'frac': 'heatwave_area_fraction'})
            .assign(heatwave_area_pct=lambda d: (d['heatwave_area_fraction'] * 100).round(2))
            .sort_values(['country', 'year']))
fig17.to_excel(writer, sheet_name='Fig_1_7_Heatwave', index=False)
del df_hw, hw, fig17

print("  Fig_1_9_Exposure …")
ep   = pd.read_parquet(BASE / 'data/processed/caribbean_adm_boundaries/exposure_panel.parquet')
ep['country'] = ep['region_name'].map(reg2ctr)
ep95 = ep[ep['year'] == 1995][['region_name', 'country', 'exposure_index']].rename(
           columns={'exposure_index': 'exposure_1995'})
ep30 = ep[ep['year'] == 2030][['region_name', 'exposure_index']].rename(
           columns={'exposure_index': 'exposure_2030'})
fig19 = (ep95.merge(ep30, on='region_name', how='inner')
             .assign(delta_exposure=lambda d: (d['exposure_2030'] - d['exposure_1995']).round(4))
             [['country', 'region_name', 'exposure_1995', 'exposure_2030', 'delta_exposure']]
             .sort_values(['country', 'region_name']))
fig19.to_excel(writer, sheet_name='Fig_1_9_Exposure', index=False)

print("  Fig_1_10_ExposureGrowth …")
ALL_C = ['Bahamas', 'Barbados', 'Dominica', 'Dominican Republic',
         'Haiti', 'Jamaica', 'Saint Lucia', 'Trinidad and Tobago',
         'Colombia', 'Guyana', 'Peru']
ep_m = ep.merge(gdf_shp, on='region_name', how='left', suffixes=('', '_shp'))
ep_m['country'] = ep_m['country'].where(ep_m['country'].notna(), ep_m['country_shp'])
ep_m = ep_m[ep_m['country'].isin(ALL_C) & ep_m['year'].isin(YEARS_EP)].copy()

lo, hi = np.percentile(ep_m['exposure_index'].dropna(), [2, 98])
ep_m['pca_idx'] = np.clip((ep_m['exposure_index'] - lo) / (hi - lo), 0, 1)

def _wmean(g):
    w = g['pop'].fillna(0).clip(lower=0)
    return (np.average(g['pca_idx'].fillna(0), weights=w)
            if w.sum() > 0 else g['pca_idx'].mean())

nat = (ep_m.groupby(['country', 'year']).apply(_wmean).reset_index(name='exposure_index_wtd'))
base = nat[nat['year'] == 1995].set_index('country')['exposure_index_wtd']
nat['delta_from_1995'] = (nat.apply(lambda r: r['exposure_index_wtd'] - base.get(r['country'], np.nan), axis=1)
                            .round(4))
nat['country_group'] = nat['country'].apply(
    lambda c: 'Insular Caribbean' if c not in ['Colombia', 'Guyana', 'Peru'] else 'Continental')
(nat[['country', 'country_group', 'year', 'exposure_index_wtd', 'delta_from_1995']]
   .sort_values(['country', 'year'])
   .to_excel(writer, sheet_name='Fig_1_10_ExposureGrowth', index=False))
del ep, ep95, ep30, ep_m, nat, fig19

print("  Tab_2_3_FixedEffects …")
fe = pd.read_csv(BASE / 'outputs/regression/fixed_effects_clean.csv')
fe['ci_95_lo'] = (fe['fixed_effect'] - 1.96 * fe['se']).round(4)
fe['ci_95_hi'] = (fe['fixed_effect'] + 1.96 * fe['se']).round(4)
fe['note'] = fe['p_value'].apply(
    lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else
              '†'   if p < 0.10  else '')
(fe[['country', 'fixed_effect', 'se', 'ci_95_lo', 'ci_95_hi', 'p_value', 'note', 'gdp_per_capita_usd']]
   .sort_values('fixed_effect', ascending=False)
   .to_excel(writer, sheet_name='Tab_2_3_FixedEffects', index=False))
del fe

print("  Fig_2_1_M6Coefs …")
M6_VARS = ['built_s', 'built_v', 'drought', 'rainfall', 'heatwave', 'wind_kt', 'log_pop', 'log_gdp']
m6c = pd.read_csv(BASE / 'outputs/regression/m6_carib_coefficients.csv')
m6c = m6c[m6c['variable'].isin(M6_VARS)].copy()
m6c['ci_95_lo'] = (m6c['coef'] - 1.96 * m6c['se']).round(4)
m6c['ci_95_hi'] = (m6c['coef'] + 1.96 * m6c['se']).round(4)
(m6c[['variable', 'coef', 'se', 'ci_95_lo', 'ci_95_hi', 'pvalue']]
   .rename(columns={'coef': 'coef_std', 'se': 'se_std', 'pvalue': 'p_value'})
   .sort_values('coef_std', ascending=False)
   .to_excel(writer, sheet_name='Fig_2_1_M6Coefs', index=False))
del m6c

print("  Fig_3_2_WindLosses …")
t51 = pd.read_csv(BASE / 'outputs/losses/table_5_1_wind_by_country_storm.csv')
for col in ['loss_low', 'loss_mid', 'loss_high']:
    t51[f'{col}_MUSD'] = (t51[col] / 1e6).round(3)
(t51[['country', 'storm_name', 'year', 'ss_category',
      'loss_low_MUSD', 'loss_mid_MUSD', 'loss_high_MUSD']]
   .sort_values(['country', 'loss_mid_MUSD'], ascending=[True, False])
   .to_excel(writer, sheet_name='Fig_3_2_WindLosses', index=False))
del t51

print("  Fig_3_3_TopEvents …")
t51b = pd.read_csv(BASE / 'outputs/losses/table_5_1_wind_by_country_storm.csv')
agg  = t51b.groupby(['storm_name', 'year', 'ss_category'])[['loss_low', 'loss_mid', 'loss_high']].sum().reset_index()
top5 = agg.nlargest(5, 'loss_mid').copy()
for col in ['loss_low', 'loss_mid', 'loss_high']:
    top5[f'{col}_BUSD'] = (top5[col] / 1e9).round(3)
(top5[['storm_name', 'year', 'ss_category',
       'loss_low_BUSD', 'loss_mid_BUSD', 'loss_high_BUSD']]
   .sort_values('loss_mid_BUSD', ascending=False)
   .to_excel(writer, sheet_name='Fig_3_3_TopEvents', index=False))
del t51b, agg, top5

print("  Fig_3_7_AEP_Modeled …")
aep_mod = pd.read_csv(BASE / 'outputs/losses/aep_modeled.csv')
for col in ['loss_low', 'loss_mid', 'loss_high']:
    aep_mod[f'{col}_MUSD'] = (aep_mod[col] / 1e6).round(3)
cols_mod = [c for c in ['name', 'year', 'category', 'rank', 'rp', 'aep_pct',
                          'loss_low_MUSD', 'loss_mid_MUSD', 'loss_high_MUSD']
              if c in aep_mod.columns]
aep_mod[cols_mod].to_excel(writer, sheet_name='Fig_3_7_AEP_Modeled', index=False)
del aep_mod

print("  Fig_3_8_InfraShare …")
pais = pd.read_csv(BASE / 'data/OSM/resumen_descarga_osm_por_pais.csv')
cats = ['escuelas', 'salud', 'gobierno', 'emergencia', 'transporte', 'servicios']
cat_en = {'escuelas': 'Education', 'salud': 'Health', 'gobierno': 'Government',
          'emergencia': 'Emergency services', 'transporte': 'Transport', 'servicios': 'Other services'}
rows38 = []
for _, row in pais.iterrows():
    for c in cats:
        n = row.get(c, 0)
        rows38.append({
            'country':           row['pais'],
            'region':            row.get('region', ''),
            'category':          cat_en[c],
            'n_facilities':      int(n),
            'total_facilities':  int(row.get('TOTAL', 0)),
            'share_pct':         round(n / row['TOTAL'] * 100, 2) if row.get('TOTAL', 0) > 0 else 0,
        })
(pd.DataFrame(rows38)
   .sort_values(['country', 'share_pct'], ascending=[True, False])
   .to_excel(writer, sheet_name='Fig_3_8_InfraShare', index=False))
del pais, rows38

print("  Fig_3_9_Students …")
city = pd.read_csv(BASE / 'outputs/exposure/city_beneficiary_indicators.csv')
cols39 = [c for c in ['region', 'country', 'city', 'urban_pop_ghsl2025',
                        'n_education_facilities', 'school_age_pop_517',
                        'students_total_est', 'students_per_edu_facility',
                        'pct_517_unwpp2024', 'ner_primary_pct', 'ner_secondary_pct']
           if c in city.columns]
(city[cols39]
   .sort_values('urban_pop_ghsl2025', ascending=False)
   .to_excel(writer, sheet_name='Fig_3_9_Students', index=False))

print("  Fig_3_10_Density …")
cols310 = [c for c in ['region', 'country', 'city', 'urban_pop_ghsl2025',
                         'n_health_facilities', 'n_education_facilities',
                         'n_total_facilities', 'facilities_per_10k_urban',
                         'exceeds_who_10k', 'health_catchment_pop']
            if c in city.columns]
city310 = city[cols310].copy()
city310['health_per_10k'] = (city['n_health_facilities'] / city['urban_pop_ghsl2025'] * 10000).round(2)
city310['edu_per_10k']    = (city['n_education_facilities'] / city['urban_pop_ghsl2025'] * 10000).round(2)
(city310.sort_values('urban_pop_ghsl2025', ascending=False)
        .to_excel(writer, sheet_name='Fig_3_10_Density', index=False))
del city, city310

print("  Fig_3_11_ReplVal …")
df_osm = pd.read_csv(BASE / 'data/OSM/osm_top_ciudades_caribe_continentales.csv')
RC_MID  = {'salud': 800*1200, 'escuelas': 450*950,
            'gobierno': 600*1200, 'emergencia': 400*1000}
CAT_EN  = {'salud': 'Health', 'escuelas': 'Education',
            'gobierno': 'Government', 'emergencia': 'Emergency services'}
SOCIAL  = list(RC_MID.keys())
df_osm['rv_mid'] = df_osm['categoria'].map(RC_MID)
agg = (df_osm[df_osm['categoria'].isin(SOCIAL)]
           .groupby(['pais', 'categoria'])['rv_mid'].sum().reset_index())
agg['rv_mid_m']    = (agg['rv_mid'] / 1e6).round(2)
agg['category_en'] = agg['categoria'].map(CAT_EN)
fig311 = (agg[['pais', 'category_en', 'rv_mid_m']]
              .rename(columns={'pais': 'country', 'category_en': 'sector',
                               'rv_mid_m': 'replacement_value_mid_MUSD'})
              .sort_values(['country', 'sector']))
fig311.to_excel(writer, sheet_name='Fig_3_11_ReplVal', index=False)
del df_osm, agg, fig311

print("  Fig_3_12_Density …")
bene = pd.read_csv(BASE / 'outputs/exposure/city_beneficiary_indicators.csv')
ca   = bene.groupby('country').agg(
    urban_pop   =('urban_pop_ghsl2025', 'sum'),
    n_health    =('n_health_facilities', 'sum'),
    n_education =('n_education_facilities', 'sum'),
).reset_index()
ca['health_per_10k'] = (ca['n_health']    / ca['urban_pop'] * 10000).round(2)
ca['edu_per_10k']    = (ca['n_education'] / ca['urban_pop'] * 10000).round(2)
ca['group'] = 'Study country'

benchmarks = pd.DataFrame([
    ('France',         8.4,  5.8),
    ('Spain',          7.2,  5.2),
    ('United States',  5.1,  4.8),
    ('Chile',          4.6,  6.3),
    ('Costa Rica',     4.1,  7.1),
], columns=['country', 'health_per_10k', 'edu_per_10k'])
benchmarks['group'] = 'International benchmark'

fig312 = (pd.concat([ca[['country', 'group', 'n_health', 'n_education', 'urban_pop',
                          'health_per_10k', 'edu_per_10k']],
                     benchmarks], ignore_index=True)
            .sort_values(['group', 'health_per_10k'], ascending=[True, False]))
fig312.to_excel(writer, sheet_name='Fig_3_12_Density', index=False)
del bene, ca, benchmarks, fig312

print("  AEP_Physical + AEP_VSL …")
pd.read_csv(BASE / 'outputs/losses/aep_emdat_physical.csv').to_excel(
    writer, sheet_name='AEP_Physical', index=False)
pd.read_csv(BASE / 'outputs/losses/aep_emdat_vsl.csv').to_excel(
    writer, sheet_name='AEP_VSL', index=False)

print("  Table_4_1 …")
rows = []
for iso, (name, vsl) in ISOS.items():
    sub = emdat_raw[
        (emdat_raw['ISO'] == iso) &
        (emdat_raw['Start Year'] >= 1980) &
        (emdat_raw['Start Year'] <= 2025) &
        (emdat_raw['Disaster Subgroup'].isin(SUBGROUPS))
    ]
    avg_deaths = sub['Total Deaths'].fillna(0).sum() / T
    avg_damage = sub["Total Damage, Adjusted ('000 US$)"].fillna(0).sum() / 1000 / T
    mort_cost  = round(avg_deaths * vsl, 2)
    total      = round(mort_cost + avg_damage, 2)
    rows.append({
        'Country':                                name,
        'Avg Annual Fatalities':                  round(avg_deaths, 2),
        'Base VSL (M USD)':                       vsl,
        'Annual Social Cost of Mortality (M USD)': mort_cost,
        'Avg Annual Physical Damage (M USD)':     round(avg_damage, 2),
        'Total Economic Impact (M USD)':          total,
    })
(pd.DataFrame(rows)
   .sort_values('Total Economic Impact (M USD)', ascending=False)
   .to_excel(writer, sheet_name='Table_4_1', index=False))
del emdat_raw, emdat_ctr, rows

writer.close()
print(f"\nSaved → {OUT}")
print("Sheets:")
import openpyxl
wb = openpyxl.load_workbook(OUT, read_only=True)
for i, name in enumerate(wb.sheetnames, 1):
    print(f"  {i:2d}. {name}")
wb.close()
