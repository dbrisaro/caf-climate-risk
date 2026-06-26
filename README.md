# CAF — Caribbean Climate Risk & Infrastructure Loss Analysis

Multi-hazard climate risk pipeline for the Caribbean, covering regional vulnerability decomposition (macro/statistical track) and storm-level infrastructure loss estimation (physical/cyclone track).

---

## Repository structure

```
CAF/
├── scripts/          # Analysis pipeline (run in order) + standalone figure scripts
├── src/              # Shared Python modules (utils, aggregation, exposure_utils, hazard_utils)
├── data/             # All inputs (raw + processed)
├── outputs/          # ALL generated outputs — no generic figures/ folder; each theme has its own dir
│   ├── exposure/     # GHS built-up/population maps + OSM infrastructure maps
│   ├── hazards/      # Rainfall, drought, heatwave diagnostics
│   ├── cyclone/      # Trajectories, wind footprints, flood timeline, service disruption
│   │   ├── trajectories/  # IBTrACS track maps (B01)
│   │   └── footprints/    # CLIMADA wind footprints — emdat/ and all/ (B02)
│   ├── vulnerability/ # Decomposition, coefficients, composite risk index
│   ├── regression/   # Panel regression models, forest plots, coefficient tables
│   ├── losses/       # Loss CSVs, AEP curves, wind loss figures
│   └── reports/      # Generated Markdown reports
├── docs/             # Manually maintained methodology notes (not pipeline outputs)
└── cache/            # Memoization cache (auto-generated, do not edit)
```

---

## Two pipeline tracks

### Track A — Macro / Statistical
Builds a region-year panel and estimates vulnerability through regression.

```
00_compute_hazards_raw.py          ERA5 raw hazard data per admin region
        ↓                            (rainfall, drought, heatwave) — Caribbean only
SA01_hazards_append.py            [continental extension] ERA5 hazards for Colombia, Peru,
        ↓                            Guyana ONLY → appends to Caribbean aggregated files.
        ↓                            Run instead of 00→03 when Caribbean data is already computed.
01_prepare_emdat.py               Cleans EM-DAT xlsx → outputs/losses/emdat_carib_1900_2024.gpkg
        ↓
02_compute_exposure.py             Population + built-up surface/volume from GHS + MERIT
        ↓
03_aggregate_hazards.py            Seasonal aggregates (SON, DJF, MAM, JJA) — full run
        ↓
04_build_panel.py                  Merges everything → master_panel.parquet
        ↓
05_vulnerability_decomposition.py  Hazard / Exposure / Residual decomposition
        ↓
06_add_emdat_losses.py             Joins EM-DAT losses to panel (fuzzy geo-matching)
        ↓
07a_vulnerability_coefficients.py   Country FE regression → vulnerability coefficients
07b_regression_panel.py            Full panel regression suite (M1–M6), forest plots,
        ↓                            stability plots, publication tables
07c_regression_split.py            [parallel] M6 split by Caribbean vs. Continental
        ↓                            → outputs/regression/ (coefficients_split.csv + forest plots)
08_regression_suite.py             12-specification regression suite → outputs/reports/
        ↓
09_composite_risk_index.py         PCA composite risk index
```

**Outputs go to:** `outputs/exposure/` · `outputs/vulnerability/` · `outputs/regression/` · `outputs/reports/` · `data/processed/caribbean_adm_boundaries/` (parquets)

---

### Track B — Physical / Cyclone
Generates storm-level wind footprints and translates them to infrastructure losses.

```
B01_cyclone_trajectory_analysis.py  IBTrACS track maps matched to EM-DAT events
        ↓
B02_cyclone_footprint.py            CLIMADA v_max wind grids (0.1°) for 2021–2025
        ↓
B03_micro_flood_analysis.py         Micro-level flood exposure via OSM + MERIT DEM
        ↓
B04_service_disruption.py           Service disruption from flood depth × OSM facilities
        ↓                            Also produces: fig_sdi_by_city, fig_elevation_cdf,
        ↓                            fig_critical_service_heatmap → outputs/cyclone/
B05_loss_estimation.py              MDR via physrisk sigmoid → USD losses (low/mid/high)
        ↓
B06_historical_loss_catalog.py      Full 1980–2025 storm catalog, AEP inputs
```

**Outputs go to:** `outputs/cyclone/` (trajectories, footprints, service disruption figures) · `outputs/losses/` (loss CSVs + wind figures) · `data/footprints/` (HDF5)

---

### Standalone figure scripts
Read processed data directly; not part of the execution chain.

| Script | What it produces | Output location |
|--------|-----------------|-----------------|
| `fig_aep_curve.py` | AEP curve + AEP CSVs (total + per country) | `outputs/losses/` |
| `fig_aep_emdat.py` | EM-DAT AEP curve (physical + VSL) | `outputs/losses/` |
| `fig_aep_modeled.py` | Modeled AEP curve from storm catalog | `outputs/losses/` |
| `fig_compare_methods.py` | Method comparison across hazard metrics | `outputs/hazards/` |
| `fig_emdat_losses.py` | EM-DAT AEP curve + loss by country | `outputs/losses/` |
| `fig_exposure.py` | Exposure summary stats + maps (5 outputs) | `outputs/exposure/` |
| `fig_exposure_dr_raster.py` | DR built-up raster change 1995–2030 | `outputs/exposure/` |
| `fig_exposure_temporal.py` | Temporal exposure evolution by country | `outputs/exposure/` |
| `fig_exposure_zoom.py` | Hispaniola zoom map | `outputs/exposure/` |
| `fig_fixed_effects.py` | Country fixed-effects bar chart | `outputs/regression/` |
| `fig_flood_timeline.py` | Flood event timeline — Haiti + Dominican Republic | `outputs/cyclone/` |
| `fig_framework_stakeholder.py` | 4-pillar analytical framework diagram | `outputs/vulnerability/` |
| `fig_hazards.py` | Rainfall/drought/heatwave fraction + map (5 outputs) | `outputs/hazards/` |
| `fig_hazards_sa.py` | Same diagnostics for South America countries | `outputs/hazards/` |
| `fig_maps.py` | Drought/rainfall raster trend and anomaly maps | `outputs/hazards/` |
| `fig_m6_results.py` | M6 coefficients + variance decomposition + FE tables + FE CSV | `outputs/regression/` |
| `fig_osm_map.py` | OSM infrastructure map | `outputs/exposure/` |
| `fig_osm_services.py` | OSM facilities summary | `outputs/exposure/` |
| `fig_panel_diagnostics.py` | Panel data QC — hazard distributions, exposure trends, correlations | `outputs/regression/` |
| `fig_q2_benchmark.py` | Benchmark combined figure (exposure vs. reference) | `outputs/exposure/` |
| `fig_q3_social_losses.py` | Social replacement value + loss share figures | `outputs/losses/` |
| `fig_regression_models.py` | Tobit regression coefficient plot | `outputs/regression/` |
| `fig_risk_matrix.py` | Country-level risk matrix bubble chart | `outputs/regression/` |
| `fig_robustness_percentiles.py` | Robustness checks — stability + IRF threshold plots | `outputs/regression/` |
| `fig_staged_regressions.py` | Staged regression coefficient + IRF + bin plots | `outputs/regression/` |
| `fig_wind_losses.py` | Wind loss maps + top-10 storms barplot | `outputs/losses/` |
| `fig_wind_results.py` | Wind losses by country + top events | `outputs/losses/` |

#### Numbered report figures (`fig_1_*`, `fig_2_*`, `fig_3_*`)
Publication-ready figures organised by report section. Read from the same processed outputs as the scripts above.

| Script | What it produces | Output location |
|--------|-----------------|-----------------|
| `fig_1_1_emdat_heatmap.py` | EM-DAT event heatmap by country × year | `outputs/losses/` |
| `fig_1_2_emdat_barplot.py` | EM-DAT loss barplot by country | `outputs/losses/` |
| `fig_1_3_emdat_annual.py` | Annual EM-DAT event count time series | `outputs/losses/` |
| `fig_1_5_drought_pixels.py` | Drought pixel fraction diagnostics | `outputs/hazards/` |
| `fig_1_6_rainfall_heatmap.py` | Seasonal rainfall anomaly heatmap | `outputs/hazards/` |
| `fig_1_7_heatwave_heatmap.py` | Heatwave fraction heatmap by country × year | `outputs/hazards/` |
| `fig_1_8_heatwave_sparklines.py` | Heatwave sparklines per country | `outputs/hazards/` |
| `fig_1_9_exposure_change.py` | Built-up / population change 1995–2030 | `outputs/exposure/` |
| `fig_1_10_exposure_temporal.py` | Temporal exposure trajectory by country | `outputs/exposure/` |
| `fig_2_1_m6_forest.py` | M6 forest plot (publication version) | `outputs/regression/` |
| `fig_2_2_fixed_effects.py` | Fixed effects bar chart (publication version) | `outputs/regression/` |
| `fig_3_2_wind_losses.py` | Wind loss map (publication version) | `outputs/losses/` |
| `fig_3_3_top_events.py` | Top loss events barplot | `outputs/losses/` |
| `fig_3_8_composition.py` | Loss composition by category | `outputs/losses/` |
| `fig_3_9_enrolled_students.py` | Enrolled students at risk | `outputs/exposure/` |
| `fig_3_10_density.py` | Infrastructure density map | `outputs/exposure/` |
| `fig_3_11_replacement_value.py` | Replacement value by country | `outputs/losses/` |
| `fig_3_12_benchmark_density.py` | Benchmark density comparison | `outputs/exposure/` |

---

### Utility scripts

#### Downloaders

| Script | Purpose |
|--------|---------|
| `D01_download_chirps.py` | Download CHIRPS v3 precipitation NetCDF files (1995–2025) |
| `D02_download_era5_sa.py` | Download ERA5 data for South America (Colombia, Peru, Guyana) |
| `D03_download_ghs_sa.py` | Download GHS rasters (built-up + population) for South America |

#### Exporters

| Script | Purpose | Output |
|--------|---------|--------|
| `E01_export_aep_csvs.py` | Export share-ready AEP CSVs (modeled + EM-DAT) | `outputs/losses/share_aep_*.csv` |
| `E02_export_report_data.py` | Compile all results into a single report Excel workbook | `outputs/reports/report_data_all_countries.xlsx` |
| `E03_export_spatial_layers.py` | Export tracks, exposure, and flood layers as GeoPackages | `outputs/cyclone/` · `outputs/exposure/` |

#### Tabular one-offs

| Script | Purpose | Output |
|--------|---------|--------|
| `T01_wind_losses.py` | Wind loss tables (Tables 5.1 and 5.2) for the report | `outputs/losses/` |
| `T02_emdat_country_impacts.py` | EM-DAT country-level impact summaries by hazard type | `outputs/losses/` |
| `T03_social_infrastructure.py` | Social infrastructure counts by country and city | `outputs/exposure/tab_social_infra_*.csv` |
| `T04_city_beneficiaries.py` | City-level beneficiary indicators | `outputs/exposure/city_beneficiary_indicators.csv` |

#### South America patch

| Script | Purpose |
|--------|---------|
| `SA01_hazards_append.py` | ERA5 hazards for COL/PER/GUY → appends to Caribbean aggregated parquets |
| `SA02_heatwave_only.py` | Heatwave computation only for South America (memory-efficient variant of SA01) |

---

## Data directory

```
data/
├── EMDAT/                    EM-DAT disaster database (xlsx)
├── GHS/                      Global Human Settlement rasters
│   ├── BUILT_S/              Built-up surface (1990–2030)
│   ├── BUILT_V/              Built-up volume (1995–2030)
│   └── POP/                  Population (1995–2030)
├── IBTrACS/                  Tropical cyclone best tracks (NetCDF)
├── OSM/                      combined_caribbean_infrastructure.gpkg — 438k buildings + roads
├── caribbean_adm_boundaries/ Admin boundary shapefile (current run)
├── chirps_v3_data/           CHIRPS v3 precipitation (1995–2025)
├── era5_hw/                  ERA5 heatwave data (1995–2025)
├── footprints/               Computed storm wind footprints (HDF5)
│   ├── footprints_16_storms.hdf5                 10 EM-DAT matched storms
│   ├── footprints_all_caribbean_1980_2025.hdf5   Full historical catalog
│   └── footprints_all_caribbean_2021_2025.hdf5   Recent storms
└── processed/
    └── caribbean_adm_boundaries/           Active run — all intermediate parquets
        ├── master_panel.parquet            Main analysis dataset
        ├── exposure_panel.parquet
        ├── vulnerability_*.parquet
        └── era5_output*/  heatwaves_computation/  ibtracs_output/  micro_flood/
```

> `data/processed/insular_caribbean_country_outlines_2020/` is an earlier run — do not overwrite it.

---

## Key vulnerability parameters (Track B)

Sigmoid model from Sealy & Strobl (2017) / physrisk:

```
v' = max(v - v_t, 0) / (v_h - v_t)
MDR = v'^3 / (1 + v'^3)
```

| Building type | Mapped from OSM `main_category` | v_h (m/s) | v_t (m/s) |
|---------------|--------------------------------|-----------|-----------|
| other | 3_Mobility_Networks, 4_Buildings | 63.0 | 25.7 |
| school | 2_Basic_Social_Services | 74.7 | 25.7 |
| government | 1_Critical_Infrastructure | 86.0 | 25.7 |

Reconstruction costs and floor areas are hardcoded in `B05_loss_estimation.py`.

---