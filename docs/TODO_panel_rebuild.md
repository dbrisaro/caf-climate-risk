# TODO: rebuild the region panel

Status: pending. The M1–M5 (`07b`) and M6 (`07d`) results in the report use a stopgap: region names shared across countries are excluded. This note lists what the full rebuild needs.

## 1. Unique region keys (name + country) in steps 02–06

`master_panel.parquet` and every upstream step (`02_compute_exposure.py`, `03_aggregate_hazards.py`, `04_build_panel.py`, `06_add_emdat_losses.py`, and `src/aggregation.py`, `src/hazard_utils.py`, `src/exposure_utils.py`) key regions by `region_name` only. In `data/caribbean_adm_boundaries/caribbean_adm_boundaries.shp`, eight names belong to more than one country:

| Name | Countries |
|---|---|
| Saint Andrew | Barbados, Dominica, Jamaica |
| Saint George | Barbados, Dominica |
| Saint James | Barbados, Jamaica |
| Saint John | Barbados, Dominica |
| Saint Joseph | Barbados, Dominica |
| Saint Peter | Barbados, Dominica |
| Saint Thomas | Barbados, Jamaica |
| Amazonas | Colombia, Peru |

Each is currently one merged series (exposure, hazards and EM-DAT losses). To do:

- Carry a unique key, `country` + `region_name`, from the shapefile through exposure, hazard aggregation, panel build and EM-DAT loss assignment.
- Add a `country` column to `master_panel.parquet`.
- Merge on the unique key everywhere, including the IBTrACS wind aggregate and the hazard excess/anomaly caches used by `07b`.
- The raw ERA5 inputs under `suyana/sources/era5/` are no longer on the server. Redo the hazard aggregation from the existing raw parquets by reassigning pixels to polygons.

## 2. Region means over the polygon, not bounding box + 20 km

`src/exposure_utils.load_raster_window` reads GHS POP, BUILT_S and BUILT_V over each region's bounding box plus a 20 km buffer, resampled to 1500×1500. `run_exposure` then takes the `nanmean` of that window. Sea cells are 0 in GHS, not nodata, so the "region means" are diluted by surrounding land and sea, and small neighbouring parishes overlap heavily.

- Mask each raster to the region polygon (`rasterio.mask`) before averaging.
- Decide whether pop and built-up area should be totals or means per 100 m cell, and update the Table B.5.1 units to match.

## 3. Rerun and compare

After rebuilding `master_panel.parquet`:

- Remove the shared-name exclusion from `07b_regression_panel.py` and `07d_caribbean_m6.py`. In 07d, also remove the `drop`/`combine` modes.
- Rerun `07b`, `07d`, `fig_2_1_m6_forest.py`, `fig_2_2_fixed_effects.py` and `E02_export_report_data.py`.
- Compare with the current report values: N, regions, within and overall R², coefficients and p-values, hazard share of the within R², country fixed effects (with n_obs and n_regions), Table B.5.1, and the M1–M5 built-environment coefficients.

Keep the cached World Bank GDP files (`data/processed/caribbean_adm_boundaries/wb_gdp_pc_NY.GDP.PCAP.KD_1995_2025*.csv`) so that GDP updates don't mix with the rebuild's effects.

## Already fixed

- Losses were summed over the four seasonal windows in `07b` and `07d` (4× too large). They are now averaged.
- World Bank GDP is cached instead of fetched live on every run.
