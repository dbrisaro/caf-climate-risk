#!/bin/bash
# run_pipeline.sh — Full CAF pipeline execution with skip-if-exists logic.
# Run with:  bash run_pipeline.sh
# Each step checks for its primary output; if it already exists the step is skipped.

set -e
PY=/opt/tljh/user/bin/python
cd "$(dirname "$0")"

# ── helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

run_if_missing() {
    # Usage: run_if_missing "label" "output_path" "script_path"
    local label="$1" output="$2" script="$3"
    if [ -f "$output" ]; then
        echo -e "  ${YELLOW}[skip]${NC} $label"
    else
        echo -e "  ${GREEN}[run ]${NC} $label"
        $PY "$script"
    fi
}

has_column() {
    # Usage: has_column "parquet_path" "column_name"
    $PY - <<EOF 2>/dev/null
import pandas as pd, sys
df = pd.read_parquet("$1", columns=[])
cols = pd.read_parquet("$1").columns.tolist()
sys.exit(0 if "$2" in cols else 1)
EOF
}

echo "======================================================================"
echo " CAF — Caribbean Climate Risk Pipeline"
echo "======================================================================"

# ── PRE-PIPELINE: CHIRPS data download ───────────────────────────────────────
echo ""
echo "--- Pre-pipeline: data ---"
CHIRPS_SENTINEL="data/chirps_v3_data/chirps-v3.0.2024.days_p05_RGS.nc"
run_if_missing "D01 CHIRPS download" "$CHIRPS_SENTINEL" "scripts/D01_download_chirps.py"

# SA ERA5 (Colombia, Peru, Guyana) — separate bbox from Caribbean ERA5
ERA5_SA_SENTINEL="$HOME/suyana/sources/era5/total_precipitation/south_america/reanalysis-era5-land_total_precipitation_2025_daily_13N_83W_19S_55W_0.1deg.nc"
run_if_missing "D02 ERA5 SA download" "$ERA5_SA_SENTINEL" "scripts/D02_download_era5_sa.py"

# GHS tiles for SA (rows R9-R12, cols C11-C13)
GHS_SA_SENTINEL="data/GHS/BUILT_S/2025/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100_V1_0_R9_C11.tif"
run_if_missing "D03 GHS SA tiles" "$GHS_SA_SENTINEL" "scripts/D03_download_ghs_sa.py"

# ── TRACK A — Macro / Statistical ────────────────────────────────────────────
echo ""
echo "--- TRACK A: Macro / Statistical ---"

# Step 00: ERA5 raw hazard extraction
# util_reassign_regions.py is a manual patch (re-runs region assignment without
# re-downloading ERA5). It is NOT run automatically here; uncomment if needed.
ERA5_SENTINEL="data/processed/caribbean_adm_boundaries/era5_output/era5_raw_data.parquet"
run_if_missing "00  ERA5 raw hazards" "$ERA5_SENTINEL" "scripts/00_compute_hazards_raw.py"
# run_if_missing "util Reassign regions" "$ERA5_SENTINEL" "scripts/util_reassign_regions.py"

run_if_missing "01  EM-DAT preparation" \
    "outputs/losses/emdat_carib_1900_2024.gpkg" \
    "scripts/01_prepare_emdat.py"

run_if_missing "02  Exposure (GHS + MERIT)" \
    "data/processed/caribbean_adm_boundaries/exposure_panel.parquet" \
    "scripts/02_compute_exposure.py"

run_if_missing "03  Aggregate hazards (seasonal)" \
    "data/processed/caribbean_adm_boundaries/era5_output_precipitation/rainfall_aggregated_data.parquet" \
    "scripts/03_aggregate_hazards.py"

run_if_missing "04  Build master panel" \
    "data/processed/caribbean_adm_boundaries/master_panel.parquet" \
    "scripts/04_build_panel.py"

run_if_missing "05  Vulnerability decomposition" \
    "data/processed/caribbean_adm_boundaries/vulnerability_decomposition.parquet" \
    "scripts/05_vulnerability_decomposition.py"

# Step 05 overwrites master_panel in place → check for its added column
echo -n "  checking 06 Add EM-DAT losses... "
if has_column "data/processed/caribbean_adm_boundaries/master_panel.parquet" "total_losses_musd"; then
    echo -e "${YELLOW}[skip]${NC} total_losses_musd already in master_panel"
else
    echo -e "${GREEN}[run ]${NC}"
    $PY scripts/06_add_emdat_losses.py
fi

run_if_missing "07a Vulnerability coefficients" \
    "data/processed/caribbean_adm_boundaries/vulnerability_coefficients.parquet" \
    "scripts/07a_vulnerability_coefficients.py"

# Optional: util_anomaly_cache.py speeds up 06b (skipped if cache exists)
run_if_missing "util Anomaly cache (optional speedup for 07b)" \
    "data/processed/caribbean_adm_boundaries/hazard_anomaly.parquet" \
    "scripts/util_anomaly_cache.py"

run_if_missing "07b Panel regression suite (M1-M6)" \
    "outputs/regression/coefficients_M1_M6.csv" \
    "scripts/07b_regression_panel.py"

run_if_missing "07c Caribbean/continental split regressions" \
    "outputs/regression/coefficients_split.csv" \
    "scripts/07c_regression_split.py"

run_if_missing "08  Extended regression suite" \
    "outputs/reports/02_regression_suite.md" \
    "scripts/08_regression_suite.py"

run_if_missing "09  Composite risk index" \
    "outputs/vulnerability/composite_risk_index_by_country.png" \
    "scripts/09_composite_risk_index.py"

# ── TRACK SA — South America extension (conditional, runs instead of 00+03) ──
# Uncomment when Caribbean data already exists and only SA data is missing.
# echo ""
# echo "--- TRACK SA: South America extension ---"
# run_if_missing "SA01 SA hazards append" \
#     "data/processed/caribbean_adm_boundaries/era5_output_precipitation/rainfall_aggregated_data.parquet" \
#     "scripts/SA01_hazards_append.py"
# run_if_missing "SA02 SA heatwave only" \
#     "data/processed/caribbean_adm_boundaries/heatwaves_computation/heatwaves_aggregated_data.parquet" \
#     "scripts/SA02_heatwave_only.py"

# ── TRACK T — Tables (prerequisite T03 must run before Track B) ──────────────
echo ""
echo "--- Tables (pre-B) ---"
run_if_missing "T03 Social infrastructure table" \
    "outputs/exposure/tab_social_infra_by_country.csv" \
    "scripts/T03_social_infrastructure.py"

run_if_missing "T04 City beneficiary indicators" \
    "outputs/exposure/city_beneficiary_indicators.csv" \
    "scripts/T04_city_beneficiaries.py"

# ── TRACK B — Physical / Cyclone ─────────────────────────────────────────────
echo ""
echo "--- TRACK B: Physical / Cyclone ---"

run_if_missing "B01 Cyclone trajectory analysis" \
    "outputs/cyclone/trajectories/overview_all_storms.png" \
    "scripts/B01_cyclone_trajectory_analysis.py"

run_if_missing "B02 Cyclone wind footprints (CLIMADA)" \
    "data/footprints/footprints_all_caribbean_2021_2025.hdf5" \
    "scripts/B02_cyclone_footprint.py"

run_if_missing "B03 Micro flood analysis" \
    "data/processed/caribbean_adm_boundaries/micro_flood/flood_events.parquet" \
    "scripts/B03_micro_flood_analysis.py"

run_if_missing "B04 Service disruption" \
    "data/processed/caribbean_adm_boundaries/micro_flood/service_disruption.parquet" \
    "scripts/B04_service_disruption.py"

run_if_missing "B05 Loss estimation (sigmoid MDR)" \
    "outputs/losses/loss_by_building.csv" \
    "scripts/B05_loss_estimation.py"

run_if_missing "B06 Historical loss catalog (1980-2025)" \
    "outputs/losses/historical_loss_catalog.csv" \
    "scripts/B06_historical_loss_catalog.py"

# ── STANDALONE FIGURES — Exposure ────────────────────────────────────────────
echo ""
echo "--- Figures: Exposure ---"
run_if_missing "fig_exposure"          "outputs/exposure/exposure_summary_stats.csv"          "scripts/fig_exposure.py"
run_if_missing "fig_exposure_temporal" "outputs/exposure/temporal_evolution_by_country.png"   "scripts/fig_exposure_temporal.py"
run_if_missing "fig_exposure_zoom"     "outputs/exposure/map_change_zoom_hispaniola.png"       "scripts/fig_exposure_zoom.py"
run_if_missing "fig_exposure_dr_raster" "outputs/exposure/exposure_dr_raster_1995_2030.png"   "scripts/fig_exposure_dr_raster.py"
run_if_missing "fig_osm_map"           "outputs/exposure/osm_map.png"                         "scripts/fig_osm_map.py"
run_if_missing "fig_osm_services"      "outputs/exposure/fig_osm_services.png"                "scripts/fig_osm_services.py"

# ── STANDALONE FIGURES — Hazards ─────────────────────────────────────────────
echo ""
echo "--- Figures: Hazards ---"
run_if_missing "fig_hazards" "outputs/hazards/rainfall_fraction_by_country.png" "scripts/fig_hazards.py"

# ── STANDALONE FIGURES — Vulnerability ───────────────────────────────────────
echo ""
echo "--- Figures: Vulnerability ---"
run_if_missing "fig_framework_stakeholder" "outputs/vulnerability/fig_framework_stakeholder.png" "scripts/fig_framework_stakeholder.py"

# ── STANDALONE FIGURES — Regression ──────────────────────────────────────────
echo ""
echo "--- Figures: Regression ---"
run_if_missing "fig_fixed_effects"      "outputs/regression/fixed_effects_by_country.png"       "scripts/fig_fixed_effects.py"
run_if_missing "fig_m6_results"         "outputs/regression/m6_coefficients.png"                "scripts/fig_m6_results.py"
run_if_missing "fig_regression_models"  "outputs/regression/fig_regression_models.png"          "scripts/fig_regression_models.py"
run_if_missing "fig_risk_matrix"        "outputs/regression/fig_regression_comparison.png"      "scripts/fig_risk_matrix.py"
run_if_missing "fig_robustness"         "outputs/regression/fig_robustness_coef_stability.png"  "scripts/fig_robustness_percentiles.py"
run_if_missing "fig_staged_regressions" "outputs/regression/fig_staged_coefs.png"               "scripts/fig_staged_regressions.py"
run_if_missing "fig_panel_diagnostics"   "outputs/regression/fig_panel_hazard_distributions.png" "scripts/fig_panel_diagnostics.py"

# ── STANDALONE FIGURES — EM-DAT losses ───────────────────────────────────────
echo ""
echo "--- Figures: EM-DAT losses ---"
run_if_missing "fig_emdat_losses" "outputs/losses/fig_emdat_aep_curve.png" "scripts/fig_emdat_losses.py"

# ── STANDALONE FIGURES — Cyclone & Losses ────────────────────────────────────
echo ""
echo "--- Figures: Cyclone & Losses ---"
run_if_missing "fig_flood_timeline"  "outputs/cyclone/fig_flood_timeline.png"            "scripts/fig_flood_timeline.py"
run_if_missing "T01 Wind losses table"     "outputs/losses/table_5_1_wind_by_country_storm.csv" "scripts/T01_wind_losses.py"
run_if_missing "fig_aep_curve"       "outputs/losses/aep_curve.png"                      "scripts/fig_aep_curve.py"
run_if_missing "fig_wind_losses"     "outputs/losses/map_wind_losses_by_country.png"      "scripts/fig_wind_losses.py"
run_if_missing "fig_wind_results"    "outputs/losses/wind_losses_by_country.png"          "scripts/fig_wind_results.py"

# ── TRACK T — Tables (remaining) ─────────────────────────────────────────────
echo ""
echo "--- Tables ---"
run_if_missing "T02 EM-DAT country impacts table" \
    "outputs/losses/tab_4_1_country_impacts.csv" \
    "scripts/T02_emdat_country_impacts.py"

# ── TRACK E — Exports ─────────────────────────────────────────────────────────
echo ""
echo "--- Exports ---"
run_if_missing "E01 AEP CSV export" \
    "share_aep_modeled.csv" \
    "scripts/E01_export_aep_csvs.py"

run_if_missing "E02 Report data Excel export" \
    "outputs/reports/report_data_all_countries.xlsx" \
    "scripts/E02_export_report_data.py"

run_if_missing "E03 Spatial layers export" \
    "outputs/cyclone/fig_1_4_cyclone_tracks.gpkg" \
    "scripts/E03_export_spatial_layers.py"

echo ""
echo "======================================================================"
echo " Pipeline complete."
echo "======================================================================"
