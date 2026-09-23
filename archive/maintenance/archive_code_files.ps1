$root = (Get-Location).Path

$groups = @{
    legacy_vecm_km = @(
        "vecm_hasbrouck2.py", "vecm_hasbrouck2_old.py", "vecm_analysis2.py",
        "hasbrouck_vecm.ipynb", "hasbrouck_vecm_old.ipynb",
        "hasbrouck_vecm_with_sampled_staleness.ipynb", "standard_vecm.ipynb",
        "km_v5_all_latency_lag_vecm_comparison.ipynb",
        "km_v5_saved_km_full_workflow_with_vecm.ipynb",
        "km_v7_all_latency_lag_vecm_comparison.ipynb",
        "km_v8_events_and_covariates_single_pull_smoke_executed.ipynb",
        "regenerate_v8_km_vecm_correlations.py"
    )
    legacy_aft = @(
        "fit_aft_v2_without_open_interest.py", "fit_aft_v3_all_covariates.py",
        "fit_aft_v5_bivariate_shock_spread.py", "fit_aft_v5_log_covariates.py",
        "fit_aft_v5_univariate_baselines.py", "fit_aft_v5_univariate_fast.py",
        "fit_aft_v6_log_price.py", "fit_aft_v8_bivariate_shock_spread.py",
        "refit_aft_without_spot_volatility.py", "reconstruct_aft_unit_agreeing_oi.py",
        "reconstruct_aft_v4_daily_price.py", "reconstruct_aft_v5_log_covariates.py",
        "reconstruct_aft_v6_log_price.py",
        "format_aft_v5_coefficients_by_contract_currency.py",
        "format_aft_v5_univariate_2delta_ll_table.py",
        "plot_aft_v2_no_oi_results.py", "plot_aft_v3_all_covariates_results.py",
        "plot_aft_v4_daily_price_correlations.py", "plot_aft_v5_log_covariate_diagnostics.py",
        "plot_aft_v5_log_covariate_results.py", "plot_aft_unit_agreeing_oi_correlations.py",
        "plot_aft_unit_agreeing_oi_results.py",
        "summarize_aft_v5_baseline_coefficients_monthly.py",
        "summarize_aft_v5_bivariate_coefficients.py",
        "summarize_aft_v5_complete_coefficients_monthly.py",
        "summarize_aft_v5_complete_coefficients_vecm_periods.py",
        "summarize_aft_v5_log_likelihood.py", "summarize_aft_v5_univariate_baselines.py",
        "summarize_aft_v5_univariate_ll_differences.py",
        "summarize_aft_v5_univariate_loglik_improvements.py"
    )
    legacy_survival = @(
        "survival_analysis.py", "survival_analysis_basis.py", "survival_analysis_basis_monthly.py",
        "survival_analysis_pipeline_final.py", "survival_analysis_workflow_final.ipynb",
        "survival_analysis_workflow_final.txt", "survival_analysis_methods.ipynb",
        "survival_analysis_methods_eventstore_v5.ipynb", "trade_data_pull_old.py"
    )
    validation_experiments = @(
        "audit_aft_original_vs_bivariate_rows.py", "audit_aft_unit_agreeing_oi_timing.py",
        "audit_hasbrouck_daily_hours.py", "audit_hasbrouck_daily_hours_2024_2025.py",
        "audit_hasbrouck_daily_hours_2025.py", "audit_hasbrouck_full_daily_fill_gaps.py",
        "audit_hasbrouck_one_hour.py", "audit_hasbrouck_stratified_other_contracts.py",
        "backfill_liquidity_metrics_missing_days.py", "benchmark_hasbrouck_full_grid.py",
        "johansen_rotating_hour_audit.py", "km_v9_confirmed_lead_december.py",
        "km_v9_deduplicated_robustness.py", "km_v9_reversal_screen_december.py",
        "run_aft_shock_size_ablation.py", "run_aft_spread_ablation.py",
        "run_hasbrouck_random_full_grid.py", "run_johansen_all_markets_parallel.py",
        "summarize_hasbrouck_full_audit.py", "summarize_random_full_grid.py",
        "summarize_random_full_grid_ils.py", "validate_hasbrouck_random_supervisor.py",
        "plot_random_full_grid_heatmaps.py"
    )
    old_analysis = @(
        "Fast_AFT_With_Daily_Covariate_Pull_Streaming.ipynb", "generate_btc_eth_vif_outputs.py",
        "plot_daily_native_volume.py", "plot_daily_perpetual_volumes.py",
        "plot_open_interest_all_perps.py", "plot_open_interest_base_position.py",
        "plot_open_interest_native.py", "plot_open_interest_notional.py", "price_graphing.ipynb",
        "pull_binance_data.py", "trade_activity_diagnostics.py"
    )
}

foreach ($group in $groups.Keys) {
    $dest = Join-Path $root (Join-Path "archive" $group)
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    foreach ($name in $groups[$group]) {
        $src = Join-Path $root $name
        if (Test-Path -LiteralPath $src) {
            $target = Join-Path $dest $name
            if (Test-Path -LiteralPath $target) {
                Write-Output "SKIP_EXISTS $name"
            } else {
                Move-Item -LiteralPath $src -Destination $dest
                Write-Output "$group`: $name"
            }
        } else {
            Write-Output "MISSING $name"
        }
    }
}
