"""Correct the ILS-dependent appendix outputs using saved data only."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from .regression import run_regression
from .plots import APPENDIX_OUTPUT, draw_regression, draw_table, draw_full_grid

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=ROOT/'hasbrouck_spec_audit/ils_correction'
OUT=ROOT/APPENDIX_OUTPUT
ACTIVITY=HERE/'data/eth_activity.csv'
LATENCIES=['10ms','50ms','100ms','200ms','500ms','1s']
REGIMES=[('2021','2021-01-01','2022-01-01'),('Jan-Jun 2022','2022-01-01','2022-07-01'),('Jul 2022-Mar 2023','2022-07-01','2023-04-01'),('Apr-Dec 2023','2023-04-01','2024-01-01'),('2024','2024-01-01','2025-01-01'),('2025','2025-01-01','2026-01-01')]
MARKETS=['btc_um','btc_cm','eth_um','eth_cm']
SPOT='log_midpoint_spot'
required_inputs = [BASE / name for name in (
    *(f'{market}_corrected_hourly.parquet' for market in MARKETS),
    'eth_cm_all_source_rows_corrected.parquet',
    'random_full_grid_corrected_hourly.parquet',
    'paired_observed_vs_full_grid_corrected.csv',
    'regime_summary_spot.csv',
)]
required_inputs += [ROOT/'hasbrouck_spec_audit/random_10_days_month_full_grid'/name for name in (
    'regime_heatmaps/spot_ils_summary.csv', 'paired_ils_signed_summary.csv',
)]
required_inputs += [ROOT/f'vecm_hasbrouck3_{market}' for market in ('eth_um', 'eth_cm')]
missing_inputs = [path.relative_to(ROOT).as_posix() for path in required_inputs if not path.exists()]
if missing_inputs:
    raise SystemExit("Missing corrected ILS inputs:\n" + "\n".join(missing_inputs))
OUT.mkdir(parents=True, exist_ok=True)
validation={}

regressions, validation['figure8_original_reproduction'] = run_regression(BASE, ROOT, ACTIVITY, OUT)
draw_regression(regressions,OUT)
print('Corrected Figure 8 and regression coefficients exported.',flush=True)

# Rebuild Figures 9 and 10 from the previously corrected full-grid audit.
full=pd.read_parquet(BASE/'random_full_grid_corrected_hourly.parquet')
full['interval']=pd.to_datetime(full.interval)
regime_rows=[]
for regime,start,end in REGIMES:
    selected=full[full.interval.ge(start)&full.interval.lt(end)]
    for (market,latency,series),group in selected.groupby(['market','latency','series']):
        regime_rows.append({'market':market,'latency':latency,'series':series,'regime':regime,'regime_start':start,'regime_end_exclusive':end,'median_ILS':group.ILS_mid.median(),'mean_ILS':group.ILS_mid.mean(),'median_ILS_original':group.ILS_mid_original.median(),'mean_ILS_original':group.ILS_mid_original.mean(),'model_intervals':len(group)})
regime_summary=pd.DataFrame(regime_rows)
regime_summary.to_csv(OUT/'full_grid_ils_by_series.csv',index=False)
spot_summary=regime_summary[regime_summary.series.eq(SPOT)].rename(columns={'median_ILS':'median_spot_ILS','mean_ILS':'mean_spot_ILS'})
spot_summary.to_csv(OUT/'full_grid.csv',index=False)
old_saved=pd.read_csv(ROOT/'hasbrouck_spec_audit/random_10_days_month_full_grid/regime_heatmaps/spot_ils_summary.csv')
old_check=spot_summary.merge(old_saved,on=['market','latency','regime'],suffixes=('','_saved'),validate='one_to_one')
assert np.allclose(old_check.median_ILS_original,old_check.median_spot_ILS_saved,rtol=0,atol=1e-14)
assert old_check.model_intervals.eq(old_check.model_intervals_saved).all()
validation['full_grid_original_summary_reproduction']={'cells':len(old_check),'max_original_median_error':float((old_check.median_ILS_original-old_check.median_spot_ILS_saved).abs().max())}
draw_full_grid(spot_summary,OUT)

# Table 4 is SPOT full-grid minus observed-bin ILS. The saved paired export
# is in the perpetual perspective, so negate its signed difference.
paired=pd.read_csv(BASE/'paired_observed_vs_full_grid_corrected.csv',parse_dates=['interval'])
paired['signed_delta_spot_pp']=100*(paired.ILS_mid_observed-paired.ILS_mid_full_grid)
paired['signed_delta_spot_pp_original']=100*(paired.ILS_mid_original_observed-paired.ILS_mid_original_full_grid)
table=paired.groupby(['market','latency']).agg(hours=('interval','size'),mean_signed_delta_spot_pp=('signed_delta_spot_pp','mean'),mean_absolute_delta_spot_pp=('absolute_difference_pp','mean'),different_leader_pct=('leadership_flip',lambda s:100*s.mean()),mean_signed_delta_spot_pp_original=('signed_delta_spot_pp_original','mean'),mean_absolute_delta_spot_pp_original=('absolute_difference_pp_original','mean')).reset_index()
table['asset']=table.market.str[:3].str.upper()
table['contract']=table.market.str[-2:].map({'um':'Linear','cm':'Inverse'})
table['sort_market']=table.market.map({name:i for i,name in enumerate(MARKETS)})
table['sort_latency']=table.latency.map({'1s':0,'100ms':1,'10ms':2})
table=table.sort_values(['sort_market','sort_latency']).drop(columns=['sort_market','sort_latency'])
table.to_csv(OUT/'robustness.csv',index=False)
old_table=pd.read_csv(ROOT/'hasbrouck_spec_audit/random_10_days_month_full_grid/paired_ils_signed_summary.csv')
check=table.merge(old_table,on=['market','latency'],validate='one_to_one')
assert np.allclose(check.mean_signed_delta_spot_pp_original,-check.signed_pp,rtol=0,atol=1e-12)
assert np.allclose(check.mean_absolute_delta_spot_pp_original,check.absolute_pp,rtol=0,atol=1e-12)
assert np.allclose(check.different_leader_pct,check.flip_pct,rtol=0,atol=1e-12)
validation['table4_original_reproduction']={'rows':len(check),'matched_hours':int(table.hours.sum()),'leader_flip_percentages_unchanged':True}

latex=[r'% Requires \usepackage{multirow}',r'\begin{table}[htbp]',r'\centering',r'\caption{Signed and absolute changes in spot ILS, and the percentage of hours with a different leader, comparing complete-grid and observed-bin models.}',r'\label{tab:ils-gap-filling}',r'\begin{tabular}{|c|c|c|r|r|r|}',r'\hline',r'Asset & Contract & Latency & \shortstack{Mean signed\\$\Delta$ILS (pp)} & \shortstack{Mean absolute\\$\Delta$ILS (pp)} & \shortstack{Different\\leader (\%)} \\',r'\hline']
for i,row in enumerate(table.itertuples(index=False)):
    asset_label=rf'\multirow{{6}}{{*}}{{{row.asset}}}' if i%6==0 else ''
    contract_label=rf'\multirow{{3}}{{*}}{{{row.contract}}}' if i%3==0 else ''
    spacing={'1s':'1 s','100ms':'100 ms','10ms':'10 ms'}[row.latency]
    latex.append(f'{asset_label} & {contract_label} & {spacing} & {row.mean_signed_delta_spot_pp:.2f} & {row.mean_absolute_delta_spot_pp:.2f} & {row.different_leader_pct:.2f} '+r'\\')
    latex.append(r'\hline' if i%6==5 else r'\cline{2-6}' if i%3==2 else r'\cline{3-6}')
latex += [r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{\textwidth}\footnotesize',r'\textit{Notes:} $\Delta\mathrm{ILS}=100(\mathrm{ILS}^{\mathrm{full\ grid}}_{\mathrm{spot}}-\mathrm{ILS}^{\mathrm{observed}}_{\mathrm{spot}})$, in percentage points. Absolute changes are averaged before aggregation. Comparisons use the same contract, date, UTC hour and latency. Only valid, unambiguous matched models are included. The sample has ten randomly selected dates per month during 2021--2025 and one random UTC hour per date.',r'\end{minipage}',r'\end{table}']
(OUT/'table_4.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
draw_table(table,OUT)
print(table[['market','latency','hours','mean_signed_delta_spot_pp','mean_absolute_delta_spot_pp','different_leader_pct']].to_string(index=False),flush=True)

# Quantiles and day-block bootstrap of the hourly-weighted mean, using 2,000
# replicates (not 2,000 sampled days). Share the draws across old/new estimates.
def summarize_bootstrap(group):
    values=group.ILS_mid.to_numpy()
    old=group.ILS_mid_original.to_numpy()
    dates=group.interval.dt.normalize()
    daily=pd.DataFrame({'date':dates,'new':values,'old':old}).groupby('date').agg(new_sum=('new','sum'),old_sum=('old','sum'),count=('new','size'))
    sums,old_sums,counts=daily.new_sum.to_numpy(),daily.old_sum.to_numpy(),daily['count'].to_numpy()
    draws=np.random.default_rng(12345).integers(0,len(daily),size=(2000,len(daily)))
    boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    old_boot=old_sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    result={'model_intervals':len(values),'n_days':len(daily),'bootstrap_replicates':2000,'bootstrap_seed':12345,'mean':values.mean(),'median':np.median(values),'std':values.std(ddof=1),'share_spot_leads':np.mean(values>.5),'ci_lower':np.quantile(boot,.025),'ci_upper':np.quantile(boot,.975),'mean_original':old.mean(),'median_original':np.median(old),'ci_lower_original':np.quantile(old_boot,.025),'ci_upper_original':np.quantile(old_boot,.975)}
    for q in [.1,.25,.75,.9]:result[f'p{int(q*100)}']=np.quantile(values,q)
    result['mean_spot_leads']=result['mean']>.5
    result['median_spot_leads']=result['median']>.5
    result['mean_median_leader_disagree']=result['mean_spot_leads']!=result['median_spot_leads']
    result['ci_contains_half']=result['ci_lower']<=.5<=result['ci_upper']
    result['mean_leader_changed_by_correction']=(result['mean']-.5)*(result['mean_original']-.5)<0
    return result

annual_rows,regime_boot_rows=[] ,[]
for market in MARKETS:
    data=pd.read_parquet(BASE/f'{market}_corrected_hourly.parquet',columns=['interval','series','latency','ILS_mid','ILS_mid_original'])
    data=data[data.series.eq(SPOT)].copy()
    data['year']=data.interval.dt.year
    for (year,latency),group in data.groupby(['year','latency']):
        annual_rows.append({'market':market,'year':int(year),'latency':latency,**summarize_bootstrap(group)})
    for regime,start,end in REGIMES:
        selected=data[data.interval.ge(start)&data.interval.lt(end)]
        for latency,group in selected.groupby('latency'):
            regime_boot_rows.append({'market':market,'regime':regime,'latency':latency,**summarize_bootstrap(group)})
    print(f'{market}: annual/regime quantiles and 2,000-replicate day-block intervals completed',flush=True)
annual=pd.DataFrame(annual_rows)
regime_boot=pd.DataFrame(regime_boot_rows)
annual.to_csv(OUT/'ils_annual_bootstrap.csv',index=False)
regime_boot.to_csv(OUT/'ils_regime_bootstrap.csv',index=False)
regime_boot[regime_boot.mean_median_leader_disagree|regime_boot.ci_contains_half|regime_boot.mean_leader_changed_by_correction].to_csv(OUT/'ils_boundary_cases.csv',index=False)

# Bracket transitions between actually sampled grid points instead of
# interpolating unobserved latencies from a rounded heatmap annotation.
main_summary=pd.read_csv(BASE/'regime_summary_spot.csv')
transitions=[]
for (market,regime),group in main_summary.groupby(['market','regime']):
    group=group.set_index('latency').reindex(LATENCIES)
    values=group.median_ILS_corrected.to_numpy()
    old_values=group.median_ILS_original.to_numpy()
    assert np.array_equal(values>.5,old_values>.5)
    brackets=[]
    for i in range(len(values)-1):
        if (values[i]-.5)*(values[i+1]-.5)<0:
            brackets.append(f'{LATENCIES[i]} to {LATENCIES[i+1]}')
    transitions.append({'market':market,'regime':regime,'transition_bracket':'; '.join(brackets),'spot_leads_at_10ms':bool(values[0]>.5),'spot_leads_at_1s':bool(values[-1]>.5),'median_leader_changed_by_correction':False})
pd.DataFrame(transitions).to_csv(OUT/'ils_transitions.csv',index=False)
validation['day_block_bootstrap']={'annual_cells':len(annual),'regime_cells':len(regime_boot),'replicates':2000,'seed':12345,'regime_mean_median_leader_disagreements':int(regime_boot.mean_median_leader_disagree.sum()),'regime_CIs_containing_half':int(regime_boot.ci_contains_half.sum()),'regime_mean_leader_changes_from_correction':int(regime_boot.mean_leader_changed_by_correction.sum())}
validation['sources']={'update_frequency_csv':ACTIVITY.relative_to(ROOT).as_posix(),'update_frequency_csv_sha256':hashlib.sha256(ACTIVITY.read_bytes()).hexdigest(),'regression_code':'regression.py','main_corrected_results':BASE.relative_to(ROOT).as_posix(),'original_pdf':'the earlier paper draft'}
(OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print(regressions[['contract_type','latency','original_coefficient','original_p_value','corrected_coefficient','corrected_p_value','corrected_ci_lower','corrected_ci_upper']].to_string(index=False),flush=True)
print(json.dumps(validation,indent=2),flush=True)
