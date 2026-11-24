"""Fit the Appendix D update-frequency regressions."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_regressions(analysis):
    regression_rows=[]
    variable='log_perp_to_spot_observed_bin_ratio'
    for (contract,latency),subset in analysis.groupby(['contract_type','latency']):
        row={'contract_type':contract,'latency':latency,'n_obs':len(subset),'n_date_clusters':subset.date.nunique()}
        for variant in ['original','corrected']:
            fit=smf.ols(f'perp_leadership_{variant} ~ {variable} + C(year)',data=subset).fit(cov_type='cluster',cov_kwds={'groups':subset.date})
            row.update({f'{variant}_coefficient':fit.params[variable],f'{variant}_standard_error':fit.bse[variable],f'{variant}_p_value':fit.pvalues[variable],f'{variant}_ci_lower':fit.conf_int().loc[variable,0],f'{variant}_ci_upper':fit.conf_int().loc[variable,1],f'{variant}_r_squared':fit.rsquared,f'{variant}_significant_5pct':bool(fit.pvalues[variable]<.05),f'{variant}_significant_1pct':bool(fit.pvalues[variable]<.01)})
        regression_rows.append(row)
    regressions=pd.DataFrame(regression_rows)
    return regressions


def run_regression(BASE, ROOT, ACTIVITY, OUT):
    SPOT = "log_midpoint_spot"
    # Reproduce the notebook's original first-duplicate selection for Figure 8.
    # This differs explicitly from the heatmap's keep-last rule.
    activity=pd.read_csv(ACTIVITY,parse_dates=['window_start','date','sample_start','sample_end'])
    assert not activity.duplicated(['window_start','latency','contract_type']).any()
    activity_parts=[]
    for market,contract in [('eth_um','linear'),('eth_cm','inverse')]:
        path=BASE/f'{market}_corrected_hourly.parquet' if market=='eth_um' else BASE/'eth_cm_all_source_rows_corrected.parquet'
        frame=pd.read_parquet(path)
        frame=frame[frame.series.eq(SPOT)].copy()
        order={path.name:i for i,path in enumerate((ROOT/f'vecm_hasbrouck3_{market}').iterdir())}
        frame['source_rank']=frame.source_file.map(order)
        frame=frame.sort_values('source_rank',kind='stable').drop_duplicates(['interval','latency'],keep='first')
        frame['contract_type']=contract
        frame=frame.rename(columns={'interval':'window_start'})
        activity_parts.append(frame[['window_start','latency','contract_type','ILS_mid','ILS_mid_original','source_file']])
    spot_ils=pd.concat(activity_parts,ignore_index=True)
    analysis=spot_ils.merge(activity,on=['window_start','latency','contract_type'],how='inner',validate='one_to_one')
    analysis['perp_leadership_original']=.5-analysis.ILS_mid_original
    analysis['perp_leadership_corrected']=.5-analysis.ILS_mid
    assert np.allclose(analysis.log_perp_to_spot_observed_bin_ratio,np.log((analysis.perp_observed_bin_count+1)/(analysis.spot_observed_bin_count+1)))
    regressions = fit_regressions(analysis)
    regressions.to_csv(OUT/'regression.csv',index=False)
    analysis.to_parquet(OUT/'appendix_figure8_hourly_inputs_corrected.parquet',index=False)

    # Values recorded in notebook cell 81 (rounded to six decimals).
    expected={('inverse','100ms'):(.003112,.815203,1464),('inverse','10ms'):(.020609,.173456,1464),('inverse','1s'):(.034094,.133621,1464),('inverse','200ms'):(.022727,.240413,1464),('inverse','500ms'):(.024698,.223137,1464),('inverse','50ms'):(.000635,.962682,1464),('linear','100ms'):(-.020868,.278372,1439),('linear','10ms'):(.063486,.000074,1439),('linear','1s'):(-.022324,.659123,1463),('linear','200ms'):(-.046094,.029686,1463),('linear','500ms'):(-.051488,.172448,1463),('linear','50ms'):(.007851,.620807,1439)}
    checks=[]
    for row in regressions.itertuples():
        coefficient,p_value,n_obs=expected[(row.contract_type,row.latency)]
        checks.append({'contract_type':row.contract_type,'latency':row.latency,'coefficient_error_vs_rounded_notebook':row.original_coefficient-coefficient,'p_value_error_vs_rounded_notebook':row.original_p_value-p_value,'n_obs_difference':row.n_obs-n_obs})
    checks=pd.DataFrame(checks)
    checks.to_csv(OUT/'regression_check.csv',index=False)
    validation={'max_coefficient_error':float(checks.coefficient_error_vs_rounded_notebook.abs().max()),'max_p_value_error':float(checks.p_value_error_vs_rounded_notebook.abs().max()),'max_n_obs_difference':int(checks.n_obs_difference.abs().max())}
    print('Original Figure 8 reproduction:', validation, flush=True)
    assert checks.coefficient_error_vs_rounded_notebook.abs().max()<5.01e-7
    assert checks.p_value_error_vs_rounded_notebook.abs().max()<5.01e-7
    assert checks.n_obs_difference.eq(0).all()
    return regressions, validation
