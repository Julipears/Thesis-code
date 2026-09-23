"""
This file contains the VECMResult class for fitting the standard VECM model and computing price discovery measures.

GenAI was used to implement CS and ILS calculations.
"""
import numpy as np
import pandas as pd
import io, contextlib
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

from numpy.linalg import cholesky
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.vecm import select_order
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
from trade_data_pull import *

COINT_RANK = 1

import numpy as np
from numpy.linalg import cholesky

def _orth_complement_2d(v):
    """
    Orthogonal complement for a 2x1 vector.
    If v = [v1, v2]', returns [v2, -v1]'.
    """
    v = np.asarray(v).reshape(-1)
    if v.shape[0] != 2:
        raise ValueError("Only implemented for 2-series systems.")
    return np.array([v[1], -v[0]], dtype=float)


def compute_price_discovery_measures(results_vecm, n_ma=50):
    """
    Compute CS, HIS lower/upper/mid, and ILS mid for a 2-series VECM/VAR system.

    Returns
    -------
    dict with:
        cs_spot, cs_perp,
        his_lower_spot, his_upper_spot, his_mid_spot,
        his_lower_perp, his_upper_perp, his_mid_perp,
        ils_mid_spot, ils_mid_perp
    """
    out = {
        'cs_spot': np.nan, 'cs_perp': np.nan,
        'his_lower_spot': np.nan, 'his_upper_spot': np.nan, 'his_mid_spot': np.nan,
        'his_lower_perp': np.nan, 'his_upper_perp': np.nan, 'his_mid_perp': np.nan,
        'ils_mid_spot': np.nan, 'ils_mid_perp': np.nan,
    }

    try:
        # ---------- Component Share (CS) ----------
        alpha = np.asarray(results_vecm.alpha).reshape(-1)   # shape (2,)
        alpha_perp = _orth_complement_2d(alpha)

        denom_cs = alpha_perp.sum()
        if np.isclose(denom_cs, 0):
            return out

        psi = alpha_perp / denom_cs   # common-trend weights normalized to sum to 1
        cs_spot, cs_perp = float(psi[0]), float(psi[1])

        # ---------- Hasbrouck Information Share (HIS) ----------
        # Long-run impact matrix from VAR MA representation
        # vma = results_var.ma_rep(maxn=n_ma)
        # Psi = np.sum(vma, axis=0)   # kept for consistency with your previous code
        Sigma = np.asarray(results_vecm.sigma_u)

        # denominator in Hasbrouck formula
        # use psi row vector as the permanent-price weights
        denom_his = float(psi @ Sigma @ psi.T)
        if denom_his <= 0 or np.isnan(denom_his):
            return out

        # ordering 1: [spot, perp]
        F1 = cholesky(Sigma)
        w1 = psi @ F1
        his1 = (w1 ** 2) / denom_his   # contributions in original order

        # ordering 2: [perp, spot], mapped back to original order
        P = np.array([[0.0, 1.0],
                      [1.0, 0.0]])
        Sigma_rev = P @ Sigma @ P.T
        F_rev_chol = cholesky(Sigma_rev)
        F2 = P.T @ F_rev_chol
        w2 = psi @ F2
        his2 = (w2 ** 2) / denom_his   # contributions in original order

        his_lower = np.minimum(his1, his2)
        his_upper = np.maximum(his1, his2)
        his_mid   = 0.5 * (his_lower + his_upper)

        his_lower_spot, his_lower_perp = map(float, his_lower)
        his_upper_spot, his_upper_perp = map(float, his_upper)
        his_mid_spot,   his_mid_perp   = map(float, his_mid)

        # ---------- Information Leadership Share (ILS) ----------
        # Putnins-style leadership ratio, converted to shares summing to 1
        if (
            his_mid_spot > 0 and his_mid_perp > 0 and
            cs_spot > 0 and cs_perp > 0
        ):
            valid = True
        else:
            valid = False
        R_spot = abs((his_mid_spot / his_mid_perp) * (cs_perp / cs_spot))
        ils_mid_spot = float(R_spot / (1.0 + R_spot))
        ils_mid_perp = float(1.0 - ils_mid_spot)
        # else:
        #     ils_mid_spot = np.nan
        #     ils_mid_perp = np.nan

        out.update({
            'cs_spot': cs_spot,
            'cs_perp': cs_perp,
            'his_lower_spot': his_lower_spot,
            'his_upper_spot': his_upper_spot,
            'his_mid_spot': his_mid_spot,
            'his_lower_perp': his_lower_perp,
            'his_upper_perp': his_upper_perp,
            'his_mid_perp': his_mid_perp,
            'ils_mid_spot': ils_mid_spot,
            'ils_mid_perp': ils_mid_perp,
            'valid': valid,
        })
        return out

    except Exception:
        return out

    
def run_tests(bidask_diff):
    dl = []
    for column in bidask_diff.columns:
        test = adf_test(bidask_diff[column],const_trend='c')
        dl.append(test)
    results1 = pd.concat(dl, axis=1)
    display(results1)

    # Test data for deterministic time trend
    dl = []
    for column in bidask_diff.columns:
        test = adf_test(bidask_diff[column],'ct')
        dl.append(test)
    results2 = pd.concat(dl, axis=1)
    display(results2)

    test = coint_johansen(bidask_diff[['log_midpoint_spot', 'log_midpoint_perp']], det_order=0, k_ar_diff=1)
    test_stats = test.lr1; crit_vals = test.cvt[:, 1]
    # Print results
    for r_0, (test_stat, crit_val) in enumerate(zip(test_stats, crit_vals)):
        print(f'H_0: r <= {r_0}')
        print(f'  Test Stat. = {test_stat:.2f}, 5% Crit. Value = {crit_val:.2f}')
        if test_stat > crit_val:
            print('  => Reject null hypothesis.')
        else:
            print('  => Fail to reject null hypothesis.')

    return results1, results2

def select_lag_coint(bidask_diff, maxlags=10, which='lag'):
    cols = ['log_midpoint_spot', 'log_midpoint_perp']

    lag_order_results = select_order(
        bidask_diff[cols],
        maxlags=maxlags,
        deterministic='co'
    )

    p = lag_order_results.aic
    k_ar_diff = max(int(p) - 1, 1)

    if which == 'both':
        coint_rank_results = select_coint_rank(
            bidask_diff[cols],
            method='trace',
            det_order=0,
            k_ar_diff=k_ar_diff
        )
        return k_ar_diff, coint_rank_results.rank
    else:
        return k_ar_diff, COINT_RANK


def fit_vecm(bidask_diff, lo, cr):
    model_vecm = VECM(bidask_diff[['log_midpoint_spot', 'log_midpoint_perp']], deterministic='co', 
        k_ar_diff=lo, 
        coint_rank=cr)
    results_vecm = model_vecm.fit()

    Pi = results_vecm.alpha@results_vecm.beta.T
    rankPi = np.linalg.matrix_rank(Pi)
    print(f'alpha = {results_vecm.alpha}')
    print(f'beta = {results_vecm.beta}')
    print(f'Pi = {Pi}')
    print(f'rank(Pi) = {rankPi}')

    return results_vecm, model_vecm

def irf_results(bidask_diff, model_vecm, irf=False):
    # make the VAR model
    model_var = VAR(bidask_diff[['log_midpoint_spot', 'log_midpoint_perp']])
    # Estimate VAR(p)
    results_var = model_var.fit(model_vecm.k_ar_diff + 1)
    # Assign impulse response functions (IRFs)
    irf = results_var.irf(10)

    # if irf:
    #     # Plot IRFs
    #     fig = irf.plot(orth=False,impulse='log_midpoint_spot',figsize=(6.5,4))
    #     fig.suptitle(" ")
    return results_var, irf


def plot_var_forecast_with_future(bidask_diff, results_var, start, steps=100, n_last=100, show_ci=True):
    """
    Plot a VAR model forecast with actual future data (if available).

    Parameters
    ----------
    df_train : pd.DataFrame
        Data used to train the model.
    df_future : pd.DataFrame
        Actual observed data beyond the training period (can be None).
    results : VARResults
        Fitted VAR model from statsmodels.tsa.api.VAR.fit().
    steps : int
        How many periods ahead to forecast.
    n_last : int
        How many past points to show before the forecast start.
    show_ci : bool
        Whether to plot 95% confidence intervals.
    """
    df_train = bidask_diff.iloc[:start]
    df_future = bidask_diff.iloc[start:start+steps]

    lag_order = results_var.k_ar
    forecast_input = df_train.values[-lag_order:]
    forecast_mean, lower, upper = results_var.forecast_interval(forecast_input, steps=steps)

    forecast_index = pd.date_range(
        start=df_train.index[-1],
        periods=steps + 1,
        freq=pd.infer_freq(df_train.index) or "D"
    )[1:]

    forecast_df = pd.DataFrame(forecast_mean, index=forecast_index, columns=df_train.columns)
    lower_df = pd.DataFrame(lower, index=forecast_index, columns=df_train.columns)
    upper_df = pd.DataFrame(upper, index=forecast_index, columns=df_train.columns)

    # Plot for each variable
    for col in df_train.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df_train.index[-n_last:], df_train[col].iloc[-n_last:], label='Train', color='C0')
        plt.plot(forecast_df.index, forecast_df[col], label='Forecast', color='C1', linestyle='--', linewidth=2)

        if df_future is not None:
            plt.plot(df_future.index, df_future[col], label='Actual Future', color='C2')

        if show_ci:
            plt.fill_between(forecast_df.index, lower_df[col], upper_df[col],
                             color='C1', alpha=0.2, label='95% CI')

        plt.title(f"VAR Forecast vs Actual for {col}")
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.legend()
        plt.tight_layout()
        plt.show()

def adf_test(data,const_trend):
    keys = ['Test Statistic','p-value','# of Lags','# of Obs']
    values = adfuller(data,regression=const_trend)
    test = pd.DataFrame.from_dict(dict(zip(keys,values[0:4])),
                                  orient='index',columns=[data.name])
    return test

def get_model_results(bidask_diff, window='1H', step='1H', maxlags=10):
    """
    Rolling VECM + Information Share computation.
    Returns DataFrame of alpha, beta, p-values, R², and IS metrics per window.
    """
    window = pd.Timedelta(window)
    step   = pd.Timedelta(step)

    bidask_diff = bidask_diff.sort_index()
    start = bidask_diff.index[0].floor('H')
    end   = bidask_diff.index[-1].ceil('H')

    idx = bidask_diff.index

    start = idx[0].floor('H')
    end   = idx[-1].ceil('H')

    # all rolling window start times
    starts = pd.date_range(start, end - window, freq=step)

    # map timestamps → integer index positions
    start_pos = idx.searchsorted(starts.values, side='left')
    end_pos   = idx.searchsorted((starts + window).values, side='right')

    error_chain = False
    results = []

    for current, i0, i1 in zip(starts, start_pos, end_pos):
        print(f"Processing window: {current} → {current + window}")

        if (i1 - i0) < 30:
            print(f"Skipping {current}: not enough data ({i1 - i0} rows)")
            continue

        subset = bidask_diff.iloc[i0:i1]
        
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                lo, _ = select_lag_coint(subset, maxlags=maxlags)


            cr = COINT_RANK

            with contextlib.redirect_stdout(io.StringIO()):
                results_vecm, model_vecm = fit_vecm(subset, lo, cr)
                # results_var, irf = irf_results(subset, model_vecm, False)

                resid = results_vecm.resid
                r2 = 1 - resid.var(axis=0) / np.var(
                    np.diff(subset[['log_midpoint_spot','log_midpoint_perp']], axis=0),
                    axis=0
                )

            alpha = results_vecm.alpha
            beta = results_vecm.beta
            alpha_se = getattr(results_vecm, "stderr_alpha", np.full_like(alpha, np.nan))
            beta_se  = getattr(results_vecm, "stderr_beta",  np.full_like(beta,  np.nan))

            with np.errstate(divide='ignore', invalid='ignore'):
                alpha_t = alpha / alpha_se
                beta_t  = beta / beta_se

            alpha_p = 2 * (1 - stats.norm.cdf(np.abs(alpha_t)))
            beta_p  = 2 * (1 - stats.norm.cdf(np.abs(beta_t)))

            # --- Compute Information Share ---
            pdm = compute_price_discovery_measures(results_vecm)

            # Default values
            const_out = getattr(results_vecm, "det_coef", None)          # outside constant
            const_in  = getattr(results_vecm, "det_coef_coint", None)    # inside constant

            # Default values
            # const_spot = np.nan
            # const_perp = np.nan
            # const_ci   = np.nan

            # # Outside constant (per equation)
            # if const_out is not None and const_out.size > 0:
            #     const_spot = const_out[0, 0]
            #     if const_out.shape[0] > 1:
            #         const_perp = const_out[1, 0]

            # # Inside constant (cointegration)
            # if const_in is not None and const_in.size > 0:
            #     const_ci = const_in[0, 0]

            results.append({
                'window_start': current,
                'window_end': current + window,
                'beta_spot': beta[0,0],
                'beta_perp': beta[1,0],
                'beta_spot_std': beta_se[0,0],
                'beta_perp_std': beta_se[1,0],
                'beta_spot_p': beta_p[0,0],
                'beta_perp_p': beta_p[1,0],
                'alpha_spot': alpha[0,0],
                'alpha_perp': alpha[1,0],
                'alpha_spot_std': alpha_se[0,0],
                'alpha_perp_std': alpha_se[1,0],
                'alpha_spot_p': alpha_p[0,0],
                'alpha_perp_p': alpha_p[1,0],
                'lag_order': lo,
                'r2_spot': r2[0],
                'r2_perp': r2[1],

                # CS
                'cs_spot': pdm['cs_spot'],
                'cs_perp': pdm['cs_perp'],

                # HIS
                'his_lower_spot': pdm['his_lower_spot'],
                'his_upper_spot': pdm['his_upper_spot'],
                'his_mid_spot': pdm['his_mid_spot'],
                'his_lower_perp': pdm['his_lower_perp'],
                'his_upper_perp': pdm['his_upper_perp'],
                'his_mid_perp': pdm['his_mid_perp'],

                # ILS
                'ils_mid_spot': pdm['ils_mid_spot'],
                'ils_mid_perp': pdm['ils_mid_perp'],
                'ils_valid': pdm['valid'],

                'n_obs': len(subset)
            })

            error_chain = False
            print(f"✓ Completed {current} (lag={lo}, r2_spot={r2[0]:.3f}, IS_spot={pdm['ils_mid_spot']:.2f})")

        except Exception as e:
            error_chain = True
            print(f"⚠️  Skipping {current}: {e}")

    # Convert results to DataFrame
    results = pd.DataFrame(results)
    # results['center_time'] = results['window_start'] + (window / 2)

    return results, error_chain

def rolling_mad_filter(s: pd.Series, window: int, z: float = 5.0, min_periods: int | None = None):
    """
    Returns a boolean mask: True = keep, False = drop as outlier,
    using rolling median and rolling MAD.
    """
    if min_periods is None:
        min_periods = max(10, window // 3)

    med = s.rolling(window, min_periods=min_periods).median()
    mad = (s - med).abs().rolling(window, min_periods=min_periods).median()

    # Convert MAD to a std-like scale (1.4826 makes MAD comparable to std for normal data)
    scale = 1.4826 * mad

    # avoid div-by-zero: if scale==0, treat as not outlier unless deviation is also 0
    zscore = (s - med).abs() / scale.replace(0, np.nan)
    keep = (zscore <= z) | zscore.isna()  # keep early points / zero-variance windows
    return keep

def rolling_z_filter(s: pd.Series, window: int, z: float = 4.0, min_periods: int | None = None):
    if min_periods is None:
        min_periods = max(10, window // 3)

    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()

    zscore = (s - mu).abs() / sd.replace(0, np.nan)
    keep = (zscore <= z) | zscore.isna()
    return keep

def rolling_outlier_mask(df: pd.DataFrame, cols, window: int, z: float = 5.0, method: str = "mad"):
    masks = []
    for c in cols:
        s = df[c].astype(float)
        if method == "mad":
            masks.append(rolling_mad_filter(s, window=window, z=z))
        else:
            masks.append(rolling_z_filter(s, window=window, z=z))
    out = masks[0].copy()
    for m in masks[1:]:
        out &= m
    return out

class VECMResults:
    def __init__(self, ticker, source, cm_um='um'):
        '''
        ticker: string
        start: datetime.datetime
        end: datetime.datetime
        agg: data aggregation interval
        period: periodicity of data sampling
        '''
        self.ticker = ticker
        self.source = source
        self.cm_um = cm_um

    def _get_and_parse_data(self, start, end, agg, lagdur='10s', **kwargs):
        maxlags = int(pd.Timedelta(lagdur) / pd.Timedelta(agg))
        data = TradeData(self.ticker, self.source, self.cm_um)
        data.grab_trades_data(end, (end-start).days)
        bidask_diff = data.agg_to_intervals(agg)
        # ff = data.get_funding_data(datetime.datetime.strftime(bidask_diff.index[0], '%Y%m%d'), datetime.datetime.strftime(bidask_diff.index[-1], '%Y%m%d'))
        ff = None
        results, error_chain = get_model_results(bidask_diff, maxlags=maxlags, **kwargs)

        return results, ff, error_chain
    
    def _get_lag_coint_data(self, start, end, agg, window, step):
        data = TradeData(self.ticker, self.source, self.cm_um)
        data.grab_trades_data(end, (end-start).days)
        bidask_diff = data.agg_to_intervals(agg)

        window = pd.Timedelta(window)
        step   = pd.Timedelta(step)

        bidask_diff = bidask_diff.sort_index()
        start = bidask_diff.index[0].floor('H')
        end   = bidask_diff.index[-1].ceil('H')

        idx = bidask_diff.index

        start = idx[0].floor('H')
        end   = idx[-1].ceil('H')

        # all rolling window start times
        starts = pd.date_range(start, end - window, freq=step)

        # map timestamps → integer index positions
        start_pos = idx.searchsorted(starts.values, side='left')
        end_pos   = idx.searchsorted((starts + window).values, side='right')

        results = []

        for current, i0, i1 in zip(starts, start_pos, end_pos):
            print(f"Processing window: {current} → {current + window}")

            if (i1 - i0) < 30:
                print(f"Skipping {current}: not enough data ({i1 - i0} rows)")
                continue

            subset = bidask_diff.iloc[i0:i1]
            
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    lo, co = select_lag_coint(subset, which='both')
                    results.append(pd.DataFrame({subset.index[0]:{'lag':lo, 'coint':co}}).transpose())
            except Exception as e:
                print(f"⚠️  Skipping {current}: {e}")
        
        return pd.concat(results)
    
    def get_lag_coint_multiperiod(self, start, end, agg, period, window, step):
        curr_start = end - datetime.timedelta(days=period)
        curr_end = end
        results = []
        while curr_start >= start:
            try:
                results_i = self._get_lag_coint_data(curr_start, curr_end, agg, window, step)
                results.append(results_i)
            except:
                pass
            
            curr_end = curr_end - datetime.timedelta(days=period)
            curr_start = curr_end - datetime.timedelta(days=period)

        results = pd.concat(results)
        return results
    
    def _parse_saved_result_files(self, agg, folder_name):
        import glob
        import os
        import re

        pattern = os.path.join(folder_name, f'results_{agg}_*.csv')
        rx = re.compile(rf"results_{re.escape(str(agg))}_(\d{{8}})_(\d{{8}})\.csv$")

        parsed = []
        for fp in glob.glob(pattern):
            m = rx.search(os.path.basename(fp))
            if not m:
                continue

            file_start = datetime.datetime.strptime(m.group(1), '%Y%m%d')
            # treat filename end as inclusive day coverage
            file_end = datetime.datetime.strptime(m.group(2), '%Y%m%d') + datetime.timedelta(days=1)

            # If possible, infer the *actual* coverage from the saved window columns.
            # This avoids missing reusable files when the filename dates are only approximate.
            try:
                tmp = pd.read_csv(fp, usecols=['window_start', 'window_end'])
                tmp['window_start'] = pd.to_datetime(tmp['window_start'])
                tmp['window_end'] = pd.to_datetime(tmp['window_end'])

                if not tmp.empty:
                    actual_start = tmp['window_start'].min()
                    actual_end = tmp['window_end'].max()

                    if pd.notna(actual_start):
                        file_start = actual_start.to_pydatetime()
                    if pd.notna(actual_end):
                        file_end = actual_end.to_pydatetime()
            except Exception:
                pass

            parsed.append({
                'path': fp,
                'start': file_start,
                'end': file_end
            })
        return parsed

    def _find_best_overlap_file(self, agg, req_start, req_end, folder_name):
        parsed_files = self._parse_saved_result_files(agg, folder_name)
        best = None
        best_overlap = datetime.timedelta(0)

        for f in parsed_files:
            # intersect [req_start, req_end) with [file_start, file_end)
            overlap_start = max(req_start, f['start'])
            overlap_end = min(req_end, f['end'])
            overlap = overlap_end - overlap_start

            if overlap > datetime.timedelta(0) and overlap > best_overlap:
                best = f
                best_overlap = overlap

        return best

    def _filter_results_to_window(self, results_i, curr_start, curr_end):
        results_i = results_i.copy()

        if 'window_start' in results_i.columns:
            results_i['window_start'] = pd.to_datetime(results_i['window_start'])
        if 'window_end' in results_i.columns:
            results_i['window_end'] = pd.to_datetime(results_i['window_end'])

        if 'window_start' in results_i.columns and 'window_end' in results_i.columns:
            results_i = results_i[
                (results_i['window_end'] > curr_start) &
                (results_i['window_start'] < curr_end)
            ].copy()
        elif 'window_start' in results_i.columns:
            results_i = results_i[
                (results_i['window_start'] >= curr_start) &
                (results_i['window_start'] < curr_end)
            ].copy()

        return results_i

    def get_data_multiperiod(self, start, end, agg, period, lagdur='10s', save_csv=True, folder_name='vecm_results', **kwargs):
        '''
        period: how frequently data is sampled, in days
        '''
        curr_start = end - datetime.timedelta(days=period)
        curr_end = end
        results = []

        while curr_end >= start:
            best_file = self._find_best_overlap_file(agg, curr_start, curr_end, folder_name)

            if best_file is not None:
                print(
                    f"Overlapping file found: {best_file['path']} "
                    f"for requested window {curr_start:%Y-%m-%d} to {curr_end:%Y-%m-%d}"
                )
                results_i = pd.read_csv(best_file['path'], index_col=0)
                # results_i = self._filter_results_to_window(results_i, curr_start, curr_end)
            else:
                results_i, ff_i, error_chain = self._get_and_parse_data(curr_start, curr_end, agg, lagdur=lagdur, **kwargs)
                if error_chain:
                    print('out of memory')
                    break

                # results_i = self._filter_results_to_window(results_i, curr_start, curr_end)

                if save_csv:
                    import os
                    os.makedirs(folder_name, exist_ok=True)
                    save_path = (
                        f"{folder_name}/results_{agg}_"
                        f"{datetime.datetime.strftime(curr_start, '%Y%m%d')}_"
                        f"{datetime.datetime.strftime(curr_end, '%Y%m%d')}.csv"
                    )
                    results_i.to_csv(save_path)
                    # ff_i.to_csv(f'{folder_name}/ff_{agg}_{datetime.datetime.strftime(curr_start, '%Y%m%d')}_{datetime.datetime.strftime(curr_end, '%Y%m%d')}.csv')

            if len(results_i) > 0:
                results.append(results_i)
            
            curr_end = curr_end - datetime.timedelta(days=period)
            curr_start = curr_end - datetime.timedelta(days=period)

        if results:
            results = pd.concat(results, ignore_index=False)

            if 'window_start' in results.columns:
                results['window_start'] = pd.to_datetime(results['window_start'])
            if 'window_end' in results.columns:
                results['window_end'] = pd.to_datetime(results['window_end'])

            dedupe_cols = [c for c in ['window_start', 'window_end'] if c in results.columns]
            if dedupe_cols:
                results = results.drop_duplicates(subset=dedupe_cols, keep='first')
            if 'window_start' in results.columns:
                results = results.sort_values('window_start')
            
            results = results[(results['window_start']>=start)&(results['window_end']<=end)]
        else:
            results = pd.DataFrame()

        # ff = pd.concat(ff)
        ff = None
        return results, ff
    