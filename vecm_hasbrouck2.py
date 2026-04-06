"""
This file contains the VECMHasbrouck2 class, which fits data to VECM models with 
flexible lag structures and computes price discovery measures.

The VECM formulation is that of Hasbrouck's 2021 paper "Price Discovery in High Resolution" (https://doi.org/10.1093/jjfinec/nbz027),
implemented through the SimpleMVAR class that is a python translation of Hasbrouck's MATLAB code.

GenAI was used to translate the MATLAB code into Python and to make SimpleMVAR compatible with lag structure inputs.
"""

from __future__ import annotations
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import datetime
from trade_data_pull import *
from timeout import *

@dataclass
class SimpleMVAR:
    prices: pd.DataFrame
    lag_structure: dict[str, list[tuple[int, int]]]
    latency: str
    interval: str = "1D"
    intercept: bool = True
    ecm: bool = True
    reference_col: Optional[str] = None
    ticker: str = 'BTCUSDT'
    source: str ='Binance'
    cm_um:str ='um'


    def __post_init__(self) -> None:
        if not isinstance(self.prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame.")
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise TypeError("prices must have a pandas DatetimeIndex.")
        if self.prices.shape[1] < 2:
            raise ValueError("prices must have at least 2 columns.")
        if not isinstance(self.lag_structure, dict) or not self.lag_structure:
            raise ValueError("lag_structure must be a non-empty dict.")
        if self.latency not in self.lag_structure:
            raise ValueError(f"latency '{self.latency}' not found in lag_structure.")

        self.prices = self.prices.astype(float).copy().sort_index()
        self.price_names = list(self.prices.columns)
        self.n_prices = self.prices.shape[1]

        if self.reference_col is None:
            self.reference_col = self.price_names[0]
        if self.reference_col not in self.price_names:
            raise ValueError("reference_col must be one of the dataframe columns.")

        self.lag_buckets = self._parse_lag_structure(self.lag_structure, self.latency)
        self.max_lag = max(end for _, end in self.lag_buckets)

        self.dp: Optional[pd.DataFrame] = None
        self.ec_terms: Optional[pd.DataFrame] = None
        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.design_index: Optional[pd.Index] = None

        self.x_names: list[str] = []
        self.y_names: list[str] = []

        self.b: Optional[np.ndarray] = None
        self.e_cov: Optional[np.ndarray] = None
        self.e_corr: Optional[np.ndarray] = None
        self.seb: Optional[np.ndarray] = None
        self.tb: Optional[np.ndarray] = None
        self.resid: Optional[np.ndarray] = None

    @staticmethod
    def _parse_lag_structure(
        lag_structure: dict[str, list[tuple[int, int]]],
        latency: str,
    ) -> list[tuple[int, int]]:
        buckets = lag_structure[latency]
        if not isinstance(buckets, list) or not buckets:
            raise ValueError("Selected lag structure must be a non-empty list of tuples.")

        cleaned: list[tuple[int, int]] = []
        seen = set()

        for item in buckets:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Each lag bucket must be a tuple (start_lag, end_lag).")
            start, end = item
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError("Lag bucket bounds must be integers.")
            if start < 1 or end < start:
                raise ValueError(f"Invalid lag bucket {(start, end)}.")

            for lag in range(start, end + 1):
                if lag in seen:
                    raise ValueError(f"Lag {lag} appears in more than one bucket.")
                seen.add(lag)

            cleaned.append((start, end))

        return cleaned

    def _build_ec_terms(self, prices: pd.DataFrame) -> pd.DataFrame:
        ref = prices[self.reference_col] # this is the spot series
        others = [c for c in self.price_names if c != self.reference_col] # this is the perp series

        ec = {}
        for c in others:
            # we take the difference at each step between the spot and perp 
            # since beta is [1, -1]
            ec[f"{self.reference_col}-{c}"] = ref - prices[c]

        return pd.DataFrame(ec, index=prices.index)

    def _build_bucketed_lags(self, dp: pd.DataFrame) -> tuple[list[pd.DataFrame], list[str]]:
        # dp is change in price
        x_parts: list[pd.DataFrame] = []
        x_names: list[str] = []

        for start, end in self.lag_buckets:
            block = None
            for lag in range(start, end + 1):
                # go through each tuple of specified lag terms
                lagged = dp.shift(lag) # for each lag, shift the dataframe by specified amount
                block = lagged if block is None else block.add(lagged, fill_value=np.nan)
                # after each lag, add the lagged series to the indep variable matrix (X)

            block = block.copy()
            if start == end:
                block.columns = [f"d{c}(t-{start})" for c in dp.columns] # in the case of a free moving coefficient
            else:
                block.columns = [f"d{c}(t-{start}:{end})" for c in dp.columns] # in the case of multiple coefficients set to be the same
                # can have dspot, dperp for the same lag bucket

            x_parts.append(block)
            x_names.extend(block.columns.tolist())

        return x_parts, x_names

    def _build_design_for_prices(
        self, prices: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, pd.Index, list[str], list[str]]:
        dp = prices.diff()
        ec_terms = self._build_ec_terms(prices) if self.ecm else None

        y = dp.copy() # this is the dependent variable on the left of equation
        y.columns = [f"d{c}" for c in self.price_names]
        y_names = list(y.columns)

        x_parts: list[pd.DataFrame] = [] # the complete X matrix
        x_names: list[str] = []

        if self.intercept:
            const = pd.DataFrame({"const": 1.0}, index=prices.index)
            # adding intercept term (not used)
            x_parts.append(const)
            x_names.append("const")

        lagged_parts, lagged_names = self._build_bucketed_lags(dp)
        # this sets up the gamma part on the right
        x_parts.extend(lagged_parts)
        x_names.extend(lagged_names)

        if self.ecm and ec_terms is not None:
            ec_lag = ec_terms.shift(1).copy()
            x_parts.append(ec_lag)
            x_names.extend(ec_lag.columns.tolist())

        X_df = pd.concat(x_parts, axis=1) # includes the ecm terms, gamma terms, intercept (if any)
        XY = pd.concat([y, X_df], axis=1).dropna() # add on the delta price

        Y = XY[y_names].to_numpy() # includes delta price for spot and perp markets
        X = XY[x_names].to_numpy() # includes constant, lagged terms, error correction terms

        return X, Y, XY.index, x_names, y_names

    def fit(self) -> "SimpleMVAR":
        X, Y, idx, x_names, y_names = self._build_design_for_prices(self.prices)

        self.X = X
        self.Y = Y
        self.design_index = idx # enumeration of all the lag terms?
        self.x_names = x_names
        self.y_names = y_names
        self.dp = self.prices.diff()
        self.ec_terms = self._build_ec_terms(self.prices) if self.ecm else None

        # solve the equation like a linear system for alpha and gammas
        xpx = X.T @ X
        xpy = X.T @ Y
        xpxi = np.linalg.inv(xpx)

        self.b = xpxi @ xpy
        self.resid = Y - X @ self.b

        n_obs = Y.shape[0]
        self.e_cov = (self.resid.T @ self.resid) / n_obs

        diag_cov = np.diag(self.e_cov)
        inv_sd = np.diag(1.0 / np.sqrt(diag_cov))
        self.e_corr = inv_sd @ self.e_cov @ inv_sd

        self.seb = np.sqrt(np.outer(np.diag(xpxi), diag_cov))
        self.tb = self.b / self.seb

        return self

    def gamma_matrix(self) -> Optional[np.ndarray]:
        if not self.ecm:
            return None
        if self.b is None:
            raise ValueError("Call fit() first.")
        k_ec = self.n_prices - 1 # all coefficients except the alpha ones
        return self.b[-k_ec:, :].T

    def beta_matrix(self) -> np.ndarray:
        if self.n_prices != 2:
            raise ValueError("beta_matrix() currently only supports 2 series.")
        return np.array([[1.0], [-1.0]])

    def phi_matrices(self) -> np.ndarray:
        if self.b is None:
            raise ValueError("Call fit() first.")

        phi = np.zeros((self.n_prices, self.n_prices, self.max_lag))
        row_start = 1 if self.intercept else 0

        for bucket_idx, (start, end) in enumerate(self.lag_buckets):
            r0 = row_start + bucket_idx * self.n_prices
            r1 = r0 + self.n_prices
            block = self.b[r0:r1, :].T 
            # basically fold the error correction term back into the equation

            for lag in range(start, end + 1):
                phi[:, :, lag - 1] = block

        return phi

    def irf(self, n_ahead: int) -> np.ndarray:
        if self.b is None:
            raise ValueError("Call fit() first.")

        phi = self.phi_matrices()
        gamma = self.gamma_matrix()

        T = self.max_lag + n_ahead + 1
        irf = np.zeros((self.n_prices, n_ahead + 1, self.n_prices))
        mphi = np.reshape(np.flip(phi, axis=2), (self.n_prices, -1))

        if self.ecm:
            B = np.hstack([
                np.ones((self.n_prices - 1, 1)),
                -np.eye(self.n_prices - 1),
            ])

        intercept_vec = self.b[0, :] if self.intercept else np.zeros(self.n_prices)

        for shock in range(self.n_prices):
            dp = np.zeros((self.n_prices, T))
            p = np.zeros(self.n_prices)

            e0 = np.zeros(self.n_prices)
            e0[shock] = 1.0

            for t in range(self.max_lag, T):
                d = intercept_vec.copy()

                if self.ecm and gamma is not None:
                    d = d + gamma @ (B @ p)

                v = dp[:, t - self.max_lag:t].reshape(-1)
                dp[:, t] = d + mphi @ v

                if t == self.max_lag:
                    dp[:, t] += e0

                p = p + dp[:, t]

            irf[:, :, shock] = np.cumsum(dp[:, self.max_lag:T], axis=1)

        return irf

    def summary(self) -> pd.DataFrame:
        if self.b is None or self.tb is None or self.seb is None:
            raise ValueError("Call fit() first.")

        rows = []
        for i, xname in enumerate(self.x_names):
            for j, yname in enumerate(self.y_names):
                rows.append({
                    "regressor": xname,
                    "equation": yname,
                    "coef": self.b[i, j],
                    "t_stat": self.tb[i, j],
                    "std_err": self.seb[i, j],
                })

        return pd.DataFrame(rows)

    @staticmethod
    def _price_discovery_from_outputs(
        price_names: list[str],
        phi: np.ndarray,
        e_cov: np.ndarray,
        alpha: np.ndarray,
    ) -> dict:
        """
        2-series case, fixed beta = [1, -1]'.
        """
        if len(price_names) != 2:
            raise ValueError("Price discovery calculations currently only support 2 series.")

        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (2, 1):
            raise ValueError(f"Expected alpha shape (2, 1), got {alpha.shape}.")

        gamma_sum = np.sum(phi, axis=2) 
        beta = np.array([[1.0], [-1.0]])
        beta_perp = np.array([[1.0], [1.0]])
        alpha_perp = np.array([[alpha[1, 0]], [-alpha[0, 0]]]) 
        # orthogonal complement of alpha, shows direction of movement for efficient price

        cs_denom = float(alpha_perp.sum())
        if np.isclose(cs_denom, 0.0):
            cs = np.array([np.nan, np.nan])
        else:
            cs = (alpha_perp / cs_denom).flatten()

        denom = float(alpha_perp.T @ (np.eye(2) - gamma_sum) @ beta_perp)
        if np.isclose(denom, 0.0):
            raise ValueError("Long-run impact denominator is near zero; shares are unstable.")

        C = beta_perp @ (alpha_perp.T / denom)
        psi = C[0, :] # long-run impact matrix

        F = np.linalg.cholesky(e_cov) # cholesky decomposition of residuals
        den = float(psi @ e_cov @ psi.T)
        his_lower = ((psi @ F) ** 2) / den

        P = np.array([[0.0, 1.0], [1.0, 0.0]])
        e_cov_rev = P @ e_cov @ P # reversed ordering
        psi_rev = psi @ P
        F_rev = np.linalg.cholesky(e_cov_rev)
        his_upper_rev = ((psi_rev @ F_rev) ** 2) / den
        his_upper = his_upper_rev[::-1]

        his_mid = 0.5 * (his_lower + his_upper)

        with np.errstate(divide="ignore", invalid="ignore"):
            il_raw = np.abs(his_mid / cs)
            ils_mid = il_raw / np.nansum(il_raw)

        return {
            "series": price_names,
            "alpha": alpha.flatten(),
            "beta": beta.flatten(),
            "alpha_perp": alpha_perp.flatten(),
            "cs": cs,
            "his_lower": his_lower,
            "his_upper": his_upper,
            "his_mid": his_mid,
            "ils_mid": ils_mid,
            "long_run_impact": C,
            "psi": psi,
            "omega": e_cov,
        }

    @timeout(360)
    def fit_by_interval(self, n_ahead_irf: int = 100) -> dict[pd.Timestamp, dict]:
        """
        Estimate one model per interval and return alpha/beta/IRF/CS/HIS/ILS.

        Returns
        -------
        dict
            Keys are interval labels. Values are dicts containing:
            - b, e_cov, e_corr, seb, tb, resid, n_obs
            - alpha, beta, phi, irf
            - cs, his_lower, his_upper, his_mid, ils_mid
        """
        results: dict[pd.Timestamp, dict] = {}

        grouped = self.prices.groupby(pd.Grouper(freq=self.interval))

        for label, chunk in grouped:
            if chunk.empty:
                continue

            X, Y, idx, x_names, y_names = self._build_design_for_prices(chunk)
            if len(Y) == 0:
                continue

            try:
                xpx = X.T @ X
                xpy = X.T @ Y
                xpxi = np.linalg.inv(xpx)

                b = xpxi @ xpy
                resid = Y - X @ b
                n_obs = Y.shape[0]

                e_cov = (resid.T @ resid) / n_obs
                diag_cov = np.diag(e_cov)
                inv_sd = np.diag(1.0 / np.sqrt(diag_cov))
                e_corr = inv_sd @ e_cov @ inv_sd

                seb = np.sqrt(np.outer(np.diag(xpxi), diag_cov))
                tb = b / seb

                # alpha
                if self.ecm:
                    k_ec = self.n_prices - 1
                    alpha = b[-k_ec:, :].T
                else:
                    alpha = None

                # beta
                beta = self.beta_matrix() if self.n_prices == 2 else None

                # phi
                phi = np.zeros((self.n_prices, self.n_prices, self.max_lag))
                row_start = 1 if self.intercept else 0
                for bucket_idx, (start, end) in enumerate(self.lag_buckets):
                    r0 = row_start + bucket_idx * self.n_prices
                    r1 = r0 + self.n_prices
                    block = b[r0:r1, :].T
                    for lag in range(start, end + 1):
                        phi[:, :, lag - 1] = block

                # interval IRF
                irf = self._irf_from_outputs(b=b, phi=phi, n_ahead=n_ahead_irf)

                out = {
                    "index": idx,
                    "x_names": x_names,
                    "y_names": y_names,
                    "n_obs": n_obs,
                    "b": b,
                    "resid": resid,
                    "e_cov": e_cov,
                    "e_corr": e_corr,
                    "seb": seb,
                    "tb": tb,
                    "alpha": alpha,
                    "beta": beta,
                    "phi": phi,
                    "irf": irf,
                }

                if self.ecm and self.n_prices == 2:
                    pd_out = self._price_discovery_from_outputs(
                        price_names=self.price_names,
                        phi=phi,
                        e_cov=e_cov,
                        alpha=alpha,
                    )
                    out.update(pd_out)

                results[label] = out

            except np.linalg.LinAlgError:
                results[label] = {
                    "index": idx,
                    "x_names": x_names,
                    "y_names": y_names,
                    "n_obs": len(Y),
                    "error": "Singular matrix during estimation.",
                }

        return results

    def _irf_from_outputs(self, b: np.ndarray, phi: np.ndarray, n_ahead: int) -> np.ndarray:
        gamma = None
        if self.ecm:
            k_ec = self.n_prices - 1
            gamma = b[-k_ec:, :].T

        T = self.max_lag + n_ahead + 1
        irf = np.zeros((self.n_prices, n_ahead + 1, self.n_prices))
        mphi = np.reshape(np.flip(phi, axis=2), (self.n_prices, -1))

        if self.ecm:
            B = np.hstack([
                np.ones((self.n_prices - 1, 1)),
                -np.eye(self.n_prices - 1),
            ])

        intercept_vec = b[0, :] if self.intercept else np.zeros(self.n_prices)

        for shock in range(self.n_prices):
            dp = np.zeros((self.n_prices, T))
            p = np.zeros(self.n_prices)
            e0 = np.zeros(self.n_prices)
            e0[shock] = 1.0

            for t in range(self.max_lag, T):
                d = intercept_vec.copy()

                if self.ecm and gamma is not None:
                    d = d + gamma @ (B @ p)

                v = dp[:, t - self.max_lag:t].reshape(-1)
                dp[:, t] = d + mphi @ v

                if t == self.max_lag:
                    dp[:, t] += e0

                p = p + dp[:, t]

            irf[:, :, shock] = np.cumsum(dp[:, self.max_lag:T], axis=1)

        return irf

    def shares_table(self, interval_results: dict[pd.Timestamp, dict]) -> pd.DataFrame:
        rows = []
        for label, res in interval_results.items():
            if "cs" not in res:
                continue
            for i, name in enumerate(res["series"]):
                rows.append({
                    "interval": label,
                    "series": name,
                    "alpha": res["alpha"][i],
                    "beta": res["beta"][i],
                    "CS": res["cs"][i],
                    "HIS_lower": res["his_lower"][i],
                    "HIS_upper": res["his_upper"][i],
                    "HIS_mid": res["his_mid"][i],
                    "ILS_mid": res["ils_mid"][i],
                    "n_obs": res["n_obs"],
                })
        return pd.DataFrame(rows)

    def interval_summary_table(self, label, res: dict) -> pd.DataFrame:
        """
        Convert one interval result dict from fit_by_interval() into a long coefficient table.
        """
        if "b" not in res or "tb" not in res or "seb" not in res:
            return pd.DataFrame()

        rows = []
        for i, xname in enumerate(res["x_names"]):
            for j, yname in enumerate(res["y_names"]):
                rows.append({
                    "interval": label,
                    "ticker": self.ticker,
                    "source": self.source,
                    "cm_um": self.cm_um,
                    "latency": self.latency,
                    "window_interval": self.interval,
                    "regressor": xname,
                    "equation": yname,
                    "coef": res["b"][i, j],
                    "t_stat": res["tb"][i, j],
                    "std_err": res["seb"][i, j],
                    "n_obs": res.get("n_obs", np.nan),
                })

        return pd.DataFrame(rows)

    def all_interval_summaries(self, interval_results: dict[pd.Timestamp, dict]) -> pd.DataFrame:
        """
        Build one long summary DataFrame for all fitted intervals.
        """
        out = []
        for label, res in interval_results.items():
            df = self.interval_summary_table(label, res)
            if not df.empty:
                out.append(df)

        if not out:
            return pd.DataFrame(
                columns=[
                    "interval", "ticker", "source", "cm_um", "latency", "window_interval",
                    "regressor", "equation", "coef", "t_stat", "std_err", "n_obs"
                ]
            )

        return pd.concat(out, ignore_index=True)


class VECMHasbrouck2:
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

    # @timeout(480)
    def _get_and_parse_data(self, start, end, aggs, interval, lag_structure=None, lag_is=[], save_csv=True, folder_name='vecm_hasbrouck2_um', prefix='hasbrouck2'):
        data_called = False
        shares_dict = {}
        summary_dict = {}
        for agg in aggs:
            data_agg = False
            if lag_structure is None:
                for i in lag_is:
                    try:
                        shares_df = pd.read_csv(f'{folder_name}/{prefix}_{interval.lower()}_{agg.lower()}_{i}_{self.cm_um}_results_{datetime.datetime.strftime(start, '%Y%m%d')}_{datetime.datetime.strftime(end, '%Y%m%d')}.csv', index_col=0)
                        summary_df = pd.read_csv(f'{folder_name}/{prefix}_{interval.lower()}_{agg.lower()}_{i}_{self.cm_um}_summary_{datetime.datetime.strftime(start, '%Y%m%d')}_{datetime.datetime.strftime(end, '%Y%m%d')}.csv', index_col=0)
                        print('File found, reading from file...')
                    
                    except FileNotFoundError:
                        if not data_called:
                            data = TradeData(self.ticker, self.source, self.cm_um)
                            data.grab_trades_data(end, (end-start).days)
                            data_called = True
                        if not data_agg:
                            bidask_diff = data.agg_to_intervals(agg)
                            data_agg = True

                        lag_structure2 = generate_multiple_lags(i, ['10ms', '50ms', '100ms', '200ms', '500ms', '1s'], '10s')
                        model = SimpleMVAR(
                            ticker = self.ticker,
                            source = self.source,
                            cm_um = self.cm_um,
                            prices=bidask_diff[["log_midpoint_spot", "log_midpoint_perp"]],
                            lag_structure=lag_structure2,
                            latency=agg,
                            interval=interval,
                            intercept=True,
                            ecm=True,
                            reference_col="log_midpoint_spot",
                        )
                        try:
                            results = model.fit_by_interval(n_ahead_irf=200)
                            shares_df = model.shares_table(results)
                            summary_df = model.all_interval_summaries(results)

                            if save_csv:
                                shares_df.to_csv(f'{folder_name}/{prefix}_{interval.lower()}_{agg.lower()}_{i}_{self.cm_um}_results_{datetime.datetime.strftime(start, '%Y%m%d')}_{datetime.datetime.strftime(end, '%Y%m%d')}.csv')
                                summary_df.to_csv(f'{folder_name}/{prefix}_{interval.lower()}_{agg.lower()}_{i}_{self.cm_um}_summary_{datetime.datetime.strftime(start, '%Y%m%d')}_{datetime.datetime.strftime(end, '%Y%m%d')}.csv')
                        except:
                            shares_df = pd.DataFrame()
                            summary_df = pd.DataFrame()
                            print("error")

                    shares_dict[(agg, i)] = shares_df
                    summary_dict[(agg, i)] = summary_df

            else:
                try:
                    shares_df = pd.read_csv(f'{folder_name}/{prefix}_results_{datetime.datetime.strftime(start, '%Y%m%d')}_{datetime.datetime.strftime(end, '%Y%m%d')}.csv', index_col=0)
                    summary_df = pd.read_csv(f'{folder_name}/{prefix}_summary_{datetime.datetime.strftime(start, '%Y%m%d')}_{datetime.datetime.strftime(end, '%Y%m%d')}.csv', index_col=0)
                    print('File found, reading from file...')
                
                except FileNotFoundError:
                    model = SimpleMVAR(
                        ticker = self.ticker,
                        source = self.source,
                        cm_um = self.cm_um,
                        prices=bidask_diff[["log_midpoint_spot", "log_midpoint_perp"]],
                        lag_structure=lag_structure,
                        latency=agg,
                        interval=interval,
                        intercept=True,
                        ecm=True,
                        reference_col="log_midpoint_spot",
                    )
                    try:
                        results = model.fit_by_interval(n_ahead_irf=200)
                        shares_df = model.shares_table(results)
                        summary_df = model.all_interval_summaries(results)

                        if save_csv:
                            shares_df.to_csv(f'{folder_name}/{prefix}_results_{datetime.datetime.strftime(start, '%Y%m%d')}_{datetime.datetime.strftime(end, '%Y%m%d')}.csv')
                            summary_df.to_csv(f'{folder_name}/{prefix}_summary_{datetime.datetime.strftime(start, '%Y%m%d')}_{datetime.datetime.strftime(end, '%Y%m%d')}.csv')
                    except:
                        shares_df = pd.DataFrame()
                        summary_df = pd.DataFrame()
                        print("error")

                shares_dict[(agg, 0)] = shares_df
                summary_dict[(agg, 0)] = summary_df

        return shares_dict, summary_dict
    
    def get_data_multiperiod(self, start, end, aggs, interval, period, **kwargs):
        '''
        period: how frequently data is sampled, in days
        '''
        curr_start = end - datetime.timedelta(days=period)
        curr_end = end
        results_dict = {}
        summary_dict = {}
        while curr_start >= start:
            results_i, summary_i = self._get_and_parse_data(curr_start, curr_end, aggs, interval, **kwargs)
            for key in results_i.keys():
                if key not in results_dict.keys():
                    results_dict[key] = [results_i[key]]
                    summary_dict[key] = [summary_i[key]]
                else:
                    results_dict[key].append(results_i[key])
                    summary_dict[key].append(summary_i[key])
            
            curr_end = curr_end - datetime.timedelta(days=period)
            curr_start = curr_end - datetime.timedelta(days=period)
            print(results_dict.keys())

        if curr_start < start and curr_end == end:
            results_i, summary_i = self._get_and_parse_data(start, curr_end, aggs, interval, **kwargs)
            for key in results_i.keys():
                if key not in results_dict.keys():
                    results_dict[key] = [results_i[key]]
                    summary_dict[key] = [summary_i[key]]
                else:
                    results_dict[key].append(results_i[key])
                    summary_dict[key].append(summary_i[key])

        for key in results_dict.keys():
            results_dict[key] = pd.concat(results_dict[key])
            summary_dict[key] = pd.concat(summary_dict[key])

        return results_dict, summary_dict

def read_files(start, end, aggs, interval, lags, cm_um, prefix, folder_name):
    folder = Path(folder_name)
    df_dict = {}

    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    for agg in aggs:
        for lag in lags:
            stem = "_".join([prefix, interval.lower(), agg.lower(), str(lag), cm_um])
            pattern = re.compile(
                rf"^{re.escape(stem)}_results_(\d{{8}})_(\d{{8}})\.csv$"
            )

            matched = []
            for f in folder.iterdir():
                if not f.is_file():
                    continue

                m = pattern.match(f.name)
                if not m:
                    continue

                file_start, file_end = m.groups()

                # overlap condition
                if file_start <= end_str and file_end >= start_str:
                    matched.append(pd.read_csv(f, index_col=0))

            if matched:
                df = pd.concat(matched).drop_duplicates()
                df['interval'] = pd.to_datetime(df['interval'])
                # 🔑 enforce datetime + filter
                df = df[(df['interval'] >= start) & (df['interval'] <= end)]

                df_dict[(agg, lag)] = df

    return df_dict
    
def price_discovery_shares(model):
    """
    Compute CS, Hasbrouck IS bounds/midpoint, and ILS for a fitted 2-price SimpleMVAR.

    Assumes:
    - exactly 2 price series
    - cointegration term is p1 - p2
    - model.fit() has already been called

    Returns
    -------
    dict
        Contains alpha, cs, his_lower, his_upper, his_mid, ils_mid, long_run_impact, psi
    """
    if model.b is None:
        raise ValueError("Call fit() first.")
    if model.n_prices != 2:
        raise ValueError("This helper currently assumes exactly 2 price series.")

    # ECM loading matrix alpha: shape (2, 1)
    alpha = model.gamma_matrix()
    if alpha is None:
        raise ValueError("Model must be estimated with ecm=True.")

    alpha = np.asarray(alpha, dtype=float)
    if alpha.shape != (2, 1):
        raise ValueError(f"Expected alpha shape (2,1), got {alpha.shape}.")

    # Sum of lagged-difference coefficient matrices
    phi = model.phi_matrices()   # shape (2, 2, max_lag)
    gamma_sum = np.sum(phi, axis=2)

    omega = np.asarray(model.e_cov, dtype=float)

    # Cointegration setup for beta = [1, -1]'
    beta_perp = np.array([[1.0], [1.0]])
    alpha_perp = np.array([[alpha[1, 0]], [-alpha[0, 0]]])

    # -------------------------
    # Component shares (CS)
    # -------------------------
    cs = (alpha_perp / alpha_perp.sum()).flatten()

    # -------------------------
    # Long-run impact matrix C
    # -------------------------
    denom = float(alpha_perp.T @ (np.eye(2) - gamma_sum) @ beta_perp)
    if np.isclose(denom, 0.0):
        raise ValueError("Long-run impact denominator is near zero; shares are unstable.")

    C = beta_perp @ (alpha_perp.T / denom)

    # In the 2-price / 1-common-trend case, both rows are identical
    psi = C[0, :]   # row vector shape (2,)

    # -------------------------
    # Hasbrouck IS: lower ordering
    # -------------------------
    F = np.linalg.cholesky(omega)
    num_lower = (psi @ F) ** 2
    den = float(psi @ omega @ psi.T)
    his_lower = num_lower / den

    # -------------------------
    # Hasbrouck IS: reversed ordering
    # -------------------------
    P = np.array([[0.0, 1.0],
                  [1.0, 0.0]])

    omega_rev = P @ omega @ P
    psi_rev = psi @ P
    F_rev = np.linalg.cholesky(omega_rev)

    num_upper_rev_order = (psi_rev @ F_rev) ** 2
    his_upper_rev_order = num_upper_rev_order / den

    # map reversed-order shares back to original variable order
    his_upper = his_upper_rev_order[::-1]

    # midpoint
    his_mid = 0.5 * (his_lower + his_upper)

    # -------------------------
    # Information leadership shares (ILS)
    # -------------------------
    # Uses midpoint HIS by default
    il_raw = np.abs(his_mid / cs)
    ils_mid = il_raw / il_raw.sum()

    out = {
        "series": model.price_names,
        "alpha": alpha.flatten(),
        "alpha_perp": alpha_perp.flatten(),
        "cs": cs,
        "his_lower": his_lower,
        "his_upper": his_upper,
        "his_mid": his_mid,
        "ils_mid": ils_mid,
        "long_run_impact": C,
        "psi": psi,
        "omega": omega,
    }

    return out

def shares_table(shares):
    return pd.DataFrame({
        "series": shares["series"],
        "CS": shares["cs"],
        "HIS_lower": shares["his_lower"],
        "HIS_upper": shares["his_upper"],
        "HIS_mid": shares["his_mid"],
        "ILS_mid": shares["ils_mid"],
    })

def generate_lag_buckets(base, latency, max_length='10s'):
    maxlags = int(pd.Timedelta(max_length) / pd.Timedelta(latency))
    lag_buckets = [(1,1)]
    curr_max = 1
    curr_power = 0
    if base > 1:
        while curr_max < maxlags:
            curr_power += 1
            if base ** curr_power > maxlags:
                lag_buckets.append((curr_max+1, maxlags))
            else:
                lag_buckets.append((curr_max+1,base ** curr_power))
            curr_max = base ** curr_power
    else:
        lag_buckets = [(i,i) for i in range(1,maxlags+1)]
    return lag_buckets

def generate_multiple_lags(base, latencies, max_length='10s'):
    buckets_dict = {}
    for latency in latencies:
        buckets_dict[latency] = generate_lag_buckets(base, latency, max_length)
    return buckets_dict

def find_period_mean(results, periods):
    """
    results is a dict of dataframes
    periods is a list of tuples consisting of start and end dates
    """
    period_list = []
    for latency, base in results.keys():
        for start, end in periods: 
            if type(start) == str:
                start_dt = datetime.datetime.strptime(start, '%Y%m%d')
                end_dt = datetime.datetime.strptime(end, '%Y%m%d')
            else:
                start_dt = start
                end_dt = end
            results[(latency, base)]['interval'] = pd.to_datetime(results[(latency, base)]['interval'])
            df = results[(latency, base)][(results[(latency, base)]['interval'] >= start_dt)&(results[(latency, base)]['interval'] <= end_dt)]
            df = df.groupby('series')[['CS', 'HIS_lower', 'HIS_upper', 'HIS_mid', 'ILS_mid']].mean().reset_index()
            df['latency'] = latency
            df['base'] = base
            df['period'] = f'{start},{end}'
            period_list.append(df)
    return period_list
