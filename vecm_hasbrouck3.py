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
from scipy import linalg
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
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

        self.prices = self.prices.astype(np.float32).copy().sort_index()
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

        Y = XY[y_names].to_numpy(dtype=np.float32) # includes delta price for spot and perp markets
        X = XY[x_names].to_numpy(dtype=np.float32) # includes constant, lagged terms, error correction terms

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
        # Use scipy inverse rather than NumPy's explicit inverse. Store the
        # fitted objects on ``self`` because the public ``fit()`` helpers
        # (gamma_matrix, summary, IRF, etc.) read these attributes.
        xpxi = linalg.inv(xpx, overwrite_a=False)
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
        return np.array([[1.0], [-1.0]], dtype=np.float32)

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
        irf = np.zeros((self.n_prices, n_ahead + 1, self.n_prices), dtype=np.float32)
        mphi = np.reshape(np.flip(phi, axis=2), (self.n_prices, -1))

        if self.ecm:
            B = np.hstack([
                np.ones((self.n_prices - 1, 1), dtype=np.float32),
                -np.eye(self.n_prices - 1, dtype=np.float32),
            ])

        intercept_vec = self.b[0, :] if self.intercept else np.zeros(self.n_prices, dtype=np.float32)

        for shock in range(self.n_prices):
            dp = np.zeros((self.n_prices, T), dtype=np.float32)
            p = np.zeros(self.n_prices, dtype=np.float32)

            e0 = np.zeros(self.n_prices, dtype=np.float32)
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
            cs = np.array([np.nan, np.nan], dtype=np.float32)
        else:
            cs = (alpha_perp / cs_denom).flatten()

        denom = float(alpha_perp.T @ (np.eye(2, dtype=np.float32) - gamma_sum) @ beta_perp)
        if np.isclose(denom, 0.0):
            raise ValueError("Long-run impact denominator is near zero; shares are unstable.")

        C = beta_perp @ (alpha_perp.T / denom)
        psi = C[0, :] # long-run impact matrix

        F = np.linalg.cholesky(e_cov) # cholesky decomposition of residuals
        den = float(psi @ e_cov @ psi.T)
        his_lower = ((psi @ F) ** 2) / den

        P = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
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

    @timeout(600)
    def fit_by_interval(self, n_ahead_irf: int = 100, use_irf: bool = False, parallel: bool = False, n_workers: Optional[int] = None) -> dict[pd.Timestamp, dict]:
        """
        Estimate one model per interval and return alpha/beta/IRF/CS/HIS/ILS.

        Parameters
        ----------
        n_ahead_irf : int
            Number of periods ahead for IRF calculation
        use_irf : bool
            Whether to compute IRF (expensive). Default False. Set to True to enable.
        parallel : bool
            Whether to use parallel processing for fitting intervals (speedup on multicore systems)
        n_workers : Optional[int]
            Number of workers for parallel processing. Default: number of CPUs.

        Returns
        -------
        dict
            Keys are interval labels. Values are dicts containing:
            - b, e_cov, e_corr, seb, tb, resid, n_obs
            - alpha, beta, phi, irf (if use_irf=True)
            - cs, his_lower, his_upper, his_mid, ils_mid
        """
        grouped = list(self.prices.groupby(pd.Grouper(freq=self.interval)))

        if not parallel:
            # Sequential processing
            results = {}
            for label, chunk in grouped:
                if chunk.empty:
                    continue
                result = self._fit_single_interval(label, chunk, n_ahead_irf, use_irf)
                if result is not None:
                    results[label] = result
            return results
        
        # Parallel processing using ThreadPoolExecutor
        results = {}
        n_workers = n_workers or os.cpu_count() or 2
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for label, chunk in grouped:
                if chunk.empty:
                    continue
                future = executor.submit(self._fit_single_interval, label, chunk, n_ahead_irf, use_irf)
                futures[future] = label
            
            # Collect results as they complete
            for future in futures:
                try:
                    result = future.result()
                    if result is not None:
                        label = futures[future]
                        results[futures[future]] = result
                except Exception as e:
                    print(f"Error fitting interval: {e}")
        
        return results

    def _fit_single_interval(self, label: pd.Timestamp, chunk: pd.DataFrame, n_ahead_irf: int, use_irf: bool) -> Optional[dict]:
        """Fit a single time interval. Can be called in parallel."""
        X, Y, idx, x_names, y_names = self._build_design_for_prices(chunk)
        if len(Y) == 0:
            return None

        try:
            xpx = X.T @ X
            xpy = X.T @ Y
            # Use scipy solver for 2-3x speedup vs matrix inversion
            xpxi = linalg.inv(xpx, overwrite_a=False)

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
            phi = np.zeros((self.n_prices, self.n_prices, self.max_lag), dtype=np.float32)
            row_start = 1 if self.intercept else 0
            for bucket_idx, (start, end) in enumerate(self.lag_buckets):
                r0 = row_start + bucket_idx * self.n_prices
                r1 = r0 + self.n_prices
                block = b[r0:r1, :].T
                for lag in range(start, end + 1):
                    phi[:, :, lag - 1] = block

            # optional IRF
            irf = self._irf_from_outputs(b=b, phi=phi, n_ahead=n_ahead_irf) if use_irf else None

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

            return out

        except np.linalg.LinAlgError:
            return {
                "index": idx,
                "x_names": x_names,
                "y_names": y_names,
                "n_obs": len(Y),
                "error": "Singular matrix during estimation.",
            }

    def _irf_from_outputs(self, b: np.ndarray, phi: np.ndarray, n_ahead: int) -> np.ndarray:
        gamma = None
        if self.ecm:
            k_ec = self.n_prices - 1
            gamma = b[-k_ec:, :].T

        T = self.max_lag + n_ahead + 1
        irf = np.zeros((self.n_prices, n_ahead + 1, self.n_prices), dtype=np.float32)
        mphi = np.reshape(np.flip(phi, axis=2), (self.n_prices, -1))

        if self.ecm:
            B = np.hstack([
                np.ones((self.n_prices - 1, 1), dtype=np.float32),
                -np.eye(self.n_prices - 1, dtype=np.float32),
            ])

        intercept_vec = b[0, :] if self.intercept else np.zeros(self.n_prices, dtype=np.float32)

        for shock in range(self.n_prices):
            dp = np.zeros((self.n_prices, T), dtype=np.float32)
            p = np.zeros(self.n_prices, dtype=np.float32)
            e0 = np.zeros(self.n_prices, dtype=np.float32)
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
        source: exchange/data source
        cm_um: Binance coin-margined (``cm``) or USDT-margined (``um``)
        '''
        self.ticker = ticker
        self.source = source
        self.cm_um = cm_um

    @staticmethod
    def _cache_value(value) -> str:
        """Return a filename-safe representation of a numeric option."""
        return str(value).replace('.', 'p').replace('-', 'm')

    def _cache_prefix(
        self,
        prefix: str,
        *,
        fill_gaps: bool,
        max_fill_gap_ms,
        drop_both_stale: bool,
        stale_after_ms,
    ) -> str:
        """Build a cache namespace that reflects the aggregation settings."""
        cache_prefix = f"{prefix}_alignedv2"
        if fill_gaps:
            cache_prefix += "_fullgrid"
        if max_fill_gap_ms is not None:
            cache_prefix += f"_fill{self._cache_value(max_fill_gap_ms)}ms"
        if drop_both_stale:
            stale_tag = (
                "all"
                if stale_after_ms is None
                else f"{self._cache_value(stale_after_ms)}ms"
            )
            cache_prefix += f"_dropstale{stale_tag}"
        return cache_prefix

    @staticmethod
    def _days_to_pull(start, end) -> int:
        """Pull enough daily files to cover the half-open range [start, end)."""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if end_ts <= start_ts:
            raise ValueError("end must be later than start.")
        # Add one day because the daily downloader is date-based and the
        # aggregation step subsequently applies the exact [start, end) filter.
        return max(1, int(np.ceil((end_ts - start_ts).total_seconds() / 86400.0)) + 1)

    def check_staleness(
        self,
        start,
        end,
        aggs,
        stale_after_ms=None,
        *,
        fill_gaps=False,
        max_fill_gap_ms=None,
        n_jobs=10,
        return_details=False,
        print_summary=True,
    ):
        """Measure only staleness introduced by forward filling.

        Ages are calculated from the timestamps of actual bid/ask-side
        observations captured before filling. A same-price trade resets the
        corresponding side age to zero. The midpoint age is the older of its
        bid-side and ask-side inputs.
        """
        if isinstance(aggs, str):
            aggs = [aggs]
        else:
            aggs = list(aggs)
        if not aggs:
            raise ValueError("aggs must contain at least one aggregation.")
        if stale_after_ms is not None and stale_after_ms < 0:
            raise ValueError("stale_after_ms must be non-negative.")
        if max_fill_gap_ms is not None and max_fill_gap_ms < 0:
            raise ValueError("max_fill_gap_ms must be non-negative.")

        data = TradeData(self.ticker, self.source, self.cm_um)
        data.grab_trades_data(
            pd.Timestamp(end).to_pydatetime(),
            self._days_to_pull(start, end),
            n_jobs=n_jobs,
        )

        summaries = []
        details = {}
        for agg in aggs:
            price, fill_detail = data.agg_last_trade_to_intervals(
                freq="10ms",
                start=start,
                end=end,
                fill_gaps=False,
                max_fill_gap_ms=None,
                return_fill_diagnostics=True,
            )

            if fill_detail.empty:
                summary = pd.DataFrame([{
                    "diagnostic_basis": STALENESS_DIAGNOSTIC_BASIS,
                    "n_rows": 0,
                }])
                detail = fill_detail
            else:
                summary, detail = check_price_staleness(
                    fill_detail,
                    stale_after_ms=stale_after_ms,
                    return_details=True,
                    print_summary=False,
                )

            summary.insert(0, "latency", agg)
            summary.insert(1, "start", pd.Timestamp(start))
            summary.insert(2, "end", pd.Timestamp(end))
            summary.insert(3, "fill_gaps", bool(fill_gaps))
            summary.insert(4, "max_fill_gap_ms", max_fill_gap_ms)
            summaries.append(summary)
            details[agg] = detail

        summary_table = pd.concat(summaries, ignore_index=True, sort=False)
        if print_summary:
            with pd.option_context(
                "display.max_columns", None,
                "display.width", 260,
            ):
                print(summary_table.to_string(index=False))

        if return_details:
            return summary_table, details
        return summary_table


    @staticmethod
    def _summarize_staleness_details(
        details: pd.DataFrame,
        stale_after_ms=None,
    ) -> pd.DataFrame:
        """Summarize precomputed observation ages after overlap removal."""
        if details.empty:
            return pd.DataFrame([{
                "diagnostic_basis": STALENESS_DIAGNOSTIC_BASIS,
                "n_rows": 0,
            }])
        return check_price_staleness(
            details,
            stale_after_ms=stale_after_ms,
            return_details=False,
            print_summary=False,
        )

    @staticmethod
    def _combine_staleness_windows(window_summary: pd.DataFrame) -> pd.DataFrame:
        """Combine disjoint fill-age windows using exact row-weighted shares."""
        if window_summary.empty:
            return pd.DataFrame()

        share_cols = [
            "spot_forward_filled_share",
            "perp_forward_filled_share",
            "either_forward_filled_share",
            "both_forward_filled_share",
            "one_sided_forward_filled_share",
            "spot_fill_stale_share",
            "perp_fill_stale_share",
            "either_fill_stale_share",
            "both_fill_stale_share",
            "one_sided_fill_stale_share",
            # Separate price-change diagnostics.
            "spot_change_share",
            "perp_change_share",
            "either_change_share",
            "neither_change_share",
            # Backward-compatible fill-age alias.
            "both_stale_share",
        ]
        rows = []
        for latency, group in window_summary.groupby("latency", sort=False):
            group = group.copy()
            weights = pd.to_numeric(group["n_rows"], errors="coerce").fillna(0.0)
            total_rows = float(weights.sum())
            row = {
                "diagnostic_basis": STALENESS_DIAGNOSTIC_BASIS,
                "latency": latency,
                "start": pd.to_datetime(group["window_start"]).min(),
                "end": pd.to_datetime(group["window_end"]).max(),
                "n_windows": int(len(group)),
                "n_rows": int(total_rows),
            }
            for col in share_cols:
                if col not in group.columns:
                    row[col] = np.nan
                    continue
                values = pd.to_numeric(group[col], errors="coerce")
                valid = values.notna() & (weights > 0)
                row[col] = (
                    float(np.average(values[valid], weights=weights[valid]))
                    if valid.any()
                    else np.nan
                )

            for col in [
                "spot_fill_age_max_ms",
                "perp_fill_age_max_ms",
                "longest_both_fill_stale_run_rows",
                "longest_both_fill_stale_run_ms",
                "spot_age_max_ms",
                "perp_age_max_ms",
            ]:
                if col in group.columns:
                    row[col] = pd.to_numeric(group[col], errors="coerce").max()

            rows.append(row)

        return pd.DataFrame(rows)

    def check_staleness_multiperiod(
        self,
        start,
        end,
        aggs,
        period=30,
        stale_after_ms=None,
        *,
        fill_gaps=False,
        max_fill_gap_ms=None,
        boundary_lookback="1H",
        n_jobs=10,
        save_csv=False,
        folder_name="staleness_diagnostics",
        prefix="staleness",
        resume=True,
        max_full_grid_rows=20_000_000,
        return_overall=False,
        print_summary=True,
    ):
        """Run staleness diagnostics over successive memory-bounded windows.

        The public call mirrors :meth:`get_data_multiperiod`: ``period`` is the
        number of days downloaded and processed at once. Only summary rows are
        retained in memory; aggregated row-level diagnostics are discarded
        after each window.

        A short overlap is pulled before every window after the first. This
        lets actual-observation fill ages carry into the reported window
        rather than resetting at the chunk boundary. Overlap rows
        are removed before the summary is stored, so windows remain disjoint
        and their row-weighted shares can be combined exactly.

        Notes
        -----
        With ``fill_gaps=True``, 10 days at 10 ms would create roughly 86.4
        million grid rows for one latency. ``max_full_grid_rows`` guards against
        accidentally constructing a grid that is too large. Use a shorter
        ``period`` (often one day or less) for full-grid diagnostics.

        Returns
        -------
        pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]
            One row per latency and diagnostic window. If ``return_overall`` is
            True, also return a compact row-weighted summary per latency.
        """
        if period <= 0:
            raise ValueError("period must be a positive number of days.")
        if isinstance(aggs, str):
            aggs = [aggs]
        else:
            aggs = list(aggs)
        if not aggs:
            raise ValueError("aggs must contain at least one aggregation.")

        start = pd.Timestamp(start).to_pydatetime()
        end = pd.Timestamp(end).to_pydatetime()
        if end <= start:
            raise ValueError("end must be later than start.")
        if stale_after_ms is not None and stale_after_ms < 0:
            raise ValueError("stale_after_ms must be non-negative.")
        if max_fill_gap_ms is not None and max_fill_gap_ms < 0:
            raise ValueError("max_fill_gap_ms must be non-negative.")

        lookback = pd.Timedelta(boundary_lookback)
        if lookback < pd.Timedelta(0):
            raise ValueError("boundary_lookback must be non-negative.")
        min_lookback = max(
            [pd.Timedelta(agg) for agg in aggs]
            + [pd.Timedelta(milliseconds=float(stale_after_ms or 0.0))]
        )
        lookback = max(lookback, min_lookback)

        output_path = None
        existing = pd.DataFrame()
        completed = set()
        if save_csv:
            output_folder = Path(folder_name)
            output_folder.mkdir(parents=True, exist_ok=True)
            grid_tag = "fullgrid" if fill_gaps else "observedbins"
            stale_tag = (
                "all"
                if stale_after_ms is None
                else f"{self._cache_value(stale_after_ms)}ms"
            )
            fill_tag = (
                "nofilllimit"
                if max_fill_gap_ms is None
                else f"fill{self._cache_value(max_fill_gap_ms)}ms"
            )
            lookback_tag = f"lookback{int(lookback.total_seconds() * 1000)}ms"
            identity = re.sub(
                r"[^A-Za-z0-9_-]+",
                "-",
                f"{self.ticker}_{self.source}_{self.cm_um}",
            )
            output_path = output_folder / (
                f"{prefix}_ffagev4_{identity}_{grid_tag}_{fill_tag}_{stale_tag}_"
                f"{lookback_tag}_{period}d_"
                f"{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}.csv"
            )
            if resume and output_path.exists():
                existing = pd.read_csv(output_path)
                for col in ["window_start", "window_end", "diagnostic_start"]:
                    if col in existing:
                        existing[col] = pd.to_datetime(existing[col], errors="coerce")
                completed = {
                    (
                        pd.Timestamp(row.window_start),
                        pd.Timestamp(row.window_end),
                        str(row.latency),
                    )
                    for row in existing.itertuples()
                    if pd.notna(row.window_start) and pd.notna(row.window_end)
                }

        summaries = []
        if not existing.empty:
            summaries.append(existing)

        curr_start = start
        window_number = 0
        total_windows = int(np.ceil(
            (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
            / (period * 86400.0)
        ))
        while curr_start < end:
            window_number += 1
            curr_end = min(end, curr_start + datetime.timedelta(days=period))
            missing_aggs = [
                agg for agg in aggs
                if (
                    pd.Timestamp(curr_start),
                    pd.Timestamp(curr_end),
                    str(agg),
                ) not in completed
            ]

            if not missing_aggs:
                if print_summary:
                    print(
                        f"Staleness window {window_number}: "
                        f"{curr_start} to {curr_end} already complete."
                    )
                curr_start = curr_end
                continue

            diagnostic_start = max(
                start,
                (pd.Timestamp(curr_start) - lookback).to_pydatetime(),
            )

            if fill_gaps and max_full_grid_rows is not None:
                duration = pd.Timestamp(curr_end) - pd.Timestamp(diagnostic_start)
                for agg in missing_aggs:
                    expected_rows = int(np.ceil(duration / pd.Timedelta(agg)))
                    if expected_rows > int(max_full_grid_rows):
                        raise MemoryError(
                            f"The {diagnostic_start} to {curr_end} full grid at "
                            f"{agg} would contain about {expected_rows:,} rows, "
                            f"above max_full_grid_rows={max_full_grid_rows:,}. "
                            "Reduce period, set fill_gaps=False, or explicitly "
                            "raise max_full_grid_rows."
                        )

            if print_summary:
                print(
                    f"Staleness window {window_number}/{max(total_windows, window_number)}: "
                    f"{curr_start} to {curr_end} ({missing_aggs})"
                )

            _, details_by_agg = self.check_staleness(
                start=diagnostic_start,
                end=curr_end,
                aggs=missing_aggs,
                stale_after_ms=stale_after_ms,
                fill_gaps=fill_gaps,
                max_fill_gap_ms=max_fill_gap_ms,
                n_jobs=n_jobs,
                return_details=True,
                print_summary=False,
            )

            new_rows = []
            for agg in missing_aggs:
                details = details_by_agg[agg]
                details = details.loc[
                    (details.index >= pd.Timestamp(curr_start))
                    & (details.index < pd.Timestamp(curr_end))
                ]
                summary = self._summarize_staleness_details(
                    details,
                    stale_after_ms=stale_after_ms,
                )
                summary.insert(0, "latency", agg)
                summary.insert(1, "window_start", pd.Timestamp(curr_start))
                summary.insert(2, "window_end", pd.Timestamp(curr_end))
                summary.insert(3, "diagnostic_start", pd.Timestamp(diagnostic_start))
                summary.insert(4, "period_days", period)
                summary.insert(5, "fill_gaps", bool(fill_gaps))
                summary.insert(6, "max_fill_gap_ms", max_fill_gap_ms)
                new_rows.append(summary)

            new_summary = pd.concat(new_rows, ignore_index=True, sort=False)
            summaries.append(new_summary)

            if save_csv and output_path is not None:
                current = pd.concat(summaries, ignore_index=True, sort=False)
                current = current.drop_duplicates(
                    subset=["window_start", "window_end", "latency"],
                    keep="last",
                ).sort_values(["window_start", "latency"])
                current.to_csv(output_path, index=False)
                # Keep only the deduplicated table to avoid accumulating copies
                # when saving after every window.
                summaries = [current]

            del details_by_agg
            curr_start = curr_end

        window_summary = (
            pd.concat(summaries, ignore_index=True, sort=False)
            if summaries
            else pd.DataFrame()
        )
        if not window_summary.empty:
            window_summary = window_summary.drop_duplicates(
                subset=["window_start", "window_end", "latency"],
                keep="last",
            )
            window_summary = window_summary[
                window_summary["latency"].isin(aggs)
                & (pd.to_datetime(window_summary["window_start"]) >= pd.Timestamp(start))
                & (pd.to_datetime(window_summary["window_end"]) <= pd.Timestamp(end))
            ].sort_values(["window_start", "latency"]).reset_index(drop=True)

        overall = self._combine_staleness_windows(window_summary)
        if print_summary and not overall.empty:
            print("\nRow-weighted summary across all completed windows:")
            with pd.option_context(
                "display.max_columns", None,
                "display.width", 220,
            ):
                print(overall.to_string(index=False))
            if output_path is not None:
                print(f"Window diagnostics saved to: {output_path}")

        if return_overall:
            return window_summary, overall
        return window_summary

    # @timeout(480)
    def _get_and_parse_data(
        self,
        start,
        end,
        aggs,
        interval,
        lag_structure=None,
        lag_is=None,
        save_csv=True,
        folder_name='vecm_hasbrouck2_um',
        prefix='hasbrouck2',
        fill_gaps=False,
        max_fill_gap_ms=None,
        drop_both_stale=False,
        stale_after_ms=None,
        n_jobs=10,
    ):
        """Pull, aggregate, and fit the VECM for each requested latency.

        The staleness options are exposed here so calls to
        :meth:`get_data_multiperiod` can pass them through unchanged.

        ``drop_both_stale=True, stale_after_ms=None`` removes every row where
        neither midpoint changes. Supplying a threshold removes a row only once
        both midpoint prices have been unchanged longer than that many
        milliseconds. Filtering creates irregular event-time spacing, so use it
        mainly as a robustness specification.
        """
        if isinstance(aggs, str):
            aggs = [aggs]
        else:
            aggs = list(aggs)
        if not aggs:
            raise ValueError("aggs must contain at least one aggregation.")
        if lag_structure is None:
            lag_is = [] if lag_is is None else list(lag_is)
            if not lag_is:
                raise ValueError(
                    "Provide at least one value in lag_is when lag_structure is None."
                )
        if stale_after_ms is not None and stale_after_ms < 0:
            raise ValueError("stale_after_ms must be non-negative.")
        if max_fill_gap_ms is not None and max_fill_gap_ms < 0:
            raise ValueError("max_fill_gap_ms must be non-negative.")
        if stale_after_ms is not None and not drop_both_stale:
            print(
                "Note: stale_after_ms is used only when "
                "drop_both_stale=True; no rows will be filtered."
            )

        cache_prefix = self._cache_prefix(
            prefix,
            fill_gaps=fill_gaps,
            max_fill_gap_ms=max_fill_gap_ms,
            drop_both_stale=drop_both_stale,
            stale_after_ms=stale_after_ms,
        )

        output_folder = Path(folder_name)
        if save_csv:
            output_folder.mkdir(parents=True, exist_ok=True)

        data = None
        shares_dict = {}
        summary_dict = {}

        def get_aggregated_prices(agg):
            nonlocal data
            if data is None:
                data = TradeData(self.ticker, self.source, self.cm_um)
                data.grab_trades_data(
                    pd.Timestamp(end).to_pydatetime(),
                    self._days_to_pull(start, end),
                    n_jobs=n_jobs,
                )

            prices = data.agg_last_trade_to_intervals(
                freq=agg,
                start=start,
                end=end,
                fill_gaps=fill_gaps,
                max_fill_gap_ms=max_fill_gap_ms,
                rename_for_vecm=True,
            )
            if prices.empty:
                raise ValueError(
                    f"No aligned spot/perpetual observations remain for {agg} "
                    f"between {start} and {end}."
                )
            required = {"log_midpoint_spot", "log_midpoint_perp"}
            missing = required.difference(prices.columns)
            if missing:
                raise KeyError(
                    f"Aggregated data is missing required columns: {sorted(missing)}"
                )
            return prices

        for agg in aggs:
            bidask_diff = None

            if lag_structure is None:
                lag_configs = [
                    (
                        i,
                        generate_multiple_lags(
                            i,
                            ['10ms', '50ms', '100ms', '200ms', '500ms', '1s'],
                            '10s',
                        ),
                    )
                    for i in lag_is
                ]
            else:
                if agg not in lag_structure:
                    raise ValueError(
                        f"latency '{agg}' is not present in lag_structure."
                    )
                lag_configs = [("custom", lag_structure)]

            for lag_key, selected_lag_structure in lag_configs:
                file_stem = (
                    f"{cache_prefix}_{interval.lower()}_{agg.lower()}_"
                    f"{lag_key}_{self.cm_um}"
                )
                result_path = output_folder / (
                    f"{file_stem}_results_{pd.Timestamp(start):%Y%m%d}_"
                    f"{pd.Timestamp(end):%Y%m%d}.csv"
                )
                summary_path = output_folder / (
                    f"{file_stem}_summary_{pd.Timestamp(start):%Y%m%d}_"
                    f"{pd.Timestamp(end):%Y%m%d}.csv"
                )

                try:
                    shares_df = pd.read_csv(result_path, index_col=0)
                    summary_df = pd.read_csv(summary_path, index_col=0)
                    print(f"File found, reading from file: {result_path.name}")
                except FileNotFoundError:
                    if bidask_diff is None:
                        bidask_diff = get_aggregated_prices(agg)

                    model = SimpleMVAR(
                        ticker=self.ticker,
                        source=self.source,
                        cm_um=self.cm_um,
                        prices=bidask_diff[
                            ["log_midpoint_spot", "log_midpoint_perp"]
                        ],
                        lag_structure=selected_lag_structure,
                        latency=agg,
                        interval=interval,
                        intercept=True,
                        ecm=True,
                        reference_col="log_midpoint_spot",
                    )

                    try:
                        results = model.fit_by_interval(
                            n_ahead_irf=200,
                            parallel=True,
                        )
                        shares_df = model.shares_table(results)
                        summary_df = model.all_interval_summaries(results)

                        if save_csv:
                            shares_df.to_csv(result_path)
                            summary_df.to_csv(summary_path)
                    except Exception as exc:
                        shares_df = pd.DataFrame()
                        summary_df = pd.DataFrame()
                        print(
                            f"Error fitting {agg}, lag={lag_key}, "
                            f"{start} to {end}: {exc}"
                        )

                result_key = (agg, 0 if lag_structure is not None else lag_key)
                shares_dict[result_key] = shares_df
                summary_dict[result_key] = summary_df

        return shares_dict, summary_dict

    def get_data_multiperiod(self, start, end, aggs, interval, period, **kwargs):
        """Fit successive non-overlapping windows covering the full range.

        ``kwargs`` are passed directly to :meth:`_get_and_parse_data`, including
        ``drop_both_stale``, ``stale_after_ms``, ``max_fill_gap_ms``, and
        ``fill_gaps``.
        """
        if period <= 0:
            raise ValueError("period must be a positive number of days.")
        start = pd.Timestamp(start).to_pydatetime()
        end = pd.Timestamp(end).to_pydatetime()
        if end <= start:
            raise ValueError("end must be later than start.")

        results_dict = {}
        summary_dict = {}
        curr_end = end

        # Work backwards as before, but use max(start, ...) so the oldest
        # partial period is not silently omitted.
        while curr_end > start:
            curr_start = max(start, curr_end - datetime.timedelta(days=period))
            results_i, summary_i = self._get_and_parse_data(
                curr_start,
                curr_end,
                aggs,
                interval,
                **kwargs,
            )

            for key, value in results_i.items():
                results_dict.setdefault(key, []).append(value)
                summary_dict.setdefault(key, []).append(summary_i[key])

            curr_end = curr_start
            print(results_dict.keys())

        for key in results_dict:
            nonempty_results = [df for df in results_dict[key] if not df.empty]
            nonempty_summaries = [df for df in summary_dict[key] if not df.empty]
            results_dict[key] = (
                pd.concat(nonempty_results, ignore_index=False)
                if nonempty_results
                else pd.DataFrame()
            )
            summary_dict[key] = (
                pd.concat(nonempty_summaries, ignore_index=False)
                if nonempty_summaries
                else pd.DataFrame()
            )

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
                df['interval'] = pd.to_datetime(df['interval'], errors='coerce')
                df = df.dropna(subset=['interval'])
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
    beta_perp = np.array([[1.0], [1.0]], dtype=np.float32)
    alpha_perp = np.array([[alpha[1, 0]], [-alpha[0, 0]]], dtype=np.float32)

    # -------------------------
    # Component shares (CS)
    # -------------------------
    cs = (alpha_perp / alpha_perp.sum()).flatten()

    # -------------------------
    # Long-run impact matrix C
    # -------------------------
    denom = float(alpha_perp.T @ (np.eye(2, dtype=np.float32) - gamma_sum) @ beta_perp)
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
                  [1.0, 0.0]], dtype=np.float32)

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
            results[(latency, base)]['interval'] = pd.to_datetime(results[(latency, base)]['interval'], errors='coerce')
            results[(latency, base)] = results[(latency, base)].dropna(subset=['interval'])
            df = results[(latency, base)][(results[(latency, base)]['interval'] >= start_dt)&(results[(latency, base)]['interval'] <= end_dt)]
            df = df.groupby('series')[['CS', 'HIS_lower', 'HIS_upper', 'HIS_mid', 'ILS_mid']].mean().reset_index()
            df['latency'] = latency
            df['base'] = base
            df['period'] = f'{start},{end}'
            period_list.append(df)
    return period_list