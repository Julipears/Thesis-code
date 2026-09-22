# Thesis-code

This repository contains the codebase for an Engineering Science thesis analyzing cryptocurrency market dynamics using Vector Error Correction Models (VECM) and survival analysis techniques. The analysis focuses on price discovery, market efficiency, and convergence times in cryptocurrency spot and perpetual futures markets.

## Overview

The thesis examines how information flows between spot and perpetual futures markets in cryptocurrency exchanges, with particular emphasis on:
- Price discovery mechanisms using Hasbrouck's information share model
- Market integration through VECM analysis
- Convergence time estimation after market shocks using survival analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Julipears/Thesis-code.git
cd Thesis-code
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For notebook execution, ensure Jupyter is installed:
```bash
pip install jupyter
```

## Usage

The active pipeline is split into three stages under `final_code/`. The
superseded notebooks and one-off experiments are retained under `archive/`;
see `PAPER_PIPELINE_MANIFEST.md` for the exact outputs and plots.
Appendix generators are nested under the corresponding section's `appendix/`
folder.

### Section 6: VECM analysis
- **`final_code/vecm_hasbrouck3.py`**: Hasbrouck information-share implementation
- **`final_code/section_4/run_hasbrouck3_full.py`**: BTC/ETH VECM batch runner
- **`final_code/section_4/vecm_plotting.py`**: VECM alpha-over-time plots, separate alpha plots,
  and regime heatmaps

### Section 7: KM event identification and AFT analysis
- **`final_code/section_5_1/`**: event identification and covariate construction
- **`final_code/section_5_2/fit_aft_v8_without_spot_volatility.py`** and **`fit_aft_v8_univariate_models.py`**: current log-logistic AFT fits
- **`final_code/section_5_2/`**: AFT figures, diagnostics, and tables

See `PAPER_PIPELINE_MANIFEST.md` for the figure/table-to-script mapping.

### Shared modules
- **`final_code/trade_data_pull.py`**: data retrieval and processing classes
- **`final_code/survival_analysis_data_processing_final.py`**, **`survival_analysis_data_pull_final.py`**, and **`survival_analysis_utils_final.py`**: KM/AFT data and utility functions
- **`final_code/timeout.py`**: timeout handling utilities

## Key Classes and Functions

### Data Retrieval
- `TradeData`: Single symbol data retrieval from multiple exchanges
- `TradeDataMulti`: Multi-symbol data retrieval with parallel processing

### Analysis
- `VECMResults`: Standard VECM model fitting and diagnostics
- `VECMHasbrouck2`: Hasbrouck information share implementation
- `SurvivalAnalysis`: Shock detection and survival model fitting

## Dependencies

- polars: High-performance DataFrame operations
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- requests: HTTP requests for data retrieval
- matplotlib: Plotting and visualization
- statsmodels: Statistical modeling (VECM, cointegration tests)
- scikit-learn: Machine learning utilities
- pytrends: Google Trends data (optional)

## Data Sources

- **Binance**: Spot and futures trade data, kline data, funding rates, metrics
- **KuCoin**: Historical trade data
- **OKX**: Trade data archives
- **Deribit**: Implied volatility data
- **The Block**: Cryptocurrency options data

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite the corresponding thesis.
