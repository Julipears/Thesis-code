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

The codebase is organized by thesis sections. Run the notebooks in order for the complete analysis pipeline.

### Section 6: VECM Analysis
- **`hasbrouck_vecm.ipynb`**: Implements Hasbrouck's information share model using VECM to measure price discovery between spot and perpetual markets
- **`standard_vecm.ipynb`**: Standard VECM analysis for testing market integration and cointegration relationships

### Section 7: Survival Analysis
- **`survival_analysis_methods.ipynb`**: Survival analysis of convergence times after market shocks, including Kaplan-Meier estimation and parametric models

### Backend Functions and Classes
- **`trade_data_pull.py`**: Core data retrieval and processing classes (`TradeData`, `TradeDataMulti`, `BinanceMetricsData`) for downloading and processing trade data from Binance, KuCoin, and OKX
- **`vecm_analysis2.py`**: Standard VECM implementation with cointegration testing and model diagnostics
- **`vecm_hasbrouck2.py`**: Hasbrouck information share model implementation
- **`survival_analysis.py`**: Survival analysis class with shock detection and convergence time estimation
- **`pull_binance_data.py`**: Additional Binance data utilities
- **`price_graphing.ipynb`**: Visualization utilities for price data
- **`timeout.py`**: Timeout handling utilities
- **`InProgress_non_parametric.ipynb`**: Work-in-progress non-parametric analysis

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
