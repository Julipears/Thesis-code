from pathlib import Path
import pandas as pd
import numpy as np

AFT_INPUT_SUBDIR = "aft_data_liquidity"
AFT_COVARIATES = [
    "basis_5min",
    "fundingRate_bps",
    "open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "taker_long_short",
    "trader_long_short",
    "vol_spot_5min_pct",
    "volume",
]

MARKETS = [
    Path("./sa_btc_um"),
    Path("./sa_btc_cm"),
    Path("./sa_eth_um"),
    Path("./sa_eth_cm"),
]

required_columns = ["Length", "Status", *AFT_COVARIATES]

def summarize_file(path: Path):
    frame = pd.read_parquet(path)
    summary = {"file": str(path)}
    summary["source_rows"] = len(frame)
    for col in required_columns:
        if col not in frame.columns:
            summary[f"missing_col_{col}"] = True
            summary[f"nonnull_{col}"] = 0
        else:
            nonnull = int(frame[col].notna().sum())
            summary[f"missing_col_{col}"] = False
            summary[f"nonnull_{col}"] = nonnull
    # Calculate rows that would remain after dropping any NA in required columns
    subset_nonnull = frame[required_columns].dropna()
    summary["complete_rows_after_dropna"] = len(subset_nonnull)
    # Also report rows with Length>0 and Status in {0,1}
    tmp = frame.copy()
    tmp["Length"] = pd.to_numeric(tmp.get("Length"), errors="coerce").replace({0:0.001})
    tmp = tmp[tmp["Length"]>0]
    tmp["Status"] = pd.to_numeric(tmp.get("Status"), errors="coerce")
    tmp = tmp[tmp["Status"].isin([0,1])]
    summary["eligible_rows_before_dropna"] = len(tmp)
    return summary

def main():
    results = []
    for market in MARKETS:
        input_dir = market / AFT_INPUT_SUBDIR
        if not input_dir.exists():
            print(f"Missing dir: {input_dir}")
            continue
        files = sorted(input_dir.glob("*.parquet"))
        for p in files:
            s = summarize_file(p)
            if s["complete_rows_after_dropna"] == 0:
                print("\n---\nZero complete rows after dropna:\")
                print(s["file"])
                for col in required_columns:
                    print(f" - {col}: non-null={s.get(f'nonnull_{col}',0)} missing_col={s.get(f'missing_col_{col}',False)}")
                print(f"eligible_rows_before_dropna={s['eligible_rows_before_dropna']}")
            results.append(s)

if __name__ == '__main__':
    main()
