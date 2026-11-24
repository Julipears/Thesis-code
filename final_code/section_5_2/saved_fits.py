"""Read monthly coefficient rows from the paper bundle or a local fit."""

from pathlib import Path

import pandas as pd

from final_code import PAPER_DATA


def read_monthly_coefficients(specification: str, folder: str, run_dir: Path | None = None) -> pd.DataFrame:
    if run_dir is None:
        table = pd.read_csv(PAPER_DATA / "section_5_2/monthly_coefs.csv.gz", float_precision="round_trip")
        return table.loc[table["specification"].eq(specification)].copy()

    rows = []
    for path in sorted((run_dir / folder).glob("*/loglogistic/coefficients/*_20??-??_coefficients.csv")):
        first, period, _ = path.stem.split("_", 2)
        rows.append(pd.read_csv(path, float_precision="round_trip").assign(
            market=path.parents[2].name, first=first, period=period,
        ))
    if not rows:
        raise FileNotFoundError(f"No monthly AFT coefficients found under {run_dir / folder}")
    return pd.concat(rows, ignore_index=True)
