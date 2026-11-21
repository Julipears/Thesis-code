"""Export three four-month PNG rows per asset-year from saved KM curves."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parents[2] / "sa_results/km_v8_final_01/appendix_km"
PLOT_SOURCE = HERE / "plot_km_curves.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    SOURCE = args.input / "data" / "monthly_km_curves_120s.csv"
    OUT = args.input / "monthly_rows_120s"
    OUT.mkdir(exist_ok=True)
    # Reuse the same plotting definitions without importing or executing the
    # event-reading or KM-fitting pipeline.
    definitions = ast.parse(PLOT_SOURCE.read_text(encoding="utf-8"))
    selected = [node for node in definitions.body if isinstance(node, ast.FunctionDef) and node.name in {"styles", "draw_panel", "decorate"}]
    plot = {"plt": plt, "STEP": 0.25}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "existing KM plot definitions", "exec"), plot)
    frame = pd.read_csv(SOURCE, dtype={"period": str})
    curves = {}
    for (period, market, first), group in frame.groupby(["period", "market", "first"], sort=False):
        group = group.sort_values("timeline")
        assert len(group) == 481 and np.array_equal(group.timeline.to_numpy(), np.arange(0, 120.25, .25))
        assert group.survival.between(-1e-14, 1 + 1e-14).all()
        curves[("monthly", market, first, period)] = dict(timeline=group.timeline.to_numpy(), survival=group.survival.to_numpy(), missing_days=int(group.missing_days.iloc[0]))
    assert len(curves) == 480
    groups = ((1, range(1, 5), "Jan-Apr"), (2, range(5, 9), "May-Aug"), (3, range(9, 13), "Sep-Dec"))
    entries = []
    for asset_index, asset in enumerate(("btc", "eth")):
        for year_index, year in enumerate(range(2021, 2026)):
            for row_number, month_numbers, label in groups:
                periods = [f"{year}-{month:02d}" for month in month_numbers]
                fig, axes = plt.subplots(1, 4, figsize=(18, 3.5), sharex=True, sharey=True)
                incomplete = False
                for axis, period in zip(axes, periods):
                    incomplete |= plot["draw_panel"](axis, asset, period, "monthly", 120, curves)
                axes[0].set_ylabel("Probability unresolved")
                plot["decorate"](fig, list(axes), asset, f"Monthly, {year}, {label}", incomplete)
                name = f"km_{asset}_{year}_120s_row{row_number}_{label}.png"
                path = OUT / name
                fig.savefig(path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                with Image.open(path) as image:
                    width, height = image.size
                    assert width > 1700 and height > 350
                    assert abs(image.info["dpi"][0] - 100) < .1
                    image.verify()
                entries.append(dict(file=name, asset=asset.upper(), year=year, source_pdf_page=asset_index * 5 + year_index + 1, row=row_number, months="; ".join(periods), incomplete_coverage=incomplete, width_px=width, height_px=height, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print(f"Exported {asset.upper()} {year}: three four-graph rows.", flush=True)
    assert len(entries) == 30
    assert pd.DataFrame(entries).groupby(["asset", "year"]).size().eq(3).all()
    pd.DataFrame(entries).to_csv(OUT / "row_index.csv", index=False)
    readme = """# Monthly KM rows, 120-second horizon

Thirty PNGs, one four-graph row per file. Each asset-year page in the
120-second monthly overview is exported as three standalone rows:

- Row 1: January–April.
- Row 2: May–August.
- Row 3: September–December.

BTC and ETH are included for 2021–2025. The original colours, line styles,
line widths, 0.25-second display grid, 120-second limits and 100-DPI image
resolution are retained. Each row includes its own title, legend and axis
labels. Asterisks and the coverage note appear only on rows containing an
affected month.

Rows are rendered directly from the previously validated saved KM curve
table, without refitting the models. The numerical survival estimates are
unchanged. [row_index.csv](./row_index.csv) maps each PNG to its original
overview page, row and months.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    source_script = Path(__file__).resolve()
    if source_script.parent != OUT:
        shutil.copyfile(source_script, OUT / source_script.name)
    validation = dict(png_rows=30, graphs_per_row=4, source_overview_pages=10, rows_per_page=3, horizon_seconds=120, source_curve_csv=str(SOURCE), source_curve_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), numerical_curve_refit=False, original_plotting_functions_reused=True, pngs_checked=True, rows=entries)
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    archive_path = OUT / "KM_monthly_rows_120s.zip"
    names = [entry["file"] for entry in entries] + ["README.md", "row_index.csv", "validation.json", source_script.name]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in names:
            archive.write(OUT / name, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None and len(archive.namelist()) == 34
    print(f"Finished and verified 30 PNG rows plus the ZIP: {OUT}", flush=True)


if __name__ == "__main__":
    main()
