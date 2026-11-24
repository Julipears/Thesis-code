"""Refresh the appendix regression figure and robustness table from saved CSVs."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from final_code import RESULTS

HERE = Path(__file__).resolve().parent
APPENDIX_OUTPUT = RESULTS / 'vecm' / 'appendix'


LATENCIES = ["10ms", "100ms", "1s"]


CONFIGS = {
    "btc_um": (Path("vecm_hasbrouck3_btc_um"), "hasbrouck3_btc_alignedv2", "um", "BTC linear"),
    "btc_cm": (Path("vecm_hasbrouck3_btc_cm"), "hasbrouck3_btc_alignedv2", "cm", "BTC inverse"),
    "eth_um": (Path("vecm_hasbrouck3_eth_um"), "hasbrouck3_eth_alignedv2", "um", "ETH linear"),
    "eth_cm": (Path("vecm_hasbrouck3_eth_cm"), "hasbrouck3_eth_alignedv2", "cm", "ETH inverse"),
}


REGIMES = [
    ("2021", "2021-01-01", "2022-01-01"),
    ("Jan–Jun 2022", "2022-01-01", "2022-07-01"),
    ("Jul 2022–Mar 2023", "2022-07-01", "2023-04-01"),
    ("Apr–Dec 2023", "2023-04-01", "2024-01-01"),
    ("2024", "2024-01-01", "2025-01-01"),
    ("2025", "2025-01-01", "2026-01-01"),
]


def table_for(summary: pd.DataFrame, market: str) -> pd.DataFrame:
    return (
        summary[summary["market"].eq(market)]
        .pivot(index="latency", columns="regime", values="median_spot_ILS")
        .reindex(index=LATENCIES, columns=[row[0] for row in REGIMES])
    )


def draw_heatmap(summary: pd.DataFrame, markets: list[str], filename: str, title: str, out: Path) -> None:
    tables = [table_for(summary, market) for market in markets]
    finite = np.concatenate([
        table.to_numpy()[np.isfinite(table.to_numpy())] for table in tables
    ])
    deviation = max(float(np.max(np.abs(finite - 0.5))), 1e-6)
    vmin, vmax = 0.5 - deviation, 0.5 + deviation

    if len(markets) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(17, 9), constrained_layout=True)
        axes = list(axes.flat)
    else:
        fig, axes = plt.subplots(1, len(markets), figsize=(17, 5), constrained_layout=True)
        axes = list(np.atleast_1d(axes).flat)

    image = None
    for ax, market, table in zip(axes, markets, tables):
        image = sns.heatmap(
            table,
            ax=ax,
            cmap="coolwarm",
            center=0.5,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            linewidths=0.4,
            cbar=False,
            mask=table.isna(),
        )
        ax.set_title(CONFIGS[market][3])
        ax.set_xlabel("Regime")
        ax.set_ylabel("Latency")
        ax.tick_params(axis="x", rotation=35)

    scalar = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap="coolwarm"
    )
    scalar.set_array([])
    fig.colorbar(
        scalar, ax=axes, shrink=0.82,
        label="Median spot information leadership share (ILS)",
    )
    fig.suptitle(title)
    fig.savefig(out / f"{filename}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def _regression_figure(frequency_results):
    plot_df = frequency_results.copy()

    latency_order = ["10ms", "50ms", "100ms", "200ms", "500ms", "1s"]
    plot_df["latency"] = pd.Categorical(
        plot_df["latency"],
        categories=latency_order,
        ordered=True,
    )

    plot_df = plot_df.sort_values(["contract_type", "latency"])

    x = np.arange(len(latency_order))

    fig, ax = plt.subplots(figsize=(8, 5))

    for contract in ["linear", "inverse"]:
        sub = plot_df.loc[plot_df["contract_type"] == contract].copy()
        sub = sub.set_index("latency").reindex(latency_order).reset_index()

        y = sub["coefficient"].to_numpy()
        lower = sub["ci_lower"].to_numpy()
        upper = sub["ci_upper"].to_numpy()

        line, = ax.plot(
            x,
            y,
            linewidth=2,
            label=contract.capitalize(),
        )

        ax.fill_between(
            x,
            lower,
            upper,
            alpha=0.2,
            color=line.get_color(),
        )

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(latency_order)
    ax.set_xlabel("Latency")
    ax.set_ylabel("Coefficient")
    ax.set_title("Effect of Relative Update Frequency on Perpetual Leadership")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    return fig, ax


def draw_full_grid(summary,out):
    # The appendix exports predate the main figures' flatter layout. Their
    # actual original renderer uses 17-by-5-inch canvases and vertical y ticks.
    display=summary.copy()
    display['regime']=display.regime.str.replace('-', '\u2013',regex=False)
    assert set(display.regime.unique())=={row[0] for row in REGIMES}
    suffix='\nFull grid: 10 sampled days per month, one random hour per day'
    for asset,number in [('btc',9),('eth',10)]:
        draw_heatmap(display,[f'{asset}_um',f'{asset}_cm'],f'figure_{number}',f'{asset.upper()} VECM price leadership by regime'+suffix, out)


def draw_regression(regressions,out):
    frequency_results=regressions[['contract_type','latency']].copy()
    for column in ['coefficient','ci_lower','ci_upper']:
        frequency_results[column]=regressions[f'corrected_{column}']
    figure,ax=_regression_figure(frequency_results)
    ax.legend(loc='lower left')
    figure.savefig(out/'figure_8.png',dpi=100,bbox_inches='tight')
    figure.savefig(out/'figure_8.pdf',bbox_inches='tight')
    plt.close(figure)


def draw_table(table,out):
    display=[]
    for row in table.itertuples(index=False):
        display.append(['','','1 s' if row.latency=='1s' else row.latency.replace('ms',' ms'),f'{row.mean_signed_delta_spot_pp:.2f}',f'{row.mean_absolute_delta_spot_pp:.2f}',f'{row.different_leader_pct:.2f}'])
    with plt.rc_context({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif']}):
        fig,ax=plt.subplots(figsize=(9.0,4.6))
        ax.axis('off')
        image_table=ax.table(cellText=display,colLabels=['Asset','Contract','Latency','Mean signed\n'+r'$\Delta$ILS (pp)','Mean absolute\n'+r'$\Delta$ILS (pp)','Different\nleader (%)'],cellLoc='center',loc='center',colWidths=[.10,.13,.13,.22,.22,.20])
        image_table.auto_set_font_size(False)
        image_table.set_fontsize(11)
        image_table.scale(1,1.6)
        for (row,col),cell in image_table.get_celld().items():
            cell.set_edgecolor('black')
            cell.set_linewidth(.6)
            if row==0:
                cell.set_height(.105)
                cell.set_text_props(weight='bold')
        for start in [1,7]:
            for row in range(start,start+6):
                image_table[row,0].visible_edges='LR'+('T' if row==start else '')+('B' if row==start+5 else '')
        for start in [1,4,7,10]:
            for row in range(start,start+3):
                image_table[row,1].visible_edges='LR'+('T' if row==start else '')+('B' if row==start+2 else '')
            image_table[start+1,1].get_text().set_text('Linear' if start in [1,7] else 'Inverse')
        fig.tight_layout()
        fig.canvas.draw()
        for start,label in [(1,'BTC'),(7,'ETH')]:
            first,last=image_table[start,0],image_table[start+5,0]
            ax.text(first.get_x()+first.get_width()/2,(first.get_y()+first.get_height()+last.get_y())/2,label,ha='center',va='center',transform=ax.transAxes,fontsize=11)
        fig.savefig(out/'table_4.png',dpi=220,bbox_inches='tight')
        fig.savefig(out/'table_4.pdf',bbox_inches='tight')
        plt.close(fig)


def read_summary(directory: Path, name: str, old_name: str) -> pd.DataFrame:
    path = directory / name
    legacy_path = directory / old_name
    return pd.read_csv(legacy_path if not path.is_file() and legacy_path.is_file() else path)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE / 'data')
    parser.add_argument('--output', type=Path, default=APPENDIX_OUTPUT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    draw_regression(read_summary(args.input, 'regression.csv', 'appendix_figure8_regression_before_after.csv'), out)
    draw_table(read_summary(args.input, 'robustness.csv', 'appendix_table4_ils_robustness_corrected.csv'), out)
    draw_full_grid(read_summary(args.input, 'full_grid.csv', 'full_grid_regime_summary_spot_corrected.csv'), out)
    print('Refreshed Figures 8-10 and Table 4 with the original appendix formatting.')
