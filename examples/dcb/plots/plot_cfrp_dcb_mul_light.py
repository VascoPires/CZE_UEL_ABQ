from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_OUT = Path(__file__).resolve().parents[3] / "runs" / "plots" / "dcb"
def apply_mul_defaults(**kwargs):
    plt.rcParams.update({"text.usetex": False, "font.size": 10})
    return (6.0, 4.0), {"green":"#488F96", "cyan":"#64B2BD", "red":"#CC4B4B", "muted":"#858374", "warm":"#836F49"}


# Explicit file names (manual import, no globbing)
FEM_CSV = BASE_DIR / "FEM_ABQ_Standard.csv"
LEFM_CSV = BASE_DIR / "LEFM.csv"
EXP_CSV = BASE_DIR / "Real_Experiment.csv"
ABQ_UEL_CSV = BASE_DIR / "ABQ_UEL.csv"
ABQ_T300_PSEUDO2D_CSV = BASE_DIR / "ABQ_T300_3D_Pseudo2D.csv"
LEFM_T300_CSV = BASE_DIR / "dcb_lefm_p_delta_T300.csv"


def read_xy(path: Path) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                x_val = float(str(row[0]).lstrip("\ufeff").strip())
                y_val = float(str(row[1]).strip())
            except ValueError:
                continue
            xs.append(x_val)
            ys.append(y_val)
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot historical comparison data.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figsize, colors = apply_mul_defaults(mode="light", two_col=False, usetex=False)
    fig, ax = plt.subplots(figsize=figsize)

    series = [
        (FEM_CSV, "FEM Abaqus/Standard", colors["green"], "-"),
        (LEFM_CSV, "Benchmark (LEFM)", colors["cyan"], "-"),
        (EXP_CSV, "Experiment", colors["red"], "-"),
        (ABQ_T300_PSEUDO2D_CSV, "ABAQUS Standard CZM", colors["muted"], "-"),
        (LEFM_T300_CSV, "Benchmark (LEFM, T300)", colors["cyan"], "--"),
        (ABQ_UEL_CSV, "UEL", colors["warm"], "-"),
    ]

    plotted_any = False
    for path, label, color, linestyle in series:
        if not path.exists():
            continue
        xs, ys = read_xy(path)
        if not xs:
            continue
        ax.plot(xs, ys, color=color, lw=1.15, ls=linestyle, alpha=0.95, label=label)
        plotted_any = True

    if not plotted_any:
        raise SystemExit(f"No valid CSV data found in {BASE_DIR}")

    ax.set_xlabel(r"Opening displacement $\delta$ [mm]")
    ax.set_ylabel(r"Load $P$ [N]")
    ax.set_xlim(0.0, 8.0)

    handles, labels = ax.get_legend_handles_labels()
    order = sorted(
        range(len(labels)),
        key=lambda i: (0 if "Benchmark" in labels[i] else 1, i),
    )
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="best",
        frameon=False,
    )

    fig.tight_layout()
    out_path = args.out_dir / "cfrp_dcb_mul_light.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
