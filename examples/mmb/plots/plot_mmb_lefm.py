from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
import csv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_OUT = Path(__file__).resolve().parents[3] / "runs" / "plots" / "mmb"
MUL_COLORS = {
    "green": "#488F96",
    "light": "#F7F1CE",
    "warm": "#CABEA6",
    "muted": "#858374",
    "cyan": "#64B2BD",
    "red": "#CC4B4B",
}


def apply_mul_dark_theme():
    plt.rcParams.update({"text.usetex": False, "font.size": 10, "axes.grid": True, "grid.alpha": 0.2})
    return (6.0, 4.0)


def read_xy(path: Path) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    with path.open("r", newline="") as handle:
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


def scale_force(values: list[float], c_val: float, l_val: float) -> list[float]:
    factor = 2*12.7*l_val / (c_val + l_val)
    return [v * factor for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot historical comparison data.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figsize = apply_mul_dark_theme()
    fig, ax = plt.subplots(figsize=figsize)

    csv_paths = sorted(BASE_DIR.glob("mmb_abaqus_standard_*.csv"))
    if not csv_paths:
        raise SystemExit("No mmb_abaqus_standard_*.csv files found in {0}".format(BASE_DIR))

    palette = [
        MUL_COLORS["cyan"],
        MUL_COLORS["green"],
        MUL_COLORS["warm"],
        MUL_COLORS["red"],
        MUL_COLORS["light"],
        MUL_COLORS["muted"],
    ]

    c_val = 41.3
    l_val = 50.4

    for idx, path in enumerate(csv_paths):
        xs, ys = read_xy(path)
        if xs and (xs[0] != 0.0 or ys[0] != 0.0):
            xs.insert(0, 0.0)
            ys.insert(0, 0.0)
        ys_scaled = scale_force(ys, c_val, l_val)
        color = MUL_COLORS["green"]
        label = "Abaqus standard (properties fitted)"
        ax.plot(xs, ys_scaled, color=color, lw=1.4, label=label)

    uel_path = BASE_DIR / "mmb_UEL_32_61.csv"
    if uel_path.exists():
        ux, uy = read_xy(uel_path)
        if ux and (ux[0] != 0.0 or uy[0] != 0.0):
            ux.insert(0, 0.0)
            uy.insert(0, 0.0)
        uy_scaled = scale_force(uy, c_val, l_val)
        ax.plot(
            ux,
            uy_scaled,
            color=MUL_COLORS["red"],
            lw=1.6,
            label="UEL",
        )

    benchmark_path = BASE_DIR / "benchmark_turon.csv"
    if benchmark_path.exists():
        bx, by = read_xy(benchmark_path)
        ax.plot(
            bx,
            by,
            color=MUL_COLORS["cyan"],
            lw=1.4,
            linestyle="--",
            label="Benchmark (Turon)",
        )

    ax.set_xlabel(r"Opening displacement $\delta$ [mm]")
    ax.set_ylabel(r"Load $P$ [N]")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )

    out_path = args.out_dir / "mmb_abaqus_standard.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
