from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def ensure_origin(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    if not xs:
        return xs, ys
    if xs[0] != 0.0 or ys[0] != 0.0:
        xs = [0.0, *xs]
        ys = [0.0, *ys]
    return xs, ys


def scale_force(values: list[float], c_val: float, l_val: float) -> list[float]:
    factor = 2 * 12.7 * l_val / (c_val + l_val)
    return [v * factor for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot historical comparison data.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figsize = apply_mul_dark_theme()
    fig, ax = plt.subplots(figsize=figsize)

    c_val = 41.3
    l_val = 50.4

    series = [
        (BASE_DIR / "mmb_UEL_32_61_tau_30.csv", r"UEL $\tau^0_{sh}=30 \mathrm{MPa}$", MUL_COLORS["green"]),
        (BASE_DIR / "mmb_UEL_32_61.csv", r"UEL $\tau^0_{sh}=61 \mathrm{MPa}$", MUL_COLORS["red"]),
        (BASE_DIR / "mmb_UEL_32_61_tau_90.csv", r"UEL $\tau^0_{sh}=90 \mathrm{MPa}$", MUL_COLORS["warm"]),
    ]

    plotted_any = False
    for path, label, color in series:
        if not path.exists():
            continue
        xs, ys = read_xy(path)
        if not xs:
            continue
        xs, ys = ensure_origin(xs, ys)
        ys_scaled = scale_force(ys, c_val, l_val)
        ax.plot(xs, ys_scaled, color=color, lw=1.5, label=label)
        plotted_any = True

    if not plotted_any:
        raise SystemExit("No valid CSV data found in {0}".format(BASE_DIR))

    ax.set_xlabel(r"Opening displacement $\delta$ [mm]")
    ax.set_ylabel(r"Load $P$ [N]")
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(bottom=0.0)
    ax.set_title("MMB UEL: Strength Comparison")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )

    out_path = args.out_dir / "mmb_uel_tau_compare.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
