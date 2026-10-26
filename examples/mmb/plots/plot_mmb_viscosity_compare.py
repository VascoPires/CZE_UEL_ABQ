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

    # neutrals
    "dark": "#2E3534",        # deep charcoal-teal
    "stone": "#B7B2A3",       # warm gray
    "sand": "#E6D8B8",        # darker than "light", still warm
    "ink": "#1F2A2E",         # near-black with cool bias

    # cool companions (work with your teal/cyan)
    "navy": "#2F4E5C",        # muted blue-teal
    "slate": "#6D7F86",       # cool gray-blue
    "mist": "#D7E3E1",        # very light cool neutral

    # warm accents (pair well with "warm" and "red")
    "rust": "#B85C4A",        # earthier than your red
    "terracotta": "#C9775A",  # warm clay
    "mustard": "#B89A3D",     # muted yellow accent

    # soft highlight accent
    "lavender": "#8B7FAF",    # muted purple (balances teal/red)
        
    # clearer, more defined accents (still not neon; good on dark bg)
    "blue": "#3A7BD5",        # crisp medium blue
    "emerald": "#2FAE8E",     # cleaner green than your base teal
    "amber": "#E3A72F",       # clear golden accent
    "magenta": "#C04A8B",     # strong pink/purple accent
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
        (BASE_DIR / "mmb_UEL_32_61_vics_1E3.csv", r"$\xi=10^{-3}$", MUL_COLORS["green"], "-"),
        (BASE_DIR / "mmb_UEL_32_61_vics_1E4.csv", r"$\xi=10^{-4}$", MUL_COLORS["warm"], "-"),
        (BASE_DIR / "mmb_UEL_32_61.csv", r"$\xi=10^{-5}$", MUL_COLORS["red"], "-"),
        (BASE_DIR / "mmb_UEL_32_61_vics_1E6.csv", r"$\xi=10^{-6}$", MUL_COLORS["amber"], "--"),
    ]

    plotted_any = False
    for path, label, color, linestyle in series:
        if not path.exists():
            continue
        xs, ys = read_xy(path)
        if not xs:
            continue
        ys_scaled = scale_force(ys, c_val, l_val)
        ax.plot(xs, ys_scaled, color=color, lw=1.5, ls=linestyle, label=label)
        plotted_any = True

    if not plotted_any:
        raise SystemExit("No valid CSV data found in {0}".format(BASE_DIR))

    ax.set_xlabel(r"Opening displacement $\delta$ [mm]")
    ax.set_ylabel(r"Load $P$ [N]")
    ax.set_title("MMB UEL: Viscosity Comparison")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )

    out_path = args.out_dir / "mmb_uel_viscosity_compare.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
