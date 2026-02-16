#!/usr/bin/env python3
"""
Plot traction-separation curves from sdv_by_increment.csv.

Default mapping (GP1, element 1):
- Normal: SDV5 (pmax) vs SDV7 (normal traction)
- Shear:  SDV4 (tmax) vs SDV6 (tangential traction magnitude)
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
THEME_DIR = BASE_DIR.parents[2] / "cohesive"
if THEME_DIR.exists():
    sys.path.insert(0, str(THEME_DIR))

MUL_COLORS = {
    "green": "#488F96",
    "light": "#F7F1CE",
    "warm": "#CABEA6",
    "muted": "#858374",
    "cyan": "#64B2BD",
    "red": "#CC4B4B",
}


def apply_mul_dark_theme() -> tuple[float, float]:
    try:
        from plot_theme import apply_theme, get_figsize, MUL_COLORS as THEME_COLORS

        apply_theme(use_dark=True, color_source="mul")
        plt.rcParams["text.usetex"] = False
        MUL_COLORS.update(THEME_COLORS)
        return get_figsize(two_col=False)
    except Exception:
        plt.rcParams.update({
            "figure.facecolor": "none",
            "axes.facecolor": "#0C1428",
            "savefig.facecolor": "none",
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "text.color": "white",
            "axes.titlecolor": "white",
            "grid.color": "white",
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.dpi": 200,
        })
        return (10.0, 4.0)


def _to_float(value):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_rows(csv_path, element, pt, step=None):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["element"]) != element or int(row["pt"]) != pt:
                continue
            if step is not None:
                row_step = int(row.get("step", 0))
                if row_step != step:
                    continue
            inc = int(row["increment"])
            row_step = int(row.get("step", 0))
            rows.append((row_step, inc, row))
    rows.sort(key=lambda x: (x[0], x[1]))
    return [r for _, _, r in rows]


def _series(rows, x_key, y_key):
    x_vals = []
    y_vals = []
    for row in rows:
        key_x = x_key
        key_y = y_key
        if key_x not in row and key_x.startswith("SDV"):
            alt = "L" + key_x
            if alt in row:
                key_x = alt
        if key_y not in row and key_y.startswith("SDV"):
            alt = "L" + key_y
            if alt in row:
                key_y = alt
        x = _to_float(row.get(key_x))
        y = _to_float(row.get(key_y))
        if x is None or y is None:
            continue
        x_vals.append(x)
        y_vals.append(y)
    return x_vals, y_vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="sdv_by_increment.csv")
    parser.add_argument("--element", type=int, default=1)
    parser.add_argument("--pt", type=int, default=1)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--normal_x", default="SDV5")
    parser.add_argument("--normal_y", default="SDV7")
    parser.add_argument("--shear_x", default="SDV4")
    parser.add_argument("--shear_y", default="SDV6")
    parser.add_argument("--outPng", default=None)
    args = parser.parse_args()

    rows = _load_rows(args.csv, args.element, args.pt, step=args.step)
    if not rows:
        raise SystemExit("No rows found for element={0}, pt={1}".format(args.element, args.pt))

    nx, ny = _series(rows, args.normal_x, args.normal_y)
    sx, sy = _series(rows, args.shear_x, args.shear_y)

    figsize = apply_mul_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(nx, ny, marker="o", color=MUL_COLORS["green"])
    axes[0].set_xlabel(args.normal_x)
    axes[0].set_ylabel(args.normal_y)
    #axes[0].set_title("Normal traction-separation")
    axes[0].grid(True)

    axes[1].plot(sx, sy, marker="o", color=MUL_COLORS["cyan"])
    axes[1].set_xlabel(args.shear_x)
    axes[1].set_ylabel(args.shear_y)
    #axes[1].set_title("Shear traction-separation")
    axes[1].grid(True)

    fig.tight_layout()
    if args.outPng:
        fig.savefig(args.outPng, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
