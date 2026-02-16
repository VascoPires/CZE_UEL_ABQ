#!/usr/bin/env python3
"""
Plot 3D traction-separation curves with a shared MUL dark theme.

Data curves:
- Normal curve: x = normal separation, y = 0, z = normal traction
- Shear curve:  x = 0, y = shear separation, z = shear traction
- Resultant curve: x = normal separation, y = shear separation, z = |T|

Pure-mode curves (from input properties):
- Mode I (pure normal): y = 0, z = |T|
- Mode II/III (pure shear): x = 0, z = |T|
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path

import scienceplots
import seaborn as sns

import argparse
import numpy as np
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401





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
        plt.rcParams["text.usetex"] = True
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
        return (6.5, 5.0)


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


def _row_is_header(row):
    for cell in row:
        if cell is None:
            continue
        cell = cell.strip().lstrip("\ufeff")
        if not cell:
            continue
        if _to_float(cell) is None:
            return True
    return False


def _load_rows(csv_path, element, pt, step=None):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return []
        if _row_is_header(first):
            header = [h.strip().lstrip("\ufeff") for h in first]
            header_lower = {h.lower(): h for h in header}
            dict_reader = csv.DictReader(f, fieldnames=header)
            has_element = "element" in header_lower and "pt" in header_lower
            has_increment = "increment" in header_lower
            le13_key = header_lower.get("le13")
            le33_key = header_lower.get("le33")
            s13_key = header_lower.get("s13")
            s33_key = header_lower.get("s33")
            if has_element and has_increment:
                for row in dict_reader:
                    if int(row[header_lower["element"]]) != element or int(row[header_lower["pt"]]) != pt:
                        continue
                    if step is not None:
                        row_step = int(row.get(header_lower.get("step", ""), 0) or 0)
                        if row_step != step:
                            continue
                    inc = int(row[header_lower["increment"]])
                    row_step = int(row.get(header_lower.get("step", ""), 0) or 0)
                    rows.append((row_step, inc, row))
            else:
                if not all([le13_key, le33_key, s13_key, s33_key]):
                    return []
                for idx, row in enumerate(dict_reader):
                    row_step = int(row.get(header_lower.get("step", ""), 0) or 0)
                    if step is not None and row_step != step:
                        continue
                    mapped = {
                        "element": str(element if element is not None else 1),
                        "pt": str(pt if pt is not None else 1),
                        "step": str(row_step),
                        "increment": str(idx),
                        "LE13": row.get(le13_key),
                        "LE33": row.get(le33_key),
                        "S13": row.get(s13_key),
                        "S33": row.get(s33_key),
                    }
                    rows.append((row_step, idx, mapped))
        else:
            data_rows = [first] + list(reader)
            for idx, row in enumerate(data_rows):
                if not row:
                    continue
                row = [cell.strip().lstrip("\ufeff") for cell in row]
                if len(row) < 4:
                    continue
                row_step = 0
                if step is not None and row_step != step:
                    continue
                mapped = {
                    "element": str(element if element is not None else 1),
                    "pt": str(pt if pt is not None else 1),
                    "step": str(row_step),
                    "increment": str(idx),
                    "LE13": row[0],
                    "LE33": row[1],
                    "S13": row[2],
                    "S33": row[3],
                }
                rows.append((row_step, idx, mapped))
    rows.sort(key=lambda x: (x[0], x[1]))
    return [r for _, _, r in rows]


def _series(rows, x_key, y_key):
    x_vals = []
    y_vals = []
    for row in rows:
        x = _to_float(row.get(x_key))
        y = _to_float(row.get(y_key))
        if x is None or y is None:
            continue
        x_vals.append(x)
        y_vals.append(y)
    return x_vals, y_vals


def load_config(config_path):
    with open(config_path, "r") as handle:
        return json.load(handle)


def build_props_from_config(config_path, material_key=None, specimen_index=None):
    config = load_config(config_path)
    mat_key = material_key or config.get("material_key", "GFRP")
    specimens = config["materials"][mat_key]["specimens"]
    spec_index = specimen_index if specimen_index is not None else config.get("specimen_index", 0)
    _ = specimens[spec_index]
    coh = config.get("cohesive", {})
    evo = coh.get("evolution", {})
    table = evo.get("table", [])
    init = coh.get("initiation", [])
    elastic = coh.get("elastic", [])
    if len(table) < 2 or len(init) < 3 or not elastic:
        raise ValueError("Cohesive data missing in config.")
    GIc = float(table[0])
    GIIc = float(table[1])
    tau3c = float(init[0])
    tau1c = float(init[1])
    tau2c = float(init[2])
    eta = float(evo.get("power", 1.0))
    K3 = float(elastic[0])
    return {
        "GIc": GIc,
        "GIIc": GIIc,
        "tau3c": tau3c,
        "tau1c": tau1c,
        "tau2c": tau2c,
        "eta": eta,
        "K3": K3,
    }


def _compute_ksh(props):
    g1c = props["GIc"]
    gsh = props["GIIc"]
    tau3c = props["tau3c"]
    tau2c = props["tau2c"]
    k3 = props["K3"]
    if gsh > 0.0 and tau3c > 0.0:
        return (g1c / gsh) * (tau2c / tau3c) ** 2 * k3
    return k3


def _compute_deltaF_from_b(props, b_var, ksh):
    g1c = props["GIc"]
    gsh = props["GIIc"]
    tau3c = props["tau3c"]
    tau1c = props["tau1c"]
    eta = props["eta"]
    k3 = props["K3"]
    kb = k3 * (1.0 - b_var) + ksh * b_var
    if ksh > 0.0 and k3 > 0.0:
        delta0_shear = tau1c / ksh
        delta0_3 = tau3c / k3
    else:
        delta0_shear = 0.0
        delta0_3 = 0.0
    aux3 = (ksh * delta0_shear ** 2 - k3 * delta0_3 ** 2) * (b_var ** eta)
    if kb > 0.0:
        delta0 = ((k3 * delta0_3 ** 2 + aux3) / kb) ** 0.5
    else:
        delta0 = 0.0
    if tau1c > 0.0 and tau3c > 0.0:
        deltaC_shear = 2.0 * gsh / tau1c
        deltaC_3 = 2.0 * g1c / tau3c
    else:
        deltaC_shear = 0.0
        deltaC_3 = 0.0
    aux4 = (ksh * delta0_shear * deltaC_shear - k3 * delta0_3 * deltaC_3)
    if kb > 0.0 and delta0 > 0.0:
        return (k3 * delta0_3 * deltaC_3 + aux4 * (b_var ** eta)) / (kb * delta0)
    return 0.0


def turon_traction(delta1, delta2, delta3, props, dmg_old):
    GIc = props["GIc"]
    GSH = props["GIIc"]
    tau3c = props["tau3c"]
    tau1c = props["tau1c"]
    tau2c = props["tau2c"]
    eta = props["eta"]
    K3 = props["K3"]
    KSH = _compute_ksh(props)

    delta3_pos = max(delta3, 0.0)
    delta_shear = (delta1 ** 2 + delta2 ** 2) ** 0.5
    aux1 = KSH * delta_shear ** 2 + K3 * delta3_pos ** 2
    aux2 = KSH ** 2 * delta_shear ** 2 + K3 ** 2 * delta3_pos ** 2
    if aux1 <= 0.0 or aux2 <= 0.0:
        b_var = 0.0
        delta = 0.0
    else:
        b_var = (KSH * delta_shear ** 2) / aux1
        delta = aux1 / (aux2 ** 0.5)

    kb = K3 * (1.0 - b_var) + KSH * b_var
    if KSH > 0.0 and K3 > 0.0:
        delta0_shear = tau1c / KSH
        delta0_3 = tau3c / K3
    else:
        delta0_shear = 0.0
        delta0_3 = 0.0

    aux3 = (KSH * delta0_shear ** 2 - K3 * delta0_3 ** 2) * (b_var ** eta)
    if kb > 0.0:
        delta0 = ((K3 * delta0_3 ** 2 + aux3) / kb) ** 0.5
    else:
        delta0 = 0.0

    if tau1c > 0.0 and tau3c > 0.0:
        deltaC_shear = 2.0 * GSH / tau1c
        deltaC_3 = 2.0 * GIc / tau3c
    else:
        deltaC_shear = 0.0
        deltaC_3 = 0.0

    aux4 = (KSH * delta0_shear * deltaC_shear - K3 * delta0_3 * deltaC_3)
    if kb > 0.0 and delta0 > 0.0:
        deltaF = (K3 * delta0_3 * deltaC_3 + aux4 * (b_var ** eta)) / (kb * delta0)
    else:
        deltaF = 0.0

    if deltaF > delta0 and delta0 > 0.0:
        rt_old = (delta0 * deltaF) / (deltaF - dmg_old * (deltaF - delta0))
    else:
        rt_old = delta0
    rt = max(rt_old, delta)
    if delta > rt_old:
        if rt > 0.0 and deltaF > delta0:
            dmg = (deltaF * (rt - delta0)) / (rt * (deltaF - delta0))
            dmg = min(max(dmg, 0.0), 1.0)
        else:
            dmg = dmg_old
    else:
        dmg = dmg_old

    t1 = KSH * (1.0 - dmg) * delta1
    t2 = KSH * (1.0 - dmg) * delta2
    if delta3 >= 0.0:
        t3 = K3 * (1.0 - dmg) * delta3
    else:
        t3 = K3 * delta3

    return t1, t2, t3, dmg


def build_pure_mode_curves(props, n_points, max_delta3=None, max_delta_shear=None):
    GIc = props["GIc"]
    GSH = props["GIIc"]
    tau3c = props["tau3c"]
    tau1c = props["tau1c"]
    K3 = props["K3"]
    KSH = _compute_ksh(props)

    deltaC_3 = 2.0 * GIc / tau3c if tau3c > 0.0 else 0.0
    deltaC_shear = 2.0 * GSH / tau1c if tau1c > 0.0 else 0.0

    if deltaC_3 > 0.0:
        max_delta3 = deltaC_3
    elif max_delta3 is None or max_delta3 <= 0.0:
        max_delta3 = 0.0
    if max_delta_shear is None or max_delta_shear <= 0.0:
        max_delta_shear = deltaC_shear
    else:
        max_delta_shear = max(max_delta_shear, deltaC_shear)

    mode1_delta = [max_delta3 * i / max(n_points - 1, 1) for i in range(n_points)]
    mode1_t = []
    dmg = 0.0
    for d in mode1_delta:
        _, _, t3, dmg = turon_traction(0.0, 0.0, d, props, dmg)
        mode1_t.append(abs(t3))
    if mode1_t:
        mode1_t[-1] = 0.0

    mode2_delta = [max_delta_shear * i / max(n_points - 1, 1) for i in range(n_points)]
    mode2_t = []
    dmg = 0.0
    for d in mode2_delta:
        t1, t2, t3, dmg = turon_traction(d, 0.0, 0.0, props, dmg)
        mode2_t.append((t1 ** 2 + t2 ** 2 + t3 ** 2) ** 0.5)

    return mode1_delta, mode1_t, mode2_delta, mode2_t


def build_mixed_mode_surface(props, n_b=12, n_steps=60):
    k3 = props["K3"]
    ksh = _compute_ksh(props)
    eps = 1.0e-6
    b_values = np.linspace(0.0, 1.0 - eps, max(n_b, 2))
    x_vals = np.zeros((len(b_values), n_steps))
    y_vals = np.zeros((len(b_values), n_steps))
    z_vals = np.zeros((len(b_values), n_steps))

    for i, b_var in enumerate(b_values):
        if b_var <= eps:
            ratio = 0.0
        else:
            ratio = ((b_var * k3) / (ksh * (1.0 - b_var))) ** 0.5
        deltaF = _compute_deltaF_from_b(props, b_var, ksh)
        if ratio == 0.0:
            delta3_max = deltaF
            delta_shear_max = 0.0
        else:
            denom = (ksh ** 2 * ratio ** 2 + k3 ** 2) ** 0.5
            scale = (ksh * ratio ** 2 + k3) / denom if denom > 0.0 else 0.0
            delta3_max = deltaF / scale if scale > 0.0 else 0.0
            delta_shear_max = ratio * delta3_max

        dmg = 0.0
        for j in range(n_steps):
            frac = j / max(n_steps - 1, 1)
            delta3 = frac * delta3_max
            delta1 = frac * delta_shear_max
            t1, t2, t3, dmg = turon_traction(delta1, 0.0, delta3, props, dmg)
            x_vals[i, j] = delta3
            y_vals[i, j] = delta1
            z_vals[i, j] = (t1 ** 2 + t2 ** 2 + t3 ** 2) ** 0.5

    return x_vals, y_vals, z_vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="abaqus_std_cohesive.csv")
    parser.add_argument("--element", type=int, default=1)
    parser.add_argument("--pt", type=int, default=1)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--normal_x", default="LE33")
    parser.add_argument("--normal_z", default="S33")
    parser.add_argument("--shear_y", default="LE13")
    parser.add_argument("--shear_z", default="S13")
    parser.add_argument("--config", default=str(BASE_DIR / "dcb_config.json"))
    parser.add_argument("--material", default=None)
    parser.add_argument("--specimen", type=int, default=None)
    parser.add_argument("--n_points", type=int, default=200)
    parser.add_argument("--pure_only", action="store_true")
    parser.add_argument("--data_points", action="store_true")
    parser.add_argument("--mixed_surface", action="store_true")
    parser.add_argument("--surface_b", type=int, default=12)
    parser.add_argument("--surface_steps", type=int, default=60)
    parser.add_argument("--surface_alpha", type=float, default=0.2)
    parser.add_argument("--save_view", default=None)
    parser.add_argument("--outPng", default=None)
    args = parser.parse_args()

    rows = _load_rows(args.csv, args.element, args.pt, step=args.step)
    if not rows:
        raise SystemExit("No rows found for element={0}, pt={1}".format(args.element, args.pt))

    nx, nz = _series(rows, args.normal_x, args.normal_z)
    sy, sz = _series(rows, args.shear_y, args.shear_z)

    tx = []
    ty = []
    tz = []
    for row in rows:
        dx = _to_float(row.get(args.normal_x))
        dy = _to_float(row.get(args.shear_y))
        tn = _to_float(row.get(args.normal_z))
        ts = _to_float(row.get(args.shear_z))
        if dx is None or dy is None or tn is None or ts is None:
            continue
        tx.append(dx)
        ty.append(dy)
        tz.append((tn ** 2 + ts ** 2) ** 0.5)

    props = build_props_from_config(args.config, args.material, args.specimen)
    max_delta3 = max(tx) if tx else None
    max_delta_shear = max(ty) if ty else None
    mode1_delta, mode1_t, mode2_delta, mode2_t = build_pure_mode_curves(
        props, args.n_points, max_delta3=max_delta3, max_delta_shear=max_delta_shear
    )

    figsize = apply_mul_dark_theme()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    if args.mixed_surface:
        surf_x, surf_y, surf_z = build_mixed_mode_surface(
            props, n_b=args.surface_b, n_steps=args.surface_steps
        )
        ax.plot_surface(
            surf_x,
            surf_y,
            surf_z,
            color=MUL_COLORS["green"],
            alpha=args.surface_alpha,
            linewidth=0.0,
            zorder=1,
            antialiased=True,
        )

    show_data = not args.mixed_surface and not args.pure_only
    if show_data:
        ax.plot(nx, [0.0] * len(nx), nz, color=MUL_COLORS["green"], lw=1.5, label="Normal (data)")
        ax.plot([0.0] * len(sy), sy, sz, color=MUL_COLORS["cyan"], lw=1.5, label="Shear (data)")
    if not args.mixed_surface and (args.data_points or args.pure_only):
        ax.plot(tx, ty, tz, color=MUL_COLORS["warm"], lw=1.2, label="Resultant (data)")
    ax.plot(mode1_delta, [0.0] * len(mode1_delta), mode1_t,
            color=MUL_COLORS["cyan"], lw=1.2, ls="-", label="Mode I (pure)")
    ax.plot([0.0] * len(mode2_delta), mode2_delta, mode2_t,
            color=MUL_COLORS["light"], lw=1.2, ls="-", label="Mode II (pure)")
    if args.mixed_surface and tx and ty and tz:
        max_tz = max(tz)
        tol = max(1.0e-12, 1.0e-6 * max_tz)
        start_idx = None
        end_idx = None
        for i, val in enumerate(tz):
            if start_idx is None and val > tol:
                start_idx = i
            elif start_idx is not None and val <= tol:
                end_idx = i
                break
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(tz) - 1
        end_idx = min(end_idx + 1, len(tz))
        ax.plot(
            tx[start_idx:end_idx],
            ty[start_idx:end_idx],
            tz[start_idx:end_idx],
            color=MUL_COLORS["red"],
            lw=1.4,
            zorder=5,
            label="Resultant (data)",
        )

    ax.set_xlabel(r"$\delta_3$")
    ax.set_ylabel(r"$\delta_{sh}$")
    ax.set_zlabel(r"$\tau$", labelpad= -2)

    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.tick_params(axis="z", labelsize=6)

    ax.set_xlim(0.0, 0.02)
    ax.set_ylim(0.0, 0.03)
    ax.grid(False)


    ax.view_init(elev=20, azim=18)
    
    
    # Tick label size (keeps same tick positions and decimal formatting)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='z', which='major', labelsize=8)  # for z-axis in 3D
    ax.tick_params(pad=-1)
    
    
    #ax.axes.set_xlim3d(left=1.0, right=1.85)
    ax.set_zlim(bottom=0.0)

    # Custom tick positions (for example, 4 ticks per axis)
    num_ticks = 3
    ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num_ticks))
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num_ticks))
    ax.set_zticks(np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], num_ticks))
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))



    ax.set_box_aspect(None, zoom=0.78)
    #ax.set_zlim(0.0, 0.05)
    #title = "3D Traction-Separation (Pure Modes Only)" if args.pure_only else "3D Traction-Separation (Data + Pure Modes)"
    #ax.set_title(title)
    #ax.legend(loc="best", frameon=False)

    if args.outPng:
        fig.savefig(args.outPng, dpi=300, bbox_inches="tight")
    else:
        def _write_view():
            view = "elev={0:.2f}, azim={1:.2f}".format(ax.elev, ax.azim)
            print("Current view:", view)
            if args.save_view:
                Path(args.save_view).write_text(view + "\n")

        def _on_key(event):
            if event.key and event.key.lower() == "v":
                _write_view()

        fig.canvas.mpl_connect("key_press_event", _on_key)
        fig.canvas.mpl_connect("close_event", lambda event: _write_view())
        plt.show()


if __name__ == "__main__":
    main()
