#!/usr/bin/env python3
"""
Compute incremental and cumulative work for a cohesive element CSV export.

Reports traction-separation work, not dissipated energy. Negative work can
also occur during elastic unloading; it does not by itself establish healing.
"""

import argparse
import csv
from pathlib import Path


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


def _split_cols(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_rows(csv_path, col_order=None, force_no_header=False):
    rows = []
    path = Path(csv_path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        first = next(reader, None)
        if first is None:
            return []
        if not force_no_header and _row_is_header(first):
            header = [h.strip().lstrip("\ufeff") for h in first]
            dict_reader = csv.DictReader(handle, fieldnames=header)
            rows.extend(dict_reader)
        else:
            if not col_order:
                raise ValueError("col_order is required for headerless CSVs.")
            data_rows = [first] + list(reader)
            for row in data_rows:
                if not row:
                    continue
                row = [cell.strip().lstrip("\ufeff") for cell in row]
                if len(row) < len(col_order):
                    continue
                rows.append(dict(zip(col_order, row)))
    return rows


def _extract_vectors(rows, delta_cols, traction_cols, thickness):
    deltas = []
    tractions = []
    for row in rows:
        delta_vec = []
        traction_vec = []
        for col in delta_cols:
            value = _to_float(row.get(col, ""))
            if value is None:
                break
            delta_vec.append(value * thickness)
        for col in traction_cols:
            value = _to_float(row.get(col, ""))
            if value is None:
                break
            traction_vec.append(value)
        if len(delta_vec) != len(delta_cols) or len(traction_vec) != len(traction_cols):
            continue
        deltas.append(delta_vec)
        tractions.append(traction_vec)
    return deltas, tractions


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="abaqus_std_cohesive.csv")
    parser.add_argument("--delta-cols", default="LE11,LE22,LE33")
    parser.add_argument("--traction-cols", default="S11,S22,S33")
    parser.add_argument("--col-order", default="LE11,LE22,LE33,S11,S22,S33")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--thickness", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    delta_cols = _split_cols(args.delta_cols)
    traction_cols = _split_cols(args.traction_cols)
    if len(delta_cols) != len(traction_cols):
        raise SystemExit("delta-cols and traction-cols must have the same length.")

    col_order = _split_cols(args.col_order) if args.col_order else None
    rows = _load_rows(args.csv, col_order=col_order, force_no_header=args.no_header)
    if not rows:
        raise SystemExit("No rows found in CSV.")

    deltas, tractions = _extract_vectors(rows, delta_cols, traction_cols, args.thickness)
    if len(deltas) < 2:
        raise SystemExit("Not enough data rows after filtering.")

    dwork = []
    cum = 0.0
    cum_list = []
    for i in range(1, len(deltas)):
        avg_t = [(t0 + t1) * 0.5 for t0, t1 in zip(tractions[i - 1], tractions[i])]
        ddelta = [d1 - d0 for d0, d1 in zip(deltas[i - 1], deltas[i])]
        dw = _dot(avg_t, ddelta)
        dwork.append(dw)
        cum += dw
        cum_list.append(cum)

    neg_indices = [i + 1 for i, val in enumerate(dwork) if val < 0.0]
    min_dw = min(dwork)
    max_dw = max(dwork)
    print("rows:", len(deltas))
    print("min dW:", min_dw)
    print("max dW:", max_dw)
    print("negative increments:", len(neg_indices))
    if neg_indices:
        sample = ", ".join(str(i) for i in neg_indices[:10])
        print("first negative indices:", sample)

    if args.out:
        out_path = Path(args.out)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["increment", "dW", "W"])
            for i, (dw, w) in enumerate(zip(dwork, cum_list), start=1):
                writer.writerow([i, dw, w])


if __name__ == "__main__":
    main()
