#!/usr/bin/env python3
"""
Parse *EL PRINT SDV tables from an Abaqus .dat file and organize by increment.

Outputs a CSV with one row per (increment, element, integration point) and
columns SDV1..SDV<N>. Optionally adds local SDV columns (LSDV1..LSDV<M>)
based on a stride layout.
"""

import argparse
import csv
import re


TABLE_MARKER = "THE FOLLOWING TABLE IS PRINTED AT THE INTEGRATION POINTS"
INCREMENT_RE = re.compile(r"\bINCREMENT\s+(\d+)\s+SUMMARY\b")
STEP_RE = re.compile(r"\bSTEP\s+(\d+)\s+INCREMENT\s+\d+\b")


def _parse_sdv_header(line):
    tokens = line.replace(",", " ").split()
    sdv_cols = [t for t in tokens if t.startswith("SDV")]
    sdv_idx = []
    for col in sdv_cols:
        try:
            sdv_idx.append(int(col[3:]))
        except ValueError:
            continue
    return sdv_idx


def parse_dat(dat_path, sdv_count):
    with open(dat_path, "r") as f:
        lines = f.readlines()

    data = {}
    current_step = None
    current_increment = None
    in_table = False
    sdv_idx = None

    i = 0
    while i < len(lines):
        line = lines[i]

        step_match = STEP_RE.search(line)
        if step_match:
            current_step = int(step_match.group(1))
            i += 1
            continue

        inc_match = INCREMENT_RE.search(line)
        if inc_match:
            current_increment = int(inc_match.group(1))
            i += 1
            continue

        if TABLE_MARKER in line:
            in_table = True
            sdv_idx = None
            i += 1
            continue

        if in_table and sdv_idx is None:
            candidate = _parse_sdv_header(line)
            if candidate:
                sdv_idx = candidate
            i += 1
            continue

        if in_table:
            stripped = line.strip()
            if not stripped:
                i += 1
                continue
            tokens = stripped.split()
            if tokens[0].upper() in ("MAXIMUM", "MINIMUM"):
                in_table = False
                sdv_idx = None
                i += 1
                continue
            if not tokens[0].lstrip("-").isdigit():
                i += 1
                continue

            elem = int(tokens[0])
            pt = int(tokens[1])
            values = tokens[2:]

            # Handle wrapped lines by consuming following lines until complete.
            j = i + 1
            while len(values) < len(sdv_idx) and j < len(lines):
                nxt = lines[j].strip()
                if not nxt:
                    j += 1
                    continue
                nxt_tokens = nxt.split()
                if nxt_tokens[0].upper() in ("MAXIMUM", "MINIMUM"):
                    break
                values.extend(nxt_tokens)
                j += 1

            step_val = current_step if current_step is not None else 0
            key = (step_val, current_increment, elem, pt)
            if key not in data:
                data[key] = ["" for _ in range(sdv_count)]

            for col_pos, sdv_number in enumerate(sdv_idx):
                if 1 <= sdv_number <= sdv_count and col_pos < len(values):
                    data[key][sdv_number - 1] = values[col_pos].replace("D", "E")

            i = j
            continue

        i += 1

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datFile", required=True, help="path to .dat file")
    parser.add_argument("--outCsv", required=True, help="output CSV path")
    parser.add_argument(
        "--sdvCount", type=int, default=71, help="number of SDV columns to write"
    )
    parser.add_argument(
        "--layoutStride",
        type=int,
        default=None,
        help="if set, add LSDV columns using stride (e.g., 20 for 1..11,21..31,...)",
    )
    parser.add_argument(
        "--layoutCount",
        type=int,
        default=11,
        help="number of local SDVs per integration point when using layoutStride",
    )
    args = parser.parse_args()

    data = parse_dat(args.datFile, args.sdvCount)
    rows = sorted(data.keys())

    with open(args.outCsv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["step", "increment", "element", "pt"] + [
            "SDV{0}".format(i + 1) for i in range(args.sdvCount)
        ]
        if args.layoutStride:
            header += ["LSDV{0}".format(i + 1) for i in range(args.layoutCount)]
        writer.writerow(header)
        for step, inc, elem, pt in rows:
            row = [step, inc, elem, pt] + data[(step, inc, elem, pt)]
            if args.layoutStride:
                offset = args.layoutStride * (pt - 1)
                local_vals = []
                for i in range(args.layoutCount):
                    sdv_idx = offset + i + 1
                    if 1 <= sdv_idx <= args.sdvCount:
                        local_vals.append(data[(step, inc, elem, pt)][sdv_idx - 1])
                    else:
                        local_vals.append("")
                row += local_vals
            writer.writerow(row)


if __name__ == "__main__":
    main()
