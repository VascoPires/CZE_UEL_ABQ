#!/usr/bin/env python3
"""
Merge ODB2VTK bulk output with UEL UNVALS/IPVALS into MultiBlock VTM frames.

This script assumes the UEL uses 8-node cohesive elements and writes either:
- Per-increment files: UNVALS_CZM_XXXXXX.txt / IPVALS_CZM_XXXXXX.txt
- Per-rank files with index: UNVALS_CZM_rank###.txt / IPVALS_CZM_rank###.txt
  and matching .idx files with byte offsets per increment

It reads the bulk model time series from a .pvd (ODB2VTK output), then
creates a MultiBlock dataset per frame with two blocks:
- "bulk" : ODB2VTK mesh
- "uel"  : UEL mesh rebuilt from .inp + UNVALS + IPVALS
"""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    import pyvista as pv
except ImportError as exc:
    raise SystemExit("PyVista is required. Install it before running.") from exc


DEFAULT_UNVALS_PATTERN = "UNVALS_CZM_{index:06d}.txt"
DEFAULT_IPVALS_PATTERN = "IPVALS_CZM_{index:06d}.txt"


def _clean_split(line):
    return [p.strip() for p in line.strip().split(",") if p.strip()]


def _as_int(token):
    try:
        return int(token)
    except Exception:
        return None


def read_inp(inp_path, element_type="U8"):
    nodes = {}
    elements = []
    el_type_key = f"type={element_type}".lower()
    lines = Path(inp_path).read_text().splitlines()
    in_part = False
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        if not raw or raw.startswith("**"):
            i += 1
            continue
        lower = raw.lower()
        if lower.startswith("*part"):
            in_part = True
            i += 1
            continue
        if lower.startswith("*end part"):
            in_part = False
            i += 1
            continue
        if lower.startswith("*node") and in_part:
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("*"):
                parts = _clean_split(lines[i])
                if len(parts) >= 4:
                    nid = _as_int(parts[0])
                    if nid is None:
                        i += 1
                        continue
                    nodes[nid] = (float(parts[1]), float(parts[2]), float(parts[3]))
                i += 1
            continue
        if lower.startswith("*element") and in_part:
            key = lower.replace(" ", "")
            if el_type_key in key:
                i += 1
                while i < len(lines) and not lines[i].lstrip().startswith("*"):
                    parts = _clean_split(lines[i])
                    if len(parts) >= 9:
                        eid = _as_int(parts[0])
                        if eid is None:
                            i += 1
                            continue
                        conn = []
                        valid = True
                        for v in parts[1:9]:
                            cv = _as_int(v)
                            if cv is None:
                                valid = False
                                break
                            conn.append(cv)
                        if valid:
                            elements.append((eid, conn))
                    i += 1
                continue
        i += 1
    if not nodes:
        raise RuntimeError("No nodes found in .inp.")
    if not elements:
        raise RuntimeError(f"No *Element block with type={element_type}.")
    return nodes, elements


def read_unvals_lines(lines):
    data = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        if len(parts) < 3:
            i += 1
            continue
        eid = int(float(parts[0]))
        nnode = int(float(parts[1]))
        i += 1
        vals = []
        for _ in range(nnode):
            while i < len(lines) and not lines[i].strip():
                i += 1
            row = [float(v) for v in lines[i].split()]
            vals.append(row[:3])
            i += 1
        data[eid] = {"nnode": nnode, "vals": vals}
    return data


def read_unvals(path):
    lines = Path(path).read_text().splitlines()
    return read_unvals_lines(lines)


def read_ipvals_lines(lines):
    data = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        if len(parts) < 3:
            i += 1
            continue
        eid = int(float(parts[0]))
        nip = int(float(parts[1]))
        nvals = int(float(parts[2]))
        scheme = None
        if len(parts) >= 4:
            scheme = int(float(parts[3]))
        i += 1
        ip_rows = []
        for _ in range(nip):
            while i < len(lines) and not lines[i].strip():
                i += 1
            row = [float(v) for v in lines[i].split()]
            if len(row) < nvals:
                row = row + [0.0] * (nvals - len(row))
            ip_rows.append(row[:nvals])
            i += 1
        data[eid] = {"nip": nip, "nvals": nvals, "scheme": scheme, "vals": ip_rows}
    return data


def read_ipvals(path):
    lines = Path(path).read_text().splitlines()
    return read_ipvals_lines(lines)


def read_index(path):
    entries = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        kinc = int(parts[0])
        kstep = int(parts[1])
        time_1 = float(parts[2])
        time_2 = float(parts[3])
        pos = int(parts[4])
        entries[kinc] = {"kstep": kstep, "time": (time_1, time_2), "pos": pos}
    return entries


def read_block_lines(file_path, offset):
    if offset <= 0:
        raise RuntimeError("Index offset must be positive.")
    lines = []
    with open(file_path, "rb") as handle:
        handle.seek(max(0, offset - 1))
        while True:
            raw = handle.readline()
            if not raw:
                break
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            if text.startswith("KINC="):
                if lines:
                    break
                continue
            lines.append(text)
    return lines


def shape_funcs(xi, eta):
    return [
        0.25 * (1.0 - xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 + eta),
        0.25 * (1.0 - xi) * (1.0 + eta),
    ]


def inv_shape_matrix(scheme):
    if scheme == 2:
        a = 1.0
    else:
        a = 1.0 / math.sqrt(3.0)
    points = [(-a, -a), (a, -a), (a, a), (-a, a)]
    N = np.zeros((4, 4), dtype=float)
    for i, (xi, eta) in enumerate(points):
        N[i, :] = shape_funcs(xi, eta)
    return np.linalg.inv(N)


def build_uel_mesh(nodes, elements, unvals, ipvals, scheme_override=None):
    if not ipvals:
        raise RuntimeError("IPVALS data is empty.")

    node_ids = sorted(nodes.keys())
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)

    disp_sum = np.zeros((num_nodes, 3), dtype=float)
    disp_count = np.zeros(num_nodes, dtype=float)

    sample = next(iter(ipvals.values()))
    nvals = sample["nvals"]
    scheme = sample["scheme"]
    if scheme_override:
        scheme = 2 if scheme_override == "nc" else 1
    if scheme is None:
        scheme = 1
    invN = inv_shape_matrix(scheme)

    field_sum = np.zeros((num_nodes, nvals), dtype=float)
    field_count = np.zeros(num_nodes, dtype=float)

    for eid, conn in elements:
        # Displacements
        if eid in unvals:
            entry = unvals[eid]
            if entry["nnode"] != 8:
                raise RuntimeError(f"Element {eid}: expected 8 nodes, got {entry['nnode']}.")
            disp_vals = entry["vals"]
            for loc, nid in enumerate(conn):
                if loc >= len(disp_vals):
                    continue
                idx = node_index[nid]
                disp_sum[idx] += disp_vals[loc]
                disp_count[idx] += 1.0

        # IP values -> nodal values (4 mid-surface nodes, duplicated to top)
        if eid not in ipvals:
            continue
        ip_entry = ipvals[eid]
        if ip_entry["nip"] != 4:
            raise RuntimeError(
                f"Element {eid}: expected 4 IPs, got {ip_entry['nip']}."
            )
        ip_rows = np.array(ip_entry["vals"], dtype=float)
        nodal_vals = np.zeros((8, nvals), dtype=float)
        for k in range(nvals):
            v_ip = ip_rows[:, k]
            v_n = invN.dot(v_ip)
            nodal_vals[0:4, k] = v_n
            nodal_vals[4:8, k] = v_n
        for loc, nid in enumerate(conn):
            idx = node_index[nid]
            field_sum[idx] += nodal_vals[loc]
            field_count[idx] += 1.0

    disp = np.zeros_like(disp_sum)
    for i in range(num_nodes):
        if disp_count[i] > 0.0:
            disp[i] = disp_sum[i] / disp_count[i]

    fields = np.zeros_like(field_sum)
    for i in range(num_nodes):
        if field_count[i] > 0.0:
            fields[i] = field_sum[i] / field_count[i]

    points = np.array([nodes[nid] for nid in node_ids], dtype=float)
    cells = np.empty((len(elements), 9), dtype=np.int64)
    cells[:, 0] = 8
    for row, (_, conn) in enumerate(elements):
        cells[row, 1:] = [node_index[nid] for nid in conn]
    celltypes = np.full(len(elements), 12, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells.ravel(), celltypes, points)
    grid.point_data["Displacement"] = disp
    grid.point_data["node_id"] = np.array(node_ids, dtype=np.int64)
    grid.cell_data["element_id"] = np.array([eid for eid, _ in elements], dtype=np.int64)

    names = [
        "ip_x",
        "ip_y",
        "ip_z",
        "delta_shear",
        "delta3",
        "T_shear_mag",
        "T_normal",
        "DMG",
        "delta",
        "B",
        "DMGv",
    ]
    if nvals > len(names):
        for idx in range(len(names) + 1, nvals + 1):
            names.append("SDV{0}".format(idx))

    for k, name in enumerate(names[:nvals]):
        grid.point_data[name] = fields[:, k]

    return grid


def read_pvd(pvd_path):
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    datasets = []
    for dataset in root.iter("DataSet"):
        file_attr = dataset.attrib.get("file")
        if not file_attr:
            continue
        timestep = float(dataset.attrib.get("timestep", "0"))
        datasets.append({"time": timestep, "file": file_attr})
    if not datasets:
        raise RuntimeError("No DataSet entries found in PVD.")
    return datasets


def write_pvd(out_path, datasets):
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for entry in datasets:
        lines.append(
            '    <DataSet timestep="{0}" group="" part="0" file="{1}"/>'
            .format(entry["time"], entry["file"])
        )
    lines.append("  </Collection>")
    lines.append("</VTKFile>")
    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", required=True, help="Abaqus .inp with U8 elements")
    parser.add_argument("--base-pvd", required=True, help="ODB2VTK .pvd file")
    parser.add_argument("--unvals-dir", help="Directory with UNVALS files")
    parser.add_argument("--ipvals-dir", help="Directory with IPVALS files")
    parser.add_argument("--unvals-file", help="UNVALS rank file (single file)")
    parser.add_argument("--ipvals-file", help="IPVALS rank file (single file)")
    parser.add_argument("--unvals-index", help="UNVALS rank index (.idx)")
    parser.add_argument("--ipvals-index", help="IPVALS rank index (.idx)")
    parser.add_argument("--out-dir", required=True, help="Output directory for VTM/PVD")
    parser.add_argument("--element-type", default="U8", help="UEL element type")
    parser.add_argument(
        "--scheme",
        choices=["gauss", "nc"],
        default=None,
        help="Override integration scheme (gauss or nc)",
    )
    parser.add_argument(
        "--unvals-pattern",
        default=DEFAULT_UNVALS_PATTERN,
        help="UNVALS filename pattern (use {index})",
    )
    parser.add_argument(
        "--ipvals-pattern",
        default=DEFAULT_IPVALS_PATTERN,
        help="IPVALS filename pattern (use {index})",
    )
    parser.add_argument(
        "--index-offset",
        type=int,
        default=0,
        help="Offset added to the frame index when reading UNVALS/IPVALS",
    )
    parser.add_argument(
        "--out-prefix",
        default="merged",
        help="Output VTM filename prefix",
    )
    parser.add_argument(
        "--allow-missing-uel",
        action="store_true",
        help="Skip UEL block when UNVALS/IPVALS are missing",
    )
    args = parser.parse_args()

    pvd_path = Path(args.base_pvd)
    base_dir = pvd_path.parent
    unvals_dir = None
    ipvals_dir = None
    unvals_file = Path(".")
    ipvals_file = Path(".")
    unvals_index = {}
    ipvals_index = {}
    use_index = any(
        [args.unvals_file, args.ipvals_file, args.unvals_index, args.ipvals_index]
    )
    if use_index:
        if not (args.unvals_file and args.ipvals_file and args.unvals_index and args.ipvals_index):
            raise SystemExit(
                "Index mode requires --unvals-file, --ipvals-file, --unvals-index, --ipvals-index."
            )
        unvals_file = Path(args.unvals_file)
        ipvals_file = Path(args.ipvals_file)
        unvals_index = read_index(args.unvals_index)
        ipvals_index = read_index(args.ipvals_index)
    else:
        if not (args.unvals_dir and args.ipvals_dir):
            raise SystemExit("Per-increment mode requires --unvals-dir and --ipvals-dir.")
        unvals_dir = Path(args.unvals_dir)
        ipvals_dir = Path(args.ipvals_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes, elements = read_inp(args.inp, element_type=args.element_type)
    datasets = read_pvd(pvd_path)

    out_entries = []
    for frame_index, entry in enumerate(datasets):
        bulk_path = (base_dir / entry["file"]).resolve()
        if not bulk_path.is_file():
            raise SystemExit(f"Bulk VTU not found: {bulk_path}")

        kinc = frame_index + args.index_offset
        if kinc < 0:
            raise SystemExit("Index offset yields negative frame index for UEL.")

        uel_mesh = None
        if use_index:
            if kinc not in unvals_index or kinc not in ipvals_index:
                if args.allow_missing_uel:
                    uel_mesh = None
                else:
                    raise SystemExit("Missing UEL index entry for kinc {0}.".format(kinc))
            else:
                un_lines = read_block_lines(unvals_file, unvals_index[kinc]["pos"])
                ip_lines = read_block_lines(ipvals_file, ipvals_index[kinc]["pos"])
                unvals = read_unvals_lines(un_lines)
                ipvals = read_ipvals_lines(ip_lines)
                uel_mesh = build_uel_mesh(
                    nodes, elements, unvals, ipvals, scheme_override=args.scheme
                )
        else:
            try:
                unvals_name = args.unvals_pattern.format(index=kinc)
                ipvals_name = args.ipvals_pattern.format(index=kinc)
            except Exception as exc:
                raise SystemExit("Invalid UNVALS/IPVALS pattern. Use {index}.") from exc

            unvals_path = (unvals_dir / unvals_name).resolve()
            ipvals_path = (ipvals_dir / ipvals_name).resolve()

            if not unvals_path.is_file() or not ipvals_path.is_file():
                if args.allow_missing_uel:
                    uel_mesh = None
                else:
                    raise SystemExit(
                        "Missing UEL files for frame {0}: {1}, {2}".format(
                            frame_index, unvals_path, ipvals_path
                        )
                    )
            else:
                unvals = read_unvals(unvals_path)
                ipvals = read_ipvals(ipvals_path)
                uel_mesh = build_uel_mesh(
                    nodes, elements, unvals, ipvals, scheme_override=args.scheme
                )

        bulk_mesh = pv.read(str(bulk_path))
        merged = pv.MultiBlock()
        merged["bulk"] = bulk_mesh
        if uel_mesh is not None:
            merged["uel"] = uel_mesh

        out_name = "{0}_{1:06d}.vtm".format(args.out_prefix, frame_index)
        out_path = out_dir / out_name
        merged.save(str(out_path))

        out_entries.append({"time": entry["time"], "file": out_name})

    pvd_out = out_dir / "{0}.pvd".format(args.out_prefix)
    write_pvd(pvd_out, out_entries)
    print("Wrote merged PVD: {0}".format(pvd_out))


if __name__ == "__main__":
    main()
