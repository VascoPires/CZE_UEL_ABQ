#!/usr/bin/env python3
"""
Simple wrapper for UEL visualization.

Supports:
- Per-rank single files + .idx offsets (recommended)
- Per-increment files (legacy)

Examples:
  python subroutine/004_dev_new_vis/run_uel_vis.py \
    --run-dir subroutine/002_dev_DCB/T300_NewtonC \
    --base-pvd subroutine/002_dev_DCB/T300_NewtonC/bulk.pvd \
    --rank 0

  python subroutine/004_dev_new_vis/run_uel_vis.py \
    --run-dir subroutine/002_dev_DCB/T300_NewtonC \
    --kinc 10 \
    --rank 0
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import pyvista as pv
except ImportError as exc:
    raise SystemExit("PyVista is required. Install it before running.") from exc

import build_model_uel_vtk as uel


DEFAULT_RUN_DIR = r"C:\Users\p2321038\Documents\GitHub\PythonLearning\subroutine\002_dev_DCB\T300_NewtonC"
DEFAULT_RANK = 0
DEFAULT_KINC = 1


def _find_inp(run_dir, inp_override=None):
    if inp_override:
        return Path(inp_override)
    candidates = sorted(Path(run_dir).glob("*.inp"))
    if not candidates:
        raise SystemExit("No .inp found in run folder.")
    for cand in candidates:
        if "CZM" in cand.name.upper():
            return cand
    return candidates[0]


def _rank_paths(run_dir, rank):
    rank_tag = "{0:03d}".format(rank)
    base = Path(run_dir)
    return {
        "rank": rank,
        "unvals": base / "UNVALS_CZM_rank{0}.txt".format(rank_tag),
        "ipvals": base / "IPVALS_CZM_rank{0}.txt".format(rank_tag),
        "unidx": base / "UNVALS_CZM_rank{0}.idx".format(rank_tag),
        "ipidx": base / "IPVALS_CZM_rank{0}.idx".format(rank_tag),
    }


def _detect_ranks(run_dir):
    base = Path(run_dir)
    ranks = []
    for path in sorted(base.glob("UNVALS_CZM_rank*.idx")):
        token = path.stem.replace("UNVALS_CZM_rank", "")
        try:
            ranks.append(int(token))
        except Exception:
            continue
    if ranks:
        return sorted(set(ranks))
    for path in sorted(base.glob("UNVALS_CZM_rank*.txt")):
        token = path.stem.replace("UNVALS_CZM_rank", "")
        try:
            ranks.append(int(token))
        except Exception:
            continue
    return sorted(set(ranks))


def _legacy_paths(run_dir, kinc):
    tag = "{0:06d}".format(kinc)
    base = Path(run_dir)
    return {
        "unvals": base / "UNVALS_CZM_{0}.txt".format(tag),
        "ipvals": base / "IPVALS_CZM_{0}.txt".format(tag),
    }


def _merge_dicts(target, source, label):
    if source is None:
        return
    for key, value in source.items():
        if key in target:
            continue
        target[key] = value


def _load_uel_frame(kinc, index_mode, index_data, file_paths):
    if index_mode:
        un_idx, ip_idx = index_data
        if kinc not in un_idx or kinc not in ip_idx:
            raise RuntimeError("Missing index entry for kinc {0}.".format(kinc))
        un_lines = uel.read_block_lines(file_paths["unvals"], un_idx[kinc]["pos"])
        ip_lines = uel.read_block_lines(file_paths["ipvals"], ip_idx[kinc]["pos"])
        unvals = uel.read_unvals_lines(un_lines)
        ipvals = uel.read_ipvals_lines(ip_lines)
    else:
        unvals = uel.read_unvals(file_paths["unvals"])
        ipvals = uel.read_ipvals(file_paths["ipvals"])
    return unvals, ipvals


def _build_uel_mesh(nodes, elements, unvals, ipvals, scheme_override=None):
    return uel.build_uel_mesh(nodes, elements, unvals, ipvals, scheme_override=scheme_override)


def _read_pvd(pvd_path):
    return uel.read_pvd(pvd_path)


def _write_pvd(out_path, datasets):
    return uel.write_pvd(out_path, datasets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", help="Run folder with UEL outputs")
    parser.add_argument("--base-pvd", help="Bulk ODB2VTK .pvd (for series)")
    parser.add_argument("--out-dir", help="Output directory", default=None)
    parser.add_argument("--out-prefix", default="merged", help="Series output prefix")
    parser.add_argument(
        "--rank",
        default="auto",
        help="Rank index, or 'all'/'auto' to merge all ranks",
    )
    parser.add_argument("--kinc", type=int, help="Extract a single increment")
    parser.add_argument("--index-offset", type=int, default=1, help="Offset for kinc/frame")
    parser.add_argument("--element-type", default="U8", help="UEL element type")
    parser.add_argument(
        "--scheme",
        choices=["gauss", "nc"],
        default=None,
        help="Override integration scheme",
    )
    parser.add_argument("--inp", help="Override .inp path")
    parser.add_argument(
        "--allow-missing-uel",
        action="store_true",
        help="Skip UEL block when data is missing",
    )
    args = parser.parse_args()
    if args.run_dir is None:
        args.run_dir = str(DEFAULT_RUN_DIR)
    if args.kinc is None and not args.base_pvd:
        args.kinc = DEFAULT_KINC

    run_dir = Path(args.run_dir)
    inp_path = _find_inp(run_dir, args.inp)
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "visu")
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes, elements = uel.read_inp(inp_path, element_type=args.element_type)

    rank_arg = str(args.rank).strip().lower()
    if rank_arg in ("auto", "all"):
        ranks = _detect_ranks(run_dir)
        if not ranks:
            ranks = [DEFAULT_RANK]
    else:
        try:
            ranks = [int(rank_arg)]
        except Exception:
            raise SystemExit("Invalid --rank value: {0}".format(args.rank))

    rank_paths_list = [_rank_paths(run_dir, r) for r in ranks]
    index_mode = all(p["unidx"].is_file() and p["ipidx"].is_file() for p in rank_paths_list)

    if index_mode:
        rank_index = {}
        for p in rank_paths_list:
            rank_index[p["rank"]] = (
                uel.read_index(p["unidx"]),
                uel.read_index(p["ipidx"]),
            )
    else:
        rank_index = {}

    if args.kinc is not None:
        kinc = args.kinc + args.index_offset
        if kinc < 0:
            raise SystemExit("Index offset yields negative kinc.")
        if index_mode:
            unvals = {}
            ipvals = {}
            for p in rank_paths_list:
                try:
                    un_idx, ip_idx = rank_index[p["rank"]]
                    u_rank, i_rank = _load_uel_frame(kinc, True, (un_idx, ip_idx), p)
                except RuntimeError:
                    continue
                _merge_dicts(unvals, u_rank, "UNVALS")
                _merge_dicts(ipvals, i_rank, "IPVALS")
            if not unvals or not ipvals:
                raise SystemExit("Missing UEL data for kinc {0}.".format(kinc))
        else:
            legacy = _legacy_paths(run_dir, kinc)
            if not legacy["unvals"].is_file() or not legacy["ipvals"].is_file():
                raise SystemExit("Missing legacy UNVALS/IPVALS files.")
            unvals, ipvals = _load_uel_frame(kinc, False, None, legacy)
        uel_mesh = _build_uel_mesh(nodes, elements, unvals, ipvals, scheme_override=args.scheme)
        out_path = out_dir / "CZM_{0:06d}.vtk".format(kinc)
        uel_mesh.save(str(out_path))
        print("Wrote VTK: {0}".format(out_path))
        return

    if not args.base_pvd:
        raise SystemExit("--base-pvd is required for series output.")

    datasets = _read_pvd(Path(args.base_pvd))
    out_entries = []
    for frame_index, entry in enumerate(datasets):
        bulk_path = (Path(args.base_pvd).parent / entry["file"]).resolve()
        if not bulk_path.is_file():
            raise SystemExit("Bulk VTU not found: {0}".format(bulk_path))

        kinc = frame_index + args.index_offset
        if kinc < 0:
            raise SystemExit("Index offset yields negative frame index for UEL.")

        uel_mesh = None
        if index_mode:
            unvals = {}
            ipvals = {}
            for p in rank_paths_list:
                un_idx, ip_idx = rank_index[p["rank"]]
                if kinc not in un_idx or kinc not in ip_idx:
                    continue
                try:
                    u_rank, i_rank = _load_uel_frame(kinc, True, (un_idx, ip_idx), p)
                except RuntimeError:
                    continue
                _merge_dicts(unvals, u_rank, "UNVALS")
                _merge_dicts(ipvals, i_rank, "IPVALS")
            if unvals and ipvals:
                uel_mesh = _build_uel_mesh(
                    nodes, elements, unvals, ipvals, scheme_override=args.scheme
                )
            elif not args.allow_missing_uel:
                raise SystemExit("Missing UEL index entry for kinc {0}.".format(kinc))
        else:
            legacy = _legacy_paths(run_dir, kinc)
            if legacy["unvals"].is_file() and legacy["ipvals"].is_file():
                unvals, ipvals = _load_uel_frame(kinc, False, None, legacy)
                uel_mesh = _build_uel_mesh(
                    nodes, elements, unvals, ipvals, scheme_override=args.scheme
                )
            elif not args.allow_missing_uel:
                raise SystemExit("Missing legacy UNVALS/IPVALS for kinc {0}.".format(kinc))

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
    _write_pvd(pvd_out, out_entries)
    print("Wrote merged PVD: {0}".format(pvd_out))


if __name__ == "__main__":
    main()
