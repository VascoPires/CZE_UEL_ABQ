#!/usr/bin/env python3
"""
End-to-end visualization helper:
1) Convert ODB -> VTU/PVD using ODB2VTK (Abaqus Python)
2) Merge bulk VTU with UEL UNVALS/IPVALS (per-rank .idx)

Defaults are set for subroutine/002_dev_DCB/T300_NewtonC.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_RUN_DIR = Path(__file__).resolve().parents[1] / "002_dev_DCB" / "T300_NewtonC"
DEFAULT_RANK = "auto"
DEFAULT_FRAME_STEP = 1


def _find_odb(run_dir, odb_override=None):
    if odb_override:
        return Path(odb_override)
    candidates = sorted(Path(run_dir).glob("*.odb"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit("No .odb found in run folder.")
    return candidates[-1]


def _header_path(odb_path):
    return odb_path.with_suffix(".json")


def _run(cmd):
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit("Command failed: {0}".format(" ".join(cmd)))


def _resolve_abaqus_cmd(user_cmd):
    if user_cmd:
        return user_cmd
    for name in ("abaqus.bat", "abaqus.cmd", "abaqus"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    return "abaqus"


def _run_abaqus(abaqus_cmd, args):
    if os.name == "nt":
        cmd = ["cmd", "/c", abaqus_cmd] + args
        _run(cmd)
    else:
        _run([abaqus_cmd] + args)


def _ensure_header(abaqus_cmd, odb2vtk_py, odb_path):
    header = _header_path(odb_path)
    if header.is_file():
        return header
    _run_abaqus(
        abaqus_cmd,
        [
            "python",
            str(odb2vtk_py),
            "--header",
            "1",
            "--odbFile",
            str(odb_path),
        ],
    )
    if not header.is_file():
        raise SystemExit("Header JSON not created: {0}".format(header))
    return header


def _pick_instance_and_step(header_path, instance_override=None, step_override=None):
    data = json.loads(header_path.read_text())
    instances = data.get("instances", [])
    steps = data.get("steps", [])
    if not instances or not steps:
        raise SystemExit("Header JSON missing instances or steps.")

    instance = instance_override or instances[0]
    step_name = step_override or steps[0][0]
    frame_names = None
    for step_entry in steps:
        if step_entry[0] == step_name:
            frame_names = step_entry[1]
            break
    if frame_names is None:
        raise SystemExit("Step not found in header: {0}".format(step_name))
    return instance, step_name, frame_names


def _frame_indices(frame_names, start=0, end=None, step=1):
    count = len(frame_names)
    if end is None or end >= count:
        end = count - 1
    if start < 0 or end < 0 or start > end:
        raise SystemExit("Invalid frame range.")
    return list(range(start, end + 1, step))


def _run_odb2vtk(abaqus_cmd, odb2vtk_py, odb_path, instance, step_name, frames):
    step_arg = "{0}:{1}".format(step_name, ",".join(str(i) for i in frames))
    base_args = [
        "python",
        str(odb2vtk_py),
        "--header",
        "0",
        "--instance",
        instance,
        "--step",
        step_arg,
        "--odbFile",
        str(odb_path),
    ]
    _run_abaqus(abaqus_cmd, base_args + ["--writePVD", "0"])
    _run_abaqus(abaqus_cmd, base_args + ["--writePVD", "1"])


def _find_pvd(odb_path):
    out_dir = odb_path.parent / odb_path.stem
    if not out_dir.is_dir():
        raise SystemExit("ODB2VTK output folder not found: {0}".format(out_dir))
    candidates = sorted(out_dir.glob("*.pvd"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit("No .pvd found in ODB2VTK output folder.")
    return candidates[-1]


def _run_merge(
    run_dir,
    base_pvd,
    rank,
    out_dir=None,
    out_prefix="merged",
    index_offset=0,
    allow_missing_uel=True,
):
    run_uel = Path(__file__).parent / "run_uel_vis.py"
    cmd = [
        sys.executable,
        str(run_uel),
        "--run-dir",
        str(run_dir),
        "--base-pvd",
        str(base_pvd),
        "--rank",
        str(rank),
        "--index-offset",
        str(index_offset),
        "--out-prefix",
        out_prefix,
    ]
    if allow_missing_uel:
        cmd.append("--allow-missing-uel")
    if out_dir:
        cmd += ["--out-dir", str(out_dir)]
    _run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", help="Run folder with UEL outputs")
    parser.add_argument("--odb", help="ODB file path (default: newest .odb in run folder)")
    parser.add_argument("--instance", help="ODB instance name (default: first)")
    parser.add_argument("--step", help="ODB step name (default: first)")
    parser.add_argument("--frame-start", type=int, default=0, help="Start frame index")
    parser.add_argument("--frame-end", type=int, help="End frame index (inclusive)")
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP, help="Frame step")
    parser.add_argument(
        "--rank",
        default=DEFAULT_RANK,
        help="Rank index, or 'auto'/'all' to merge all ranks",
    )
    parser.add_argument("--out-dir", help="Output directory for merged VTM/PVD")
    parser.add_argument("--out-prefix", default="merged", help="Output prefix")
    parser.add_argument("--index-offset", type=int, default=1, help="Index offset for UEL")
    parser.add_argument("--abaqus-cmd", default="abaqus", help="Abaqus command")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_RUN_DIR
    odb_path = _find_odb(run_dir, args.odb)
    odb2vtk_py = Path(__file__).resolve().parents[1] / "ODB2VTK-main" / "python" / "odb2vtk.py"
    if not odb2vtk_py.is_file():
        raise SystemExit("ODB2VTK script not found: {0}".format(odb2vtk_py))

    abaqus_cmd = _resolve_abaqus_cmd(args.abaqus_cmd)
    header_path = _ensure_header(abaqus_cmd, odb2vtk_py, odb_path)
    instance, step_name, frame_names = _pick_instance_and_step(
        header_path, args.instance, args.step
    )
    frames = _frame_indices(frame_names, args.frame_start, args.frame_end, args.frame_step)
    _run_odb2vtk(abaqus_cmd, odb2vtk_py, odb_path, instance, step_name, frames)

    base_pvd = _find_pvd(odb_path)
    _run_merge(
        run_dir,
        base_pvd,
        args.rank,
        args.out_dir,
        args.out_prefix,
        args.index_offset,
        allow_missing_uel=True,
    )


if __name__ == "__main__":
    main()
