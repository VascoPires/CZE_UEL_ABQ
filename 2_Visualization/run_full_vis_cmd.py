#!/usr/bin/env python3
"""
Windows-friendly wrapper that launches CMD and runs the full
ODB2VTK + UEL merge pipeline via run_full_vis.py.

Defaults target subroutine/002_dev_DCB/T300_NewtonC.
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT_DIR / "subroutine" / "002_dev_DCB" / "T300_Gaussian"
RUN_FULL = ROOT_DIR / "subroutine" / "004_dev_new_vis" / "run_full_vis.py"
DEFAULT_RANK = "auto"


def _build_cmd(args):
    cmd = [
        sys.executable,
        str(RUN_FULL),
        "--run-dir",
        str(args.run_dir),
        "--rank",
        str(args.rank),
        "--index-offset",
        str(args.index_offset),
        "--out-prefix",
        args.out_prefix,
        "--abaqus-cmd",
        args.abaqus_cmd,
    ]
    if args.out_dir:
        cmd += ["--out-dir", str(args.out_dir)]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, help="Run folder")
    parser.add_argument(
        "--rank",
        default=DEFAULT_RANK,
        help="Rank index, or 'auto'/'all' to merge all ranks",
    )
    parser.add_argument("--out-dir", help="Output directory")
    parser.add_argument("--out-prefix", default="merged", help="Output prefix")
    parser.add_argument("--index-offset", type=int, default=1, help="UEL index offset")
    parser.add_argument("--abaqus-cmd", default="abaqus", help="Abaqus command")
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep the CMD window open after completion",
    )
    args = parser.parse_args()

    if not RUN_FULL.is_file():
        raise SystemExit("run_full_vis.py not found: {0}".format(RUN_FULL))

    cmd = _build_cmd(args)

    if sys.platform.startswith("win"):
        cmd_str = subprocess.list2cmdline(cmd)
        shell_cmd = ["cmd", "/k" if args.keep_open else "/c", cmd_str]
        proc = subprocess.run(shell_cmd)
    else:
        proc = subprocess.run(cmd)

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
