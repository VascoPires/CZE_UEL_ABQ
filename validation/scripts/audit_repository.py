"""Read-only Phase 0 inventory. Never launches Abaqus, a compiler, or a builder.

Run from any directory: python validation/scripts/audit_repository.py --out NEW_DIR
The output directory must not already exist. No solver conclusions are inferred
from a successful input parse or from a manifest's submitted flag.
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def fingerprint(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def read_xy(path):
    rows = []
    for row in csv.reader(path.read_text(encoding="utf-8-sig").splitlines()):
        try:
            rows.append([float(row[0]), float(row[1])])
        except (ValueError, IndexError):
            continue
    return rows


def collect():
    source = ROOT / "src/3D_CZM.f"
    spec = importlib.util.spec_from_file_location("audited_run_case", ROOT / "scripts/run_case.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)  # main() is guarded; no file creation/submission.
    decks = []
    for path in sorted((ROOT / "examples").rglob("*.inp")):
        item = fingerprint(path)
        text = path.read_text()
        try:
            item.update(uses_uel=runner.validate_deck(text), interface_parse="PASS")
        except ValueError as exc:
            item.update(interface_parse="FAIL", error=str(exc))
        item["uel_properties"] = re.findall(r"^\*UEL PROPERTY[^\n]*\n([^\n]+)", text, re.I | re.M)
        item["step_headers"] = re.findall(r"^\*STEP[^\n]*", text, re.I | re.M)
        decks.append(item)
    syntax = []
    for folder in ["scripts", "examples", "visualization"]:
        for path in sorted((ROOT / folder).rglob("*.py")):
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                syntax.append({"path": str(path.relative_to(ROOT)), "status": "PASS"})
            except SyntaxError as exc:
                syntax.append({"path": str(path.relative_to(ROOT)), "status": "FAIL", "error": str(exc)})
    current_hash = fingerprint(source)["sha256"]
    runs = []
    for folder in sorted((ROOT / "runs").iterdir()):
        if not folder.is_dir():
            continue
        copies = list(folder.glob("*.f"))
        stats = list(folder.glob("*.sta"))
        if not copies and not stats:
            continue
        item = {"folder": str(folder.relative_to(ROOT)),
                "sources": [dict(fingerprint(p), matches_current=hashlib.sha256(p.read_bytes()).hexdigest() == current_hash) for p in copies],
                "status_files": [dict(fingerprint(p), completed="THE ANALYSIS HAS COMPLETED SUCCESSFULLY" in p.read_text(errors="replace")) for p in stats],
                "odb_files": [fingerprint(p) for p in folder.glob("*.odb")]}
        manifest = folder / "run.json"
        if manifest.is_file():
            item["manifest"] = json.loads(manifest.read_text())
        runs.append(item)
    curves = [dict(fingerprint(p), numeric_rows=len(read_xy(p)))
              for family in ["dcb", "mmb"] for p in sorted((ROOT / "examples" / family / "data").glob("*.csv"))]
    a = read_xy(ROOT / "examples/mmb/data/benchmark_turon.csv")
    b = read_xy(ROOT / "examples/mmb/data/benchmark_turon_backup_original.csv")
    shifts = [{"row": i+1, "dx_mm": x[0]-y[0], "dy_N": x[1]-y[1]} for i, (x, y) in enumerate(zip(a,b))]
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "root": str(ROOT),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
            "source": fingerprint(source), "python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "commands": {p: shutil.which(p) for p in ["abaqus", "abaqus2023", "ifort", "ifx", "gfortran", "cmd.exe", "pdftotext", "pdfinfo", "pdftoppm"]},
            "installed_windows_files": {p: Path(p).is_file() for p in [
                "/mnt/c/SIMULIA/Commands/abaqus.bat", "/mnt/c/SIMULIA/Commands/abq2023.bat",
                "/mnt/c/SIMULIA/EstProducts/2023/win_b64/code/bin/SMALauncher.exe",
                "/mnt/c/Program Files (x86)/Intel/oneAPI/compiler/2023.0.0/windows/bin/intel64/ifort.exe",
                "/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"]},
            "references": [fingerprint(p) for p in sorted((ROOT / "validation/references").glob("*.pdf"))],
            "decks": decks, "python_syntax": syntax, "historical_runs": runs, "historical_curves": curves,
            "benchmark_copy_comparison": {"row_counts": [len(a), len(b)], "row_differences": shifts},
            "abaqus_executed": False, "compiler_executed": False,
            "license": "NOT TESTED", "constitutive_verification": "NOT RUN", "specimen_verification": "NOT RUN"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output directory; historical audit files are preserved.")
    data = collect()
    args.out.mkdir(parents=True)
    (args.out / "inventory.json").write_text(json.dumps(data, indent=2) + "\n")
    print("Recorded", len(data["references"]), "PDFs,", len(data["decks"]), "decks,",
          len(data["historical_runs"]), "historical run folders. No analyses launched.")


if __name__ == "__main__":
    main()
