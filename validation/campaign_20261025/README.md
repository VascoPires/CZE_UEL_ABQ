# Executed campaign — 25 October 2026

Current authorization: [Prompt0.md](../../Prompt0.md), decisions D13–D17. The original canonical source and examples are preserved. This directory contains actual Abaqus2023/Standard executions, independent Fortran checks, derived results and failed attempts. A successful solver termination alone is not a verification PASS.

- [Investigation and interventions](INVESTIGATION.md)
- [Execution plan](../PLAN.md)
- [Technical report](../REPORT.md)
- [Run inventory](results/run_inventory.csv), [17-case matrix](results/verification_validation_matrix.csv), [figure captions](results/FIGURE_CAPTIONS.md)
- Final execution sequence: `queue_open_mmb` → `queue_diagnostics_final` → `queue_final_retry` → `queue_convergence_followup`. Each queue has a `.json` task list and `.status.json` execution record; the supplementary logger is `queue_diagnostics_final_v2.log`. Earlier stopped controllers are explicitly superseded. Read the current technical report for pending work.

## Evidence layout

`runs/<name>/` retains exact `.inp`, copied UEL when active, `run.json`, solver `.sta/.msg/.dat/.odb`, stdout, extraction-attempt logs, nodal CSV, global energies, available kinematic checks and `assessment/`. Parameters carry source/assumption classifications in the reference cards; `run.json` records actual executed choices. Earlier decks retain their original end-constraint issue and are diagnostics. Revision2 uses corrected end constraints; direct E/LE checks establish dimensionality for each available run.

`material/` contains actual constitutive routines extracted from the canonical Fortran, Intel compiler logs, drivers, input histories, response CSVs and assessments. These are material checks outside Abaqus, not specimen tests. `element_tangent/` contains full assembled UEL residual/Jacobian tests with serial runtime stubs. `sources/` contains isolated, hashed perturbation and I/O variants and patches. `reference_data/` contains digitized published curves/markers and extraction metadata; rejected extractions are labelled and excluded.

## Reproduce

Run from the repository root using WSL Python. The runner invokes Windows Abaqus through `cmd.exe` and `wslpath`. Keep one Abaqus solver active at a time. Existing executed folders are protected against resubmission/overwriting.

Check an exact retained run without processing:

```console
PYTHONDONTWRITEBYTECODE=1 python validation/scripts/replay_run.py dcb_ps03 --check
```

Replay the exact retained deck/source in a **fresh** folder:

```console
PYTHONDONTWRITEBYTECODE=1 python validation/scripts/replay_run.py dcb_ps03 --name dcb_ps03_replay --run
PYTHONDONTWRITEBYTECODE=1 python validation/scripts/campaign.py extract validation/campaign_20261025/runs/dcb_ps03_replay
PYTHONDONTWRITEBYTECODE=1 python validation/scripts/analyze_campaign.py specimen validation/campaign_20261025/runs/dcb_ps03_replay
```

The exact replay avoids dependence on earlier generator versions. The current `structured_specimen.py` regenerates the corrected constrained models; its later output requests and optional controls differ from the earliest diagnostic decks. Unit batches use `campaign.py elements`. Ordinary generator arguments prepare a fresh deck and submit only with `--run`.

Regenerate derived comparisons, inventory and vector/raster scientific figures:

```console
PYTHONDONTWRITEBYTECODE=1 python validation/scripts/reduce_structure.py
PYTHONDONTWRITEBYTECODE=1 python validation/scripts/assessment_diagnostics.py
PYTHONDONTWRITEBYTECODE=1 python validation/scripts/report_campaign.py --figures
PYTHONDONTWRITEBYTECODE=1 python validation/scripts/write_technical_report.py
```

SVG and PDF figures are exported at85 or180mm widths with editable SVG text; PNGs are review copies. Read captions before thesis use: failed traces, graphical read-off uncertainty, reconstructed apparatus and non-independent experimental parameters remain explicit.

The reference extraction scripts refuse existing output directories. For a fresh extraction, use a new output name/directory in a copied script and verify PDF hashes/path IDs. No reference curve is shifted or fitted to a simulation.
