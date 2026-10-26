# Elastic dimensionality study: execution record

26 October 2026. **INCOMPLETE: 1 new job completed, 47 prepared and deferred.**
The original campaign's 106 attempts and scientific verdicts are unchanged.
No matched 2D/3D equivalence or width-convergence conclusion is available yet.

## New solver result

Actual Abaqus 2023/Standard job `CPS4_dx0p5_a34p5` completed its full elastic
opening path. Native plane stress, dx=0.5 mm, five elements per arm, a=34.5 mm,
1 mm section thickness scaled to 20 mm specimen width. No active UEL,
stabilization, viscosity or contact.

| Quantity | Result |
|---|---:|
| Full opening | 0.200000003 mm |
| One-arm force, scaled to specimen width | 4.401704669 N |
| Compliance | 0.04543694273 mm/N |
| GI from VCCT | 0.001780586903 N/mm |
| GII from VCCT | 0 N/mm |
| Opposing reaction imbalance and tip tie jump | 0 at retained precision |
| Relative ALLSE versus reaction-work difference | 1.49e-8 |
| Relative compliance spread over loaded frames | 7.00e-8 |

These are elastic quantities at **a=34.5 mm**, not a propagation peak or a
result at the nominal 35 mm crack. A compliance derivative needs the other
two crack lengths. A dimensional comparison needs corresponding plane-strain
and 3D controls. Do not compare this single result with the old a=35 mm
incompatible-mode result and attribute the difference solely to dimensionality.

Termination, hashes, commands and timestamps remain in the run manifest.
Python 2 ODB path and JSON scalar conversion errors were corrected without
rerunning the solver; failed extraction logs and partial exports are retained.

## Extraction and baseline compatibility verification

The new extractor was run on a copy of the previously completed
`fixed_dcb_a35p0` ODB. It reproduces original reaction-based compliance and GI
exactly at retained precision: C=0.04696448645 mm/N,
stiffness=21.29268466 N/mm, GI=0.001702051250 N/mm.
The newly prepared corresponding C3D8I deck has identical bulk connectivity
and equation constraints, with maximum coordinate difference 5.55e-17 mm.
See `control_verification.json` and `scripts/verify_control.py`.
This checks post-processing and part of generator compatibility; it does not
replace a fresh 3D solve or prove native 2D equivalence. Native 2D extraction
was exercised by the completed CPS4 result.

All 48 prepared decks pass connectivity, positive rectangular geometry,
crack-station, dependent-DOF uniqueness and common-grid checks. Solver keyword
acceptance is established only for the executed deck. Preservation checks
cover 2,957 first-campaign/source/report files with no detected changes:
input/source/assessment hashes plus sizes/modification times of other artifacts.

## Resource hold and continuation

The separate transverse-cracking controller continued through t0229, t0230,
t0233 and t0240 during this work. The first preflight found no active solver,
allowing the single CPS4 execution. Once continued external queue activity was
established, a persistent hold was installed to prevent submission during
future gaps. No background controller is active for this study.
`queue_status.json` records the last process check.

Next is `CPS4_dx0p5_a35p0`. The 48-job matrix includes two in-plane mesh levels
and a separate 3/7/15/25-element finite-width DCB sensitivity series. Input
decks are unchanged; source revisions/reasons remain in `plan.json` and its
archived copies. Resume after the external campaign ends, or after explicit
user authorization to change the recorded concurrency rule. Post-processing
and checks can continue during the hold.

See [README](README.md) for reproduction and definitions, and
[mesh audit](MESH_SOURCE_AUDIT.md) for source mesh counts and the failed pilot.
