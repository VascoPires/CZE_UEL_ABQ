# Continue this research session

Updated 26 October 2026. This is the entry point for a fresh agent working in
this same folder. The conversation need not be available: current decisions,
results and remaining questions are recorded below and in the linked files.

**Latest continuation:** the user commissioned the matched elastic study.
See [its report](validation/elastic_dimensions_20261026/REPORT.md) and handoff
section 22. One native CPS4 job completed; 47 jobs are prepared and deferred
behind an unrelated active transverse-cracking queue. A persistent hold
prevents submissions during gaps between its jobs. No matched equivalence or
width-convergence conclusion is available. The first campaign is unchanged.

## Read first

1. This file, then [Prompt0.md](Prompt0.md) for execution authority. The original
   campaign has already finished; do not restart its sequence blindly.
2. [Decisions](validation/DECISIONS.md), especially D13–D18. Older checkpoint
   stops are historical and superseded by the user's autonomous protocol.
3. [Current handoff sections 19–21](validation/HANDOFF.md#19-campaign-execution-and-assessment-complete--2026-10-26).
   Earlier sections record historical states, including queues that have ended.
4. [Technical report](validation/REPORT.md) for evidence and scientific limits.
5. [Dimensionality and DCB explanation](validation/DIMENSIONALITY_AND_DCB.md)
   for the latest discussion and the proposed next investigation.
6. [UEL interface](docs/uel-interface.md) and
   [optional-input checks](validation/tangent_input_20261026/README.md) before
   preparing new UEL models.

## Completed work and evidence boundaries

- First campaign: **106 actual Abaqus attempts, 50 completed and 56 failed**.
  All final queues ended; the last conditional retry was skipped because its
  baseline completed. No solver/controller was active at the last handoff.
  Check live processes before launching anything in a new session.
- The 17-case scientific matrix has **one FAIL and 16 INCONCLUSIVE** verdicts.
  Solver completion and individual diagnostic passes are not full validation.
- The zero-viscosity DCB reaches 4 mm opening: peak 51.4275 N at 0.3 mm mesh
  spacing and 51.2219 N at 0.075 mm. Agreement of about 0.84%/0.44% is with the
  published **numerical** peak, not the elementary-beam LEFM initiation load.
  Nonmonotonic mesh robustness and time resolution limit convergence claims.
- The matched viscous DCB increment refinement lowers energy drift from about
  112.78% to 0.4146%, but changes the peak by about 2.11%; time convergence is
  unresolved. Constant step damping with 15–16% stabilization work was rejected.
- Higher-penalty MMB20 failed with the old perturbation and completed with the
  otherwise identical smaller perturbation. This demonstrates a numerical
  tangent improvement, not a repair of the physical formulation.
- General geometric energy verification **fails**: changing interface area and
  rotating its frame produce nonzero closed-cycle work. Reference-area scaling
  fixes only the restricted area cycle; the frame cycle still fails. Do not
  hide these defects with damping or claim a general energy-consistent UEL.
- Independent fixed-crack elastic/VCCT checks support loading and energy-release
  definitions. Full specimen propagation and independent experimental validation
  remain unestablished. Turon 2018 comparisons are internal/appendix checks.

The [campaign guide](validation/campaign_20261025/README.md) documents exact replay
and regeneration. Its [results folder](validation/campaign_20261025/results/)
contains the run inventory, matrix, controls and audits, plus 16 SVG/PDF/PNG
figure sets in `figures/`, captions and `overview.png`. Read each caption before
thesis use. Failed traces and uncertain references remain explicitly labelled.

## Current source versus archived campaign source

The user explicitly requested the optional perturbation input; this extension
is **implemented and verified**, with no outstanding feature work.

- Current `src/3D_CZM.f`: **v1.4**, optional real `PROPS(8)=zeta`, positive finite
  length units; omission preserves **1e-6**. Explicit **1e-8** is supported.
  Integer `JPROPS(1)` remains the integration scheme and follows all real values;
  for eight real properties, put the integer on the next data line. Use `UNSYMM`.
- Converter setting: `cohesive_UEL.tangent_perturbation`. Both validation
  builders expose `--tangent-perturbation`. Do not edit a Fortran constant to
  select a new perturbation.
- Current source SHA256:
  `b93cadaf39fe8ad34d97e1819d77ef53c9803673a3655ca4358f87522bc36ab8`.
- Archived first-campaign v1.3 source:
  [3D_CZM_v1_3_baseline.f](validation/campaign_20261025/sources/3D_CZM_v1_3_baseline.f),
  SHA256 `bfa62fce8fef54fc8f3ba6f04946b3e0c5b68e4dbed5fb60987c8b27cc8d455c`.
- [Feature verification](validation/tangent_input_20261026/verification.json) is
  PASS: exact assembly/material compatibility, three completed actual Abaqus
  patch batches, expected invalid-input rejection and workflow checks. These
  jobs are separate from the 106 first-campaign attempts. All retained campaign
  decks/source copies and original example files were preserved.

## Latest question and proposed next study

The user asked whether our models and benchmarks are 2D or 3D, and why
`03_dcb_reference.png` has different LEFM stiffness and peak load.

The completed structural models use intended **ordinary plane-strain constrained
3D slices**, not generalized plane strain: mainly C3D8I bulk, 1 mm width, one
element across width, equal in-plane front/back motion and zero width motion.
Forces are scaled to specimen width. Exact native 2D equivalence is unproven.
The separate finite-width Krueger pilot failed. **No actual CPE4/CPE4I/CPS4
solver model has been run.** Selected Krueger critical curves originate from
2D models; Turon 2018 uses 2D CPE4I; Turon 2010 contradicts itself (plane-stress
wording versus CPE4). Published AS4/PEEK experiments are physical 3D specimens.

Figure 03's elementary-beam LEFM predicts stiffness 26.056 N/mm and sharp-crack
initiation 56.227 N. The existing independent elastic FE control already gives
21.293 N/mm and 52.633 N, while the UEL early-load fit is 21.241 N/mm and its
recorded maximum 51.427 N. This compliance gap exists before cohesive damage;
its separate causes have not been quantified. See the
[annotated figure](validation/campaign_20261025/results/explanation_03/03_dcb_reference_explained.pdf).
The early-load fit is not certified undamaged stiffness.

**Originally proposed task, now started in the separate study above:** compare matched true-2D plane-stress and
plane-strain elastic fixed-crack models (CPS4/CPE4/CPE4I) against the constrained
3D slice, with no active UEL. Preserve geometry, material orientation, loading
and force/width definitions; compare compliance and G by compliance derivative
and VCCT where available. Isolate dimensional/element effects before changing
strengths, toughness or penalty. This was originally a recommendation; the
subsequent user request commissioned it. See the separate current study status;
routine justified numerical checks are already authorized by Prompt0.md.

## Working constraints and environment

For work shared between computers, follow the
[Git research handoff guide](docs/research-handoff.md). Input/source snapshots,
manifests, CSV/JSON evidence, reports, figures and validation diagnostics are
eligible for Git; solver databases and literature PDFs require separate
transfer. Ignore rules alone do not commit or push any files.

- Preserve completed/failed evidence. New studies use fresh folders and record
  exact inputs, source hashes, controls, provenance and termination states.
  Do not resubmit an existing run folder or blindly regenerate frozen reports.
- No peak-load stopping criterion is built into the UEL. Prescribed step end,
  convergence failure and resource limits govern termination; maximum accepted
  force in an incomplete trace is not automatically a verified peak.
- Keep viscosity, step stabilization, increment size, mesh and tangent
  perturbation separate. Record changes and actual artificial energy. Smaller
  mesh sometimes helps convergence; it does not guarantee monotonic robustness.
- Do not fit material/geometry to targets or silently reformulate the geometric
  energy law. Missing values and reconstructed apparatus must stay labelled.
- Abaqus 2023/Standard runs on Windows from WSL through `cmd.exe`; launcher
  `C:\SIMULIA\Commands\abaqus.bat`, compatible Intel oneAPI/ifort and VS2022.
  Inspect retained `run.json` and `execute.cmd` for exact working commands.
  Keep **one Abaqus solver at a time**. Replay `--check` is nonprocessing;
  actual exact replay requires a fresh name and `--run` (see campaign guide).
- The repository already has migration deletions and many untracked files.
  Do not reset, restore, clean, stage or commit them without user instruction.
- Local papers (`validation/references/`) and solver artifacts include ignored
  files. This handoff assumes the **same workspace**. A clone or another machine
  needs a separate transfer of those files; Git alone is not the whole record.

## Suggested message for a new session

> Continue the CZE_UEL_ABQ research in this folder. Read START_HERE.md first,
> then the current handoff, decisions and report it links. Do not restart the
> completed campaign. Preserve its evidence and distinguish verified results
> from proposed work. First briefly confirm the current state and recommended
> next investigation.
