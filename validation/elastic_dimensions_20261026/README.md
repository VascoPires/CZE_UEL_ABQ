# Matched 2D/3D elastic fixed-crack DCB comparison

26 October 2026. User-authorized continuation of the proposed study in
`START_HERE.md`, under `Prompt0.md` and decisions D9/D14/D17. The completed
first campaign and canonical UEL are preserved. This is a separate study.

**Status: one new native CPS4 job completed and extracted; 47 jobs prepared and
deferred behind a separate active solver queue.** See
`queue_status.json` for the latest submission check and `results/assessment.json`
for actual job statuses. An unrelated six-CPU Abaqus/Standard transverse-cracking
study was active during preparation. The runner refuses to overlap it; it does
not stop it, poll for a gap between its jobs, or start a background controller.

## Contract and comparison

The question is whether dimensional state and element formulation explain the
DCB elastic compliance gap before cohesive damage. No material fitting,
cohesive elements, contact, viscosity, stabilization or finite-displacement
effects enter these diagnostic controls. They cannot verify propagation or
resolve the known UEL geometric-energy defects.

Use Turon 2010 geometry and orthotropic constants already transcribed in
`../sources/turon2010CS.md` and the completed fixed-crack control: length 150 mm,
width 20 mm, arms 1.55 mm, initial crack 35 mm. Material direction 1 is length,
2 is arm thickness, 3 is width. Units are N, mm, MPa. End-section kinematic
couplings allow in-plane rotation; each arm's load reference has U1=0. Full
relative opening is 0.2 mm, with +0.1/-0.1 mm prescribed at the upper/lower
reference points. P is the force on one arm, not the sum of both magnitudes.
Coincident upper/lower interface nodes are tied from the crack tip onward.
This matches the previous diagnostic loading; it is not a new claim about the
paper's complete apparatus.

| Group | Discretization | Crack lengths [mm] | Jobs |
|---|---|---|---:|
| Matched dimensional/element comparison | CPS4, CPS4I, CPE4, CPE4I, C3D8 and C3D8I; dx=0.5/0.25 mm with 5/10 elements per arm | 34.5, 35, 35.5 | 36 |
| Finite-width sensitivity | C3D8I; dx=0.5 mm, 5 per arm, 3/7/15/25 across the full 20 mm width | 34.5, 35, 35.5 | 12 |

All crack lengths share exactly the same x stations at each mesh level. This
avoids changing the discretization while differentiating compliance. The two
levels refine longitudinal and thickness spacing together; they are an initial
sensitivity screen, not separate convergence proofs for each direction.
The width series holds the in-plane mesh fixed.

Native 2D models have section thickness 1 mm; constrained 3D slices have width
1 mm, equal U1/U2 at front/back and U3=0, including coupled ends. Both use a
force multiplier of 20. Finite-width models use the full width, no plane-strain
equations and no force multiplier. A single width anchor removes rigid
translation. Their end-section in-plane couplings are an explicitly imposed
diagnostic loading assumption. They are DCB controls, not reproductions of
Krueger's finite-width MMB apparatus or its failed UEL pilot.

## Evaluation, when solver results exist

Compute C=opening/P and stiffness 1/C. Use the central derivative at a=35 mm,
`dC/da=[C(35.5)-C(34.5)]/1 mm`, then `G=P^2/(2*b)*dC/da`.
VCCT uses the upper-tip bulk nodal force and the relative crack-face motion one
element behind. Sum nodal closure work before dividing by `b*dx`; do not
multiply a mean force by a mean opening for a varying finite-width front.
Retain nodal front values with half-width tributary areas at free edges.
The central compliance derivative uses total force and full specimen width;
VCCT uses the modeled section width, so the two definitions give the same
energy units. An LEFM initiation estimate scales elastic load by
`sqrt(GIc/G)`; it is not a cohesive peak prediction.

Check opposing reaction balance, linearity across saved frames, tie jumps,
out-of-plane strains/stresses as appropriate, and `ALLSE=0.5*P*opening` with
consistent width scaling. Working diagnostic screens inherited from the first
study are 1% energy agreement, 1% successive mesh change and 1% VCCT/compliance
agreement. Also report the raw differences. A 0.1% agreement screen between
CPE4/C3D8 and CPE4I/C3D8I tests near equivalence; it is not assumed in advance.
Missing, failed or deferred runs cannot pass these screens. Refine the crack
derivative interval separately if it controls the comparison. Finite-width
front variation and global compliance need separate convergence assessment.

## Reproduction

From the repository root, preparation and checks are nonprocessing:

```console
python -B validation/elastic_dimensions_20261026/scripts/study.py --prepare
python -B validation/elastic_dimensions_20261026/scripts/check_inputs.py
python -B validation/elastic_dimensions_20261026/scripts/study.py --check
```

Preparation refuses an existing study plan. Existing attempts are never
resubmitted. The retained plan includes input and script hashes. `--check`
checks prepared input hashes, hashes of first-campaign inputs/source/assessment
data, and size/modification times of other retained artifacts; it does not
rehash large binary solver files or certify solver acceptance.

Actual execution, when no other solver is active:

```console
python -B validation/elastic_dimensions_20261026/scripts/study.py --run
python -B validation/elastic_dimensions_20261026/scripts/assess.py
python -B validation/elastic_dimensions_20261026/scripts/study.py --check
```

Run one controller only. `--run` checks Windows solver processes before every
submission, returns DEFERRED_EXTERNAL_SOLVER if occupied, and resumes only
PREPARED jobs on a later explicit invocation. It preserves and stops on any
failed/nonfresh attempt. It uses the existing serial launcher, one CPU and
Abaqus 2023; no Fortran compilation is needed. Input acceptance and ODB
extraction remain unverified for the 47 unrun jobs. A persistent
`external_queue_hold.json` also prevents submissions during gaps in the known
external controller. Release it only after verifying that the external
campaign has ended or after explicit user authorization for concurrency.

The first preflight found no active solver and one CPS4 job ran successfully.
The external controller subsequently resumed. This exposed the limitation of
a process snapshot, prompting the persistent hold. Initial extraction attempts
exposed Python 2 unicode and single-precision JSON conversion issues. Their
logs and partial exports are retained; plan amendments record corrected hashes.

`scripts/verify_control.py` checks the new extractor against a copy of the
completed campaign's fixed-crack ODB. Force, compliance and GI match the
original exports exactly at retained precision. The prepared C3D8I baseline
has identical bulk connectivity and equation constraints, with coordinate
differences below 1e-9 mm. This reuses old solver evidence, not a new 3D run.

Read [the study report](REPORT.md) for numerical results and limits.

See [mesh-source audit](MESH_SOURCE_AUDIT.md) for the width-mesh question.
