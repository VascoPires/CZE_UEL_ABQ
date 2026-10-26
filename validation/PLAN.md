# Autonomous verification and validation execution plan

**26 October continuation:** the first campaign below is completed. The
user-requested matched elastic 2D/3D study has a separate
[plan and execution record](elastic_dimensions_20261026/README.md): one native
CPS4 job completed,47 prepared, actual submission deferred behind an unrelated
active queue. Preserve its persistent external-queue hold and follow handoff22.

Updated 2026-10-25. Authority: current [Prompt0.md](../Prompt0.md), D13–D17 in [DECISIONS.md](DECISIONS.md). This is an execution record, not a new approval gate. Preserve the canonical UEL and all reference/historical data. Store models, controls, failed attempts and source hashes under `campaign_20261025/`. Run one Abaqus/Standard job at a time, with one CPU initially. Post-processing and independent material checks may run alongside that job.

The old Checkpoint 2 stop is superseded by the current autonomous protocol. Consequential constitutive changes or calibration remain researcher decisions. Unknown technical details may be used for explicitly labelled diagnostic models; such runs do not become faithful published reproductions merely by completing.

## 1. Parameters and evidence

Published energies, elastic properties and eta stay with their source case. Turon 2006 Table 3 supplies the primary experimental observations; the 2002 table remains separate. Exclude both historical `benchmark_turon` files from quantitative targets. Ordinary plane-strain slices enforce equal in-plane displacements across width and zero width displacement; they do not substitute for finite-width 3D models.

| Family | Source baseline | Adjustable numerical inputs / limitations |
|---|---|---|
| Unit element | Existing test GIc=.187, GIIc=.781 N/mm; normal/shear strengths38/52 MPa; eta1.22; K3=1e5 N/mm³ | K3=1e4,1e5,1e6,1e7; both integration schemes; rho=0,1e-4,1e-3; temporal refinement |
| Turon 2010 DCB/MMB | Source geometry/materials; K3=1e6; strengths30/58.9 MPa for the adjusted-strength branch; eta2 | K sweeps1e5/1e6/1e7; strength15/29.45 and60/117.8 sensitivity cases retain baseline separately. Published Fig.4 pure-I strength30 MPa remains the central DCB comparison. L=75 mm is the approved interpretation. |
| Krueger MMB | Published geometry152.4×25.4×4.5 mm; a0=25.4; published energies/eta | Strengths32/61 MPa and K3=170000 are assumed project defaults. Screen K3=170000,500000,1e6,1.7e6,5e6,1e7. Vary strengths separately with documented purpose; never fit them to Pcrit. |
| AS4/PEEK experimental family | Turon 2006 per-test cracks, energies, strengths80/100 and eta2.284 | Numerical convergence first. Changing a strength or stiffness is a labelled sensitivity, not replacement of the original prediction. Original experimental scatter remains unavailable. |
| Turon 2018 MMB | Published a0=32.64, eta2, K3=1e5; central extracted strength pair30/80 | Reconstructed weight-free companion apparatus, explicitly labelled. Its global fractions need checking; it is not an exact author-model reproduction. |

Source cards supply page/table provenance. `run.json` supplies every actual model choice, including assumed controls. Baseline source choices are not overwritten by a better-conditioned numerical alternative.

## 2. Penalty assessment

Use Turon et al. 2007, *Engineering Fracture Mechanics*74,1665–1682, pp.1667–1668, Eqs.4–7 (DOI10.1016/j.engfracmech.2006.08.025). For an adjacent sublaminate thickness t, alpha=K3*t/E3 and the simple transverse-series model gives Eeff/E3=alpha/(1+alpha). This is a screening estimate, not the actual DCB compliance. State whether t is an arm or a ply; here the beam-arm thickness is used and sensitivity to that scale is explicit.

For Krueger, t=2.25 mm and E3=11380 MPa give alpha≈33.61 at K3=170000 N/mm³, or≈2.89% stiffness loss in that screening model. At K3=1e6, alpha≈197.72 and loss≈0.50%. The shear penalty is recomputed through the actual UEL relation, Ksh/K3=(GIc/GIIc)(tau_sh/tau3)^2; holding Ksh fixed while varying K3 would change the intended law.

Also record delta0=normal_strength/K3, deltaf=2GIc/normal_strength and zeta/delta0. Raising K3 with the fixed1e-6-mm numerical perturbation may degrade tangent accuracy even when interface compliance improves. Test the actual extracted Fortran material tangent against independent central differences away from onset/compression branch switches. If a perturbation problem is demonstrated, test a separate smaller-perturbation source copy, preserve the original and compare both responses.

## 3. Single-element verification

Fully prescribe independent unit-area U8 patches, avoiding mechanisms after complete separation. Retain normal/shear/local-axis mapping, all three reaction components, separations, D/Dv, stored energy and accumulated dissipation. Do not equate external work to damage energy while elastic energy remains.

| Check | Configuration / reference | Required outputs and failure meaning |
|---|---|---|
| Pure I,II,III | Monotonic opening/sliding to separation; theoretical triangle areaGIc/GIIc | Peak traction,onset separation,traction continuity,final D,area under traction–separation,ALLPD/SDVs |
| Proportional mixed mode | Local energy fractions0.2,0.5,0.8; expected additiveBK energy | Compute displacement ratio from K3/Ksh; do not substitute an unweighted displacement ratio |
| Unload/reload | Partial damage, unload, reload, then separate; Gauss and Newton–Cotes | D must not heal; unloading follows damaged stiffness; zero spurious dissipation during constant-D unloading |
| Closure | Separate then compress | Normal compression stiffness remains K3; final external work includes recoverable compression energy |
| Changing mode | Change shear sign/direction and opening before separation | Compare traction work minus stored energy with accumulated dissipation at successively finer increments; no single prescribed final nominalGc |
| Material tangent | Compile the actual constitutive routines; central finite differences using the same old state | Normed error by active branch and K. Exclude nondifferentiable onset/closure points from smooth-branch tangent verdicts; report them separately |
| Full element | Actual Abaqus residual/response and matched integration results | Material checks alone cannot establish assembly or finite-rotation correctness |
| K/rho sensitivity | Same law with named K values; rho0 versus1e-4/1e-3, unit step time | Track zeta/onset scale, excess dissipated energy and dt/rho. Positive rho is an extension, not the rate-independent baseline |

Initial solver batches use maximum normalized increments0.005 and0.001; sampled material paths explicitly resolve elastic onset and softening. Where reaction-work error remains above the working criterion, refine the path near onset or lower the increment before diagnosing a law defect. Retain unsuccessful attempts.

## 4. DCB execution

Start with Turon 2010: length150, width20, arm thickness1.55, crack35 mm. A structured C3D8I/U8 slice provides explicit control of mesh and constraints. Start dx0.6 mm and5 elements/arm, then0.3 and0.15 mm; increase thickness resolution independently if needed. Scale all forces and energies by full-width/slice-width, never displacement. The source's CPE4 versus plane-stress contradiction remains documented; the selected state is plane strain. Use free in-plane rotation at the loaded sections and ±half-opening; preserve the precise coupling geometry in the manifest.

The initial diagnostic baseline is small-displacement bulk kinematics for comparison with beam references, with a separately labelled NLGEOM sensitivity. The UEL itself updates its local basis from deformed coordinates. A discrepancy between bulk/UEL kinematics must be investigated rather than hidden by changing material properties.

Execute the following adaptively, retaining one-factor comparisons: mesh0.6→0.3→0.15; penalty1e5→1e6→1e7; Gauss versus Newton–Cotes; increment0.005→0.0025 or finer; rho0 versus a small positive value; strength15/30/60 MPa; and NLGEOM off/on. Cross-check elastic behaviour with a fully bonded fixed-crack reference before attributing a beam-theory compliance gap to the UEL. Source mesh and numerical-element differences limit claims about an exact author-model reproduction.

Extract one-arm force, full relative opening, onset/peak, stable propagation, crack-front position, energy and solver attempts. Define a fully separated front with D≥0.995 and check D≥0.999 sensitivity; report onset and cohesive-zone lengths separately. Do not treat these numerical thresholds as published crack measurements.

Analytical comparison: Appendix A1–A2 exactly as printed, plus clearly labelled root/shear/large-rotation diagnostics if needed. Initial beam compliance and the published graphical initial slope may differ; propagation-only transcription agreement cannot settle the initial crack-length correction. Preserve such disagreement.

## 5. MMB and ENF execution

After the DCB investigation is assessed and documented, continue automatically to the agreed mixed-mode cases. A linear, weight-free equivalent lever enforces w_load=(1+c/L)w_mid−(c/L)w_end and preserves virtual work; load/output is at the lever coordinate. It supplies a documented diagnostic reconstruction. Check its force split and energy against statics before using it. A finite-rotation rigid-apparatus comparison is required before claiming equivalence to a source's explicitly modeled lever when rotations matter.

Turon 2010 uses c149.05/63.18/41.06 mm at nominal20/50/80%; retain the adjusted-strength/LEFM targets. Compute both nominal-constant-Gc and evolving-LEFM-fraction references as separate policies. Krueger uses c92.9/41.3/27.3 mm; compare critical points128.5/385/751 N and1.64/1.34/1.65 mm, but distinguish equilibrium snap-back points from displacement-controlled jump paths. The finite-width benchmark requires a separate3D representation; width scaling cannot supply crack-front variation.

The 2018 apparatus is reconstructed using the approved companion approach and L51 mm. Formula-derived c values are assumptions derived from the stated ideal load split, not reported2018 dimensions. Check achieved analytical fractions rather than forcing local cohesiveB to remain equal to nominal mixity. Its old-formulation series is a counterexample.

Experimental AS4/PEEK DCB/MMB/ENF retain the2006 per-test crack lengths. Use the companion numerical lever lengths for MMB, retaining experimental apparatus/weight differences in the uncertainty record. Precrack contact must prevent interpenetration in ENF and mixed bending; after debonding, retain the UEL's normal compression response.

## 6. Metrics and working diagnostic criteria

These are declared screening criteria for reproducible assessment. They are not represented as previously approved thesis claims or universal literature requirements. Final validation claims remain researcher judgment.

- Relative error: (prediction−reference)/abs(reference), with signed and absolute values retained.
- Unit proportional fracture energy: ≤1%; damage irreversibility within output precision; no negative dissipation beyond numerical tolerance.
- Smooth material tangent: normalized Frobenius error≤1%; report maxima and percentiles away from branch switches. Full element tangent is a separate requirement.
- External work balance: |W−(recoverable+dissipated)|/max(peak |W|,small floor)≤1% after refinement; account for stabilization and bulk energy separately.
- DCB initial compliance: fit the common undamaged interval from5–20% of the reference initiation/peak load, require linear fitR²≥0.999, compare source beam and numerical fixed-crack references separately. A1–A2 disagreement alone does not prove a UEL bug.
- Peak comparison: ≤2% for a source-equivalent numerical benchmark. A model with unresolved physical assumptions is INCONCLUSIVE even if this screen passes.
- Propagation: linear interpolation on the common stable postpeak displacement interval; start after the larger of the two peak displacements plus5% of that interval; end at the smaller available endpoint. Report normalized mean absolute error and maximum error relative to reference peak; mean target≤2%. Do not interpolate across a snap-back jump or extend a curve beyond its data.
- Mesh/K/increment sensitivity: successive peak changes≤1%, with propagation/error and onset reported too. A two-level near-cancellation is insufficient for a convergence claim; examine three levels where available.
- Experimental comparisons: report discrepancies and numerical uncertainty, but do not apply the2% numerical target as an experimental validation limit without scatter/source uncertainty. Existing toughness fitting and missing raw observations remain explicit.

A completed job is necessary but insufficient for PASS. Use FAIL for a demonstrated failed quantitative check; INCONCLUSIVE for inadequate reference equivalence/uncertainty; NOT RUN for unexecuted tests. Distinguish a passed diagnostic identity from a final physical validation verdict.

## 7. Reproducibility, cost and failure handling

Every run has a fresh folder, exact input/source hashes, a command, start/end timestamps and status in run.json. Failure logs are preserved and corrected decks receive new names. Post-processing retries preserve their earlier logs/partial data. Canonical UEL modifications are prohibited; isolated experimental variants carry their own hash and patch record.

Unit batches contain16 independent patches per job and take tens of seconds on this host. Structural timing is measured by the first DCB pilot; mesh refinement and finite-width3D can cost substantially more. Do not extrapolate a successful input check into a runtime promise. Refine adaptively rather than launching a full Cartesian product. Maintain one active solver and check disk usage; the initial campaign started with about341GiB free.

Figures read retained CSV/JSON results. Report all failed and incomplete comparisons, peak and propagation differences, sensitivity results, solver warnings and reference limitations. Keep the original audit as historical evidence and append current campaign results to REPORT.md. There is no phase-end permission request under the updated protocol.

## Execution refinements from actual evidence

The live queues and interventions are recorded in HANDOFF.md and campaign_20261025/INVESTIGATION.md. Corrected2018 lever reconstructions enforce the nominal initial global mode fraction using the corrected beam equations, without force/displacement fitting; original ideal-lever cases remain separate. The0.1mm control matches the source spacing locally in the fracture region, while the far field/loading-end mesh remains graded; it is not a uniform author-mesh reproduction.

An elastic changing-area full-UEL counterexample is now part of element verification. Constant-area energy checks do not close this defect. A nominal/reference-area source copy is tested only as an isolated measure hypothesis; canonical adoption and the intended finite-deformation traction/energy measure remain researcher decisions. Explicit adaptive stabilization and all energy/contact screens are checked independently of solver termination. A mechanically admissible completed adaptive control may supply a provisional peak after a failed baseline; selection uses completion, postpeak, energy and gap checks, never target agreement. Baselines remain in the inventory and overall convergence/source-equivalence verdicts stay separate.


### Mesh and regularization follow-up

Execute queue_convergence_followup.json after queue_final_retry.json: DCB0.075mm and AS4 MMB80 0.1mm unstabilized refinement, followed by matched corrected-constraint DCB0.3mm rho=1e-4 at maximum step increments0.02/0.001 and a conditional fine-increment adaptive retry. Keep material, K, viscosity and numerical controls fixed within each one-factor comparison. Follow-up INC=3000 is a declared computation cap, unrelated to load peak. Report incomplete traces and actual solver reasons without substituting maxima for peaks. Assess maximum ETOTAL drift, ALLSD/peak work and recorded postpeak before using any stabilized prediction. Refresh all tables and figures after the four final queues have ended.

The same follow-up queue includes a separate corrected-lever2018 MMB20 K3=1e6 control, compared with its publishedK3=1e5 baseline. The arm-thickness series screen gives6.08% versus0.643%; actual MMB compliance and convergence must be assessed from the retained runs.

The final follow-up task list is frozen at seven entries: five unconditional runs and two conditional retries (higher-K smaller perturbation; fine-increment viscous adaptive stabilization). The high-K tangent retry is justified by the directly measured2.4224% error; it changes no material response at fixedK. All queued work must end or be explicitly skipped before final reporting.

Reported early-load compliance values in legacy metrics.json use5–20% of achieved maximum force; this interval is not independently certified undamaged. The report labels these early-load fits and does not award the planned undamaged-compliance PASS from them. Tied fixed-crack elastic controls remain separate evidence. This is an evidence limitation, not a relaxed acceptance criterion.


## Final execution status — 2026-10-26

The agreed campaign and seven-entry convergence follow-up are complete, including the justified conditional tangent retry and the skipped adaptive viscous retry. All106 Abaqus attempts are retained and assessed (50 completed,56 failed). The report,17-case matrix and16 scientific figure sets distinguish passed diagnostics from1 overall FAIL and16 INCONCLUSIVE benchmark verdicts. Known geometric energy defects, failed paths, unresolved apparatus equivalence, incomplete mesh/time/width convergence and experimental independence limits are documented. The refined viscous run passes energy but retains a2.114% peak change; it does not establish temporal convergence. Canonical source is unchanged. No additional computation is required by this recorded campaign; physical reformulation or target calibration would be a separate researcher decision.


### Subsequent authorized interface extension

D18 exposes optional tangent perturbation in v1.4 with the v1.3 default retained. The separate tangent_input_20261026 verification is complete, including actual Abaqus and compiled equivalence checks. It does not rerun or alter the first campaign, change the material law, or resolve its physical/geometric limitations.
