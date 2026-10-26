# Abaqus UEL verification and validation investigation — 25 October 2026

Campaign started25 October2026; assessment generated 2026-10-26T00:24:55.838495+00:00.

**Execution status: EXECUTION AND ASSESSMENT COMPLETE for the recorded campaign.**

The campaign executes the updated autonomous [Prompt0.md](../Prompt0.md), under [decisions D6–D17](DECISIONS.md). It includes actual Abaqus2023/Standard jobs, independent constitutive and assembled-element Fortran checks, numerical sensitivity studies, published graphical comparisons and provisional experimental comparisons. The canonical UEL is unchanged. An elastic changing-area test demonstrates an energy defect even at rho=0. This report records passed subchecks, failed checks and insufficient evidence separately; it does not declare the implementation fully validated.

The initial reference-only audit is retained in [REPORT_historical.md](audit/phase0_2026-10-25/REPORT_historical.md). Its old NOT RUN/checkpoint status is historical. Current source cards, actual model manifests and results have distinct roles.

## Scope and evidence

| Evidence | Actual execution / retained record |
|---|---|
| Abaqus jobs | 56 FAILED, 50 COMPLETED; [full run inventory](campaign_20261025/results/run_inventory.csv) |
| Unit patches |16 independent patches per batch; both integration schemes, six proportional modes, history changes, closure, K and rho studies |
| Material checks |Canonical and smaller-perturbation copies use the same23 histories; an additional49-history run includes20 tightly bracketed onset probes and six viscous refinements |
| Full UEL assembly |Nine residual/Jacobian states per source variant, including rigid rotations0/15/30degrees; serial runtime stubs, separate from Abaqus solver-interface evidence |
| Structural controls |Native cohesive comparisons, ordinary-plane-strain checks, elastic fixed-crack ties, independent VCCT and compliance derivatives where completed |
| Experiments |Turon2006 Table3 is primary; Camanho2002 Table4 remains separate. Published experimental markers are digitized; original instrument records/repeat scatter remain unavailable |

All working numerical thresholds are declared in [PLAN.md](PLAN.md). They are diagnostic screens, not previously approved experimental validation tolerances: proportional energy1%, smooth tangent1%, work balance1%, successive numerical peak changes1%, source-equivalent numerical peak2% and stable-branch mean error2%. Reference uncertainty, unresolved apparatus equivalence or failed convergence can prevent PASS even when an individual error falls below a threshold. No force/displacement curve is shifted, and no strength/toughness is fitted to the target observations.

## Seventeen-case verification and validation matrix

| Case | Verdict | Evaluated job | Solver | Resolved peak [N] | Peak difference |
|---|---|---|---|---:|---:|
| element | FAIL | [element_dt001](campaign_20261025/runs/element_dt001/run.json) | COMPLETED | — | — |
| turon2010CS_dcb | INCONCLUSIVE | [dcb_ps03](campaign_20261025/runs/dcb_ps03/run.json) | COMPLETED | 51.43 | +0.839% |
| turon2010CS_enf | INCONCLUSIVE | [t2010_enf_pc](campaign_20261025/runs/t2010_enf_pc/run.json) | FAILED | — | — |
| turon2010CS_mmb20 | INCONCLUSIVE | [t2010_mmb20_open](campaign_20261025/runs/t2010_mmb20_open/run.json) | FAILED | — | — |
| turon2010CS_mmb50 | INCONCLUSIVE | [t2010_mmb50_open](campaign_20261025/runs/t2010_mmb50_open/run.json) | FAILED | — | — |
| turon2010CS_mmb80 | INCONCLUSIVE | [t2010_mmb80_open](campaign_20261025/runs/t2010_mmb80_open/run.json) | FAILED | — | — |
| krueger2012CR_mmb20 | INCONCLUSIVE | [kr_mmb20_K170k_open](campaign_20261025/runs/kr_mmb20_K170k_open/run.json) | FAILED | — | — |
| krueger2012CR_mmb50 | INCONCLUSIVE | [kr_mmb50_K170k](campaign_20261025/runs/kr_mmb50_K170k/run.json) | FAILED | — | — |
| krueger2012CR_mmb80 | INCONCLUSIVE | [kr_mmb80_K170k](campaign_20261025/runs/kr_mmb80_K170k/run.json) | FAILED | — | — |
| turon2006MM_exp_dcb | INCONCLUSIVE | [as4_dcb_ps](campaign_20261025/runs/as4_dcb_ps/run.json) | FAILED | — | — |
| turon2006MM_exp_enf | INCONCLUSIVE | [as4_enf](campaign_20261025/runs/as4_enf/run.json) | FAILED | — | — |
| turon2006MM_exp_mmb20 | INCONCLUSIVE | [as4_mmb20_adaptive](campaign_20261025/runs/as4_mmb20_adaptive/run.json) | COMPLETED | 91.47 | -15.383% |
| turon2006MM_exp_mmb50 | INCONCLUSIVE | [as4_mmb50](campaign_20261025/runs/as4_mmb50/run.json) | FAILED | — | — |
| turon2006MM_exp_mmb80 | INCONCLUSIVE | [as4_mmb80](campaign_20261025/runs/as4_mmb80/run.json) | FAILED | — | — |
| turon2018CS_mmb20 | INCONCLUSIVE | [t2018_mmb20_lefmlever](campaign_20261025/runs/t2018_mmb20_lefmlever/run.json) | COMPLETED | 99.57 | +2.454% |
| turon2018CS_mmb50 | INCONCLUSIVE | [t2018_mmb50_lefmlever_adaptive](campaign_20261025/runs/t2018_mmb50_lefmlever_adaptive/run.json) | COMPLETED | 273.2 | +0.902% |
| turon2018CS_mmb70 | INCONCLUSIVE | [t2018_mmb70_lefmlever](campaign_20261025/runs/t2018_mmb70_lefmlever/run.json) | FAILED | — | — |

A resolved peak requires a completed loading path and a recorded postpeak drop. Energy/contact/convergence screens are additional requirements. Failed-job maxima are retained in the inventory as **maximum reached force**, never substituted for a verified peak. Experimental differences use the primary2006 table; numerical differences use the registered graphical series. The full machine-readable matrix includes the reason for every verdict.

[Solver termination register](campaign_20261025/results/solver_terminations.csv) preserves actual error messages, including preprocessing failures. An ODB without response frames yields an inconclusive assessment with unavailable values; it is not treated as a zero-force or zero-energy result. [Submission interval audit](campaign_20261025/results/submission_interval_audit.json) checks serial execution from retained timestamps.

When an unstabilized baseline fails, a named adaptive control or the finer unstabilized AS4MMB80 control is used for the reported resolved peak only if it completes with a postpeak drop, work error≤1%, stabilization/work≤1% and the applicable inactive-contact check passes. The finer unstabilized control takes precedence over an adaptive control when both qualify. This selection does not use agreement with target data; the original baseline remains in the inventory. The overall benchmark verdict still requires convergence and source equivalence. [Diagnostic subcheck matrix](campaign_20261025/results/diagnostic_subcheck_matrix.csv) records individual PASS/FAIL/INCONCLUSIVE checks separately.

## Constitutive and element verification

The refined Abaqus unit batch has a maximum proportional/closed-cycle fracture-energy discrepancy of 0.0258%. DAT rounding limits the reported precision. The element area is1mm²; reaction work and dissipated energy are normalized consistently. Closure keeps compressive recoverable energy, so total external work is not equated to fracture energy alone.

| Unit history | Fracture-energy error | Final work-balance error | Minimum sampled damage increment |
|---|---:|---:|---:|
| B0.0 | +0.000% | +0.033% | 0 |
| B0.2 | -0.000% | +0.023% | 0 |
| B0.5 | +0.001% | +0.010% | 0 |
| B0.8 | +0.026% | +0.032% | 0 |
| B1.0 | -0.026% | +0.021% | 0 |
| modeIII | +0.026% | +0.030% | 0 |
| unload_reload_s1 | +0.000% | +0.001% | 0 |
| unload_reload_s2 | +0.000% | +0.001% | 0 |
| closure | +0.000% | +0.060% | 0 |
| mode_change_s1 | — | +0.072% | 0 |
| mode_change_s2 | — | +0.072% | 0 |
| K10000.0 | +0.001% | +0.002% | 0 |
| K1000000.0 | +0.001% | +0.031% | 0 |
| K10000000.0 | +0.001% | +0.380% | 0 |
| rho0.0001 | — | +0.005% | 0 |
| rho0.001 | — | +0.045% | 0 |

Twenty independent onset probes bracket the predicted initiation separation at0.999 and1.001times its analytical value. The material checks retain the old state when perturbing the three separation components; they do not differentiate along an already-updated damage history. Smooth-branch tangent errors exclude onset/closure kinks, which have no unique classical derivative.

| Normal penalty [N/mm³], B=.5 | Canonical maximum tangent error | Smaller-perturbation test |
|---|---:|---:|
| 10000 | +0.017% | +0.000% |
| 100000 | +0.149% | +0.001% |
| 1e+06 | +1.435% | +0.015% |
| 1e+07 | +12.576% | +0.145% |

The four actual structural material-property sets are also tested at five proportional fractions each, independently of the generic penalty sweep. Krueger strengths/stiffness remain assumed project inputs.

| Structural property set, B=.5 | Canonical tangent error | Smaller-perturbation test |
|---|---:|---:|
| turon2010 | +1.602% | +0.016% |
| krueger2012 | +0.266% | +0.003% |
| turon2006_dcb | +0.658% | +0.007% |
| turon2018 | +0.245% | +0.002% |

For the separate2018 K3=1e6 sensitivity, five additional proportional histories per source copy are executed. At B=.5 the canonical tangent error is +2.422% versus +0.024% with the smaller perturbation. These are standalone material checks, separate from the higher-K Abaqus structural run.

**Identified canonical numerical defect:** the fixed1e-6mm forward perturbation loses tangent accuracy as the onset separation becomes smaller at high K. The isolated1e-8mm source copy improves the tested smooth tangents. [Material response equivalence](campaign_20261025/sources/material_patch_response_equivalence.json) checks traction/history/energy columns separately from tangent columns. This patch has not replaced the canonical source.

The full assembled UEL residual-versus-tangent diagnostic has about0.382% normalized error in the tested softening state; reducing the material perturbation barely changes it. The source updates the local basis and area but omits their full geometric derivatives. The restricted rotated-patch tests do not establish a generally consistent finite-deformation tangent. Native cohesive DCB controls also fail near the same coarse-mesh opening, so the structural convergence problem is not assigned solely to this UEL tangent.

## DCB numerical sensitivity and reference comparison

| Job | Solver | Resolved peak [N] | Maximum accepted [N] | Early-load C [mm/N] | Work-balance error | Stabilization / work |
|---|---|---:|---:|---:|---:|---:|
| dcb_adaptive_dx0p15 | COMPLETED | 51.56 | 51.56 | 0.0473201 | +0.008% | +0.098% |
| dcb_adaptive_dx0p3 | COMPLETED | 51.43 | 51.43 | 0.0470797 | +0.180% | +0.005% |
| dcb_adaptive_dx0p6 | FAILED | — | 51.34 | 0.0469366 | +0.084% | +0.004% |
| dcb_local015_rho1e5 | COMPLETED | 51.57 | 51.57 | 0.0473186 | +252.730% | +0.000% |
| dcb_native_dx06 | FAILED | — | 52.86 | 0.0469359 | +0.000% | +0.000% |
| dcb_native_local015 | FAILED | — | 51.5 | 0.0473197 | +0.000% | +0.000% |
| dcb_native_local03 | FAILED | — | 51.39 | 0.0470789 | +0.000% | +0.000% |
| dcb_ps0075 | COMPLETED | 51.22 | 51.22 | 0.0473944 | +0.018% | +0.000% |
| dcb_ps015 | FAILED | — | 51.55 | 0.0473194 | +0.005% | +0.000% |
| dcb_ps015_stab2e5 | COMPLETED | 57.82 | 57.82 | 0.0442561 | +0.014% | +15.380% |
| dcb_ps03 | COMPLETED | 51.43 | 51.43 | 0.0470789 | +0.180% | +0.000% |
| dcb_ps03_K1e5 | COMPLETED | 51.02 | 51.02 | 0.047255 | +0.017% | +0.000% |
| dcb_ps03_K1e7 | COMPLETED | 51.45 | 51.45 | 0.0470593 | +0.182% | +0.000% |
| dcb_ps03_gauss | FAILED | — | 51.81 | 0.0470961 | +0.004% | +0.000% |
| dcb_ps03_inc005 | FAILED | — | 51.47 | 0.0470789 | +0.001% | +0.000% |
| dcb_ps03_nlgeom | FAILED | — | 51.52 | 0.0470649 | +0.142% | +0.000% |
| dcb_ps03_refarea | COMPLETED | 51.39 | 51.39 | 0.0470789 | +0.180% | +0.000% |
| dcb_ps03_rho1e4_dt001 | COMPLETED | 51.83 | 51.83 | 0.0470789 | +0.415% | +0.000% |
| dcb_ps03_rho1e4_dt02 | COMPLETED | 52.95 | 52.95 | 0.0470789 | +112.777% | +0.000% |
| dcb_ps03_stab2e4 | COMPLETED | 51.44 | 51.44 | 0.0470885 | +0.178% | +0.050% |
| dcb_ps03_stab2e5 | COMPLETED | 58.68 | 58.68 | 0.0421652 | +0.088% | +16.245% |
| dcb_ps06 | FAILED | — | 51.34 | 0.0469358 | +0.085% | +0.000% |
| dcb_rho1e3 | COMPLETED | — | 60.61 | 0.0470795 | +14.844% | +0.000% |
| dcb_rho1e4 | COMPLETED | 52.95 | 52.95 | 0.0470789 | +112.777% | +0.000% |

The corrected0.3mm DCB baseline gives 51.4275N versus 50.9998N in the published30MPa graphical series (+0.839%). The common stable propagation mean absolute error, normalized by the graphical peak, is +0.771%. Graphical uncertainty is±0.03mm/±0.8N, so these error digits are computational descriptors, not reference accuracy. The0.15mm unstabilized mesh and0.3mm smaller-increment job fail; source-equivalent mesh/increment convergence is therefore not established.

The0.075mm longitudinal refinement completes the full4mm path without viscosity or stabilization, whereas the0.15mm run failed. Its peak is 51.2219N (-0.400% versus0.3mm), and maximum energy drift is +0.018%. The recorded peak moves from 2.64 to 2.56mm; the maximum displacement increment is0.08mm in both runs, so the peak locations are coarsely sampled. This demonstrates a beneficial mesh effect but does not establish monotonic solution robustness or temporal convergence.

The tabulated early-load compliance is fitted over5–20% of each achieved maximum force where enough samples exist. This diagnostic interval is not certified undamaged; local cohesive damage can begin before the global peak. It must not be interpreted as a pure elastic penalty-compliance measurement. Independent tied fixed-crack controls provide separate elastic evidence.

Printed Appendix A1–A2 give initial DCB compliance0.0383785mm/N, whereas the solid FE controls give about0.047mm/N. This discrepancy is retained. The published numerical initial slope is closer to the FE result; root rotation/shear are plausible contributors. The root-correction expression in the paper’s MMB appendix gives a separate diagnostic scale; it is not silently inserted into the printed DCB equation or fitted by changing crack length. Agreement on the propagation branch alone does not verify initial compliance.

A stabilization input fraction is not a bound on actual artificial energy. Solver messages show that STABILIZE=2e-4 activated adaptive damping, whereas2e-5 used constant damping. The latter completed with roughly15–16% stabilization/work and materially higher forces; those runs are rejected as converged predictions. Later named adaptive controls explicitly hold ALLSDTOL=.005. Their actual energy and force sensitivity remain the deciding evidence.

The user-directed follow-up adds0.075mm DCB and0.1mm AS4 MMB80 meshes without changing material or regularization, then holds DCB rho=1e-4 fixed while reducing the maximum normalized step increment0.02→0.001. Step duration is1. A conditional adaptive retry is a separate intervention. These follow-ups have a declared3000-increment computation cap. No run stops because peak load was detected. [Matched numerical-control table](campaign_20261025/results/numerical_control_comparisons.csv) retains failed paths, resolved peaks, successive eligible peak differences and energy screens; missing peak differences are not zero errors.

## Penalty and strength choices

Normal/shear strengths and K are source-specific. Krueger’s VCCT benchmark supplies no cohesive strengths or stiffness: 32/61MPa and170000N/mm³ are labelled project assumptions. The sensitivity study preserves that baseline, increases K separately and refines the mesh; it does not infer strengths from the target critical load. The DCB strength15/30/60MPa tests are named sensitivities, and failed attempts remain in the inventory.

The simple series-compliance screen in Turon et al.2007, Eqs4–7, gives alpha=K*t/E3 and relative transverse stiffness loss1/(1+alpha). Using the Krueger arm thickness2.25mm and E3=11380MPa, K=170000 gives about2.89% loss, versus about0.50% at1e6. This suggests the inherited penalty is not negligible; it does not measure actual MMB compliance. The thickness scale is explicitly an arm, not an individual ply. [Penalty screen](campaign_20261025/penalty_screen.csv) and the source record retain all tested values. Increasing K also shrinks onset separation and worsens the canonical fixed-perturbation tangent, so higher K alone is not an established remedy.

The2018 source explicitly states K3=1e5N/mm³. Its arm-thickness series screen gives6.08% loss; a separate K3=1e6 control gives0.643% in that screen. This diagnostic changes only the penalty of the completed corrected-lever20% case, retaining the published-K baseline and the constitutive normal/shear stiffness ratio. It tests compliance and convergence sensitivity without fitting the published force curve. Actual completed/incomplete responses are retained in the matched numerical-control table.
If that higher-K canonical-perturbation run fails, a conditional smaller-perturbation retry holds the same K and solver controls, separating the directly measured tangent error from the physical penalty sensitivity. Both copies remain isolated from canonical source.

The higher-K canonical-perturbation case failed at 9.66272mm of11mm. The same-K smaller-perturbation case completed at 11mm, with peak 99.4873N and maximum energy drift +0.189%. Both use zero viscosity and zero step stabilization. This comparison changes the Jacobian approximation while preserving the tested constitutive response; it does not cure the independent geometric work defects.


In the actual Krueger50% runs, the first accepted secant compliance changes from0.003520747mm/N at K170000 to0.003505011mm/N at K1e6, about0.45%. These are single early-load secants, not multi-point undamaged fits. Both paths terminate near1.34mm, so their maxima are not verified peaks. Raising K does not resolve that benchmark discrepancy.

## MMB loading, dimensionality and independent energy-release checks

The linear, weight-free equivalent lever enforces u_load=(1+c/L)u_mid−(c/L)u_end and uses the reaction at that load coordinate. It preserves virtual work for that linear reconstruction; it does not automatically reproduce a finite-rotation rigid apparatus. The fixed-crack MMB50 control independently checks its force split through bulk NFORC and verifies its displacement identity. Abaqus2023 exports separate scalar NFORC1..6 fields; the extractor handles those explicitly. The VCCT sign was calibrated with the pure-I control before comparison with mixed loading.

Direct hard contact was incompatible with the paired-width equations and caused initial zero pivots. Explicit penalty contact resolves ENF initial loading and has a separate1e6-versus1e7 stiffness check. Initially opening MMB contact controls still failed near zero gap. A contact-free fixed-crack MMB control has no face penetration and correct lever statics; subsequent opening-MMB cases omit inactive initial-face contact and check the gap at every saved frame. The evolving UEL retains normal compression. A contact-free run that develops overlap cannot establish physical equivalence. ENF retains contact because its faces close.

Ordinary-plane-strain nodal constraints enforce width-invariant in-plane motion and zero width displacement. An initial end-coupling defect allowed small U3 motions because Abaqus ignored extra boundary conditions on coupling nodes; revision2 fixes the coupling and excludes those redundant BCs. Linear C3D8I controls directly show near-zero out-of-plane strains. Finite-displacement C3D8I reports small internal LE33 residuals despite constrained nodal motion; these values are retained rather than calling the full nonlinear element exactly identical to CPE4I. The finite-width Krueger control is a separate coarse3D diagnostic, not a width-scaled plane-strain validation.

| Fixed-crack control | Solver | B=GII/GT | Critical P [N] | Critical displacement [mm] |
|---|---|---:|---:|---:|
| fixed_as4_dcb | COMPLETED | 0 | 139.5 | 4.183 |
| fixed_dcb_a34p5 | COMPLETED | 0 | 53.34 | 2.406 |
| fixed_dcb_a35p0 | COMPLETED | 0 | 52.63 | 2.472 |
| fixed_dcb_a35p5 | COMPLETED | 0 | 51.94 | 2.538 |
| fixed_krueger2012_mmb20_open | COMPLETED | 0.1897 | 128.1 | 1.637 |
| fixed_krueger2012_mmb50_dx025_open | COMPLETED | 0.4799 | 383.5 | 1.346 |
| fixed_krueger2012_mmb50_open | COMPLETED | 0.4806 | 384 | 1.345 |
| fixed_krueger2012_mmb80_open | COMPLETED | 0.7834 | 743 | 1.637 |
| fixed_turon2010_mmb20_open | COMPLETED | 0.1976 | 40.07 | 7.811 |
| fixed_turon2010_mmb50 | FAILED | — | — | — |
| fixed_turon2010_mmb50_n2s | FAILED | — | — | — |
| fixed_turon2010_mmb50_open | COMPLETED | 0.4971 | 127.7 | 6.83 |
| fixed_turon2010_mmb80_open | COMPLETED | 0.7976 | 247.4 | 8.443 |

The independent DCB compliance derivative gives G=0.001702203N/mm, compared with VCCT G=0.001702051N/mm (-0.009%). This uses0.5mm virtual closure and a1mm central crack-length interval; it is a diagnostic consistency check, not a crack-front convergence proof.

The printed MMB LEFM reference is evaluated twice: with evolving analytical B and with Gc held at nominal B. Both CSVs are retained per run. Local cohesive B, analytical modal partition and finite-width VCCT partition are distinct quantities. Krueger symbols describe critical equilibrium, including snap-back, and are not a displacement-controlled jump path. No curve is interpolated across a detected jump or extended beyond its reference domain.

The first2018 ideal-lever reconstruction achieves initial corrected-beam fractions below nominal (0.18605 for nominal0.20). A separate reconstruction analytically inverts the corrected beam partition at the source initial crack, giving c100.8067375/42.9266764/31.9981592mm for nominal20/50/70% under the adopted L51mm. This uses geometry, elastic properties and requested mixity, without force/displacement fitting. Both policies remain in the record. The0.1mm control matches the source longitudinal spacing locally in the fracture region; the far field/loading end remains graded, so it is not a uniform author-mesh reproduction.

## Experimental comparison and independence limits

Turon2006 Table3 and the digitized black square markers from Figs9–13 are the primary published observations. Camanho2002 Table4 remains a separate comparison; the later publication date does not establish newer experiments or corrections. The marker extraction retains only printed markers: no origin, missing measurements, or experimental interpolation is inserted. Read-off uncertainty is case-specific; ENF has only three markers. The source also notes sparse propagation observations.

| Baseline | Solver | Resolved peak [N] | 2006 observed peak [N] | Difference | Compared experimental markers |
|---|---|---:|---:|---:|---:|
| as4_dcb | FAILED | — | 147.5 | — | 2 |
| as4_dcb_ps | FAILED | — | 147.5 | — | 2 |
| as4_dcb_ps_adaptive | FAILED | — | 147.5 | — | 3 |
| as4_enf | FAILED | — | 748.4 | — | 2 |
| as4_enf_adaptive | FAILED | — | 748.4 | — | 2 |
| as4_mmb20 | FAILED | — | 108.1 | — | 3 |
| as4_mmb20_adaptive | COMPLETED | 91.47 | 108.1 | -15.383% | 6 |
| as4_mmb50 | FAILED | — | 275.3 | — | 3 |
| as4_mmb50_adaptive | FAILED | — | 275.3 | — | 3 |
| as4_mmb80 | FAILED | — | 518.7 | — | 2 |
| as4_mmb80_adaptive | COMPLETED | 471 | 518.7 | -9.203% | 9 |
| as4_mmb80_dx01 | FAILED | — | 518.7 | — | 1 |

The unstabilized AS4 MMB80 local0.1mm refinement fails at 4.99632mm of its6mm target. Its maximum accepted force is 471.066N, not a verified full-path peak. Material, penalty, viscosity and step controls match the0.3mm baseline; mesh refinement alone does not recover this MMB path.

**No independent experimental validation claim is made.** Toughness and eta were already fitted to related experiments; raw histories, repeat scatter and full apparatus metadata are missing. Unresolved numerical convergence or apparatus assumptions further limit interpretation. Peak and common-marker errors are provisional baseline-prediction comparisons, not a new calibration or a held-out validation score.

The fixed BK fit also differs from individual measured toughnesses: at nominal20%, it gives0.987994N/mm versus1.103N/mm in the2006 table, a10.43% energy difference. Under the same-geometry linear LEFM scaling alone this corresponds to about5.36% lower load. This is an explanatory estimate, not a corrected prediction; it does not account for geometry, achieved partition, finite process zones or numerical errors. [All five nominal toughness comparisons](campaign_20261025/results/experimental_toughness_fit_screen.csv) are retained without refitting eta or endpoint toughnesses.

## Energy consistency

**Elastic changing-area failure:** an actual full-UEL standalone cycle opens a unit patch to0.0001mm, stretches area from1 to1.2mm² at fixed gap, closes it, then restores the original geometry. Damage and final stored/dissipated energies are zero, but the canonical residual gives net external work−0.0001Nmm, or20% of peak loading work. The100- and1000-interval tests agree. This failure is distinct from damage quadrature and does not disappear with time refinement. The current-area force/energy weighting has no conjugate in-plane area-change force.

An isolated [reference-area experiment](campaign_20261025/sources/reference_area_experiment.json) gives net work about−1.26e-19Nmm in the same restricted cycle. It interprets traction and energy per undeformed area; a current-area model with additional variational area/basis forces is another possible formulation. The intended measure requires researcher judgment before adoption. The variant does not replace the canonical source or fix missing basis derivatives, the fixed material perturbation, or viscous quadrature. Constant-area unit and matched specimen tests are retained separately; this is not a general finite-deformation validation claim.

**Constant-area basis-change failure:** a second elastic full-UEL cycle rotates the midsurface0→45degrees while the small global gap direction stays fixed, then closes and restores the original geometry. This is not a rigid-body rotation of the entire separated configuration. Both canonical and reference-area variants leave about+0.000137909Nmm external work with zero final stored/dissipated energy, about27.58% of peak loading work. The reference-area variant therefore does not cure the missing frame/geometric work terms. Small residual/Jacobian error and correct net force alone do not prove that the residual is the gradient of the reported energy.

The authoritative additional assessment uses maximum variation of the solver-reported ETOTAL, divided by peak absolute ALLWK; final variation is retained separately. This includes host contact-energy accounting. The subset ledger |ALLWK−ALLIE−ALLSD−ALLVD| is also retained but omits contact elastic energy/discontinuity work. Individual ALLCCE/ALLCCDW histories were not requested in the early decks; ETOTAL captures their net contribution. UEL damage dissipation is already part of ALLIE through ALLPD; it must not be counted again. These models have no frictional dissipation. Stabilization and artificial strain energy are reported separately. Earlier metrics.json records retain their original final/subset definitions; additional_metrics.json and this report use the complete history criterion. All listed energies are slice energies unless force/energy width scaling is explicitly applied; relative errors are invariant under that scaling.

Positive-rho DCB runs can complete with severe energy errors. For example, dcb_rho1e4 reports W=7.076822Nmm and IE=15.057832Nmm. The canonical viscous dissipation uses endpoint trapezoidal integration of Y dDv. For fixed basis, nonnegative opening and diagonal K, the independently verified discrete identity is:

`Delta psi + Q - trapezoidal traction work = .25 DeltaDv (Delta delta)^T K (Delta delta)`.

Actual extracted Fortran tests with100/1000/10000 uniform increments give energy excess0.00990610/0.0000990610/0.000000990610Nmm for both tested rho values. The error decreases with refinement and matches the identity; it does not license an unrefined structural energy claim. Changing basis/area and nonproportional structural paths require their own work checks. No reported energy is altered merely to force a balance.

The matched corrected-constraint rho=1e-4 DCB comparison holds material, geometry, mesh and viscosity fixed. Reducing the maximum normalized step increment0.02→0.001 lowers maximum energy drift from +112.777% to +0.415%. Both runs complete4mm without step stabilization. Their recorded peaks change from 52.9467 to 51.8272N (-2.114%). The fine run passes the1% energy screen, but the peak change still exceeds the1% increment-sensitivity screen. These two levels therefore demonstrate strong increment sensitivity, not established temporal convergence or a zero-viscosity limit. The conditional adaptive retry is skipped because the fine run completes.


[Matched area-measure comparisons](campaign_20261025/results/reference_area_comparison.json) retain separate unit and specimen evidence.
The 16 constant-area unit histories have maximum dissipation difference 0Nmm between these copies at retained DAT precision.

| Area-measure test | Baseline peak [N] | Test peak [N] | Peak change | Baseline work error | Test work error |
|---|---:|---:|---:|---:|---:|
| dcb_ps03_refarea | 51.43 | 51.39 | -0.064% | +0.180% | +0.180% |
| t2018_mmb20_lefmlever_refarea | 99.57 | 99.39 | -0.189% | +0.384% | +0.381% |


## Reproducibility and retained limitations

[Exact-run replay and regeneration instructions](campaign_20261025/README.md) retain decks, source copies, command lines, hashes, solver diagnostics, post-processing failures and derived CSV/JSON data. Original models are never overwritten. The run inventory distinguishes UEL, native and tied elastic controls. One Abaqus solver is submitted at a time, initially with one CPU. The repository was dirty before the campaign; file hashes, not HEAD alone, identify the baseline.

Two older controllers had advanced beyond their intended stopping boundary, leaving stale RUNNING manifests after the solver had independently failed. Original manifests and queue ownership-guard errors are retained. Status was recovered from terminal STA evidence with no live lock; the original outputs were extracted and fresh opening-MMB replacements were executed. The submission audit records this resolution separately from the still-failed solver results and checks for overlapping execution intervals.

[Preservation check](campaign_20261025/results/preservation_check.json) compares62 original source/example files, including historical curves. The canonical SHA256 remains `bfa62fce8fef54fc8f3ba6f04946b3e0c5b68e4dbed5fb60987c8b27cc8d455c`. The unproven historical benchmark_turon CSVs are preserved but excluded from targets. Isolated I/O-off and smaller-perturbation copies are separately hashed and tested.
[Final artifact audit](campaign_20261025/results/final_artifact_audit.json) records ended queues, applicable assessment coverage, source/deck preservation, serial execution, local links, Python syntax and figure export checks. These integrity checks do not replace the scientific verdicts.

The remaining scientific limits are source-equivalent numerical convergence, exact finite-rotation apparatus equivalence, finite-width front convergence, a general finite-deformation UEL tangent/energy assessment, and independent experimental validation with raw observations/scatter. A failed or inconclusive benchmark remains a legitimate result; it is not relabelled PASS because a nearby run completes.

## Figures and source records

Figures are generated from retained numerical/reference data as SVG/PDF, with PNG review copies, at85 or180mm widths. Read the [figure captions](campaign_20261025/results/FIGURE_CAPTIONS.md) before thesis use. Dashed traces/crosses indicate incomplete jobs; open experimental markers are digitized published observations.

[Figure directory](campaign_20261025/results/figures/) · [Source cards](sources/) · [Digitization register](DIGITIZATION.md) · [Intervention log](campaign_20261025/INVESTIGATION.md) · [Full run inventory](campaign_20261025/results/run_inventory.json)

Primary methodological references: [Abaqus coupling constraints](https://docs.software.vt.edu/abaqusv2025/English/SIMACAECSTRefMap/simacst-c-coupling.htm), [static stabilization controls](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-static.htm), [surface behavior/contact enforcement](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-surfacebehavior.htm), [nodal element forces](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEOUTRefMap/simaout-c-std-wholeelementvariables.htm). [Contact energy accounting](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEEXARefMap/simaexa-c-contactenergy.htm) explains why ETOTAL rather than the incomplete element-energy subset is used for the final global check. The installed solver is2023; online2025 documentation explains the control semantics, and actual2023 messages/decks are the execution evidence. Published materials/geometry/observations are traced to local PDF pages in the case/source cards. The additional penalty reference is [Turon et al.2007](https://doi.org/10.1016/j.engfracmech.2006.08.025).
