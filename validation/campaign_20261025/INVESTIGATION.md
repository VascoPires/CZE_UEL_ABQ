# Investigation record

All dates are 25 October 2026. The canonical `src/3D_CZM.f` is preserved. Exact solver times, hashes and decks are in each `runs/*/run.json`. Failed jobs are retained.

## Material tangent

The canonical forward perturbation is a fixed 1e-6 mm. Independent central differences of the same old-state constitutive response demonstrate increasing smooth-branch error as penalty stiffness increases: at local B=.5, the maximum normalized error is 0.149%, 1.435% and 12.576% for K=1e5,1e6,1e7 N/mm³. The isolated 1e-8-mm variant reduces these to 0.00150%,0.0146%,0.145%. Stored traction/history/energy responses are unchanged in these proportional histories. This establishes a numerical tangent accuracy problem, not an approved constitutive reformulation.

The full assembled UEL tangent has a separate error: about 0.382% for the tested softening state, including rigid rotations up to30 degrees. Reducing the material perturbation barely changes this error, consistent with missing geometric derivatives when the basis/area update. These restricted tests do not establish a general finite-deformation tangent.

## DCB convergence

Uniform dx=.6-mm UEL and native COH3D8 controls both fail near2.68-mm opening with almost identical undamaged compliance and last accepted load. Line search and a smaller material perturbation do not remove this failure. A local dx=.3-mm UEL mesh completes to4 mm, but dx=.15 and .6 local variants fail. A finer native control fails too. These observations rule out attributing the convergence failure solely to the fixed material perturbation, but do not prove the generator or UEL is correct. No mesh-converged DCB result is claimed.

## Viscous energy quadrature

Positive rho permits several DCB solutions to complete but can produce unacceptable work/energy differences. For example, `dcb_rho1e4` ends with external work7.076822 Nmm, internal energy15.057832 Nmm and damage dissipation10.278909 Nmm. These are slice energies before width scaling; relative errors are unaffected by scaling. Solver completion does not pass the energy check.

An independent identity explains one discretization contribution. For fixed local basis, nonnegative opening, positive diagonal penalty K, Y=delta^T K delta/2 and psi=(1-Dv)Y, the canonical increment Q=.5(Y_old+Y_new)DeltaDv gives

`Delta psi + Q - trapezoidal_traction_work = .25 DeltaDv (Delta delta)^T K (Delta delta)`.

Thus this combination of endpoint quadratures overestimates energy during damage increments. It tends to zero with increment refinement and is not a negative thermodynamic dissipation result. In the actual canonical material routines, rho=1e-4 and1e-3 monotonic histories with100,1000,10000 uniform increments give energy excess0.00990610,0.0000990610,0.000000990610 Nmm, matching the identity to rounding. Their relative work errors drop from about2.2% to0.022% to0.00022%. A structural run also changes local basis/area, so this identity alone does not allocate its entire error. Do not repair the reported energy merely to force a balance or accept a viscous structural result without its own checks.

## Plane-strain constraint correction

Direct ODB checks identify small U3 motions at the nodes in the end kinematic coupling. Abaqus reports that its separate U3 boundary conditions on these coupling nodes are ignored. Other nodal width motions are essentially zero, and in-plane width-pair motion is equal. This is a generator defect affecting the initial diagnostic runs; their label describes intended dimensionality, not a completed strain-equivalence proof.

Generator constraint revision2 couples translations1–3 at slice end sections, fixes RP U3/UR1/UR2, and applies zero U3 only to uncoupled bulk nodes. Finite-width models keep their separate unconstrained width behavior. Elastic pilot jobs compare C3D8I, C3D8 and C3D8R and directly inspect bulk E33/E13/E23 (or LE for NLGEOM). Do not presume that constrained nodal motion also suppresses all C3D8I internal incompatible strain modes.

The original MMB queue controller was stopped after `as4_dcb` terminated and before any MMB job was submitted, so unstarted tasks can use the corrected generator. The completed/failed decks remain unchanged. The additional strain extractor records Abaqus E for small-displacement jobs and LE for finite-displacement jobs; a Python2 single-precision JSON serialization error was corrected separately and the failed attempt retained.

The corrected elastic C3D8I pilot has maximum absolute out-of-plane strain below2e-17; the completed propagating `dcb_ps03` has E33/E13/E23 below4e-16 and nodal U3 below1e-35mm. This directly supports ordinary plane strain for these models. The C3D8 comparison reports nonzero E33 despite constrained nodal motion; do not substitute nodal constraints for checking the element's reported internal strain field.

## Krueger geometry cross-check

The test-apparatus sketch in Fig.1 (printedp24/PDF28) includes tabs outside the supports. The actual FE model in Fig.3a–b (printedp25/PDF29) places the left support/hinge at the modeled left end, defines a0 from that origin, and extends the beam beyond the right support. The generator follows this FE coordinate convention: hinge/supportx0, midspan50.4, right support100.8 and modeled end152.4mm. Do not infer a symmetric25.8-mm left overhang for the FE reproduction from the overall/support lengths alone. The explicit rigid-apparatus versus linear-equivalent-lever distinction remains unresolved for finite rotations.

## Turon2010 MMB graphical extraction

Fig.12 (printedp1862/PDF6) is the adjusted-strength30/58.9MPa comparison. Numerical and analytical series share colours. A first raster-envelope attempt was rejected after visual review found isolated green analytical dots contaminating the numerical series. Its directory carries `REJECTED.md`. The v2 method retains connected coloured components and excludes the legend, then selects the lower curve cluster where the analytical hook lies above the numerical branch. The retained review overlay follows the numerical series. It remains a conservative graphical envelope estimate (±0.05mm,±3N), not original solver points; peak/jump neighbourhoods have extra ambiguity. Reference CSVs are in `reference_data/turon2010CS_mmb_v2/`.

## Stabilization defaults and actual energy

The low-parameter comparison is not a valid one-factor study of damping intensity: solver messages confirm that STABILIZE=2e-4 uses adaptive automatic stabilization, whereas2e-5 uses constant damping. The0.3mm constant case reports58.681N and ALLSD/W=16.245%; the0.15mm case57.824N and15.380%. Both fail the artificial-energy screen despite successful termination. New tests explicitly fix ALLSDTOL=.005. Actual stabilization energy is always assessed separately; the input fraction is an initial setting, not a guarantee of the energy ratio.

## Contact and loading verification

Direct hard contact produces zero pivots from redundant contact/MPC constraints before ENF/MMB loading. Explicit linear penalty contact permits elastic ENF loading. Raising contact penalty1e6 to1e7N/mm3 reduces maximum penetration from about1.57e-7 to1.57e-8mm in the small-load control. This parameter is separate from cohesive K3. The full ENF contact-corrected case reaches the expected load-drop region before a later convergence failure.

Initially opening MMB controls still fail near zero contact gap, with both surface and independent-node formulations. A linear tied fixed-crack MMB50 control without initially inactive contact completes, with zero precrack overlap, plane-strain strains near machine zero and relative lever-force-split errors below2e-8. Opening-MMB source cases therefore omit inactive precrack contact and retain explicit per-frame gap checks; ENF keeps contact. The UEL compression response ahead of and behind its evolving front is unchanged. If an opening-MMB run develops overlap, it cannot establish source-equivalent behavior.

Independent fixed-crack VCCT uses upper-arm bulk nodal forces and relative behind-tip displacements, not cohesive SDVs. Abaqus2023 supplies separate scalar NFORC1..6 fields. A postprocessor sign was corrected using the pure-I control: positive upper-tip NFORC2 and positive opening give positive GI. The original wrong-sign derived JSON is archived; this was a post-processing correction, not a UEL defect. Subsequent compliance-versus-crack-length checks provide an independent total-G comparison.

## Experimental graphical observations

The black square markers in Turon2006 Figs9-13 are now digitized from registered filled vector rectangles, with axis/path IDs and PDF hash in reference_data/turon2006_experimental_markers/digitization.json. No origin or missing observations are inserted. These are republished graphical observations; the original raw instrument records and repeat scatter remain unavailable. The source explicitly fitted eta to the toughness data, so the load comparisons are not held-out validation. Figure13 has only three experimental markers. The primary2006 table and separate2002 table remain intact.

## Controller status recovery

Two uncooperative controller stops occurred after the controller had advanced beyond the intended boundary: t2010_mmb50 and kr_mmb20_K170k were left with RUNNING manifests, although their solver STA files had already terminated unsuccessfully and no lock/solver child remained. Their original manifests are retained as run_before_controller_recovery.json; FAILED status and a recovery note were recorded from terminal solver evidence. No completed source deck was altered. Fresh opening-MMB jobs with `_open` names replace those diagnostic configurations in the primary matrix. run_queue.py now supports a `.stop` sentinel checked between jobs; do not terminate a controller merely because an earlier job manifest has ended. The actual active solver was checked through process/execute.cmd ownership before status recovery.

## Reconstructed2018 lever policy

The original ideal lever gives an initial corrected-beam GII/GT=0.18605 for nominal0.20. That completed run has peak90.443N versus the graphical97.190N and stable-branch NMAE2.20%, with0.249% work imbalance and no projected face overlap. This is not accepted as a source-equivalent result.

The source defines global mode fractions using LEFM but does not give lever/span details. A second, explicitly labelled reconstruction analytically inverts the corrected beam partition at the published initial crack. With gamma=1.18sqrt(E11*E22)/G13 and chi from2010 A6, set q=sqrt(3(1-B)/B)*(a+.42chi*h)/(a+chi*h), c=L*(1+q/2)/(3-q/2). For nominal20/50/70%, this gives c100.8067375/42.9266764/31.9981592mm and exactly the requested initial beam fractions under the adopted L51mm. No published force or displacement value enters this calculation. Original material parameters, strengths and stiffness remain unchanged. Three0.3mm cases and a0.1mm20% mesh control are prepared in separate folders and queued. The0.1mm mesh follows the source longitudinal mesh; apparatus equivalence remains reconstructed, not proved.

## Elastic area-cycle counterexample and isolated measure hypothesis

A new standalone full-UEL test (actual Fortran, serial runtime stubs, no Abaqus solver interface) opens a unit interface elastically to0.0001mm, stretches its in-plane area from1 to1.2mm2 at fixed gap, closes it, then returns the in-plane geometry to its initial state. K3=1e5N/mm3 and normal strength38MPa keep damage identically zero. The canonical UEL reports final external work -0.00010000000000000018Nmm, zero stored energy and zero dissipation. This is20% of the maximum loading work and persists for this exact elastic cycle; it is not the viscous damage quadrature issue.

Mechanism: current area weights normal force and stored energy, but the residual has no conjugate in-plane area-change force. Loading costs A0*psi, in-plane stretching costs zero UEL work, and closing returns A1*psi, giving net work -(A1-A0)*psi. All states and the driver are retained under element_tangent/area_cycle_canonical.

An isolated `3D_CZM_reference_area_experiment.f` uses the undeformed midsurface integration measure for residual, tangent and energies, with the same constitutive routines and current local basis. Its identical cycle ends with work about -1.26e-19Nmm and zero stored/dissipated energy. This tests a nominal/reference-area traction-energy interpretation. It is NOT promoted to canonical and is NOT a complete finite-deformation fix: missing basis terms remain, and a current-area model with variational area/basis forces is a different possible formulation. Researcher judgment is required before adopting a traction/energy measure. Constant-area16-patch Abaqus, matched DCB and matched2018MMB20 comparisons are queued in fresh folders. Only this restricted correction is claimed; full specimen validation of the variant is not established.

## Constant-area basis-cycle limit of the area hypothesis

Both canonical and reference-area full-UEL tests rotate the midsurface0→45degrees at fixed small global gap direction, close the interface and restore the initial geometry, with area constant. This is not a rigid-body rotation of the entire separated configuration. Both leave +1.379091724864132e-4Nmm external work and zero final stored/dissipated energy, about27.58% of peak loading work. The reference-area variant therefore fails a second independent geometric work check. The local constitutive response has different normal and shear stiffness, so rotating the basis changes reported energy even though paired nodal motion at fixed global gap produces no corresponding residual work. Missing frame/geometric work terms remain. Passing a residual/Jacobian check does not establish a variational energy-consistent residual.

These are full actual Fortran element-assembly tests with serial runtime stubs, not Abaqus solver-interface runs. Constant-area, constant-basis proportional and history tests remain separate. Neither the source measure nor a complete geometric reformulation is silently adopted; canonical source is preserved. The area-only hypothesis is retained and tested to define its restricted benefit and its failure, rather than promoted as a complete correction.


## 17. User-directed mesh and regularization follow-up

The user asked whether UEL execution stops at peak load and then emphasized mesh refinement as a convergence control. There is no load-peak termination criterion: runs target prescribed displacement, and peaks are identified afterward. Canonical PNEWDT is declared but never assigned and no STOP/XIT is present. Failed jobs retain their actual Abaqus error. A finer mesh may improve cohesive-zone resolution; it is not assumed to improve every nonlinear solution.

queue_convergence_followup.json contains five serial tasks and waits for queue_final_retry.json. It adds unstabilized DCB local spacing0.075mm and AS4 MMB80 local spacing0.1mm, retaining the corresponding baseline material, regularization and step controls. It then compares corrected-constraint DCB0.3mm at fixed rho=1e-4 with maximum normalized increments0.02 and0.001. A conditional fine-increment adaptive retry is run only if the latter fails. Step duration is1; viscosity and increment size are separate controls. Each follow-up uses a declared3000-increment resource cap, not a peak-load stop. Solver completion, postpeak coverage, energy drift, stabilization work and mesh sensitivity remain separate acceptance checks. The current supplementary MMB80 adaptive solve is still advancing toward6mm; do not abandon it based solely on reaching peak or on an earlier status snapshot.

Report generators now include the follow-up queue when deciding whether execution is complete. Figure04 includes the0.075mm DCB control and explicit adaptive mesh comparisons; figure12 includes the0.1mm MMB80 control; figure15 records the matched viscosity/increment comparison. Figures show only retained executed data. Canonical source remains unchanged.


The convergence follow-up now has six tasks: a separate t2018_mmb20_lefmlever_K1e6 diagnostic is inserted after the two mesh refinements. It preserves the completed corrected-lever20% case except K3 rises from the source-stated1e5 to1e6N/mm3 (shear penalty follows the unchanged constitutive ratio). The Turon2007 arm-thickness series screen changes from6.08% to0.643%; these are transverse series estimates, not actual MMB compliance changes. The publishedK baseline is retained, and no result is selected by target agreement. HigherK also decreases onset gap and can worsen the fixed-perturbation tangent. This closes the user-requested stiffness/compliance check without changing canonical source or source-case inputs.


### Higher-penalty2018 material check

Five proportional histories per copy use the exact2018 properties except K3=1e6, independently of the queued structural sensitivity. At B=.5 the canonical/smaller-perturbation tangent errors are 2.422397%/0.024433%. The five proportional fracture energies agree within2.5e-15 relative. All non-tangent response columns are exactly identical between these two source copies (sources/turon2018_K1e6_response_equivalence.json). The sourceK3=1e5 material test remains separate. This verifies the higher-K tangent tradeoff directly; no stiffness or perturbation change is promoted to canonical.

A conditional t2018_mmb20_lefmlever_K1e6_z8 structural retry is added only if the same higher-K canonical-perturbation run fails, because the exact higher-K material check exceeds the1% tangent screen. It holds material/penalty and all solver controls fixed, changing only the independently checked perturbation. The follow-up task list now has seven entries, of which two are conditional. No further cases are planned.
