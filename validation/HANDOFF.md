# Session handoff: validating the CZE UEL against the literature

**Current entry point, updated 2026-10-26:** read [START_HERE.md](../START_HERE.md)
first. Sections **19–21** below describe the completed campaign, verified v1.4
optional perturbation input, and latest dimensionality/LEFM discussion. Earlier
sections are a chronological record: their pending checkpoints, live queues,
and unchanged-canonical-source statements describe earlier dates, not current
work. The current source is v1.4; first-campaign v1.3 is archived separately.

The original planning record began on 2026-10-25. The papers are in
`validation/references/` (local only, not in git). Current execution authority
is the root `Prompt0.md` with decisions D13–D18; do not restart completed work.

## 1. Goal

Vasco (PhD candidate) wrote `src/3D_CZM.f`, an Abaqus/Standard 8-node cohesive user element
(U8) implementing the cohesive law of **Turon et al. 2018** (mode-dependent penalty
stiffness, B-K propagation, numerical tangent by perturbation). He never managed to match
published results with it. The goal is to **verify the UEL against well-defined literature
benchmarks**, then use those results in a thesis chapter ("A cohesive user element for
delamination").

The chapter copies the *structure and figure types* of Germain (2020), PhD thesis, Ch. 3-4
(cohesive law, implementation, single-element tests, DCB, MMB, interface fields, solver
behaviour, integration-point paths, viscous regularisation). It does **not** reproduce
Germain's results: his T700/M21 data are dropped.

## 2. Method: Castro's research workflow (mandatory)

Vasco has a skill for this (`research-ai-workflow`, after S. G. P. Castro, TU Delft, 2026).
Rules that matter here:

- **Never guess a source.** Every value carries paper key + page/table/figure.
- **Report faithfully.** Never tune parameters to match a curve; keep failed runs with a note.
- **Ask, don't assume.** Ambiguous definitions, unstated BCs and choices go to Vasco as
  numbered questions; his answers are dated in `DECISIONS.md`.
- **Two checkpoints.** Checkpoint 1: Vasco checks the reference cards and answers the
  questions. Checkpoint 2: Vasco approves the plan (models, sensitivities, acceptance
  criteria). Nothing is run before Checkpoint 2.
- **One key per case** (folder, data card, BibTeX). Units N, mm, MPa (N/mm for G, N/mm3 for K).
- **Figures as code.** Data plots in matplotlib from result files; schematics in TikZ.
- **Stop and ask** before changing `src/3D_CZM.f`.

## 3. Agreed benchmark set (Vasco agreed in chat on 2026-10-25)

| Order | Case key | What | Reference curve | Role |
| --- | --- | --- | --- | --- |
| 0 | `element` | Single element: dissipated energy = B-K G_c(B) at all mixities, unloading, mode changing | analytical | verification |
| 1 | `turon2010CS_dcb`, `_enf`, `_mmb` | Turon 2010 set: DCB, ENF, MMB 20/50/80 %, strength sweeps | closed-form LEFM (Turon 2010 App. A) + digitised Turon figures | verification, main text |
| 2 | `krueger2012CR_mmb` | Krueger MMB IM7/8552, G_II/G_T = 0.2/0.5/0.8, 2D and 3D | Krueger VCCT benchmark + LEFM | verification, main text |
| 3 | `turon2006MM_exp` | AS4/PEEK DCB, MMB 20/50/80 %, ENF vs Reeder & Crews experiments | experimental peak loads (Turon 2006 Table 3 / Camanho 2002 Table 4) | validation, main text |
| 4 | `turon2018CS_mmb` | Turon 2018 AS4/PEEK MMB 20/50/70 %, strength sweep | Turon 2018 Figs. 2-4 + LEFM | internal check ("for us"); appendix at most |

Thesis figures: one per case, showing the UEL, closed-form LEFM and the published curve on the
same axes. Turon 2018 is the green light for the rest because it is the source paper.

Dropped: Germain T700/M21; Camanho T300/977-2 DCB; Krueger 2008 DCB (see section 6).

## 4. What each paper defines (summary; full values in `references/NOTES.md`)

- **Turon 2018** (CS 184:506-511), MMB: AS4/PEEK, 2L 102, b 25.4, arms 1.56, a0 32.64 mm;
  2D plane strain CPE4I, 5 el./arm, 0.1 mm, 4-node cohesive; K3 1e5; perturbation 1e-6;
  G_Ic 0.969, G_IIc 1.719, eta 2; tau3 5-80, tau_sh 80-120 MPa; 20/50/70 %. **Not stated:**
  lever length c, how the mixity is imposed (lever or load split), NLGEOM, solver.
- **Turon 2010** (CS 92:1857-1864): 150 x 20, arms 1.55, a0 35; CPE4 (text says plane stress:
  inconsistent), 0.15 mm, 10 el. through thickness; E11 120, E22 10.5, G12 5.25, G23 3.48 GPa;
  tau3 15, tau_sh 30/45/60; G_Ic 0.260, G_IIc 1.002, eta 2.0; c = 63.18 mm at 50 %; DCB, ENF,
  MMB 20/50/80 %; LEFM closed forms in Appendix A. Penalty not yet checked.
- **Turon 2006** (MM 38:1072-1089) and **Camanho & Dávila 2002** (NASA/TM-2002-211737):
  AS4/PEEK Reeder & Crews tests; 102 x 25.4, arms 1.56; 3D decohesion elements; K 1e6;
  Table 2: G_c and a0 per mixity; eta 2.284; lever not modelled, end and mid-span loads from
  Camanho App. B; predicted vs experimental peak loads.
- **Krueger NASA/CR-2012-217562**: IM7/8552 [0]24; b 25.4, 2h 4.5, 2L 100.8, 2L* 152.4,
  a0 25.4; c = 92.9 / 41.3 / 27.3 mm (0.2/0.5/0.8); 2D CPE4I (6 through 2h, 0.5 mm) with
  rigid R2D2 lever and MPCs (tip: same u, w, free rotation; mid-span: same w, free u;
  lower supports w = 0); 3D C3D8I, 25 across width; w_crit 1.64/1.34/1.65 mm; P_crit 385 N
  (0.5), 751 N (0.8).

## 5. Key finding of this session

The repo's MMB examples (`examples/mmb/`) are IM7/8552 in **Krueger's geometry exactly**
(L 50.4, c 41.3 or 92.9, a0 25.4, 2h 4.5, b 25.4), so they are valid Krueger models. But
they were compared with `examples/mmb/data/benchmark_turon.csv`, labelled "Turon". No Turon
paper uses IM7/8552 for the MMB, so that curve cannot be a Turon 2018 result. This is the
most likely reason the UEL "never matched Turon". Trace that CSV's source before using it.

## 6. Known inconsistencies to put in front of Vasco (Checkpoint 1)

1. `benchmark_turon.csv` provenance (above).
2. MMB force conversion in plotters: `2 * 12.7 * 50.4 / (41.3 + 50.4)` uses c = 41.3 while
   `mmb_config.json` has C = 92.9. Which load quantity is plotted (lever load P?)
3. `MMB_Thesis_*Visc1E6*` decks actually use rho = 1e-5; all historical MMB decks use
   rho 1e-3 to 1e-5 (viscosity raises the propagation branch). Benchmarks should use rho = 0.
4. Historical curves were not regenerated with the current source v1.3.
5. Krueger 2008 DCB: elastic data T300/1076, but the figure and fracture data say T300/914C.
6. Repo T300_9772 DCB matches the Camanho 2002 geometry (b written as 2.0 vs 20 mm) but its
   cohesive set (G_Ic 0.352, 30 MPa) does not match Camanho Table 1 (0.268, 45 MPa).
7. Turon 2010 says "plane stress elements (CPE4)": CPE4 is plane strain.
8. Turon 2018 leaves c, the load introduction and NLGEOM unstated. Proposed: take them from
   Turon 2006 / Camanho 2002 (same specimen and material) and compute c or the load split
   from Camanho App. B / ASTM D6671. Vasco's decision.
9. The repo models are pseudo-2D 3D (U8 + C3D8), while Turon 2010/2018 and Krueger's
   benchmark use 2D plane strain. A 2D cohesive element does not exist in the repo; decide
   whether a one-element-wide 3D slice with plane-strain constraints is acceptable.

## 7. Where things are

- UEL: `src/3D_CZM.f`; props: K3, GIc, GIIc, tau3, tau_sh, eta, rho (+ JPROPS(1): 1 Gauss,
  2 Newton-Cotes). Interface docs: `docs/uel-interface.md`; validation status:
  `docs/validation.md`.
- Generators: `examples/dcb/main_script.py`, `examples/mmb/main_script.py` (Abaqus CAE),
  configs `*_config.json`; runner `scripts/run_case.py`; converter `scripts/convert_input.py`.
- Single-element decks: `examples/element/` (energy check included).
- Abaqus runs only on Vasco's machine (Abaqus 2023 + Fortran). A cloud agent can write
  inputs, scripts and plots, but cannot solve.
- Thesis chapter draft: Vasco's thesis repo, `chapters/cohesive-uel.tex`; style
  `plotting/thesis_style.py` (matplotlib), `style/vp-colors` (TikZ); specimen schematics
  `figures/czm/specimen_schematics.tex`. Teal = identity, crimson = damage only.
- Plan document (Claude Doc): https://claude.ai/artifact/2PvbqEb7D3D8ShsSeFXS6Q

## 8. Next step

Phase 1: reference cards. Follow `PROMPT_phase1_reference_cards.md`. End at Checkpoint 1
with the questions in section 6 (plus anything new) answered by Vasco.

## 9. Audit update — 2026-10-25, Prompt0 execution

The Phase 0 audit and Phase 1 reference-card preparation now have an
[index](INDEX.md), [environment record](ENVIRONMENT.md), [report](REPORT.md),
[source audit](SOURCE_AUDIT.md), and [Checkpoint 1](CHECKPOINT1.md).
Seventeen case cards and five shared source cards were prepared. Seven graphical
series were extracted with source hashes, axes and uncertainty; the full requested
digitization remains partial, as recorded in [DIGITIZATION.md](DIGITIZATION.md).
No Abaqus command, model builder or compiler was executed. The UEL, examples,
historical results and DECISIONS.md were preserved.

Corrections to the earlier sections above, retained here without rewriting history:

- The actual real-property order is GIc, GIIc, tau3, tau_sh, eta, K3, rho, followed
  by the integer integration property. The older symbol listing is not input order.
- The MMB examples share Krueger properties and several dimensions but are not
  established reproductions: overall length, actual pseudo-2D width, element
  formulation and apparatus constraints differ. Both generators force width to
  2 mm when pseudo-2D mode is enabled.
- All 14 non-origin displacements in benchmark_turon.csv are 0.1 mm below its
  backup; the reason is unknown. Publication/figure and raw force definitions
  remain unresolved. Keep both files as historical data.
- Two pre-existing completed element folders have the current source hash:
  fix_closure_energy and fix_regularized_energy_final. Other copied sources differ.
  This does not establish a completed current-source DCB/MMB comparison.
- The default DCB UEL strength is 60 MPa; 30 MPa belongs to the native block.
- Krueger's 20% critical load is explicitly annotated as 128.5 N in Fig. 10.
- Turon 2006 and Camanho 2002 have different experimental peak tables and ENF
  initial cracks. Their observations must remain separate.
- Turon 2010 Figs. 6–7 are earlier-law counterexamples; adjusted-strength
  Figs. 11–12 have a different role. The definition of span L is also ambiguous.
- Windows Abaqus 2023 and compiler installations are visible from this WSL
  environment. Live execution/licence readiness was not tested.
- Prompt0's current sequence requires element verification and DCB before
  authorized MMB work; the earlier suggestion to make 2018 MMB the first
  structural gate is superseded.

Next action: researcher review of Q1.1–Q1.9 in CHECKPOINT1.md. No new scientific
decision has been recorded, and Phase 2 has not begun. Full constitutive and
specimen verification, sensitivity studies and experimental comparisons remain
NOT RUN in this session. The DCB analytical/graphical agreement check assesses
reference transcription only.

## 10. Checkpoint responses — 2026-10-25

The user approved Q1.1, Q1.2, Q1.3, Q1.7 and Q1.9; authorized a 3D slice if
necessary and delegated its dimensional constraint choice; and allowed labelled
project defaults when cohesive strengths/penalty are missing. These responses
are recorded as D6–D12 in DECISIONS.md. Ordinary plane strain is selected for
the plane-strain benchmark reproduction, subject to later equivalence checks.
Generalized plane strain is not adopted for those reference cases merely
because the UEL requires 3D geometry.

The user requested explanations of Q1.6 (the more recent experimental table)
and Q1.8 (excluding the historical curve) before continuing. These remain
pending; do not treat the explanation request as approval. No Phase 2 plan,
model or Abaqus execution was started. The detailed nominal-versus-evolving
mixity policy for A5 toughness must also stay explicit in the next plan.

## 11. Autonomous campaign started — 2026-10-25

Q1.6 and Q1.8 are resolved. D13–D17 record the latest user decisions and updated root protocol. The old checkpoint stops are superseded: continue execution, preserve failed runs, and escalate only consequential unresolved scientific choices. The canonical UEL remains unchanged.

Execution scripts are in validation/scripts/campaign.py, structured_specimen.py and extract_odb.py. All new solve folders are under validation/campaign_20261025/runs/. run.json records the exact deck/source hashes, controls, command and termination state. PLAN.md is the working execution plan rather than a stop gate. Do not mistake the older phase audit's NOT RUN statements for the current campaign status; append current results to REPORT.md.

## 12. Corrected constrained-slice campaign

Current authority is the autonomous Prompt0.md, not historical phase stops. Canonical source remains unchanged. Completed material/unit/DCB diagnostics and isolated perturbation patches are retained under campaign_20261025. The first generator allowed small U3 motions at end-coupling nodes because Abaqus ignored their separate U3 boundary conditions. Revision2 couples all three translations at those slice ends and constrains other U3 nodes separately. The C3D8I elastic pilot and completed dcb_ps03 directly verify out-of-plane strains near machine zero.

The original queue_mmb_experimental controller was stopped after as4_dcb failed, before any MMB submission, so remaining jobs use corrected constraints. Live tasks are in queue_corrected.json/status.json/log, with one Abaqus solver at a time. Continue until all tasks are assessed; do not resubmit completed/running folders. Supplemental diagnostics go in fresh names. The viscous DCB runs fail global energy balance despite solver completion; material refinement identifies a positive endpoint-quadrature excess. Details are in campaign_20261025/INVESTIGATION.md.

## 13. Contact troubleshooting and independent MMB statics

The prior queue_corrected and queue_contact_corrected controllers were stopped at solver boundaries. Their stopped.json files identify superseding work; incomplete queue status arrays do not imply live solvers. Hard contact conflicted with paired-width constraints. Explicit penalty enforcement fixes ENF initial loading: contact_enf_elastic and K1e7 controls complete with penetration about1.57e-7/1.57e-8mm. The full t2010_enf_pc advances to the load-drop region before failing. Both surface-to-surface and independent-node contact versions of an initially opening MMB control fail near zero contact gap; keep their diagnostics.

fixed_turon2010_mmb50_open (linear tied fixed-crack control) completes without initially inactive precrack contact. Its saved frames show zero precrack face penetration, exact tip ties, plane-strain strains near machine zero, and lever force split error below2e-8. It independently returns GI/GII from bulk nodal forces. Abaqus2023 exports NFORC as separate scalar NFORC1..6 fields; the extractor was corrected for this, and its pure-I sign convention calibrated against fixed_dcb_a35p0. Earlier incorrect derived files/logs remain archived.

The active source-case controller is queue_open_mmb.json/status.json/log (16 tasks; started with t2010_mmb20_open). It omits initially inactive contact for MMB, retaining the UEL compression law and checking precrack gap every frame. ENF retains contact. Do not claim source equivalence if penetration is detected. queue_diagnostics_final.json contains later explicit-adaptive DCB checks, element perturbation comparison, fixed-crack controls and conditional stabilization retries; it has not yet been launched at this record update. Launch only after validating the first opening-MMB trace or wait for queue_open_mmb.json with run_queue.py --wait-for.

Important stabilization finding: naked STABILIZE=2e-4 activated adaptive damping, while2e-5 used constant damping. The latter completed but actual stabilization energy was15-16% of external work and its peak rose materially; reject it as a converged prediction. New controls explicitly use ALLSDTOL=.005 at each damping level. Canonical source remains unchanged. No final independent experimental validation is established.

## 14. Live queues and stale-status correction

Current main solver controller: queue_open_mmb.json/status.json/log. Current waiting supplementary controller: queue_diagnostics_final.json, logging to queue_diagnostics_final_v2.log, awaiting the16 main tasks. Supplementary tasks now total24. Two earlier controllers had advanced beyond the intended stop boundary, leaving t2010_mmb50 and kr_mmb20_K170k manifests stale RUNNING. Their STA files show terminal failure, no lock/process remained, and status recovery is explicitly recorded with original manifest backups. Fresh primary jobs t2010_mmb50_open, t2010_mmb80_open and kr_mmb20_K170k_open are at the head of the supplementary queue. All other fresh source cases use their original names with precrack_contact_active:false for MMB. Do not reuse old contact-failed folders as source-equivalent baselines.

The first opening-MMB20 trace advances through loading and fails later near8mm rather than at the initial contact solve. The successful fixed_turon2010_mmb50_open independently verifies GI/GII, zero overlap and lever statics. The extractor now handles scalar NFORC components and correct pure-I closure-work sign. assessment_diagnostics.py reproduces onset/viscous identities and fixed-crack checks. report_campaign.py builds inventories/matrix/figures; write_technical_report.py writes the full dynamic report. Refresh reduce_structure.py, assessment_diagnostics.py, report_campaign.py --figures, then write_technical_report.py after queues finish. Check all physics/energy/contact limits before final conclusions.

## 15. Additional source-policy and area-energy findings

queue_diagnostics_final.json now contains32 tasks (task list is read only after the main queue completes). Four corrected-LEFM-lever2018 models and three reference-area experimental models were pre-prepared so the waiting controller can replay exact decks despite its earlier imported builder. Do not regenerate or overwrite those folders. The corrected-lever policy analytically enforces initial global beam mode fraction, without force/displacement fitting; exact material parameters remain unchanged. Original ideal-lever cases are retained. One0.1mm20% case checks the source mesh.

Actual full-UEL standalone area cycle demonstrates a rho=0 elastic energy defect: canonical final work -1e-4Nmm, zero energy/damage and original geometry restored. An isolated nominal/reference-area variant gives about -1.26e-19Nmm in the same cycle. This changes the traction/energy measure and is not a canonical correction or general finite-deformation validation. The source variant, patch, manifest and area-cycle drivers/data are retained. Constant-area unit batch plus matched DCB and mixed-mode tests are queued. Canonical source remains unchanged. Include this counterexample prominently in the final report and element verdict; constant-area energy agreement cannot support general energy consistency.

## 16. Final queued work and completed geometric counterexamples

Main queue_open_mmb has ended. The active supplementary controller reads queue_diagnostics_final.json (32 tasks), logging to queue_diagnostics_final_v2.log. A final one-task queue_final_retry.json waits for that queue and retries corrected-lever2018MMB50 with explicit adaptive stabilization if its baseline failed. Do not finalize until both queues end and every required assessment is refreshed.

Two additional canonical/reference-area standalone basis cycles are complete: rotate the midsurface0→45degrees at fixed small global gap, close, restore orientation, with area constant. Both leave +1.379091724864132e-4Nmm work, zero final stored/dissipated energy; canonical and reference-area remain inconsistent in this test. This is not rigid-body rotation of the entire opened patch. The reference-area hypothesis therefore only cures the area-cycle counterexample and must not be called a complete geometric fix. Data are in element_tangent/basis_cycle_canonical and basis_cycle_reference; area cycles remain in their three separate folders. The1000-interval canonical area cycle reproduces -1e-4Nmm exactly, so its error is not an increment quadrature artifact.

Actual structural property-set material checks are complete in material/reference_properties_canonical and reference_properties_z8, each20 proportional histories. At B=.5 canonical/smaller-perturbation tangent errors in percent are: Turon2010 1.6017/0.0162; Krueger0.2657/0.0027; AS4 0.6576/0.0066; Turon2018 0.2453/0.0025. Proportional fracture energies agree to about1.2e-14 relative. Krueger strengths/K remain assumed inputs, not published values. The BK-fit versus measured-toughness screen is retained separately; nominal20% gives0.987994 versus1.103N/mm, explaining a5.36% same-geometry LEFM load-scaling difference without changing any curve.

report_campaign.py now has14 figures, a17-case matrix plus diagnostic subcheck matrix, and mechanically screened adaptive-control selection (completion/postpeak, work<=1%, stabilization<=1%, applicable gap check PASS; no agreement-based selection). Primary2018 cases are corrected-lever variants; original ideal variants remain controls. Preserve the original unit/canonical code. Refresh reports only after all queued runs and reductions; inspect final plots and hashes. Submission interval audit currently found no overlapping Abaqus execution windows, including the recovered stale manifests.


## 17. User-directed mesh and regularization follow-up

The user asked whether UEL execution stops at peak load and then emphasized mesh refinement as a convergence control. There is no load-peak termination criterion: runs target prescribed displacement, and peaks are identified afterward. Canonical PNEWDT is declared but never assigned and no STOP/XIT is present. Failed jobs retain their actual Abaqus error. A finer mesh may improve cohesive-zone resolution; it is not assumed to improve every nonlinear solution.

queue_convergence_followup.json contains seven serial tasks and waits for queue_final_retry.json. It adds unstabilized DCB local spacing0.075mm and AS4 MMB80 local spacing0.1mm, retaining the corresponding baseline material, regularization and step controls. It then compares corrected-constraint DCB0.3mm at fixed rho=1e-4 with maximum normalized increments0.02 and0.001. A conditional fine-increment adaptive retry is run only if the latter fails. Step duration is1; viscosity and increment size are separate controls. Each follow-up uses a declared3000-increment resource cap, not a peak-load stop. Solver completion, postpeak coverage, energy drift, stabilization work and mesh sensitivity remain separate acceptance checks. The current supplementary MMB80 adaptive solve is still advancing toward6mm; do not abandon it based solely on reaching peak or on an earlier status snapshot.

Report generators now include the follow-up queue when deciding whether execution is complete. Figure04 includes the0.075mm DCB control and explicit adaptive mesh comparisons; figure12 includes the0.1mm MMB80 control; figure15 records the matched viscosity/increment comparison. Figures show only retained executed data. Canonical source remains unchanged.


The convergence follow-up now has six tasks: a separate t2018_mmb20_lefmlever_K1e6 diagnostic is inserted after the two mesh refinements. It preserves the completed corrected-lever20% case except K3 rises from the source-stated1e5 to1e6N/mm3 (shear penalty follows the unchanged constitutive ratio). The Turon2007 arm-thickness series screen changes from6.08% to0.643%; these are transverse series estimates, not actual MMB compliance changes. The publishedK baseline is retained, and no result is selected by target agreement. HigherK also decreases onset gap and can worsen the fixed-perturbation tangent. This closes the user-requested stiffness/compliance check without changing canonical source or source-case inputs.

The waiting convergence-follow-up controller disappeared before submitting any job (empty log, no status file, all five unconditional decks still PREPARED). It was relaunched detached with setsid -f and the same --wait-for final-retry dependency. No solve folder was resubmitted or overwritten; verify this controller remains live until its queue starts.

Actual higher-K2018 material tests find2.4224% canonical tangent error at B=.5 versus0.02443% with the isolated smaller perturbation, with identical non-tangent responses. The final follow-up queue has seven tasks: it now includes a conditional K1e6_z8 retry only if the otherwise identical K1e6 structural run fails. This separates the verified tangent defect from the physical penalty change. No further cases are planned beyond these direct mesh/increment/penalty controls.

Current live state after supplementary completion: queue_diagnostics_final has ended32/32 tasks. queue_final_retry is executing t2018_mmb50_lefmlever_adaptive; the detached follow-up controller PID20360 waits for its one-task completion before loading the seven follow-up tasks. The MMB80 adaptive solve is complete (6mm, energy drift1.103%); its earlier live description above is historical.


## 18. Live state after mesh and tangent follow-up

All32 supplementary tasks and the final2018 MMB50 adaptive retry have ended. That retry completes6mm, peak273.228N, maximum energy drift0.09668%, stabilization0.04215%; source graphical peak difference+0.9017%. The detached follow-up controller PID20360 executes queue_convergence_followup.json with seven tasks. Five have ended; the sixth, dcb_ps03_rho1e4_dt001, is running at roughly half of its unit-duration step (nominal1000 increments, cap3000), target4mm, rho1e-4, maximum increment0.001. The seventh is its conditional adaptive retry and must be skipped if this run completes. Do not finalize while either remains pending.

New retained outcomes: dcb_ps0075 completes4mm with zero rho/stabilization, peak51.2219095N, -0.39974% versus0.3mm, maximum energy drift0.018419%. Peak displacement2.56 versus2.64mm is only one0.08mm maximum load increment, so no finely resolved peak-position convergence is claimed. as4_mmb80_dx01 fails at4.996318mm/6mm, maximum accepted471.06553N, not a verified peak. t2018_mmb20_lefmlever_K1e6 fails at9.662719mm/11mm; its same-K z8 retry completes11mm, peak99.4872959N, maximum energy drift0.189396%, zero rho/stabilization. Coarse corrected-constraint dcb_ps03_rho1e4_dt02 completes4mm but has about112.78% energy drift. The fine-increment viscous result is still pending.

The report script now exports16 scientific figure sets, including15 (matched viscosity/increment) and16 (MMB penalty/tangent), plus a reproducible overview collage. Most recent rendered set has15 until the next --figures run. Root README.md and docs/validation.md now link to current evidence and disclose the geometric energy defects; their preceding versions are backed up in audit/phase0_2026-10-25. All17 case cards link to generated execution_result.json. Early-load fitted compliances are labelled as5–20% of achieved maximum, not certified undamaged values. Original source/examples remain protected.

Two original main-queue ownership errors are historical stale-manifest guards, resolved by documented terminal-STA recovery, preserved original manifests, extraction and fresh *_open replacements. execution_audit records this resolution separately from still-failed solver results, and the technical-report completion guard checks unresolved controller errors as well as pending tasks. No overlapping ended submission intervals have been found.

After the last queue ends: reduce_structure.py; assessment_diagnostics.py; report_campaign.py --figures; write_technical_report.py. Inspect figures04/08/12/14/15/16, PDF dimensions/text, current Markdown links, all106-or107 run deck/source hashes and62 protected files, ended queue entries and absence of live solvers. Update this handoff and PLAN with final counts/outcomes. The source-area hypothesis still fails the basis cycle; canonical source must remain unchanged. No further numerical cases are planned.


## 19. Campaign execution and assessment complete — 2026-10-26

All required final queues have ended: queue_open_mmb16/16, queue_diagnostics_final32/32, queue_final_retry1/1 and queue_convergence_followup7/7. The final conditional adaptive viscous retry is SKIPPED because its fine-increment baseline completed. No solver or controller remains live. There are106 actual Abaqus jobs:50 COMPLETED and56 FAILED; all have their applicable retained assessments. The17-case matrix has1 FAIL (general element-energy verification) and16 INCONCLUSIVE structural/experimental cases. Individual subchecks remain separate from these final scientific verdicts.

The fixed-rho=1e-4 corrected-constraint DCB comparison is complete. Maximum increment0.02 gives peak52.9466915N and112.7768279% maximum energy drift;0.001 gives peak51.8272018N and0.4145592% drift. Both reach4mm with zero step stabilization. The peak changes about2.114%, so temporal convergence is not established despite the refined run passing the1% energy screen. The refined result is not a zero-viscosity limit or a new fitted material law. The successful0.075mm zero-viscosity DCB and same-K smaller-perturbation MMB results are recorded in section18 and the report.

The final technical report is validation/REPORT.md. Results include the17-case matrix, diagnostic subchecks, numerical-control comparisons, solver terminations, controller-recovery/serial-execution audit, reference-area comparisons and16 SVG/PDF/PNG scientific figure sets plus overview.png. The16th figure shows the same-K numerical-tangent experiment; figure15 shows the matched viscosity/increment study; figure11 now compares completed unstabilized0.3/0.075mm crack-front definitions. Full regeneration is documented in campaign_20261025/README.md. A final display adjustment confines figure15's energy axis to its relevant range; no data are changed.

Preservation checks confirm62 original source/example files and every106 retained deck/source pair, with no mismatch. Canonical SHA256 remains bfa62fce8fef54fc8f3ba6f04946b3e0c5b68e4dbed5fb60987c8b27cc8d455c. No ended execution intervals overlap and no controller errors remain unresolved. Root README/docs and reference-card execution pointers link to the current findings; original public-scope text is backed up in the phase0 audit. Early-load compliance fits are not certified undamaged values.

This completes the recorded computational investigation, including failed/inconclusive outcomes. It does not approve a canonical UEL replacement or establish full specimen validation. Remaining scientific choices concern the intended traction/energy measure and a general geometrically consistent residual/tangent/energy formulation; missing source apparatus and independent raw experimental observations remain limitations. Do not hide those defects with damping or fit parameters to targets. No further numerical cases are pending or planned in this campaign.


## 20. User-requested optional tangent input — 2026-10-26

D18 authorizes exposing the perturbation parameter. src/3D_CZM.f is now v1.4: optional real PROPS(8)=zeta, finite positive length units, defaults to1e-6 for old seven-property decks. All real properties precede the integer integration scheme; eight-real decks put that integer on the next data line. Converter/config support cohesive_UEL.tangent_perturbation; run_case accepts seven/eight real properties and multiline values. Both validation builders accept --tangent-perturbation. The material/assembly diagnostic drivers now pass the parameter rather than editing the Fortran constant.

The v1.3 source is archived at campaign_20261025/sources/3D_CZM_v1_3_baseline.f, preserving its original bfa62f... SHA. Existing106 first-campaign run copies/decks/results are unchanged; the report now carries a source-version annotation. New checks live separately under tangent_input_20261026. All passed: five compiled nine-state full-UEL configurations (exact residual/Jacobian/200SVAR/eight-energy equivalence for the relevant three pairs); two material response files byte-identical to their9600-sample v1.3 references; three actual Abaqus16-patch batches (legacy default, explicit1e-6, explicit1e-8), byte-identical response/SDV files; explicit Fortran rejection of zero zeta; eighteen example decks, six converter layouts and twelve invalid workflow cases. See tangent_input_20261026/verification.json and README.md for exact scope and source hash. This feature is backward compatible in the tested cases and does not repair the geometric work defects or establish full specimen validation.


## 21. Dimensionality and figure03 explanation

The user asked whether our/reference models are2D or3D and why figure03 has a different LEFM stiffness/peak. The first campaign contains100 intended ordinary-plane-strain constrained3D structural slices and one finite-width3D Krueger50 pilot (width25.4mm, nz3, failed). No actual CPE4/CPE4I/CPS4 solver model was run. Main slices are1mm wide with tied in-plane front/back motion and U3zero; linear strains were checked, but exact CPE4I equivalence is not established, especially for nonlinear incompatible modes. The2006 published AS4 numerical models use3D decohesion and width discretization, so our slices are not proven full-width reproductions. Krueger selected critical curves specifically originate from2D fixed-crack models in §§4.1–4.3; separate3D models appear elsewhere. Turon2010 has the known plane-stress/CPE4 contradiction;2018 clearly uses2D CPE4I.

Figure03 brown curve uses elementary beam A1–A2: C(a0)=.03837848mm/N, stiffness26.056N/mm, ideal sharp-crack initiation56.227N at2.1579mm. The straight elastic branch is our C(a0) extension; its declining branch is the printed crack-growth relationship. Grey is the independent published numerical curve (approximately51N), not that analytical initiation result. Existing fixed-crack C3D8I/VCCT control has stiffness21.293N/mm and initiation52.633N; UEL0.3 early-load stiffness21.241N/mm and maximum51.427N at2.64mm. The bulk elastic control already shows the compliance gap, so it cannot be assigned solely to cohesive softening. Root/shear/state effects remain unquantified separately.

New explanation note: DIMENSIONALITY_AND_DCB.md. Reproducible annotated figure and comparison.json are in campaign_20261025/results/explanation_03/, generated by scripts/explain_dcb_reference.py. Original16 figures and all solver/reference data are unchanged. The initial explanatory rendering is retained in explanation_03_initial. A matched true2D plane-stress/plane-strain versus constrained3D elastic study is proposed but has NOT RUN. The perturbation-input feature is complete and verified; no solver/controller remains active.

## 22. Matched elastic study and mesh-source audit — 2026-10-26

The user commissioned the proposed comparison and asked whether 3D source
papers specify width resolution. New isolated work is in
`elastic_dimensions_20261026/`; read its README, REPORT and MESH_SOURCE_AUDIT.
The completed first campaign, original report and canonical UEL are preserved.
The last preservation check covers 2,957 files with no detected changes
(input/source/assessment hashes; other artifact sizes/modification times).

The new matrix has 48 prepared elastic DCB jobs: CPS4/CPS4I/CPE4/CPE4I and
constrained C3D8/C3D8I at dx0.5/0.25 with 5/10 per arm, each at crack lengths
34.5/35/35.5 mm. All crack variants share the same in-plane grid. Twelve
additional jobs within that total use finite-width C3D8I with 3/7/15/25 width
elements and the coarser in-plane mesh. No active cohesive law or fitting.
These full-width DCB controls do not reproduce Krueger's MMB apparatus.

One actual native CPS4 job, `CPS4_dx0p5_a34p5`, COMPLETED: C0.04543694273 mm/N,
GI0.001780586903 N/mm at opening0.2mm. Other47 remain PREPARED. This is the
first native 2D solver result in this workspace; statements in sections19–21
about no native2D run describe the first campaign/historical state. It is not
yet a matched dimensionality or width-convergence result.

Initial extraction failed on Python2 unicode and JSON single-precision values;
those attempts/partial files remain. Corrected extraction succeeds. New
extractor reproduces original force/C/GI from a copied fixed_dcb_a35p0 ODB
exactly at retained precision. Prepared C3D8I baseline connectivity/equations
match its archived counterpart; coordinates agree within5.55e-17mm. Script
amendments are recorded with hashes and retained plan copies.

External Windows Abaqus transverse-cracking queue was active (t0229 through
t0240 observed under E:\Transverse_Cracking_0p5mm\study_2410\L4_L70_noscale_s201).
The first preflight found a gap and submitted one small job; the external
controller resumed afterward. A persistent `external_queue_hold.json` now
prevents submission during gaps. Do not infer whole-queue completion from one
empty process snapshot. There is no live controller for this new study.
The last attempted invocation returns DEFERRED_EXTERNAL_SOLVER; next job is
CPS4_dx0p5_a35p0. Resume when the external campaign has ended, or obtain an
explicit concurrency-policy change from the user. The generic user
"continue" and disabled filesystem sandbox do not themselves change the
one-solver instruction. An asynchronous question offered waiting versus
allowing these elastic jobs alongside the other queue; no explicit choice was
received at this handoff.

Mesh audit directly confirms Krueger §3.1 p4/PDF8:25 across width, six total
thickness,0.5mm crack-tip spacing. Turon2006 p1082/PDF11 and Camanho2002
p22/PDF26 state four interface elements across width for ENF/MMB,150 along
length, and102 along length for DCB without separately specifying its width
count. Our failed Krueger pilot used three across25.4mm, three per arm and
requested1mm longitudinal spacing. Its coarse width is a credible sensitivity
concern, not a demonstrated cause of failure. One-element constrained slices
impose plane strain and need a different adequacy test from full-width fronts.
