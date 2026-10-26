# Phase 0 audit and Phase 1 preparation — 2026-10-25

A traceable basis for Checkpoint 1 is prepared: seventeen case cards, five shared source cards, seven graphical series and paired experimental-peak transcriptions. **Full figure digitization remains partial. No new constitutive/specimen verification or experimental validation is claimed. No Abaqus command or numerical model builder was executed.**

Decisions D1–D5 are preserved. Prompt0's sequence governs the next work: element verification, DCB, a DCB report and review, then authorized MMB and experimental comparisons. The older handoff's suggestion to make 2018 MMB the first structural gate is superseded by that explicit sequence. ENF stays in the selected reference set; it has not been scheduled or modeled. No Phase 2 plan or acceptance criteria have been approved.

## Repository and environment audit

The working tree was already dirty. A baseline manifest records the actual files before additions. The UEL, examples, historical decks/curves/results and decisions were preserved. HEAD alone cannot reproduce this state.

The existing workflow is:

1. Example launchers call `scripts/build_model.py`, which invokes Abaqus CAE and the specimen builder. Generation requires Abaqus even without `--submit`; it was not invoked.
2. The builder writes a native model/deck, then converts a separate deck to U8. Conversion preserves labels and coordinates; it does not establish benchmark equivalence.
3. `scripts/run_case.py` validates interface declarations, creates a new folder, copies the source and records hashes in `run.json`. It submits only with `--submit`. A later manual submission may leave `submitted:false` in a preparation manifest, so solver evidence must also be inspected.
4. Element extraction uses DAT parsing and an Abaqus-Python ODB energy checker. Specimen plotters read historical CSVs; a source-verified structural extraction pipeline is not established.

All 17 example decks (14 U8, 3 native) passed the existing runner's interface parser; all 21 existing Python files passed AST parsing. These are static checks, not solver or physics tests. The [environment audit](ENVIRONMENT.md) records visible Windows Abaqus 2023 and compiler installations, while live execution/licence readiness remains INCONCLUSIVE.

## Constitutive implementation inspection

| Feature | Observed code | Evidence boundary |
|---|---|---|
| Property interface | GIc, GIIc, normal/shear strengths, eta, K3, optional rho; JPROPS(1) selects integration (lines 116–128) | Input order confirmed; material values are not validated |
| Modal stiffness | KSH = (GIc/GIIc)(tau_sh/tau3)^2 K3 (225–231) | Algebra follows Turon 2018 Eq. 1; response still requires tests |
| Mode fraction | Instantaneous K-weighted squared separations, B = KSH delta_sh²/(KSH delta_sh² + K3 positive(delta3)²) (610–622) | Implements Eq. 17; no stored accumulated modal-work integrals |
| Equivalent jump | AUX1/sqrt(AUX2), with KB = K3(1−B)+KSH B | Tracks Eqs. 12/18 algebraically; no independent test performed |
| History | Stored D and Dv; threshold reconstructed from old D using current onset/final jumps (658–686) | Historical B is not stored. Nonproportional response remains a verification task |
| Regularization | Dv = s D + (1−s) Dv_old, s = DTIME/(rho+DTIME) | Optional delayed-damage extension; positive rho differs from the rate-independent source law |
| Compression | Normal compression retains K3; shear is degraded (690–699) | Closure, reversal and unloading need numerical assessment |
| Material tangent | Forward perturbation of 1e-6 mm, using the same old history for all three columns (531–559) | Method identified; numerical consistency not independently tested |
| Element assembly | Deformed midsurface basis; transformed Bᵀ Ct B; RHS assembled with first-face-positive/opposite-face-negative B (343–400, 497–525, 739–884) | Since jump is second minus first, RHS sign corresponds to minus internal force. Full geometric derivatives are absent; the finite-rotation element tangent remains unverified |
| Quadrature | Four Gauss points at ±1/sqrt(3), or four Newton–Cotes corners at ±1; unit weights (886–907) | Selection confirmed; integration sensitivity NOT RUN |
| Recoverable energy | psi = (1−Dv)Y plus compression energy; area-weighted ENERGY(2) | Stored energy, distinct from external work |
| Dissipation | For rho = 0, endpoint Gc(B) times an increment in r(D); for rho > 0, trapezoidal Y dDv; accumulated in ENERGY(4) (705–737) | Variable-mode and viscous paths need increment refinement. There is no independent dashpot energy contribution |
| Area | Current Jacobian weights new energy; prior dissipation is not rescaled (370–383) | Fixed-area tests cannot establish general finite-area/rotation balance |
| Output | Per-IP stride 20: D at 8, B at 10, Dv at 11, energy/dissipation at 12–15 | Text output can include iterations/retries. Quantitative work needs converged DAT/ODB histories |

Static correspondence with source equations is not numerical verification. The routine assumes a suitable nondegenerate eight-node interface and static use. No change to `src/3D_CZM.f` was made or concluded necessary from this inspection alone.

## Historical evidence

Nine historical run folders contain source/status evidence. Five element status files report completion. Only `fix_closure_energy` and `fix_regularized_energy_final` carry copied source hashes matching the current UEL. Other completed opening, mixed and regularized folders use earlier hashes. These are pre-existing runs; ODB acceptance metrics were not re-extracted here.

The DCB/MMB generation/datacheck folders carry other source hashes and provide no status-file evidence of completed loading in this inventory. Historical specimen CSV job/configuration associations remain incomplete. The [source audit](SOURCE_AUDIT.md) records the unproven reference CSV, its 0.1-mm shift, force conversion, width override and publication discrepancies.

## Reference assessment

The [index](INDEX.md) covers the complete selected family: one element card; five Turon 2010 cards; three Krueger cards; five Turon 2006 experimental cards; and three Turon 2018 cards. Missing quantities are null in the parameter records. These are reference records, not approved model configurations.

The independent Turon 2010 DCB Appendix A1–A2 calculation differs from 41 visible samples of Fig. 4's analytical line by at most **0.254155 N (0.510%)**. All vertical differences are below the declared graphical read-off estimate of 0.3 N; horizontal uncertainty is 0.01 mm. This supports the transcription at that precision, not the UEL. The source beam compliance at a0 = 35 mm is 0.0383784812 mm/N. [Metrics](cases/turon2010CS_dcb/results/reference_check_2026-10-25/metrics.json) and [figure](cases/turon2010CS_dcb/results/reference_check_2026-10-25/reference_check.svg) are retained.

ENF/MMB equation functions require explicit span and mixity policy; no case-specific calculation is claimed where those are unresolved. Krueger critical symbols remain separate from load/displacement-controlled jump paths. The missing 2018 apparatus inputs remain missing. The [digitization register](DIGITIZATION.md) identifies every retained and outstanding curve family.

## Scientific status

| Metric/family | Reference | UEL | Deviation | Tolerance | Status |
|---|---|---|---|---|---|
| Element energy, tangent, history and sensitivities | Source equations and independent checks | Unavailable in this phase | — | Not approved | NOT RUN |
| DCB initial compliance | 0.0383784812 mm/N, Turon 2010 A2 | Unavailable | — | Not approved | NOT RUN |
| DCB onset, peak, propagation, energy and mesh | LEFM and published CZM | Unavailable | — | Not approved | NOT RUN |
| MMB critical/propagation response and mixity | Published numerical/analytical references | Unavailable | — | Not approved | NOT RUN |
| AS4/PEEK experimental comparison | Separate published observation tables | Unavailable | — | Not approved | NOT RUN |
| Historical reference provenance | Publication/figure/axes/job association | Unresolved | — | Not applicable | INCONCLUSIVE |
| Live solver readiness | Licence/solver/compiler execution | Not tested | — | Not applicable | INCONCLUSIVE |

No new physical PASS/FAIL claim is made. The prompt's proposed 1%/2% tolerances remain proposals for Checkpoint 2. They were not applied retrospectively to historical data or reference transcription. **Checkpoint 1 is the next researcher action.**
