# Validation and data provenance

## Current scope

The public repository contains a research UEL and reusable workflows. Organizing the code for release does not establish specimen-level accuracy for a particular application.

The [current campaign report](../validation/REPORT.md) and [17-case matrix](../validation/campaign_20261025/results/verification_validation_matrix.csv) record actual Abaqus executions, independent compiled Fortran checks, source provenance and numerical sensitivities. Restricted unit checks pass, but canonical elastic changing-area/changing-basis tests fail energy balance; full specimen validation remains inconclusive. The first campaign’s source copies and historical examples are preserved. Current campaign models use the separate documented builder under `validation/scripts/`, with exact per-run decks and hashes.

That first campaign used v1.3 and its retained experimental copies. The subsequent
user-requested v1.4 extension makes the numerical tangent perturbation an optional
input property, preserving the old default for seven-property decks. Its
[separate checks](../validation/tangent_input_20261026/README.md) distinguish
interface compatibility from the first campaign's scientific findings. The
extension does not repair the geometric energy defects described below.

| Component | Evidence and limits |
| --- | --- |
| Constitutive routines | Prior standalone compiled checks covered the material response, numerical tangent, fracture energy, unloading, compression, delayed damage, and changing-mode energy refinement. |
| Abaqus element response | Prior Abaqus 2023 checks covered proportional opening/mixed-mode energy, unloading/closure, and a regularized history. Reproduce the [energy example](../examples/element/energy_check/README.md) for your environment. |
| DCB/MMB generators | Previously checked with Abaqus 2023 through generation, UEL conversion, compilation/linking, and datacheck. Datacheck does not establish completed loading or agreement with reference curves. |
| DCB/MMB comparison curves | Historical development outputs. Exact source/configuration associations are incomplete, and curves have not been regenerated with the current source. |
| New campaign results | Actual completed and failed unit/DCB/ENF/MMB jobs, fixed-crack elastic/VCCT controls, mesh/increment/penalty/stabilization studies and provisional published experimental comparisons. Refer to the current report for execution status and evidence limits. |
| Public automation | Python syntax and isolated example preparation only. No Abaqus license or solver run is provided by CI. |

The development regression suite is not included in the public tree. Single-element decks, the ODB energy checker, and the work diagnostic remain available as runnable examples. Before changing the material law, reproduce proportional and changing-mode response, unloading/closure, the numerical tangent, and energy behavior with a configured compiler and Abaqus installation.

## Numerical limits

The material routine uses a numerical displacement perturbation of `1E-6` (mm in the supplied examples). The element tangent retains the geometric-term approximation described in the source. Select a consistent unit system and assess sensitivity when changing its scale.

The generator defaults and basic element inputs use zero viscosity. Positive `rho` enables delayed damage and requires time-increment sensitivity checks. `ALLPD` is the accumulated cohesive damage dissipation contributed by this UEL; `ALLSE` is recoverable energy. `ALLVD` is zero because the model has no independent dashpot. Reaction work computed from finite increments can differ from internal energy; assess this through refinement.

Fixed-area element checks do not establish a general finite-area or finite-rotation energy balance. Full specimen studies still require completed loading histories and mesh, increment, and viscosity sensitivity checks.

The campaign demonstrates two elastic geometric work counterexamples with zero final stored/dissipated energy. The canonical changing-area cycle leaves −1e-4Nmm work; both canonical and isolated reference-area variants leave about+1.3791e-4Nmm in the constant-area basis cycle. The latter rotates the midsurface at fixed global gap direction and is not rigid-body rotation of the entire separated patch. The reference-area hypothesis therefore cures only the restricted area-cycle defect. It is not adopted as a general correction. Smaller material perturbations improve tested tangents, but do not repair these geometric work defects or establish specimen convergence.

UEL text output can include solver iterations and retries. The exporter is intended for visualization; consult converged DAT/ODB output for quantitative checks. Multi-step increment indexing and bulk/UEL frame alignment have further limits described in the [visualization guide](../visualization/README.md).

## Figures and reference data

Images under `docs/images/` were extracted from the project presentation `2026_01_27_UEL_Subroutine_KKV.pptx`:

| Image | Presentation slide |
| --- | --- |
| `fracture-modes.png` | 7 |
| `element-test.png` | 26 |
| `dcb-results.png` | 30 |
| `mmb-results.png` | 31 |

The original presentation is not bundled. The DCB/MMB result images show historical comparisons.

Each specimen example separates its files into:

- `inputs/`: supplied full input models, with historical filenames retained; the guides identify known parameter/name discrepancies.
- `data/`: historical solver curves and analytical or digitized reference curves.
- `figures/`: supplied comparison images.
- `plots/`: scripts for regenerating comparisons into `runs/plots/` by default.

DCB data include supplied UEL/native cohesive results and a T300 LEFM reference. MMB data include supplied UEL/native cohesive results, LEFM references, and a digitized Turon comparison. Neither data family establishes experimental validation. The exact publication/digitization metadata and source job association are incomplete; these files should be treated as development references. `benchmark_turon_backup_original.csv` preserves an earlier reference copy and is not used by the plotters.

MMB simulation plotters retain the original force conversion `2 * 12.7 * 50.4 / (41.3 + 50.4)`; the digitized reference retains its supplied scaling. Historical filename labels do not reliably establish the parameters used for a curve.
