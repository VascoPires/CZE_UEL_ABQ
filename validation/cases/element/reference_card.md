# element — reference card

Current campaign execution: [verdict and evidence paths](execution_result.json), [technical report](../../REPORT.md). The original reference-audit status below is historical.

Status: **Checkpoint1 pending; UEL comparison NOT RUN**. Role: constitutive verification, before DCB. Prepared2026-10-25.

Read the complete [paper card](../../sources/turon2018CS.md) for bibliographic identity, shared elastic values, model definitions, source limitations and page mapping. Source SHA256: [audit inventory](../../audit/phase0_2026-10-25/inventory.json). These are reference inputs, not an approved model configuration. No tolerance has been approved.

| Quantity | Value | Unit | Page/table/figure | Classification |
|---|---|---|---|---|
| interface_area | 1.0 | mm^2 | examples/element/energy_check/uel_proportional.inp nodes | Derived:1×1 surface |
| GIc | 0.187 | N/mm | same deck *UEL PROPERTY | Repository baseline, not literature input |
| GIIc | 0.781 | N/mm | same | Repository baseline, not literature input |
| normal_strength | 38 | MPa | same | Repository baseline, not literature input |
| shear_strength | 52 | MPa | same | Repository baseline, not literature input |
| eta | 1.22 | 1 | same | Repository baseline, not literature input |
| K3 | 100000.0 | N/mm^3 | same | Repository baseline, not literature input |
| rho | 0 | time | same | Repository baseline, not literature input |
| integration | 2 | 1=Gauss,2=Newton-Cotes | same | Repository baseline, not literature input |
| final_global_top_displacement | [0.02, 0.02, 0] | mm | same *BOUNDARY | Repository baseline, not literature input |

Every property above belongs to a repository test, not the AS4/PEEK paper. The full intended StageA suite remains **NOT RUN**: pureI,pureII,proportional mixed mode,unloading/reloading,onset/evolution,complete separation/work and dissipation,changing mode,tangent,Gauss/Newton–Cotes,and K/rho sensitivity. Existing opening/mixed/direction-change/native/energy decks supply starting points, not evidence that this complete suite has passed.

The code localB is instantaneous K-weighted squared separation (Turon2018 Eq.17). For proportional loading it matches the modal-energy partition used in the theoretical BK target. For changing paths record localB, modal work integrals, stored energy, and cumulative dissipation separately; do not force the terminal energy to one nominal Gc. Old historiesD,Dv are stored; historicalB and separate modal work integrals are not. The current endpoint-Gc dissipation update explicitly requires increment refinement for changing mode. No independent dashpot energy is tracked. Full element tangent finite differences must include geometry effects, unlike a material-tangent-only check.

The source implementation is statically consistent with many intended formula definitions; this card does not certify numerical correctness. The pre-existing current-source closure/regularized runs are documented in the audit, without being substituted for a fresh StageA suite.

The listed repository baseline requires no new literature assumption. Approval
of the full verification suite and its acceptance criteria belongs to Checkpoint 2.
Data availability and outstanding curves are in the [digitization register](../../DIGITIZATION.md).
No new simulated response is supplied. Missing values remain null in `parameters.json`.
