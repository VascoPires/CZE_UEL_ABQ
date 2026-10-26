# Optional tangent perturbation input — 26 October 2026

The user requested exposing the first campaign's tested tangent perturbation as a
UEL input. Version1.4 reads optional real `PROPS(8)=zeta`; omission preserves the
v1.3 default of1e-6 in the model's length units. The integer integration scheme
follows all real properties. See the [interface examples](../../docs/uel-interface.md).

The old source is retained as
[`3D_CZM_v1_3_baseline.f`](../campaign_20261025/sources/3D_CZM_v1_3_baseline.f).
`baseline.json` records its hash. The first campaign's106 run decks, source copies
and scientific results are unchanged; `first_campaign_REPORT.md` retains its
report before the source-version annotation. This extension does not correct
the reported geometric energy defects or establish full specimen validation.

## Checks — passed

- [Assembly comparisons](assembly_comparisons.json): nine full-UEL states per
  configuration, including elastic, softening and separated responses at three
  orientations. Residuals, Jacobians, all200 state variables and eight energy
  slots are exactly equal for v1.3 versus the omitted default, omitted versus
  explicit1e-6, and the fixed1e-8 variant versus input-selected1e-8. These are
  compiled standalone UEL tests with serial Abaqus-runtime stubs.
- [Material comparisons](material_comparisons.json): each five-history,
  9600-sample response file is byte-identical to the corresponding first-campaign
  canonical/smaller-perturbation reference, including tangent-error columns.
- [Workflow checks](workflow_checks.json): old/new input preparation, conversion,
  scheme ordering, invalid values and the structured UEL/native/tied builders.
- [Abaqus comparisons](solver_comparisons.json): three Abaqus2023/Standard
  batches completed, with16 patch histories each. Nodal reaction/displacement
  and SDV CSVs are byte-identical to the matching v1.3 reference batches.
- A deliberately invalid zero-perturbation deck was submitted directly to test
  the Fortran guard. It terminated with the expected diagnostic; see
  [invalid_input_check.json](invalid_input_check.json).
- [verification.json](verification.json) records completion and confirms that
  the original example files and all106 first-campaign run inputs/source copies
  remain unchanged. [source_patch.diff](source_patch.diff) records the source change.

Run the preparation checks from the repository root:

```console
PYTHONDONTWRITEBYTECODE=1 python validation/tangent_input_20261026/check_workflows.py
PYTHONDONTWRITEBYTECODE=1 python validation/tangent_input_20261026/check_results.py
```

Standalone assembly tests use `validation/scripts/element_tangent_check.py`
with `--out`, `--source`, `--export-state` and optional
`--tangent-perturbation`. Material checks use `material_check.py --out ...
--material turon2018 --K 1e6 --zeta ...`. Output folders must be fresh.
Existing Abaqus run manifests retain exact commands, input/source hashes,
properties and termination state. Reusing a completed run folder is refused.
