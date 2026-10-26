# Interface energy checks

`uel_proportional.inp` is a unit-area, fully prescribed U8 interface. Its top face opens and shears proportionally to complete decohesion, using the unregularized law. The expected total fracture energy is **0.3290707125 N mm**, obtained from the configured BK toughness at that local mode ratio.

```console
python scripts/run_case.py examples/element/energy_check/uel_proportional.inp --name energy_mixed --submit
abaqus python examples/element/energy_check/check_odb_energy.py --odb runs/energy_mixed/energy_mixed.odb --expected 0.32907071245835584
```

The ODB checker verifies nonnegative, nondecreasing damage dissipation and compares its final value with the supplied total energy. Pass `--expected-elastic` to check a final recoverable energy as well. It reports the difference between reaction work and internal energy; use step refinement to assess that difference. The pure-opening counterpart is obtained by setting the prescribed `TOP, 1, 1` displacement to zero; its expected energy is **0.187 N mm**. Run it under a new job name. The ODB is written in single precision by Abaqus; use a suitable tolerance for comparisons.

The UEL reports recoverable energy in `ALLSE` and cohesive damage dissipation in `ALLPD`. `ALLVD` remains zero because no separate dashpot energy is defined; delayed-damage dissipation, if enabled, is included in `ALLPD`. Recoverable energy decreases during unloading, while accumulated dissipation must not decrease. External reaction work uses finite-increment integration and should be checked by refining the step. The inviscid material energy update is exact for proportional loading; variable-mode and viscous histories require increment-refinement checks. These unit-area tests do not establish a general finite-rotation energy balance.

## Native changing-mode work diagnostic

The native COH3D8 deck continues the development study with reversed X shear and added Z shear in its second step. It fixes the bottom face and prescribes all top-face components; `INTERFACE` identifies the cohesive element. The deck explicitly uses unit constitutive thickness and requests stress/strain/damage fields. It is a separate loading path from `direction_change.inp`.

```console
python scripts/run_case.py examples/element/energy_check/coh_mixed_mode_direction_change.inp --name energy_native --submit
python examples/element/energy_check/compute_incremental_energy.py --csv exported.csv --delta-cols E13,E23,E33 --traction-cols S13,S23,S33 --thickness 1 --out work.csv
```

Export an ordered history for one element/integration point, including step boundaries. Supply matching **work-conjugate traction and separation components** in a consistent basis. The example column names assume the exported nominal cohesive strain components use those names; change them to match the actual export. For native cohesive strain, multiply by its constitutive thickness to recover separation. For true separation columns, use thickness 1. Avoid treating logarithmic strain as an exact finite-displacement jump.

The script retains the original trapezoidal work calculation. Its historical default column names are placeholders; always pass appropriate columns explicitly. It reports total incremental traction-separation work per area, including recoverable elastic work. Negative work alone is not evidence of negative dissipation or material healing. A thermodynamic check also needs the change in stored free energy and the complete relevant history.
