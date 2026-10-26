# Single-element checks

| Deck | Loading |
| --- | --- |
| `inputs/opening.inp` | Fixed bottom; prescribed top displacement in global direction 2 |
| `inputs/mixed_mode.inp` | Fixed bottom; simultaneous prescribed top shear and opening |
| `inputs/direction_change.inp` | Two steps, changing the relative prescribed displacements |
| `inputs/standard_direction_change.inp` | Native Abaqus cohesive comparison, no UEL required |

`BOTTOM`, `TOP` and `ALL` define the face and output sets. `INTERFACE` is the UEL element set. Connectivity `1,4,3,2,5,8,7,6` makes the interface lie in the global x–z plane. Inspect the prescribed displacements rather than inferring local mode directions from the global coordinate number. Other unprescribed degrees of freedom follow the deck's boundary conditions.

Run the commands below from the repository root. Prepare any deck using `scripts/run_case.py`; add `--submit` to solve. The runner detects native decks and omits the Fortran source from their solver command.

For the UEL cases, read state variables from `*EL PRINT`:

```console
python examples/element/parse_elprint_sdv_by_increment.py --datFile runs/opening_solved/opening_solved.dat --outCsv runs/opening_solved/sdv.csv --sdvCount 200
python examples/element/plot_sdv_curves.py --help
```

The UEL packs four integration points into a single element state vector with stride 20. Use SDV1–11 for GP1, SDV21–31 for GP2, SDV41–51 for GP3 and SDV61–71 for GP4. The `.dat` row's integration-point number is not automatically the UEL's internal GP number; avoid applying the parser's optional `--layoutStride` blindly. The parser preserves step and increment columns. The plotter exposes options for selecting columns and steps.

Use `.dat` output for the two-step case: the historical VTK index reader keys entries only by increment, which restarts in each step.

The Windows `run_job_and_parse.cmd` shortcut solves the direction-change case and parses/plots its DAT output in `runs/direction_change_solved`. It uses Python and Abaqus from PATH and refuses to reuse an existing run folder. The 3D plotters use `element_properties.json` for their analytical overlays (matching mixed-mode/direction-change properties); pass another `--config` when plotting a material with different properties, including the opening test. These overlays are analytical comparisons, not solver validation.

The [energy-check study](energy_check/README.md) adds the native cohesive direction-reversal deck and a traction-separation work diagnostic. Its output is work per area, not a direct dissipation measure.

The basic UEL inputs use the rate-independent law (`rho=0`) and declare `UNSYMM`. Positive viscosity remains available in `*UEL PROPERTY`. The [proportional energy case](energy_check/README.md) fully prescribes the interface motion through complete decohesion and checks the ODB energy output.
