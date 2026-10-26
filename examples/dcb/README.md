# Double cantilever beam

`T300_Gaussian.inp` and `T300_NewtonC.inp` retain the historical T300 geometry, constraints, load and element sets; their declaration now correctly reserves seven real properties and one integer. Both supplied decks actually specify `JPROPS(1)=1` (Gauss). To study Newton–Cotes, make a new copy and set the final `*UEL PROPERTY` entry to `2`; do not infer it from the filename.

```console
python scripts/run_case.py examples/dcb/inputs/T300_Gaussian.inp --name dcb --submit
```

Each deck defines the bulk solid and `COHESIVE_LAYER` interface. Read its boundary, equation and output blocks before adapting the geometry. Historical comparison CSVs are in `data/`, supplied figures in `figures/`, and plotting scripts in `plots/`. Full input decks are in `inputs/`. Run the commands below from the repository root.

## Generate a model

```console
python examples/dcb/main_script.py --out runs/dcb_build
```

Edit `dcb_config.json` or select another file with `--config`. The default creates:

```text
runs/dcb_build/
  dcb_config.json          Configuration used for this model
  T300_Gaussian_CZM.cae    Native cohesive CAE model
  T300_Gaussian_CZM.inp    Native COH3D8 input
  T300_Gaussian_UEL.inp    Converted U8 input
  3D_CZM.f                User subroutine, ready for submission
```

`run_name` controls the output prefix. The generator builds and meshes the specimen in Abaqus CAE, then converts a copy of the input using the common converter. It preserves labels, connectivity, boundary conditions and loading. The original native input remains available for comparison. The `.cae` model remains native; rerunning `writeInput` from CAE does not reproduce the UEL modifications automatically.

| Configuration | Controls |
| --- | --- |
| `material_key`, `specimen_index` | Bulk material and specimen selected from `materials` |
| `geometry`, `mesh` | Length, mesh sizes and elements through thickness |
| `def_u2`, `Pseudo_2D`, `Symmetry` | Displacement and model constraints |
| `cohesive_standard` | Native cohesive material and layer used by the CAE model |
| `cohesive_UEL` | Fracture energies, strengths, penalty stiffness, viscosity and integration for U8 |
| `Run Subroutine` | Whether to create the UEL input; defaults to true if omitted |

The native and UEL material values are configured separately; check both when comparing formulations. Available material/specimen settings are options, not a claim that every combination was tested.

```console
python examples/dcb/main_script.py --config examples/dcb/dcb_config_standard.json --out runs/dcb_native --native
python examples/dcb/main_script.py --out runs/dcb_solve --submit --cpus 2
```

Use `--abaqus "C:/SIMULIA/Commands/abaqus.bat"` if needed. Run these commands with ordinary Python 3; the launcher invokes Abaqus CAE itself. Choose a new output folder for each run. To solve an already generated UEL input, open an Abaqus command environment in its output folder and run:

```console
abaqus job=T300_Gaussian_UEL input=T300_Gaussian_UEL.inp user=3D_CZM.f interactive
```

To convert another existing native COH3D8 deck without rebuilding geometry:

```console
python scripts/convert_input.py model.inp --config examples/dcb/dcb_config.json --output model_UEL.inp
```

Conversion supports one explicit COH3D8 block and one cohesive section. Expand includes first. UEL properties are read from `cohesive_UEL`; unused native material definitions are retained.

## Plot the comparison

From the repository root:

```console
python examples/dcb/plots/plot_cfrp_dcb_mul_light.py
```

This writes `runs/plots/dcb/cfrp_dcb_mul_light.png`, comparing the available T300 UEL, native cohesive and LEFM curves. Python reads the CSVs relative to the script, so the command also works from another working directory. Use `--out-dir path/to/plots` to select another output folder. No LaTeX or interactive display is needed.

The supplied CSVs are development results; their exact job configurations have not all been established, and the current source has not been used to reproduce them. Optional CFRP/experimental CSVs mentioned in the plotter are absent and skipped. These comparisons do not establish experimental validation.

Model generation and the converted UEL input have been checked with Abaqus 2023, including Fortran compilation/linking and datacheck. These checks do not replace a full analysis or reproduce the supplied comparison curves.

The generator defaults to zero viscosity in both material blocks. Set `cohesive_UEL.viscosity` explicitly for the optional delayed-damage extension; the named sensitivity-study decks retain their supplied viscosity values. Native damage evolution follows `evolution.type` (`B-K`/`BK` or `POWER_LAW`), and generated U8 declarations include `UNSYMM`.
