# Mixed-mode bending

The eight `MMB_Thesis_*.inp` decks are ports of the historical IM7/8552 UEL studies. The original two shear-strength entries were identical in every selected deck; the port uses one shared strength and sets `PROPERTIES=7`. Geometry, connectivity, supports, loading and other solver settings remain as supplied. The current v1.3 source differs from the historical source, so these ports require fresh runs before comparison with historical curves.

`standard_GIIGT_0p2.inp` is a separate native cohesive model. It is not a matched native counterpart to every UEL sweep. Inspect the support, loading and equation sets in each full input before changing it.

```console
python scripts/run_case.py examples/mmb/inputs/MMB_Thesis_visc_SDEG_1_Visc1E5_MscEq.inp --name mmb --submit
```

| Variant suffix | Actual normal/shear strength (MPa) | Actual rho |
| --- | --- | --- |
| `tau_30` | 32 / 30 | 1E-5 |
| `tau_90` | 32 / 90 | 1E-5 |
| `tau_120` | 32 / 120 | 1E-5 |
| `tau_5_60` | 5 / 80 | 1E-5 |
| `Visc1E3_MscEq` | 32 / 61 | 1E-3 |
| `Visc1E4_MscEq` | 32 / 61 | 1E-4 |
| `Visc1E5_MscEq` | 32 / 61 | 1E-5 |
| `Visc1E6_MscEq` | 32 / 61 | **1E-5** |

These discrepancies existed in the original filenames. They are preserved, not silently corrected to guessed intentions. No unique source-deck association has been established for each historical CSV.

## Generate a model

```console
python examples/mmb/main_script.py --out runs/mmb_build
```

The generator uses `mmb_config.json` to build the specimen and loading apparatus in Abaqus CAE. It writes `MMB_IM7_8552_GIIGT_0p2_CZM.cae`, the native `MMB_IM7_8552_GIIGT_0p2_CZM.inp`, a separate `MMB_IM7_8552_GIIGT_0p2_UEL.inp`, the selected configuration and `3D_CZM.f` into the run folder. `run_name` sets the output prefix. The CAE model remains native; the UEL conversion is applied to a separate input file.

| Configuration | Controls |
| --- | --- |
| `material_key`, `specimen_index` | Bulk material and specimen selected from `materials` |
| `geometry.l` | Specimen length |
| `geometry.L`, `geometry.C`, `geometry.loading_height` | Support/lever geometry and loading height |
| `mesh` | Bulk/fine mesh sizes and elements through thickness |
| `def_u2`, `Pseudo_2D`, `Symmetry` | Applied displacement and model constraints |
| `cohesive_standard` | Native cohesive material and layer used to generate the mesh |
| `cohesive_UEL` | Fracture energies, strengths, penalty stiffness, viscosity and integration for U8 |
| `Run Subroutine` | Whether to create the UEL input; defaults to true if omitted |

The default retains the IM7/8552 geometry from the original model generator and uses the rate-independent default (`rho=0`). The static sensitivity-study decks above have their own parameters. The generator's native and UEL property blocks are configured separately; check both when comparing responses.

```console
python examples/mmb/main_script.py --config my_mmb_config.json --out runs/my_mmb
python examples/mmb/main_script.py --out runs/mmb_native --native
python examples/mmb/main_script.py --out runs/mmb_solve --submit --cpus 2
```

Run the launcher with ordinary Python 3. It invokes Abaqus CAE; use `--abaqus "C:/SIMULIA/Commands/abaqus.bat"` if Abaqus is outside PATH. Choose a new output folder each time. To solve an already generated UEL input, run from its output folder in an Abaqus command environment:

```console
abaqus job=MMB_IM7_8552_GIIGT_0p2_UEL input=MMB_IM7_8552_GIIGT_0p2_UEL.inp user=3D_CZM.f interactive
```

To convert an existing native cohesive deck separately:

```console
python scripts/convert_input.py model.inp --config examples/mmb/mmb_config.json --output model_UEL.inp
```

Conversion replaces one COH3D8 block and its cohesive section with U8 and `*UEL PROPERTY`, preserving node/element labels, sets and loading. Expand included input files first.

## Plot the comparisons

Full input models are in `inputs/`. Historical comparison CSVs, supplied figures, and plotting scripts are separated into `data/`, `figures/`, and `plots/`. From the repository root:

```console
python examples/mmb/plots/plot_mmb_lefm.py
python examples/mmb/plots/plot_mmb_tau_compare.py
python examples/mmb/plots/plot_mmb_viscosity_compare.py
```

Each script writes its PNG under `runs/plots/mmb/` and locates supplied data relative to itself. Use `--out-dir path/to/plots` to select another output folder. No LaTeX or interactive display is needed.

The scripts retain the original load conversion `2 * 12.7 * 50.4 / (41.3 + 50.4)` for simulation curves; the digitized reference retains its supplied scaling. Inspect each script's series before comparing curves. The `32_61` baseline is labelled 61 MPa. These are existing development comparisons, not fresh verification runs; the exact source/configuration for each CSV has not been established. Review digitization and curve provenance before publication.

Model generation and the converted UEL input have been checked with Abaqus 2023, including Fortran compilation/linking and datacheck. These checks do not replace a full analysis or reproduce the supplied comparison curves.
The original apparatus ties and constraints are retained; the generated model reports tie/boundary and inactive-degree-of-freedom warnings during datacheck. Review these when adapting the support or loading arrangement.

The generator defaults to zero viscosity in both material blocks. Set `cohesive_UEL.viscosity` explicitly for the optional delayed-damage extension; the named sensitivity-study decks retain their supplied viscosity values. Native damage evolution follows `evolution.type` (`B-K`/`BK` or `POWER_LAW`), and generated U8 declarations include `UNSYMM`.
