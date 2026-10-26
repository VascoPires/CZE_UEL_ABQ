# CZE UEL for Abaqus

An eight-node cohesive user element for mixed-mode delamination in Abaqus/Standard. The fixed-form Fortran implementation follows the mode-dependent penalty stiffness approach of Turon's 2018 formulation. Modes II and III share a shear strength and fracture energy.

The repository includes the UEL, input conversion and job preparation tools, configurable DCB and MMB examples, and ParaView exporters.

![Delamination fracture modes](docs/images/fracture-modes.png)

## Quick start

Use Python 3.10 or newer for the command-line tools. Solving requires a licensed Abaqus/Standard installation and a compatible Fortran compiler configured for user subroutines. Model generation runs inside Abaqus CAE; the supplied generators have been checked with Abaqus 2023.

From the repository root, prepare an opening example:

```console
python scripts/run_case.py examples/element/inputs/opening.inp --name opening
```

This creates `runs/opening/` with the input deck, `3D_CZM.f`, and `run.json` recording input hashes and the solver command. Preparation uses only the Python standard library. To solve, use a new run folder and add `--submit`:

```console
python scripts/run_case.py examples/element/inputs/opening.inp --name opening_solved --submit
```

Use `--abaqus "C:/SIMULIA/Commands/abaqus.bat"` when Abaqus is outside `PATH`, `--cpus 2` to select solver CPUs, or `--out path/to/new-folder` for another output location. Existing run folders are never overwritten.

## Examples

| Example | Contents | Guide |
| --- | --- | --- |
| Single element | Opening, mixed mode, changing direction, and energy checks | [Element guide](examples/element/README.md) |
| Double cantilever beam | T300 models, configurable CAE generator, historical comparisons | [DCB guide](examples/dcb/README.md) |
| Mixed-mode bending | IM7/8552 models, configurable CAE generator, historical sensitivity studies | [MMB guide](examples/mmb/README.md) |

Generate a specimen model with ordinary Python; the launcher invokes Abaqus CAE:

```console
python scripts/build_model.py dcb --out runs/dcb_build
python scripts/build_model.py mmb --out runs/mmb_build
```

Each command retains the native cohesive input and writes a separate UEL input. Use `--config custom.json`, `--native`, or `--submit` as needed. The example guides describe configuration fields and historical input variants.

## Post-processing

Install the plotting dependencies only if needed:

```console
python -m pip install -r requirements.txt
python examples/dcb/plots/plot_cfrp_dcb_mul_light.py
```

Comparison plots are written to `runs/plots/`; `--out-dir` selects another folder. Supplied CSVs and figures remain unchanged.

For UEL visualization, install the additional dependencies and export a solved increment:

```console
python -m pip install -r requirements-visualization.txt
python visualization/run_uel_vis.py --run-dir runs/opening_solved --kinc 1 --rank auto
```

Open `runs/opening_solved/visu/CZM_000001.vtk` in ParaView. See the [visualization guide](visualization/README.md) for fields, bulk-mesh merging, and output limitations.

## Repository layout

```text
src/                 Canonical Fortran UEL
scripts/             Input conversion, model generation, and job preparation
examples/
  element/           Single-element inputs and post-processing tools
  dcb/               DCB generator, configurations, inputs, data, figures, plots
  mmb/               MMB generator, configurations, inputs, data, figures, plots
visualization/       UEL VTK export and bulk/UEL series merge
docs/                Interface reference, validation scope, and project images
runs/                Local generated jobs and plots; excluded from Git
```

For an existing COH3D8 input, use [scripts/convert_input.py](scripts/convert_input.py). The [UEL interface reference](docs/uel-interface.md) defines the property order, integration selector, node ordering, and required `UNSYMM` declaration.

## Verification scope

The current implementation has element-level verification. The supplied DCB/MMB curves are historical development results and have not been reproduced with the current source. Full specimen validation and mesh, increment, and viscosity sensitivity remain outstanding. See [validation and data provenance](docs/validation.md) before interpreting the comparisons.

A lightweight GitHub workflow checks Python syntax and prepares example jobs without launching Abaqus. It does not establish solver or physical validation.

## Credits and license

Implementation: Vasco D. C. Pires, Chair of Designing Plastics and Composite Materials, Technical University of Leoben. Project figures originate from `2026_01_27_UEL_Subroutine_KKV.pptx`; [provenance notes](docs/validation.md#figures-and-reference-data) identify their slides and the status of comparison data.

Distributed under the [GNU Affero General Public License v3](LICENSE). Abaqus and the optional external ODB2VTK converter are not bundled. See [CONTRIBUTING.md](CONTRIBUTING.md) for reproducible reports and contributions.
