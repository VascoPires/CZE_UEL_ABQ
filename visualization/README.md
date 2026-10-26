# View UEL fields in ParaView

Install `requirements-visualization.txt` in ordinary Python. Run the commands below from the repository root. Abaqus Python is only needed for reading an ODB through the external bulk converter.

## Export the cohesive elements

```console
python visualization/run_uel_vis.py --run-dir runs/opening_solved --kinc 1 --rank auto
```

This reads the input connectivity and UEL output and writes `visu/CZM_000001.vtk`. `--kinc` selects the actual increment number. `--rank auto` merges the detected MPI process ranks; rank numbers do not count solver threads. Each rank writes text and index files:

```text
UNVALS_CZM_rank000.txt    UNVALS_CZM_rank000.idx
IPVALS_CZM_rank000.txt    IPVALS_CZM_rank000.idx
```

In ParaView open the VTK, apply **Warp by Vector**, select `Displacement`, and color by `DMG`, `DMGv`, `T_normal`, or another named field. The exporter extrapolates four IP values to nodes and averages shared-node contributions; nodal colors are visualization interpolants and can overshoot IP extrema.

## Add the bulk mesh

The bulk converter remains an external dependency: provide the `python/odb2vtk.py` file from the ODB2VTK distribution used in your Abaqus environment. It is not bundled with this release.

```console
python visualization/run_full_vis.py --run-dir runs/dcb --odb2vtk C:/tools/ODB2VTK/python/odb2vtk.py --abaqus-cmd abaqus --rank auto
```

Alternatively merge an already exported bulk PVD:

```console
python visualization/run_uel_vis.py --run-dir runs/dcb --base-pvd runs/dcb/bulk.pvd --rank auto --index-offset 1
```

Open the resulting `merged.pvd` in ParaView. The historical series merger skips bulk frame 0 and maps later frames by ordinal index plus `--index-offset`; it does not match physical time or independently verify frame correspondence. Use only when output frequency and increment sequence agree. Check correspondence explicitly for sparse frames, skipped increments or failed increments.

## Output limitations

The text writer runs inside UEL evaluations, including solver iterations and retries. These files are not a guaranteed converged-state history. The reader keeps the last encountered element block for an increment; use the converged `.dat`/ODB history to check quantitative results. The current index reader keys by `KINC`, so multi-step jobs with repeated increment numbers are ambiguous; use `.dat` for the direction-change element test. Restart histories are not supported by the release workflow.

The input reader supports flat element decks and the supplied single-part models. It does not apply arbitrary assembly instance transforms or disambiguate repeated node labels in multiple parts. The retained zero H1 gradients are format placeholders. The state-variable layout is listed below.

## State variables and output files

Each of four internal integration points occupies 20 state entries, starting at `20*(ip-1)+1`. Allocate 200 variables as in the supplied examples; the code also uses diagnostic slots including SDV108.

| Local SDV | Meaning |
| --- | --- |
| 1–3 | Deformed mid-surface IP coordinates |
| 4 | Shear separation magnitude |
| 5 | Normal separation |
| 6 | Shear traction magnitude |
| 7 | Normal traction |
| 8 | Rate-independent damage DMG |
| 9 | Equivalent separation |
| 10 | Mode-mixity variable B |
| 11 | Viscously regularized damage DMGv |
| 12 | Recoverable energy density |
| 13 | Accumulated damage dissipation per unit area |
| 14 | Accumulated area-weighted dissipation at this integration point |
| 15 | Thermodynamic damage force Y |
| 16–20 | Reserved/zero placeholders |

UNVALS contains element label, eight nodes, three displacement components, and zero gradient placeholders. IPVALS contains element label, four integration points, 20 values per point, and the integration-scheme selector. A `KINC=` header starts each increment block. Index rows contain increment, step, step time, total time and a Fortran stream position; readers convert the one-based position before seeking.

The Fortran writes relative to the solver's working directory. `scripts/run_case.py` copies the source into an isolated run folder. No source-line-number patch or machine-specific output path is required.

`ENERGY(2)` supplies recoverable energy (`ALLSE`); `ENERGY(4)` supplies accumulated cohesive damage dissipation (`ALLPD`). The latter includes dissipation from the delayed damage variable when viscosity is enabled. `ALLPD` is the Abaqus label for this UEL contribution; compare the appropriate total damage dissipation when examining a native cohesive model, rather than assuming identical output labels. There is no independent viscous dashpot in this formulation, so `ENERGY(5)`/`ALLVD` is zero and must not be interpreted as the total influence of viscosity. Do not add SDV13 to SDV14: they express the same accumulated dissipation with different area weighting.

For `rho=0`, the damage-energy increment is exact under proportional loading and uses the current mode mixity during a changing-mode increment. For `rho>0`, it integrates the damage force against the change in delayed damage using trapezoidal quadrature (the delayed-damage evolution itself is backward Euler). Check increment refinement for changing-mode or regularized histories. Accumulated area-weighted dissipation is retained without rescaling old increments when the surface area changes. Recoverable energy uses the current mid-surface area; a full finite-area/finite-rotation energy balance is not established by the fixed-area element checks.
