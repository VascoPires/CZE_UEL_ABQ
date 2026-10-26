# Contributing

For a bug report, include the smallest self-contained input deck that reproduces it, material values, Abaqus/compiler versions, the exact command, and the relevant error or output comparison. Use GitHub issues for reproducible problems and pull requests for changes.

Keep the canonical UEL in `src/3D_CZM.f`. Ordinary Python tooling belongs in `scripts/`, visualization in `visualization/`, and specimen-specific models and post-processing in the corresponding example. Keep generator configuration separate from generated input, and preserve native inputs when converting to U8.

Use `scripts/run_case.py` to prepare isolated jobs and retain `run.json`. Generated jobs, solver binaries, VTK exports, and scratch files belong under ignored `runs/` folders. Add only deliberately selected comparison data with the originating input, source revision, solver/compiler versions, units, and extraction method documented. Identify analytical, digitized, historical, and newly computed curves explicitly.

Before submitting tooling changes, check Python syntax and prepare the affected example in a new run folder. GitHub automation performs these portable checks without Abaqus. For constitutive changes, also compare proportional/changing-mode loading, unloading, compression, tangent response, energy balance, and increment/viscosity sensitivity with a configured compiler and Abaqus installation. Record commands and results in the pull request; distinguish element checks from specimen validation.

Preserve the [GNU AGPL v3 license](LICENSE) and existing attribution. Document the origin and applicable permissions of any added third-party code, figures, or reference data.
