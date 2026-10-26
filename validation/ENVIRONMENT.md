# Environment audit — 2026-10-25

**Current execution update:** Abaqus2023 and the Intel Fortran toolchain have now compiled and executed actual campaign jobs, including successful UEL and native controls. The licence was available for these recorded one-CPU jobs. The “not tested” descriptions below are the preserved initial audit, not the current execution status. Actual evidence is in `campaign_20261025/runs/*/execution_stdout.txt`, `.sta`, compiler logs and `run.json`. Intel Fortran classic reports2021.8.0 Build20221119 from the oneAPI2023 installation. Ordinary WSL Python and Abaqus Python remain separate; Abaqus post-processing uses its bundled Python2.7-compatible API.

## Initial read-only audit

No Abaqus process, CAE builder, datacheck, compiler, ODB extraction or licence checkout was invoked. Ordinary Python performed inventories, static parsing, PDF extraction and the Phase 1 analytical reference check.

| Component | Observed state | Limit |
|---|---|---|
| Host | WSL2 Linux 6.6.87.2; Windows 10.0.26200.9168 | Read-only platform and `cmd.exe /c ver` observations |
| Python | 3.13.9, `/home/vasco/miniconda3/bin/python` | Separate from Abaqus Python |
| Reference packages | numpy 2.4.1; matplotlib 3.10.8; pypdf 6.4.1; Pillow 12.1.0 | Installed locally; reference requirements are pinned |
| PDF utilities | pdftotext, pdfinfo, pdftoppm, pdftocairo | Available locally |
| WSL PATH | No abaqus, abaqus2023, ifort, ifx or gfortran resolved | This does not establish Windows tool availability |
| Windows solver | `where abaqus` resolves `C:/SIMULIA/Commands/abaqus.bat` | Launcher was inspected, never executed |
| Launcher chain | abq2023.bat initializes Intel oneAPI 2023.0.0 and VS 2022, then calls the 2023 SMALauncher.exe | Installation is configured for 2023; live release and functionality untested |
| Compiler/C++ tools | ifort.exe and VS Community vcvars64.bat exist | ifort/ifx/cl did not resolve in plain Windows cmd; initialized toolchain not tested |
| Licence | NOT TESTED | No checkout attempted |
| Execution readiness | INCONCLUSIVE | Installation paths do not prove that a licensed solver/compile job will succeed |

The inherited claim that this is necessarily a solver-less cloud environment is too strong: local Windows Abaqus and compiler installations are visible. Equally, their presence does not prove execution readiness. Defer a live test to the approved phase.

The current source is version 1.3, dated 16.10.2026, SHA256 `bfa62fce8fef54fc8f3ba6f04946b3e0c5b68e4dbed5fb60987c8b27cc8d455c`. HEAD is recorded in the [inventory](audit/phase0_2026-10-25/inventory.json), but the working tree was already dirty and the modern layout is untracked relative to that commit. Use file hashes for provenance.

Re-audit into a fresh directory:

```console
PYTHONDONTWRITEBYTECODE=1 python validation/scripts/audit_repository.py --out /tmp/cze-phase0-recheck
```

This script launches neither Abaqus nor a compiler. Its PASS entries describe input parsing and Python syntax only.
