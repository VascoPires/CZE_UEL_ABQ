# turon2018CS_mmb20 — reference card

Current campaign execution: [verdict and evidence paths](execution_result.json), [technical report](../../REPORT.md). The original reference-audit status below is historical.

Status: **Checkpoint1 pending; UEL comparison NOT RUN**. Role: internal constitutive-formulation check;appendix at most. Prepared2026-10-25.

Read the complete [paper card](../../sources/turon2018CS.md) for bibliographic identity, shared elastic values, model definitions, source limitations and page mapping. Source SHA256: [audit inventory](../../audit/phase0_2026-10-25/inventory.json). These are reference inputs, not an approved model configuration. No tolerance has been approved.

| Quantity | Value | Unit | Page/table/figure | Classification |
|---|---|---|---|---|
| length | 102 | mm | §3.1 p.509 | Directly stated |
| width | 25.4 | mm | §3.1 | Directly stated |
| arm_thickness | 1.56 | mm | §3.1 | Directly stated |
| a0 | 32.64 | mm | §3.1 | Directly stated |
| E11 | 122700 | MPa | Table1 p.509 | Directly stated |
| E22_E33 | 10100 | MPa | Table1 | Directly stated |
| G12_G13 | 5500 | MPa | Table1 | Directly stated |
| G23 | 3700 | MPa | Table1 | Directly stated |
| nu12_nu13 | 0.25 | 1 | Table1 | Directly stated |
| nu23 | 0.45 | 1 | Table1 | Directly stated |
| GIc | 0.969 | N/mm | Table1 | Directly stated |
| GIIc | 1.719 | N/mm | Table1 | Directly stated |
| eta | 2.0 | 1 | Table1 | Directly stated |
| K3 | 100000.0 | N/mm^3 | §3.1 | Directly stated |
| perturbation | 1e-06 | mm | §3.1 | Directly stated |
| strength_pairs | [[5, 80], [15, 80], [30, 80], [45, 80], [80, 106.5], [80, 120]] | MPa | Figs.2–4 legends | Directly stated |
| mesh | CPE4I plane strain;5/arm;0.1-mm length;4-node interface | — | §3.1 p.509 | Directly stated |
| c | None | mm | §3.1 | Missing |
| support_span | None | mm | §3.1 | Missing |
| BC | None | — | §3.1 | Missing |
| crack_initialization | None | — | §3.1 | Missing |
| solver_controls | None | — | §3.1 | Missing |
| outputs | lever-applied total load vs lever displacement | N,mm | Figs.2–4 | Directly stated |
| nominal_GII_GT | 0.2 | 1 | Fig.2 p.509 | Directly stated |
| reference | Fig.2 new-law markers and analytical line | — | p.509 / PDF4 | Directly stated |

The AS4/PEEK identification follows the shared elastic/toughness set and companion context; do not import the IM7/8552 skin-stringer parameters from §3.2. The new-law tau30/80 series has graphical data with explicit axes/uncertainty. An independent LEFM comparison is **not established** because c/supports/load introduction remain missing. `lefm.py` records the blocker; it supplies no guessed c. The old-law tau15/80 Fig.2 curve is a counterexample.

Questions: Q1.7. Data availability and all pending curves: [digitization register](../../DIGITIZATION.md). Graphical estimates stay in `data/`; analytical calculations in `results/`; no simulated response is supplied. Missing values remain null in `parameters.json`.
