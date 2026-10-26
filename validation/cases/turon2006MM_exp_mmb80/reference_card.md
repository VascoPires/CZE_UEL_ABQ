# turon2006MM_exp_mmb80 — reference card

Current campaign execution: [verdict and evidence paths](execution_result.json), [technical report](../../REPORT.md). The original reference-audit status below is historical.

Status: **Checkpoint1 pending; UEL comparison NOT RUN**. Role: provisional experimental validation target. Prepared2026-10-25.

Read the complete [paper card](../../sources/turon2006MM.md) for bibliographic identity, shared elastic values, model definitions, source limitations and page mapping. Source SHA256: [audit inventory](../../audit/phase0_2026-10-25/inventory.json). These are reference inputs, not an approved model configuration. No tolerance has been approved.

| Quantity | Value | Unit | Page/table/figure | Classification |
|---|---|---|---|---|
| length | 102 | mm | §4.1 p.1082 | Directly stated |
| width | 25.4 | mm | §4.1 p.1082 | Directly stated |
| arm_thickness | 1.56 | mm | §4.1 p.1082 | Directly stated |
| E11 | 122700 | MPa | Table1 p.1080 | Directly stated |
| E22_E33 | 10100 | MPa | Table1 | Directly stated |
| G12_G13 | 5500 | MPa | Table1 | Directly stated |
| G23 | 3700 | MPa | Table1 | Directly stated |
| nu12_nu13 | 0.25 | 1 | Table1 | Directly stated |
| nu23 | 0.45 | 1 | Table1 | Directly stated |
| GIc | 0.969 | N/mm | Table1 | Directly stated |
| GIIc | 1.719 | N/mm | Table1 | Directly stated |
| eta | 2.284 | 1 | §4.1 p.1083 | Directly stated |
| normal_strength | 80 | MPa | Table1 | Directly stated |
| shear_strength | 100 | MPa | Table1 | Directly stated |
| original_equal_K | 1000000.0 | N/mm^3 | §4.1 p.1082 | Directly stated |
| bulk_element_and_thickness_mesh | None | — | §3; §4.1 | Missing |
| solver_controls | None | — | §4.1 | Missing |
| NLGEOM_rho | None | — | §4.1 | Missing |
| nominal_GII_GT | 0.8 | 1 | Table2 p.1080 | Directly stated |
| a0 | 31.4 | mm | Table2 p.1080 | Directly stated |
| measured_Gc | 1.376 | N/mm | Table2 p.1080 | Directly stated |
| experimental_Pmax | 518.7 | N | Table3 p.1085 | Directly stated |
| published_model_Pmax | 496.9 | N | Table3 p.1085 | Directly stated |
| companion_a0 | 31.4 | mm | Camanho2002 Table3 p.22 | Directly stated |
| companion_experimental_Pmax | 518.66 | N | Camanho2002 Table4 p.23 | Directly stated, rounding differs |
| interface_mesh | 8-node;150 along length,4 across width | — | §4.1 p.1082 | Directly stated |
| crack_initialization | open decohesion elements retain compression contact | — | §4.1 p.1082 | Directly stated |
| reference | Fig.12,Table3;companion Fig.8/Table4 | — | pp.1083–1085; Camanho2002 p.23 | Directly stated |
| c_numerical_companion | 28.4 | mm | Camanho2002 Appendix B Table5 p.33 | Directly stated |
| c_experimental_companion | 27.6 | mm | Camanho2002 Appendix B Table5 p.33 | Directly stated |
| BC | end/midspan force split;lever absent;reconstruct lever displacement | — | §4.1 p.1083;companion Appendices B–C | Directly stated |
| outputs | total lever P versus lever load-point displacement | N,mm | §4.1 p.1083;Figs.10–12 | Directly stated |

The companion is [Camanho2002](../../sources/camanho2002TM.md); its inputs and observations remain separate. The full experimental traces, scatter and specimen-level independence are unresolved. `data/experimental_peak_sources.csv` transcribes both tables with no averaging. The 2006 and 2002 published numerical predictions are distinct earlier laws and are not UEL results. Use the additive BK law with its independently documented parameters; do not set its mixed-mode energy equal to measured Gc without an approved calibration.

Questions: Q1.6,Q1.9. Data availability and all pending curves: [digitization register](../../DIGITIZATION.md). Graphical estimates stay in `data/`; analytical calculations in `results/`; no simulated response is supplied. Missing values remain null in `parameters.json`.
