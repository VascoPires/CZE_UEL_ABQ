# turon2010CS_dcb — reference card

Current campaign execution: [verdict and evidence paths](execution_result.json), [technical report](../../REPORT.md). The original reference-audit status below is historical.

Status: **Checkpoint1 pending; UEL comparison NOT RUN**. Role: numerical/analytical verification. Prepared2026-10-25.

Read the complete [paper card](../../sources/turon2010CS.md) for bibliographic identity, shared elastic values, model definitions, source limitations and page mapping. Source SHA256: [audit inventory](../../audit/phase0_2026-10-25/inventory.json). These are reference inputs, not an approved model configuration. No tolerance has been approved.

| Quantity | Value | Unit | Page/table/figure | Classification |
|---|---|---|---|---|
| length | 150 | mm | §3 p.1859 | Directly stated |
| width | 20 | mm | §3 p.1859 | Directly stated |
| arm_thickness | 1.55 | mm | §3 p.1859 | Directly stated |
| a0 | 35 | mm | §3 p.1859 | Directly stated |
| E11 | 120000 | MPa | Table1 p.1860 | Directly stated |
| E22_E33 | 10500 | MPa | Table1 p.1860 | Directly stated |
| G12_G13 | 5250 | MPa | Table1 p.1860 | Directly stated |
| G23 | 3480 | MPa | Table1 p.1860 | Directly stated |
| nu12_nu13 | 0.3 | 1 | Table1 p.1860 | Directly stated |
| nu23 | 0.5 | 1 | Table1 p.1860 | Directly stated |
| GIc | 0.26 | N/mm | Table1 p.1860 | Directly stated |
| GIIc | 1.002 | N/mm | Table1 p.1860 | Directly stated |
| eta | 2.0 | 1 | Table1 p.1860 | Directly stated |
| original_equal_penalty | 1000000.0 | N/mm^3 | §3 p.1859 | Directly stated |
| cohesive_length | 0.15 | mm | §3 p.1859 | Directly stated |
| bulk_state | CPE4 named; text plane stress | — | §3 p.1859 | Contradictory |
| thickness_mesh | 10 elements; allocation per arm/full thickness unclear | count | §3 p.1859 | Directly stated |
| solver_controls | None | — | §3; Appendix A | Missing |
| rho | None | time | not specified | Missing |
| normal_strengths | [15, 30, 60] | MPa | Fig.4 p.1860 | Directly stated |
| shear_strength_for_UEL | None | MPa | pure-I benchmark does not fix the required UEL shear input | Missing |
| crack_initialization | remove initial cohesive elements | — | §3 p.1859 | Directly stated |
| outputs | one-arm force vs full relative opening | N, mm | Fig.3; Appendix A1-A2 | Directly stated |
| reference | Fig.4 analytical and numerical curves; A1-A2 | — | pp.1860,1863 | Directly stated |

The existing T300 DCB is a different case. `lefm.py` reproduces the AppendixA1–A2 reference; its graphical-domain comparison includes the formal extension to a<a0 visible in Fig.4. That domain extension is not a physical initial loading path. DCB beam theory here omits root-rotation/shear corrections because the source equations omit them; any enhanced analytical model must be separately labelled.

Questions: Q1.1–Q1.3. Data availability and all pending curves: [digitization register](../../DIGITIZATION.md). Graphical estimates stay in `data/`; analytical calculations in `results/`; no simulated response is supplied. Missing values remain null in `parameters.json`.
