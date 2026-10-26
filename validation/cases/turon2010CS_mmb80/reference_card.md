# turon2010CS_mmb80 — reference card

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
| nominal_GII_GT | 0.8 | 1 | §3 pp.1859–1860 | Directly stated |
| c | 41.06 | mm | §3 pp.1859–1860 | Directly stated |
| L | None | mm | Fig.3 versus Appendix A2 | Contradictory |
| strength_families | 15 with30/45/60;30 with60/90/120 | MPa | Table1; Figs.6–7 | Directly stated |
| adjusted_strength_pairs | 15/29.4 and30/58.9 | MPa | §4.1 p.1862; Fig.11 | Directly stated |
| crack_initialization | remove initial cohesive elements | — | §3 p.1859 | Directly stated |
| outputs | lever-applied total P vs lever displacement | N, mm | Appendix A5-A7; Figs.6–7,11–12 | Directly stated |
| reference | Appendix A5-A7; corrected Fig.12 | — | pp.1862–1863 | Directly stated |

Earlier-law Figs.6–7 are diagnostic counterexamples. Agreement targets for the updated law must be chosen explicitly (Q1.1). No c from another paper is used. LEFM equilibrium mode fraction may vary with crack length at fixedc; the source nominal percentage and UEL localB remain separate. `lefm.py` requires explicit L and toughness/mixity policy before a calculation.

Questions: Q1.1–Q1.3. Data availability and all pending curves: [digitization register](../../DIGITIZATION.md). Graphical estimates stay in `data/`; analytical calculations in `results/`; no simulated response is supplied. Missing values remain null in `parameters.json`.
