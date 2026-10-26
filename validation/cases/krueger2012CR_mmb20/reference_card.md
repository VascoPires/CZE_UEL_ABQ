# krueger2012CR_mmb20 — reference card

Current campaign execution: [verdict and evidence paths](execution_result.json), [technical report](../../REPORT.md). The original reference-audit status below is historical.

Status: **Checkpoint1 pending; UEL comparison NOT RUN**. Role: VCCT numerical benchmark. Prepared2026-10-25.

Read the complete [paper card](../../sources/krueger2012CR.md) for bibliographic identity, shared elastic values, model definitions, source limitations and page mapping. Source SHA256: [audit inventory](../../audit/phase0_2026-10-25/inventory.json). These are reference inputs, not an approved model configuration. No tolerance has been approved.

| Quantity | Value | Unit | Page/table/figure | Classification |
|---|---|---|---|---|
| length | 152.4 | mm | Fig.1 p.24 / PDF28 | Directly stated |
| width | 25.4 | mm | Fig.1 | Directly stated |
| total_thickness | 4.5 | mm | Fig.1 | Directly stated |
| a0 | 25.4 | mm | Fig.1 | Directly stated |
| L | 50.4 | mm | 2L=100.8 in Fig.1 | Derived: divide by2 |
| layup | [0]24 | — | §2 p.3; Fig.1 | Directly stated |
| E11 | 161000 | MPa | Table I p.19 | Directly stated |
| E22_E33 | 11380 | MPa | Table I p.19 | Directly stated |
| G12_G13 | 5200 | MPa | Table I p.19 | Directly stated |
| G23 | 3900 | MPa | Table I p.19 | Directly stated |
| nu12_nu13 | 0.32 | 1 | Table I | Directly stated |
| nu23 | 0.45 | 1 | Table I | Directly stated |
| GIc | 0.212 | N/mm | Table II p.19 | Directly stated |
| GIIc_GIIIc | 0.774 | N/mm | Table II | Directly stated |
| eta | 2.1 | 1 | Table II | Directly stated |
| K3 | None | N/mm^3 | VCCT benchmark | Missing |
| normal_strength | None | MPa | VCCT benchmark | Missing |
| shear_strength | None | MPa | VCCT benchmark | Missing |
| mesh_2D | CPE4I;6 through2h;0.5mm at crack;R2D2 lever | — | §3.1 p.3 | Directly stated |
| mesh_3D | C3D8I;6 through2h;25 acrossb;0.5mm tip;R3D4 lever | — | §3.1 p.4 | Directly stated |
| BC | tip u,w matched/free rotation;midspan samew/slidingu;lower supports | — | Fig.3b-c p.25 | Directly stated |
| solver | Standard6.10;static;NLGEOM in Appendix example;no material viscosity | — | §3; Appendix pp.22–23 | Directly stated |
| outputs | lever P,w; growth a*=a-a0 | N, mm | Figs.10–21 | Directly stated |
| nominal_GII_GT | 0.2 | 1 | §§4.1–4.3 | Directly stated |
| c | 92.9 | mm | Fig.3 p.25 | Directly stated |
| Pcrit | 128.5 | N | Fig.10 p.30 annotation | Directly stated |
| wcrit | 1.64 | mm | Fig.10 p.30 annotation | Directly stated |
| reference | critical symbols Fig.10; control-specific benchmark paths separately labelled | — | p.30 / PDF34 | Directly stated |

The extracted critical symbols include equilibrium points that may not be traversed under prescribed displacement. Keep their order and do not sort by opening. Original VCCT release tolerance/contact stabilization do not define a UEL rho. A formal Turon2010 LEFM beam formula is only a separately labelled analytical comparison, not a reconstruction of Krueger’s finite-width VCCT results. No independent closed-form check is claimed here.

Questions: Q1.4–Q1.5. Data availability and all pending curves: [digitization register](../../DIGITIZATION.md). Graphical estimates stay in `data/`; analytical calculations in `results/`; no simulated response is supplied. Missing values remain null in `parameters.json`.
