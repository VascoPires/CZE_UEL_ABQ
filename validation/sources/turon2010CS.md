# Turon 2010 source card

Turon A, Camanho PP, Costa J, Renart J. *Accurate simulation of delamination growth under mixed-mode loading using cohesive elements: Definition of interlaminar strengths and elastic stiffness*. Composite Structures 92 (2010), 1857–1864. DOI: https://doi.org/10.1016/j.compstruct.2010.01.012. Local PDF: `../references/turon2010CS.pdf` (8 pages). Identity checked on PDF page 1. Journal pages are PDF page + 1856.

Read: constitutive definitions §§2.1–2.1.1, pp.1858–1859; all specimen/model text §3, pp.1859–1861; corrections §4, pp.1862–1863; Appendix A, pp.1863–1864; figure captions and legends 3–12. PDF hashes are in `../audit/phase0_2026-10-25/inventory.json`.

| Item | Value and units | Source | Classification |
|---|---|---|---|
| Material/layup | Unidirectional carbon/epoxy; ply count not stated | §3 p.1859 | Directly stated / missing count |
| Geometry | Length 150, width 20, arm thickness 1.55, initial crack 35 mm | §3 p.1859; Fig.3 | Directly stated |
| Moduli | E11=120000; E22=E33=10500; G12=G13=5250; G23=3480 MPa | Table1 p.1860 | Directly stated, GPa converted ×1000 |
| Poisson ratios | nu12=nu13=0.3; nu23=0.5 | Table1 p.1860 | Directly stated |
| Toughness | GIc=0.260; GIIc=1.002 N/mm; eta=2 | Table1 p.1860 | Directly stated, 1 kJ/m²=1 N/mm |
| Original penalty | k=1e6 N/mm³, equal modal penalties in the §3 calculations | §3 p.1859; Eqs.5,11,12 | Directly stated |
| Strengths | Table1 pairs normal 15 with shear 30/45/60, normal 30 with shear 60/90/120 MPa; figures include additional pure-mode strengths | Table1; Figs.4–7 p.1860–1861 | Directly stated; distinguish each legend |
| Bulk elements | Text says plane stress, element code CPE4 means plane strain | §3 p.1859; official Abaqus element definitions | Contradictory; Q1.2 |
| Mesh | 0.15 mm cohesive length; “10 ... along the thickness” | §3 p.1859 | Directly stated; per-arm vs full-thickness allocation unclear |
| Crack | DCB/MMB remove cohesive elements over precrack; ENF uses pre-damaged elements to retain compression contact | §3 p.1859 | Directly stated |
| Support/lever | Fig.3 identifies L from midpoint to support, but Appendix A2 calls L specimen length; numeric support span not stated explicitly | Fig.3 p.1859; Appendix A2 p.1863 | Contradictory/ambiguous; Q1.3 |
| MMB c | 149.05 / 63.18 / 41.06 mm for 20 / 50 / 80% | §3 pp.1859–1860 | Directly stated |
| Solver | Abaqus UEL; displacement/load-control algorithm, NLGEOM, viscosity, increments, stabilization not established | §3 and Appendix A | Missing, not zero by implication |
| Outputs | DCB arm force P vs relative opening; ENF central P vs midpoint displacement; MMB lever P vs lever displacement, all total N | Fig.3; Figs.4–7,11–12; Appendix A | Directly stated/defined by drawings and captions |

Figs.6–7 deliberately show erroneous strength-dependent mixed-mode propagation using the earlier constant-penalty law. They are diagnostic counterexamples, not obligatory agreement targets for the updated 2018 law. Figs.11–12 use the adjusted strengths of Eq.23: tau_sh=tau3 sqrt(GIIc/GIc), rounded to 29.4/58.9 MPa. Fig.12 also displays 70%, although the selected case set remains 20/50/80%. The current UEL instead enforces Ksh/K3=(GIc/GIIc)(tau_sh/tau3)^2. Matching every old strength sweep would test a different law. Q1.1 resolves the target.

Independent reference equations: A1–A2 DCB; A3–A4 ENF; A5–A7 MMB. DCB uses elementary beam theory without a root correction. MMB includes chi*h and 0.42*chi*h corrections. The formulas do not determine a stable trajectory through snap-back. The LEFM modal fraction obtained by differentiating A7 is distinct from local UEL B. Missing span/mixity choices must be resolved before running ENF/MMB reference calculations.

Graphical data: the visible black DCB analytical branch is extracted with registered axes/path IDs. Its gaps where other curves obscure it remain gaps. The other 2010 curve families are registered as pending in `../DIGITIZATION.md`; no synthetic samples replace them.
