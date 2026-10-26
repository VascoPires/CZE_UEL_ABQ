# Turon 2006 experimental-source card

Turon A, Camanho PP, Costa J, Dávila CG. *A damage model for the simulation of delamination in advanced composites under variable-mode loading*. Mechanics of Materials 38 (2006), 1072–1089. DOI: https://doi.org/10.1016/j.mechmat.2005.10.003. Local PDF `../references/turon2006MM.pdf`, 18 pages; printed page = PDF page + 1071.

Read: variable-mode formulation §§2.3–2.4 pp.1076–1080; discretization §3 pp.1081–1082; complete experimental specimen section §4.1 pp.1082–1085 including captions Figs.8–13 and Tables1–3. Original experimental sources are cited in the paper, but their raw specimen records are not in the supplied folder.

| Item | Value and units | Source | Classification |
|---|---|---|---|
| Specimens | Unidirectional AS4/PEEK; 102 long, 25.4 wide, two 1.56-mm arms | §4.1 p.1082 | Directly stated; ply count absent |
| Moduli | E11=122700; E22=E33=10100; G12=G13=5500; G23=3700 MPa | Table1 p.1080 | Directly stated, converted |
| Poisson ratios | nu12=nu13=0.25; nu23=0.45 | Table1 | Directly stated |
| Cohesive properties | GIc=0.969; GIIc=1.719 N/mm; tau3=80; tau_sh=100 MPa; K=1e6 N/mm³; eta=2.284 | Table1 p.1080; §4.1 pp.1082–1083 | Directly stated, converted |
| Per-test a0 | 32.9 / 33.7 / 34.1 / 31.4 / 39.2 mm at 0 / 20 / 50 / 80 / 100% | Table2 p.1080 | Directly stated |
| Measured Gc | 0.969 / 1.103 / 1.131 / 1.376 / 1.719 N/mm at those fractions | Table2 | Directly stated; retain separately from BK predictions |
| Interface mesh | 8-node 3D decohesion; ENF/MMB 150 elements along length and 4 across width; DCB 102 along length | §4.1 p.1082 | Directly stated; DCB transverse count not independently specified |
| Bulk mesh | Standard volume elements; exact code and through-thickness counts not established for these specimens | §3 and §4.1 | Missing |
| Precrack | Open interface elements retain contact over each specified a0 | §4.1 p.1082 | Directly stated; initialization mechanism needs further detail |
| MMB application | Lever not meshed; end/midspan forces, lever displacement reconstructed from specimen displacements | §4.1 pp.1082–1083; Fig.8 | Directly stated; procedure referred to Camanho2003, available 2002 report is a companion, not silent substitution |
| Solver | Abaqus UEL; geometric terms omitted from tangent; NLGEOM card, loading magnitude/control, increments/stabilization not specified here | §3 p.1082; §4.1 | Directly stated approximation / missing settings |
| Experimental outputs | DCB arm force/full opening; MMB lever force vs lever load-point displacement; ENF load vs midpoint displacement | Figs.8–13; §4.1 | Directly stated |
| Experimental Pmax | 147.5 / 108.1 / 275.3 / 518.7 / 748.4 N | Table3 p.1085 | Directly stated, reported precision only |
| Published model Pmax | 152.4 / 100.5 / 263.9 / 496.9 / 697.1 N | Table3 | Published numerical prediction, not the current UEL |

Do not replace the BK curve by per-mixity measured Gc while retaining the claim of a predictive constitutive test. Eta=2.284 was fitted to the same toughness set (p.1083), and the authors identify Gc(20%)=0.988 from the fit versus measured1.103 N/mm as a source of load discrepancy (p.1084). Parameter provenance and force-response independence therefore need qualification. Re-fitting eta or changing toughness is a separately approved calibration study.

This paper's ENF peak748.4 N and crack39.2 mm disagree with the companion Camanho2002 values733.96 N and39.3 mm. DCB peaks differ as well,147.5 vs147.11 N; MMB values differ slightly in rounding. Keep both tables, not an average. Experimental scatter, per-specimen measurements and original force data have not been recovered. Graphical curves have sparse propagation points (authors' explicit limitation on p.1085). No experimental validation has been performed.
