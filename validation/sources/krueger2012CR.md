# Krueger 2012 source card

Krueger R. *Development and Application of Benchmark Examples for Mixed-Mode I/II Quasi-Static Delamination Propagation Predictions*. NASA/CR-2012-217562, NIA Report 2012-01, April 2012. Local PDF: `../references/krueger2012CR217562.pdf` (62 pages). Printed report page = PDF page − 4 after front matter. Conference companion `krueger2012MMB_conf.pdf` is a distinct 50% publication, not the source for the 20/80% definitions.

Read: §§2–4 pp.2–7; 2D/3D differences §5.1 pp.7–9; §5.2 p.9–10, §5.3 pp.10–13, §5.4 pp.13–15; Appendix pp.20–23; Tables I–II p.19; Figs.1–6 pp.24–28 and Figs.7–29 pp.29–40; later propagation curves/captions were distinguished from the static benchmark. In particular, Figs.10,14,19 occur on printed pp.30,32,35 (PDF pages 34,36,39). The old prompt mixed printed and PDF page numbers.

| Item | Value and units | Source | Classification |
|---|---|---|---|
| Material/layup | IM7/8552 [0]24 | §2 p.3; Fig.1 p.24 | Directly stated |
| Geometry | b=25.4; 2h=4.5; support span 2L=100.8; overall 2L*=152.4; a0=25.4 mm | Fig.1 p.24 / PDF28 | Directly stated |
| Moduli | E11=161000; E22=E33=11380; G12=G13=5200; G23=3900 MPa | Table I p.19 / PDF23 | Directly stated, converted |
| Poisson ratios | nu12=nu13=0.32; nu23=0.45 | Table I | Directly stated |
| Toughness | GIc=0.212; GIIc=GIIIc=0.774 N/mm; eta=2.1 | Table II p.19 | Directly stated, converted |
| Cohesive K/strength | No cohesive strengths or penalty stiffness defined for the VCCT benchmark | §§3–4; Table II | Missing for a cohesive reproduction; not inferable from VCCT |
| 2D discretization | CPE4I plane strain, 6 through total thickness 2h, tip spacing 0.5 mm, rigid R2D2 lever | §3.1 p.3; Figs.3–4 | Directly stated |
| 3D discretization | C3D8I, 6 through total thickness, 25 across width, tip spacing 0.5 mm, rigid R3D4 lever | §3.1 p.4; Figs.5–6 | Directly stated |
| Crack | Coincident upper/lower meshes; unbonded initial region and bonded nodes ahead; VCCT release | §§3.1–3.2 pp.3–4 | Directly stated |
| Constraints | Tip: same in-plane u,w, free rotation; midspan: same w, sliding u; lower left u,w fixed, right w fixed | Fig.3b–c p.25; Appendix model | Directly stated; retain equivalent rigid-body suppression |
| c | 92.9 / 41.3 / 27.3 mm at nominal 0.2 / 0.5 / 0.8 | Figs.3–4 pp.25–26 | Directly stated |
| Pcrit,wcrit | 128.5 N,1.64 mm (20%); 385 N,1.34 mm (50%); 751 N,1.65 mm (80%) | Fig.10 p.30; Fig.14 p.32; Fig.19 p.35 | Directly annotated, not fitted |
| Outputs | Total load P and vertical displacement w at applied lever load; a*=increase in crack length, not absolute a | Figs.1,10–21 | Directly stated |
| Solver | Abaqus/Standard 6.10; Appendix example uses NLGEOM in both steps; static propagation | §3.1 p.3; Appendix p.22–23 | Directly stated for example, not proof every source run had identical control cards |
| Numerical controls | No automatic/static stabilization or viscosity; release-tolerance sweeps, with/without contact stabilization cs=1e-6 | §3.2 p.5; §5; Appendix | Directly stated for VCCT studies; these are not UEL settings |
| Applied-displacement examples | Step1 w=1.5 / 1.2 / 1.5 mm at 20/50/80%; Step2 w=8 mm; initial increment 0.005, minimum 1e-18, max5000 increments in Step2 | §§5.2–5.4; Appendix pp.22–23 | Directly stated; implementation transfer needs a separate plan |

The static benchmark was constructed from fixed-crack FE/VCCT models, scaling the load and displacement by sqrt(Gc/GT), Eq.3 pp.5–6. It is a numerical reference, not an independent LEFM closed form or an experimental force curve. At 50/80%, the critical equilibrium curve includes snap-back: prescribed-displacement and load-control paths differ. Critical symbol data have been extracted separately; they do not by themselves implement either jump rule.

The selected critical-load reference curves in Figs.10,14,19 are constructed
from the **two-dimensional** fixed-crack models in §§4.1–4.3. The same report
also contains separate three-dimensional models and a dimensional comparison
in §5.1; that does not make these selected critical curves full-width3D targets.

Three-dimensional fronts and local mixities differ from 2D. §5.1 and Figs.22–29 explicitly discuss variation across width and changes with crack length; nominal mode fraction is not a uniform cohesive-process-zone B. A narrow constrained 3D slice cannot reproduce the finite-width 3D benchmark merely by multiplying force by width.

The repository shares some dimensions and properties, but its configured overall length is 150 mm (not 152.4), its pseudo-2D width becomes 2 mm, its bulk is C3D8R (not CPE4I/C3D8I), and its apparatus uses ties instead of the published MPC configuration. The handoff's claim that these are already “valid Krueger models” is unsupported. Their later equivalence must be demonstrated.
