# Turon 2018 source card

Turon A, González EV, Sarrado C, Guillamet G, Maimí P. *Accurate simulation of delamination under mixed-mode loading using a cohesive model with a mode-dependent penalty stiffness*. Composite Structures184 (2018),506–511. DOI: https://doi.org/10.1016/j.compstruct.2017.10.017. Local PDF `../references/turon2018CS.pdf`,6 pages; printed page = PDF page+505.

Read complete formulation §2 pp.507–508, MMB §3.1 p.509/Table1/Figs.2–4 with legends, and separate skin-stringer §3.2 pp.509–510 to prevent mixing its IM7/8552 material with AS4/PEEK MMB.

| Item | Value and units | Source | Classification |
|---|---|---|---|
| MMB material/layup | AS4/PEEK; ply count not stated | §3.1/Table1 p.509 (same elastic set as AS4/PEEK companion); material identity contextual rather than explicit label in table | Derived identity/context; no ply count |
| Geometry |102 long,25.4 wide,two1.56-mm arms,a0=32.64 mm | §3.1 p.509 | Directly stated |
| Elastic moduli |E11=122700,E22=E33=10100,G12=G13=5500,G23=3700 MPa | Table1 p.509 | Directly stated, converted |
| Poisson ratios |nu12=nu13=.25,nu23=.45 | Table1 | Directly stated |
| Toughness |GIc=.969,GIIc=1.719 N/mm;eta=2 | Table1 | Directly stated; eta differs from2006's2.284 |
| Penalty/tangent |K3=1e5 N/mm³;zeta=1e-6 mm | §3.1 p.509 | Directly stated |
| Modal penalty |Ksh/K3=(GIc/GIIc)(tau_sh/tau3)^2 | Eq.1 p.506 | Directly stated |
| Strength combinations |(5,80),(15,80),(30,80),(45,80),(80,106.5),(80,120) MPa | Figs.2–4 legends p.509 | Directly stated; Table1 only gives ranges |
| Mesh |CPE4I plane strain;5 elements per arm;0.1-mm longitudinal length;4-node cohesive | §3.1 p.509 | Directly stated |
| Crack initialization |Initial length given; exact removal/pre-damage/contact implementation not specified | §3.1 | Missing mechanism |
| Mixities |20%,50%,70%, global fraction computed using LEFM | §3.1/Figs.2–4 | Directly stated; local B need not be uniform |
| Apparatus/BC |c, support span, lever versus equivalent load split, constraints, load height not established | §3.1/captions | Missing; do not borrow silently |
| Solver |Abaqus UEL; displacement/control algorithm, increments, NLGEOM, viscosity, stabilization not stated in MMB description | §§2.4,3.1 | Missing |
| Outputs |Total load N vs lever displacement mm | Figs.2–4 captions and axes | Directly stated |
| Role |Internal formulation check, appendix at most | DECISIONS D3 | Prior researcher decision |

Source Eq.17 gives B=Ksh*delta_sh²/(Ksh*delta_sh²+K3*positive(delta3)²); Eq.12 gives the weighted equivalent jump; Eq.18 gives KB. The source introduces B through modal work integrals, Eqs.14–16, then gives Eq.17. Under nonproportional loading, an instantaneous elastic-energy ratio need not equal the ratio of accumulated modal work. The current UEL implements Eq.17, not accumulated modal-integral state variables. This distinction must survive the StageA assessment.

A few printed expressions appear typographically inconsistent (notably Eq.23 lacks the additive plus between GIc and the BK increment; its pure-mode limits would otherwise fail). The intended additive BK law is independently supported by2010 Eq.15,2006 Eq.33 and2018 Eq.24. This is a source-transcription warning, not evidence that the current implementation is correct.

The old-formulation tau15/80 series appears only as a counterexample in Fig.2. It must remain separately labelled. The new-formulation tau30/80 square-marker series is now extracted from all three plots; other strength series and the analytical line are pending. Missing apparatus inputs prevent an independent absolute LEFM comparison now. Do not claim the analytical overlay has been reproduced.
