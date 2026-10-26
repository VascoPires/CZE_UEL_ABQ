# Camanho and Dávila 2002 companion-source card

Camanho PP, Dávila CG. *Mixed-Mode Decohesion Finite Elements for the Simulation of Delamination in Composite Materials*. NASA/TM-2002-211737, June2002. Local PDF `../references/camanho2002TM211737.pdf`,42 pages. Report printed page = PDF page−4. Identity checked on cover. Used as a companion for the selected AS4/PEEK cases, not to revive the dropped T300/977-2 benchmark (D4).

Read: §3.2 pp.21–24, Tables2–4 and Fig.8; Appendix B pp.33–34, Tables5–7 and Eqs.68–78; Appendix C p.35, Eqs.79–81; Appendix D pp.36–37. The force/kinematic appendices are usable evidence; their assumptions and sign conventions must be retained.

| Item | Value and units | Source | Classification |
|---|---|---|---|
| AS4/PEEK geometry/material | Length102, width25.4, arms1.56 mm; E11=122700,E22=E33=10100,G12=G13=5500,G23=3700 MPa; nu12=nu13=.25,nu23=.45 | §3.2 p.21; Table2 p.22 | Directly stated, converted |
| Cohesive law | GIc=.969,GIIc=1.719 N/mm; normal80,shear100 MPa; eta2.284 | Table2; p.22; Appendix D | Directly stated; earlier law, not the 2018 formulation |
| Penalty units | Text p.21 prints K=1e6 N/mm, inconsistent with traction/separation stiffness | §3.2 p.21 vs law dimensions; Turon2006 p.1082 | Contradictory unit; do not copy without qualification |
| a0 |32.9/33.7/34.1/31.4/39.3 mm at0/20/50/80/100% | Table3 p.22 | Directly stated; ENF differs from Turon2006 |
| Gc measured |.969/1.103/1.131/1.376/1.719 N/mm | Table3 | Directly stated |
| Experimental peaks |147.11/108.09/275.35/518.66/733.96 N | Table4 p.23 | Directly stated; distinct record from Turon2006 |
| Published model peaks |153.27/86.95/236.60/479.86/695.94 N | Table4 | Published numerical, not current UEL |
| Lever numerical c |109.4/44.4/28.4 mm at20/50/80% | Table5 p.33 | Directly stated |
| Lever experimental c |97.4/42.2/27.6 mm | Table5 | Directly stated; lever weight accounted for in experimental lengths |
| Load split | Pm=P(c+l)/l; Pe=Pc/l, neglect lever weight | Eqs.71–72 p.33 | Directly stated; opposite load directions in specimen equilibrium |
| Mid/end ratios |1.46/2.14/2.79 | Table6 p.34 | Directly stated, rounded |
| Modal components | PI=P(3c−l)/(4l); PII=P(c+l)/l | Eqs.75–76 p.34 | Directly stated; these are not identical to Pe and Pm |
| Load-point displacement | LambdaLP=((3c−l)/(4l))*LambdaI+((c+l)/l)*(LambdaM+LambdaI/4), only a<l | Eq.81 p.35 | Directly stated; signs follow paper coordinate conventions |

Numerical c values follow the idealized, weight-free split and differ from experimental apparatus c. The symbol l is the half specimen span: the table values are consistent with51 mm, not102 mm, when Eq.70 is used (derived dimensional interpretation). They are not interchangeable with Turon2010 c or the missing Turon2018 apparatus geometry. For a32.64-mm Turon2018 crack, also distinguish that initial length from all per-test a0 in Table3. Any transfer of this apparatus to 2018 is an explicit hybrid reconstruction requiring Q1.7 approval.

T300 audit only: §3.1 pp.19–20/Table1 gives a150-mm specimen, width20 mm, two1.98-mm arms, a0=55 mm, GIc=.268 N/mm, strength45 MPa. The current DCB UEL configuration uses GIc=.352,strength60 MPa; the native block strength30 MPa is a separate entry. Neither source block is a matched Table1 reproduction. D4 keeps this case outside the selected campaign.
