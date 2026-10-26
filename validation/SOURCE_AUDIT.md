# Source and provenance audit — 2026-10-25

All seven PDFs exist and were hash-inventoried. Their identity was checked against the supplied publications. Original sources take precedence over the inherited NOTES.md and HANDOFF.md, which remain preserved. The [index](INDEX.md) links five shared source cards and seventeen case cards with page-level inputs.

| PDF | Pages | Role |
|---|---:|---|
| turon2010CS.pdf | 8 | Primary DCB/ENF/MMB numerical reference; DOI 10.1016/j.compstruct.2010.01.012 |
| turon2018CS.pdf | 6 | Constitutive formulation and internal MMB check; DOI 10.1016/j.compstruct.2017.10.017 |
| turon2006MM.pdf | 18 | Published experimental observations and earlier-law predictions; DOI 10.1016/j.mechmat.2005.10.003 |
| camanho2002TM211737.pdf | 42 | Companion AS4/PEEK tables and load/kinematic appendices; distinct from the 2003 journal paper |
| krueger2012CR217562.pdf | 62 | Primary IM7/8552 MMB numerical reference |
| krueger2012MMB_conf.pdf | 19 | Conference companion; not substituted for the full report |
| krueger2008TM215123.pdf | 71 | Inventory of the dropped DCB source; no new benchmark extracted |

The dropped 2008 material-identity ambiguity was not independently adjudicated: it remains UNRESOLVED and outside the selected campaign. PDF counts were checked with pypdf. Bibliographic completeness does not imply that original experimental raw data exist.

## Inherited issues and new findings

| ID | Finding | Status and consequence |
|---|---|---|
| A01 | Actual property order is GIc, GIIc, normal strength, shear strength, eta, K3, rho, then the integer scheme | CONFIRMED in source lines 116–128 and interface docs. The symbol listing in Prompt0/old handoff is not an input order. |
| A02 | `benchmark_turon.csv` lacks publication, figure, series and digitization metadata | Provenance UNRESOLVED. A filename cannot establish the source. Exclude from quantitative targets pending Q1.8. |
| A03 | Active/backup curves have 15 rows; all 14 non-origin x values differ by −0.1 mm | CONFIRMED by row comparison. Y differs only by rounding, below 4.3e-8 N. Reason and author are unknown; preserve both. |
| A04 | Plotters use c = 41.3, L = 50.4 and factor 2×12.7×L/(c+L); active config uses C = 92.9 | Formula/config mismatch CONFIRMED; impact on each old CSV is UNRESOLVED without its job and force definition. Do not substitute a new factor blindly. |
| A05 | Eight historical MMB UEL decks use rho from 1e-3 to 1e-5; the file labelled Visc1E6 uses 1e-5 | CONFIRMED at `*UEL PROPERTY`. The effect on propagation needs approved reruns; viscosity is not established as the sole explanation. |
| A06 | Historical source equality is mixed | Two completed element folders match the current source; other copies differ. No completed DCB/MMB loading curve is tied to the current hash in this inventory. |
| A07 | MMB shares Krueger parameters but differs in overall length, actual width, bulk elements and apparatus constraints | CONFIRMED: configured length 150 versus 152.4 mm; forced width 2 mm; C3D8R versus CPE4I/C3D8I; ties versus published MPCs. The handoff's claim of an already valid Krueger model is unsupported. |
| A08 | Both generators force width to 2 mm in pseudo-2D mode and request front/back U3 = 0 with one element across width | CONFIRMED in constructor and mesh/constraint methods. This supplies part of a plane-strain construction; equivalence remains unverified. |
| A09 | DCB UEL config uses GIc = .352 and normal strength 60; the native block uses strength 30 | CONFIRMED. Camanho Table 1 instead gives .268 N/mm and 45 MPa. The old attribution of 30 MPa to the UEL was wrong. The dropped T300 case stays dropped. |
| A10 | Turon 2010 calls CPE4 plane stress | Source contradiction CONFIRMED in §3, p. 1859. CPE4 is plane strain; CPS4 is plane stress. Author intent remains unresolved. |
| A11 | Turon 2018 MMB omits lever length, support span, load introduction and detailed solver settings | Omissions CONFIRMED in §3.1 and captions. Any companion-based reconstruction needs explicit approval. |
| A12 | AS4/PEEK observation tables differ | CONFIRMED: ENF peaks 748.4 versus 733.96 N and cracks 39.2 versus 39.3 mm; DCB peaks 147.5 versus 147.11 N. Keep both sources separate. |
| A13 | Turon 2010 Figs. 6–7 demonstrate earlier-law errors; Figs. 11–12 use adjusted strengths | CONFIRMED by §§3–4. Select agreement targets explicitly at Q1.1. |
| A14 | Turon 2010 gives inconsistent descriptions of L | Fig. 3 draws midpoint-to-support L; Appendix A2 calls it specimen length. Numeric span is not explicit. Q1.3 resolves the interpretation. |
| A15 | Krueger's 20% critical load is explicitly annotated as 128.5 N | CONFIRMED in Fig. 10, printed p. 30/PDF page 34. It need not be estimated from the plotted path. |
| A16 | Both historical DCB decks use scheme 1 and rho = 1e-6, including the NewtonC filename | CONFIRMED in property blocks. Names do not define actual integration or viscosity. |
| A17 | A finite initial face gap could enter the UEL as separation | A suspected default-gap problem was DISPROVED for the inspected first elements: generators collapse interface nodes; paired nodes coincide in inspected historical DCB/MMB and generated DCB decks. Do not diagnose a gap solely from native layer thickness. This is not a check of every element. |

For A10, [official Abaqus element documentation](https://docs.software.vt.edu/abaqusv2025/English/TsoUserMap/tso-c-usr-solver-abaqus-elements.htm) confirms the element definitions. It cannot resolve the authors' actual model.

The code and property-block observations are confirmed facts; their physical effects generally remain unresolved. The full source cards retain missing controls, per-figure strengths, crack initialization and output definitions. No parameter was changed to improve agreement.

## Experimental evidence and methodological source

Original Reeder/Crews specimen records, force traces and scatter are absent. Eta = 2.284 was fitted to the reported toughness set. A future comparison must distinguish fitted fracture inputs from independent force-response observations. Published numerical agreement and successful termination are not experimental validation.

Castro's workflow reference was checked at [Zenodo record 23478287](https://zenodo.org/records/23478287), dated 19 October 2026. It supplies the method, not cohesive benchmark values. Web references accessed 2026-10-25.

## Executed implementation checks — 25 October 2026

Current results supersede historical NOT RUN statements only where executed evidence is available. See REPORT.md and campaign_20261025/INVESTIGATION.md for complete scope.

- Canonical fixed1e-6mm forward perturbation fails the1% smooth-tangent screen at high penalty; an isolated1e-8mm copy is tested separately. Constant-area traction/history/energy responses are unchanged in matched material histories.
- Full assembled residual/Jacobian tests retain a roughly0.382% softening discrepancy, consistent with missing geometric terms. This restricted check is not a generally consistent finite-deformation tangent proof.
- A rho=0 elastic closed geometry cycle with area1→1.2→1 gives canonical net work-1e-4Nmm and zero final stored/dissipated energy, unchanged at100 versus1000 intervals. The current-area weight changes force/energy but no conjugate area-change force is assembled. This is a demonstrated energy counterexample, distinct from viscous quadrature.
- An isolated nominal/reference-area hypothesis closes that restricted cycle. It is not canonical, not an approved traction-measure choice, and not a complete geometric formulation. Constant-area and matched specimen checks are separate.
- Positive-rho endpoint Y*dDv quadrature has a separately verified positive discrete energy excess that decreases under refinement. Structural completed jobs with large energy imbalance are rejected.
- An end-coupling width-BC defect in early generated models is corrected in revision2. Initially opening MMB contact controls fail near zero gap; successful contact-free fixed-crack controls verify no overlap and correct lever statics, and opening cases retain per-frame gap checks. ENF uses measured penalty contact.

The canonical src/3D_CZM.f and original examples remain unchanged. Do not infer validated specimen behavior from static source algebra, a successful solver termination, or a constant-area unit test.
