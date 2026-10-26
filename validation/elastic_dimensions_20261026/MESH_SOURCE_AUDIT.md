# What the 3D references report about mesh resolution

Checked directly against the retained PDFs on 26 October 2026, including the
mesh paragraphs rather than only inherited source cards.

| Reference | Reported discretization | Exact location |
|---|---|---|
| Krueger NASA/CR-2012-217562 | C3D8I; 25 uniform elements across width; six through total thickness; 0.5 mm element length at the crack tip; graded length mesh | §3.1, printed p.4 / PDF p.8, Figs.5–6; `../references/krueger2012CR217562.pdf` |
| Turon et al. 2006, Mechanics of Materials 38 | ENF/MMB: 150 eight-node decohesion elements along length and four across width. DCB: 102 along length; that paragraph does not separately give its transverse count. Bulk volume elements are described, but the exact specimen bulk code/through-thickness count is not established. | §4.1, printed p.1082 / PDF p.11; `../references/turon2006MM.pdf` |
| Camanho and Dávila NASA/TM-2002-211737 | Companion AS4/PEEK ENF/MMB models: 150 decohesion elements along length, four across width; DCB: 102 along length. Do not silently assign four to DCB from this sentence. | §3.2, printed p.22 / PDF p.26; `../references/camanho2002TM211737.pdf` |
| Turon 2010 | 2D; contradictory plane-stress wording/CPE4 designation is retained in the source audit | §3, printed p.1859; `../sources/turon2010CS.md` |
| Turon 2018 | 2D CPE4I plane strain | §3.1, printed p.509; `../sources/turon2018CS.md` |

Krueger §5.1 and Figs.26–29 explicitly resolve variation of energy release and
mode fraction across width. Later discussion also says front shapes may improve
with width refinement. Even its 25-element mesh is not a universal convergence
certificate for every local quantity or another formulation.

## Our actual coarse full-width pilot

Retained manifest: `../campaign_20261025/runs/kr_mmb50_3d_w3/run.json`.
This was C3D8I, 25.4 mm full width, **three elements across width**, three per
arm (six total thickness), requested longitudinal spacing 1 mm and 103 length
intervals. The width spacing was about 8.467 mm, versus Krueger's 1.016 mm.
The arm-thickness spacing was 0.75 mm, giving width/thickness element aspect
ratio about 11.29. The requested longitudinal spacing was also coarser than
the paper's 0.5 mm crack-tip spacing. This was a coarse diagnostic and cannot
establish finite-width convergence or faithful source-mesh reproduction.

Its retained MSG ends with too many attempts for an increment, 14 cutbacks,
166 iterations and 99 negative-eigenvalue warnings. These establish failed
nonlinear convergence; they do not identify width discretization as its cause.
The first campaign also contains failed constrained slices and native cohesive
controls. A causal claim needs matched mesh refinements with unchanged
formulation, loading and numerical controls.

Three across width is not a sufficient general adequacy argument, and neither
is a blanket rule that it can never work. The target matters: average elastic
compliance may settle earlier than free-edge mixity or crack-front shape. The
papers themselves include a four-element interface-width mesh. Published mesh
counts are reproduction inputs, not replacements for our convergence checks.

The one-element-wide ordinary-plane-strain slices are a distinct case. Their
equal in-plane face displacements and zero width displacement impose width
invariance. Adding unconstrained elements across their width would change the
physical problem. Native 2D versus constrained 3D comparison is the direct test
of their intended elastic equivalence; full-width refinement tests free-edge
resolution separately.

Element formulation also matters in bending: the Abaqus documentation describes
parasitic shear stiffening in fully integrated first-order elements and the
purpose of incompatible modes. See [Abaqus solid elements](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEELMRefMap/simaelm-c-solidcont.htm).
This supports checking CPE4/C3D8 separately from CPE4I/C3D8I; it does not predict
the magnitude of their difference in these unrun models. Documentation is
2025; actual study execution is intended for the installed 2023 release.
