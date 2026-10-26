# Model dimensionality and the DCB analytical reference

This note explains the completed first-campaign models and original figure03.
It introduces no new solver results, material fitting or changed reference equations.

**Subsequent study:** the user commissioned the proposed investigation. See
[its current report](elastic_dimensions_20261026/REPORT.md): one native CPS4
elastic job completed,47 prepared/deferred. Statements below about the first
campaign remain historical; no matched equivalence conclusion exists yet.

## Executed model representations

The first campaign has100 structural runs labelled ordinary-plane-strain
constrained3D slices and one finite-width3D diagnostic. Earlier imperfect end
constraints are retained as diagnostics; the main completed comparisons use
constraint revision2. No native CPE4/CPE4I/CPS4 model was solved in this campaign.

- Bulk elements are principally C3D8I; elastic formulation controls also use
  C3D8 and C3D8R. Cohesive models use the eight-node UEL, with native COH3D8
  controls where recorded.
- Main slices are1mm wide with one element across width, matched in-plane
  displacements at corresponding front/back nodes and zero width displacement.
  Total forces are scaled by specimen-width/slice-width. This is an intended
  ordinary plane-strain representation, not generalized plane strain or a
  demonstrated finite-width reproduction.
- Linear controls verify negligible out-of-plane strains. C3D8I internal modes
  and finite-displacement behavior prevent assuming exact CPE4I equivalence
  from nodal constraints alone.
- The full-width Krueger MMB50 pilot uses width25.4mm and three elements across
  width. It failed and does not establish width convergence.
- Single-element verification uses actual3D eight-node UEL patches.

## Published references

| Reference | Published representation |
|---|---|
| [Turon2010](sources/turon2010CS.md), §3 p.1859 | Two-dimensional; text says plane stress but specifies CPE4, a plane-strain code. We adopted plane strain following the element code; author intent remains ambiguous. |
| [Krueger2012](sources/krueger2012CR.md), §§3–5 | Selected critical benchmark curves are built from2D CPE4I models. Separate finite-width3D C3D8I models use25 elements across width. |
| [Turon2018](sources/turon2018CS.md), §3.1 p.509 | Two-dimensional CPE4I plane strain, with four-node cohesive elements. |
| [Turon2006](sources/turon2006MM.md), §4.1 pp.1082–1083 | Physical3D experiments; published numerical models use volume elements and eight-node3D decohesion elements, with four interface elements across width stated for ENF/MMB. |

Thus the constrained slices are not full-width reproductions of the2006
numerical models or the experimental specimens. Dimensional equivalence remains
a limitation, especially for free-edge behavior, front shape and local mixity.

## Meaning of figure03

Three different predictions are overlaid:

1. Current UEL response, using the0.3mm constrained-slice model.
2. Graphical extraction of the published30MPa numerical curve, approximately51N peak.
3. Elementary-beam compliance combined with LEFM, using Turon2010 A1–A2 exactly.

For force P on one arm and full relative opening u,

`C(a)=u/P=8a^3/(E11*b*h^3)`

`P(a)=sqrt(GIc*b^2*h^3*E11/(12*a^2))`.

The original straight brown line was constructed from C(a0), from the origin to
the ideal sharp-crack initiation point at a=a0. The declining brown branch is
parameterized by crack lengths a≥a0. The elastic extension is not a digitized
loading curve from the paper. Its corner is not the cohesive model's peak criterion.

| Model | Initial or early-load stiffness [N/mm] | Initiation load or recorded maximum [N] | Opening there [mm] |
|---|---:|---:|---:|
| Printed elementary-beam LEFM |26.056|56.227|2.158|
| Independent tied elastic FE/VCCT control, no active UEL |21.293|52.633|2.472|
| UEL0.3mm, early-load fit |21.241|51.427|2.640|

The fixed-crack control already differs in compliance from the elementary beam
before activating a cohesive law. Shear deformation, root rotation and the
dimensional/element assumptions need separate checks before assigning that gap.
Finite cohesive-process-zone effects can additionally affect peak/onset behavior.
None of those explanations has been used to fit crack length or alter toughness.

The earlier0.84% agreement for0.3mm and0.44% for0.075mm refers to the **published
numerical peak**, not the elementary-beam initiation prediction. Figure03 does
not establish complete agreement with LEFM initial stiffness and initiation.

The [annotated figure](campaign_20261025/results/explanation_03/03_dcb_reference_explained.pdf)
separates the elastic beam extension from its growth branch and adds the existing
elastic FE initiation marker. The [calculation record](campaign_20261025/results/explanation_03/comparison.json)
contains the numerical values. Regenerate into a fresh directory with
`python validation/scripts/explain_dcb_reference.py --out <new-folder>`.

A direct elastic fixed-crack comparison between actual2D CPE4/CPE4I/CPS4
models and the constrained3D slice would isolate element/state effects before
changing strengths, toughness or penalty stiffness. That comparison has not
yet been executed.
