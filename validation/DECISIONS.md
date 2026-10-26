# Decisions

Vasco's decisions, dated. Add new entries at the bottom; never edit an old one, supersede it.

| ID | Date | Decision | Source |
| --- | --- | --- | --- |
| D1 | 2026-10-25 | The chapter follows Germain (2020) Ch. 3-4 in structure and figure types only; Germain's T700/M21 results are not reproduced. | chat |
| D2 | 2026-10-25 | Benchmark set: single element; Turon 2010 (DCB, ENF, MMB 20/50/80 %); Krueger MMB (CR-2012-217562, 0.2/0.5/0.8, 2D and 3D); validation on Reeder & Crews AS4/PEEK experiments via Turon 2006 / Camanho 2002. | chat |
| D3 | 2026-10-25 | Turon 2018 MMB (AS4/PEEK, 20/50/70 %) is reproduced as an internal check; in the thesis it is mentioned and shown in an appendix at most. Main-text figures compare UEL, closed-form LEFM and the published curves on the same axes. | chat |
| D4 | 2026-10-25 | Dropped: Germain T700/M21, Camanho T300/977-2 DCB, Krueger 2008 DCB. | proposed, not objected |
| D5 | 2026-10-25 | Work follows Castro's workflow with two checkpoints; no runs before Checkpoint 2. | chat |

## Open questions for Checkpoint 1

Listed in `HANDOFF.md` section 6. Record each answer here as a new D-entry with the question
number.

## Checkpoint 1 responses — 2026-10-25

The user answered Q1.1–Q1.9 after reviewing the initial deliverables. These
entries preserve the scope of each answer; requests for explanation are not
treated as approvals.

| ID | Date | Decision | Source |
| --- | --- | --- | --- |
| D6 | 2026-10-25 | Q1.1 approved: LEFM and the adjusted-strength Turon 2010 curves are agreement targets; earlier-law Figs. 6–7 remain diagnostic counterexamples. | User reply 1: Yes |
| D7 | 2026-10-25 | Q1.2 approved: interpret the named CPE4 model as plane strain, retaining the contradictory source wording in the audit. | User reply 2: Yes |
| D8 | 2026-10-25 | Q1.3 approved: use L = 75 mm as the explicit interpretation and report the analytical energy fraction versus crack length. The answer did not separately specify whether A5 toughness is fixed at nominal mixity or evaluated at that energy fraction; this distinction must remain explicit in the later plan. | User reply 3: Yes |
| D9 | 2026-10-25 | Q1.4: a 3D slice is authorized if needed; the user delegates selection of ordinary versus generalized plane strain. Agent selection: ordinary plane strain for reproducing the selected plane-strain references, with zero width strain/shear and width-invariant in-plane displacements; retain a separate finite-width 3D comparison. This is a model choice for planning, not a demonstrated equivalence result. | User reply 4 and delegated modelling judgement |
| D10 | 2026-10-25 | Q1.5: prefer recent relevant source information, using the more complete source when necessary; if cohesive strengths/penalty are absent, existing project defaults may be used as explicitly labelled assumed numerical inputs and varied later. No literature property is silently replaced and no fitting campaign is approved. Candidate project defaults for Krueger remain normal strength 32 MPa, shear strength 61 MPa, K3 = 170000 N/mm³; these are not universal Abaqus defaults or Krueger-reported values. | User reply 5; values from existing mmb_config.json |
| D11 | 2026-10-25 | Q1.7 approved: reconstruct the missing 2018 apparatus using the companion sources, explicitly labelled as a reconstruction. Retain the 2018 crack length, toughness and eta rather than silently importing another case's entire parameter set. Exact assumptions will be exposed in the Phase 2 plan. | User reply 7: Yes |
| D12 | 2026-10-25 | Q1.9 approved: published experimental peaks may be used provisionally while original traces/scatter are sought; this alone does not establish full independent experimental validation. | User reply 9: Yes |

### Answers requiring explanation before progression

- **Q1.6 remains pending:** the user proposes the more recent source and asks for
  an explanation before continuing. Turon 2006 is the newer of the two selected
  experimental tables. Its later publication date does not establish that its
  observations are newer experiments or corrections to the 2002 values.
- **Q1.8 remains pending:** the user asks what excluding the historical curve
  means. The recommendation concerns its role as a quantitative target; it does
  not propose deleting the old CSVs, historical runs or backup.
- This response stage records decisions and supplies clarification. Phase 2,
  model generation, source changes and solver execution have not begun.

## Checkpoint resolution and autonomous execution — 2026-10-25

| ID | Date | Decision | Source |
|---|---|---|---|
| D13 | 2026-10-25 | Q1.6 resolved: Turon 2006 is the primary experimental table; Camanho 2002 remains a separate cross-check. | User: On 6, Okay lets do that |
| D14 | 2026-10-25 | Q1.4 resolved: proceed with ordinary plane strain in a constrained 3D U8 slice for the plane-strain comparisons; retain the finite-width 3D distinction. | User: On 4, proceed with your recommendation |
| D15 | 2026-10-25 | Cohesive strengths and penalty stiffness are adjustable, explicitly documented numerical inputs when absent from a source. Investigate whether the current penalty is too low; preserve published-source baselines and record every change, its reason and effect. This authorizes sensitivity checks, not undisclosed fitting of fracture energies to target curves. | User: On 5, those can be changed; penalty stiffness might be too low; document checks |
| D16 | 2026-10-25 | Q1.8 resolved: preserve historical MMB CSVs, but exclude them from quantitative agreement targets until provenance and shifts are explained. | User: On 8, Okay |
| D17 | 2026-10-25 | The current Prompt0.md is an Autonomous Execution Protocol. It explicitly authorizes generation, compilation, analyses, troubleshooting and continuous progression through the campaign without routine phase approval stops. It supersedes the old stop-at-Checkpoint-2 restriction for execution; canonical UEL preservation and consequential scientific-decision safeguards remain. | Current root Prompt0.md, operating mode/execution authority/immediate instruction; user: Proceed |

The original D5 and checkpoint records remain historical. Their ordinary phase-stop clauses are superseded by D17. Q1.3's analytical toughness convention will be tested explicitly as two labelled policies; no curves are shifted to conceal their difference. Working diagnostic tolerances are declared in PLAN.md and are not represented as previously approved thesis acceptance criteria.


## D18 — Optional tangent perturbation input, 2026-10-26

The user requested making the tested perturbation parameter a UEL input. Implement optional real PROPS(8)=zeta, with positive finite length units. Preserve1e-6 when omitted; allow explicit1e-8 without editing source. Retain the v1.3 source archive and all first-campaign run copies/results. This is an authorized numerical interface extension, not a new material law or an approved geometric-energy reformulation. The integer integration scheme follows all real properties.

## D19 — Matched elastic dimensionality comparison, 2026-10-26

The user explicitly requested continuing the proposed matched 2D/3D elastic
fixed-crack comparison and investigating mesh adequacy/source mesh reporting.
Preserve the completed campaign. Implement the separate elastic study and
document its numerical mesh controls under existing autonomous authority.
This does not approve fitting, constitutive changes or overriding the recorded
one-solver-at-a-time constraint. Actual resource deferral and incomplete study
status are recorded in `elastic_dimensions_20261026/REPORT.md`.
