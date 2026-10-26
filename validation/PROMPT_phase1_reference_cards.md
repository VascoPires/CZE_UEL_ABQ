# Prompt: Phase 1, reference cards

Paste this into a new session that has access to the CZE_UEL_ABQ repository. Use the
`research-ai-workflow` skill.

```
KEY  = turon2010CS | krueger2012CR | turon2006MM | turon2018CS   (one card each)
PDF  = validation/references/<file>.pdf    (see validation/references/README.md)
CASE = <KEY>_<specimen>   e.g. turon2010CS_mmb50, krueger2012CR_mmb20, turon2018CS_mmb70
```

## Goal

Write one reference card per paper so that every value needed to build the benchmark models
in Phase 3 is recorded with its source, and every value that is missing or inconsistent is a
numbered question for Vasco. Do not build or run models in this phase.

## Read first

1. `validation/HANDOFF.md` (context, agreed cases, known inconsistencies)
2. `validation/DECISIONS.md`
3. `validation/references/NOTES.md` (values already extracted, with pages)
4. The PDFs themselves, all of each relevant section, captions and footnotes included.

## Deliverables (in `validation/cases/<CASE>/`)

- `reference_card.md`: geometry, layup, elastic properties, interface properties (G_Ic,
  G_IIc, eta, strengths, penalty), element type and size, model dimension (2D/3D, plane
  strain/stress, width), boundary conditions and load introduction (lever/MPC/load split, c),
  loading (displacement or load, magnitude), solver settings (NLGEOM, viscosity, stabilisation),
  output quantity (what P and w mean, per width or not), reference curve type (LEFM, VCCT,
  experiment). Each row: value, unit, source (page/table/figure), status (stated / derived /
  missing / inconsistent).
- `data/`: digitised reference curves as CSV with header `# source: <key>, Fig. X, p. Y;
  digitised <date> with <tool>; est. error <value>`. Curves to digitise:
  - Turon 2010: Figs. 4-7 (pp. 4-5) and 11-12 (p. 6)
  - Turon 2018: Figs. 2-4 (p. 4)
  - Krueger CR-2012-217562: benchmark curves Figs. 10, 14, 19 (pp. 34, 36, 39);
    displacement/load vs a* Figs. 11, 15, 16, 20, 21
  - Turon 2006 Figs. 9-13 and Table 3; Camanho 2002 Table 4 (experimental peak loads)
- `lefm.py` (per case, where applicable): the closed-form LEFM curve from Turon 2010
  Appendix A / Camanho 2002 Appendix B, plotted against the digitised curve as a sanity check
  of the card (the LEFM curve must reproduce the paper's LEFM curve within digitisation error).
- A single `validation/CHECKPOINT1.md`: every open question, numbered Q1.1, Q1.2, ... with
  your recommendation and the evidence, so Vasco can answer quickly. Start from
  `HANDOFF.md` section 6.

## Ground rules

- Never guess a value. Missing means missing: write it as a question.
- Units N, mm, MPa throughout; state them in every table and CSV header.
- Do not modify `src/3D_CZM.f`, existing examples or historical data. New files only, under
  `validation/`.
- Do not commit the PDFs (they are git-ignored).

## Stop and ask

Stop at Checkpoint 1 after writing `CHECKPOINT1.md`. Do not start Phase 2 (plan) until Vasco
has answered and the answers are in `DECISIONS.md`.

## After Checkpoint 1 (for reference)

Phase 2 plan: model choices per case, verification checks (single element, mesh, penalty,
Gauss vs Newton-Cotes, increment size, rho = 0), acceptance criteria. Proposed tolerances
(Vasco decides): single-element energy 1 %, initial compliance 1 %, peak load 2 %,
propagation branch 2 % mean deviation, mesh/penalty sensitivity of the peak < 1 %.
Checkpoint 2: Vasco approves. Phase 3 onward: models through the existing generators and
`scripts/run_case.py`, solved on Vasco's machine.
