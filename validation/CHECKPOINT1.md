# Checkpoint 1 — researcher review

Prepared 2026-10-25. **Stop before Phase 2.** Seventeen case cards, five shared source cards and the audit are ready for review. Graphical extraction is partial. No Abaqus command or model builder was executed. Decisions D1–D5 remain unchanged; no proposed answer below has been entered as an approved decision.

Review the [case index](INDEX.md), [report](REPORT.md), [source audit](SOURCE_AUDIT.md), [environment](ENVIRONMENT.md) and [digitization register](DIGITIZATION.md).

### Q1.1 — Turon 2010 agreement targets

**Recommendation:** use LEFM and the adjusted-strength curves in Figs. 11–12 as targets. Keep Figs. 6–7 as diagnostic counterexamples: the paper deliberately shows erroneous mixed-mode propagation from the earlier constant-penalty law (§§3–4, pp. 1860–1862). The current UEL implements the updated modal-penalty relation. Approve this distinction, or identify a separate earlier-law reproduction to retain. The previously selected benchmark family stays unchanged.

### Q1.2 — Plane stress or plane strain in Turon 2010

**Recommendation:** follow the named CPE4 element as plane strain, while recording the contradictory “plane stress” wording in §3, p. 1859. The alternative is to follow the text and use a plane-stress reference. Official Abaqus definitions settle the element name, but cannot establish what the authors actually ran. Either interpretation requires later evidence that the constrained 3D U8 model represents the intended state.

### Q1.3 — Span and analytical mode fraction in Turon 2010

**Recommendation:** approve L = 75 mm as an explicit interpretation of Fig. 3 for a centred 150-mm specimen. The figure shows L from midpoint to support, whereas Appendix A2 calls it the specimen length. A numerical support span is not explicitly supplied. Using 150 instead of 75 mm changes ENF/MMB compliance.

Also approve reporting the LEFM energy fraction derived from A7 as a function of crack length, separately from nominal mixity. Decide whether the toughness in the A5 comparison should use that fraction or remain fixed at the nominal value. The implemented functions require an explicit span and policy; neither has been silently selected or evaluated.

### Q1.4 — Krueger 2D and finite-width 3D comparisons

Both 2D and 3D comparisons are already selected in D2. The unresolved choice is
whether a constrained 3D U8 slice may represent the 2D cohesive calculation.
**Recommendation:** allow that construction subject to explicit plane-strain
equivalence checks, while keeping the finite-width 3D benchmark separate. A
width-scaled 2-mm slice cannot automatically replace the latter. Krueger uses
CPE4I/C3D8I and rigid levers with MPCs; the current generators use C3D8R, ties
and a forced 2-mm width. Approve the slice representation, or require a different
2D implementation. The later plan will justify constraints, elements and observables.

### Q1.5 — Cohesive properties absent from the VCCT reference

Krueger supplies fracture energies and eta, but no cohesive stiffness or strengths. The repository values K3 = 170000 N/mm³, normal strength 32 MPa and shear strength 61 MPa are existing model choices.

**Recommendation:** classify them as numerical-study inputs, and choose the documented ranges at Checkpoint 2 while holding the published energies fixed. They must not be called Krueger material inputs or fitted to the critical load. Approve that classification, or supply an independent source for the strengths and penalty.

### Q1.6 — Primary AS4/PEEK experimental record

**Recommendation:** use Turon 2006 Table 3 as the primary published observation record and Camanho 2002 Table 4 as a separate cross-check, pending original records. ENF peaks are 748.4 versus 733.96 N; ENF initial cracks are 39.2 versus 39.3 mm. DCB peaks also differ: 147.5 versus 147.11 N. Both sets are retained. Approve the primary record, or choose the companion/original source. Do not average or combine the records.

### Q1.7 — Reconstruction of the incomplete 2018 apparatus

**Recommendation:** hold the absolute LEFM comparison until you approve an explicitly labelled reconstruction from Camanho 2002/Turon 2006, or obtain the missing model details. Turon 2018 does not specify the lever length, support span, load introduction, crack initialization or NLGEOM settings. A companion-based reconstruction would be a hybrid assumption. Its 32.64-mm initial crack and eta = 2 must remain distinct from the 2006 inputs. The extracted 20/50/70% reference curves remain usable as graphical records.

### Q1.8 — Historical MMB curve and output provenance

**Recommendation:** exclude `benchmark_turon.csv` and its backup from quantitative verification; use fresh source-traceable references. Preserve both old files. Every non-origin displacement in the active copy is 0.1 mm below the backup, with no recorded reason. Plotters use c = 41.3 mm while the active configuration uses 92.9 mm.

If available, provide the original publication/figure, digitization project, source job or raw force/displacement definitions. Otherwise approve proceeding without these curves as targets. No re-alignment or material fitting is proposed.

### Q1.9 — Provisional experimental evidence

**Recommendation:** begin with published peaks as provisional comparisons and seek original Reeder/Crews traces and scatter before claiming independent experimental validation. Eta = 2.284 was fitted to the toughness observations; at 20%, measured Gc = 1.103 N/mm differs from the fitted 0.988 N/mm. The later plan must distinguish numerical error, input provenance and model discrepancy. The prompt's proposed 2% tolerance is not automatically an experimental acceptance limit. Approve this provisional scope, or require original records first.

Reply with the question numbers and any changes. Answers will be appended as dated entries in DECISIONS.md. Applicable questions must be resolved before the Phase 2 plan; that plan needs separate Checkpoint 2 approval before any numerical analysis.

The stop is required by [Prompt0.md](../Prompt0.md): “Do not proceed to Phase 2 until Checkpoint 1 is resolved.” It is not an environment failure.
