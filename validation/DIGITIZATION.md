# Graphical-data register — 2026-10-25

**Partial extraction.** Seven graphical series and paired experimental-peak transcriptions for five cases are retained. The complete requested figure family is not claimed as digitized. No reference was synthesized from the UEL or shifted to improve agreement.

Graphical CSV comments record source, figure, printed/PDF page, date, method, uncertainty and source hash. [reference_extraction.json](reference_extraction.json) records exact PDF coordinates, linear-axis calibration, series identity and path IDs. The extractor reads vector paths and transforms; it uses circle/square centres for markers.

The black DCB analytical stroke is interrupted by overplotted curves. Of 80 sampling positions, 41 have visible bounded stroke outlines; no missing spans were interpolated. Tiny negative or off-zero coordinates from calibration are preserved. CSV precision does not imply original-data accuracy. Critical point order is retained through snap-back.

The uncertainty estimates below are conservative graphical read-off estimates, not statistical confidence bounds, experimental errors or approved UEL tolerances.

| Source | Retained data | Estimated uncertainty (mm, N) | Limit |
|---|---|---|---|
| Turon 2010 Fig. 4, p. 1860/PDF 4 | 41 visible analytical propagation points | ±0.01, ±0.3 | Formal analytical branch includes a < a0; numerical strength series remain pending |
| Krueger Fig. 10, p. 30/PDF 34 | 32 critical GT = Gc symbols | ±0.01, ±1 | Explicit 128.5 N/1.64 mm annotations are separate source values |
| Krueger Fig. 14, p. 32/PDF 36 | 23 critical symbols | ±0.01, ±2 | Equilibrium points, not a control-specific jump path |
| Krueger Fig. 19, p. 35/PDF 39 | 30 critical symbols | ±0.01, ±3 | Same distinction; no finite-width front information |
| Turon 2018 Fig. 2, p. 509/PDF 4 | 155 new-law tau30/80 square markers | ±0.01, ±1 | One strength series only |
| Turon 2018 Fig. 3, p. 509/PDF 4 | 174 new-law tau30/80 square markers | ±0.01, ±2 | One strength series only |
| Turon 2018 Fig. 4, p. 509/PDF 4 | 167 new-law tau30/80 square markers | ±0.01, ±3 | Main plot only; inset not duplicated |

[Inspect the extracted series](audit/reference_figures_2026-10-25/reference_extracts.svg). These are references, not simulated UEL responses.

## Outstanding requested data

| Requested curves | Status and next step |
|---|---|
| Turon 2010 Fig. 4 numerical series, Fig. 5 ENF, Figs. 6–7 old-law MMB | NOT EXTRACTED as reliable complete series. Filled outlines are fragmented at line/marker overlaps. Each series needs marker/axis registration and a checked overlay. Figs. 6–7 must retain their counterexample role. |
| Turon 2010 Figs. 11–12 | PENDING individual registration of overlapping adjusted-strength families. Fig. 12 also contains 70%. An independent absolute check needs Q1.3 resolved. |
| Krueger Figs. 10, 14, 19 control-specific benchmark paths | PARTIAL: critical symbols extracted; jump trajectories remain separate pending series registration. Sorting/interpolating the critical points is not a substitute. |
| Krueger Figs. 11, 15, 16, 20, 21 versus a* | PENDING independent axes/branch registration. Printed/PDF pages: 31/35; 33/37 for Figs. 15–16; 35/39 for Fig. 20; 36/40 for Fig. 21. |
| Other Turon 2018 strengths, analytical line, old-law counterexample | PARTIAL: only uniquely identified tau30/80 squares accepted. Other markers, legend and inset separation need registration. Missing apparatus prevents independent LEFM comparison. |
| Turon 2006 Figs. 9–13 experimental and numerical traces | NOT EXTRACTED as reliable experimental records. Original specimen data/scatter are absent and the paper explicitly notes sparse propagation measurements (pp. 1084–1085). Visible experimental markers require separate calibration; sparse gaps must not become invented measurements. |
| Turon 2006 Table 3 / Camanho 2002 Table 4 | TRANSCRIBED for all five cases in `data/experimental_peak_sources.csv`, with both identities and no averaging. Table transcription is distinct from curve digitization. |

The PDFs exist. The outstanding tasks concern reliable series identification and scientific definitions, rather than missing filenames. The present extracts do not justify calling the full digitization phase complete.

## Reproduction

Use fresh output paths; existing outputs are preserved:

```console
PYTHONDONTWRITEBYTECODE=1 python validation/scripts/extract_reference_curves.py --out /tmp/cze-reference-recheck
PYTHONDONTWRITEBYTECODE=1 python validation/cases/turon2010CS_dcb/lefm.py --out /tmp/cze-dcb-reference-recheck
PYTHONDONTWRITEBYTECODE=1 python validation/scripts/plot_reference_extracts.py --out /tmp/cze-reference-plots-recheck
```

The extractor writes `<out>/<case>/data/`; the DCB checker and plotter read retained data under `validation/cases/`. Local PDFs are required and hash-checked. Ordinary Python package versions are in [requirements-reference.txt](requirements-reference.txt). No command invokes Abaqus.

## Autonomous campaign additions — 25 October 2026

The original extraction register above is historical. The campaign adds the Turon2010 DCB30MPa numerical curve, ENF60MPa numerical curve, adjusted-strength MMB20/50/80% curves, and five sets of Turon2006 experimental square markers. Together with the original seven series, this gives17 usable graphical series: sixteen structural agreement/reference series and one analytical-transcription check. These are graphical estimates, not raw datasets. Other strengths/earlier-law counterexamples remain unextracted and are not agreement targets.

| Addition | Retained data / metadata | Limit |
|---|---|---|
| Turon2010 Fig4 DCB30MPa |campaign_20261025/reference_data/turon2010CS_dcb/ |RGB raster curve estimate;±0.03mm/±0.8N |
| Turon2010 Fig5 ENF60MPa |campaign_20261025/reference_data/turon2010CS_enf/ |RGB raster curve estimate;±0.04mm/±5N; no jump interpolation |
| Turon2010 Fig12 adjusted-strength MMB |campaign_20261025/reference_data/turon2010CS_mmb_v2/ |Conservative lower drawn-curve envelope; shared analytical/numerical colours;±0.05mm/±3N; initial extraction rejected and labelled |
| Turon2006 Figs9–13 experimental markers |campaign_20261025/reference_data/turon2006_experimental_markers/ |Registered vector square centres; no origin insertion, missing-marker interpolation, raw-record or scatter claim |

Reproduction scripts are extract_dcb_numerical.py, extract_enf_numerical.py, extract_mmb2010_numerical.py and extract_experimental_markers.py under scripts/. Metadata retain PDF hashes, page/figure, axis mapping, colour/path selection, uncertainties and review figures. The first MMB raster attempt carries REJECTED.md and is excluded from all comparisons. Experimental2006/2002 table transcriptions remain separate; the secondary ENF row's different initial crack prevents a matched-specimen error calculation at the primary geometry.
