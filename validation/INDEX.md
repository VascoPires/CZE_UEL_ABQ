# Verification and validation —2026-10-25

Start with [REPORT.md](REPORT.md), [PLAN.md](PLAN.md), [DECISIONS.md](DECISIONS.md), and the [campaign record](campaign_20261025/INVESTIGATION.md). The autonomous campaign has executed actual Abaqus jobs. The canonical UEL and original examples are preserved. Queue manifests under `campaign_20261025/` are the live execution authority. The reference cards below retain their original audit status text; each card links to its generated execution_result.json and the current report.

| Case | Role | Questions |
|---|---|---|
| [turon2010CS_dcb](cases/turon2010CS_dcb/reference_card.md) | numerical/analytical verification | Q1.1–Q1.3 |
| [turon2010CS_enf](cases/turon2010CS_enf/reference_card.md) | numerical/analytical verification | Q1.1–Q1.3 |
| [turon2010CS_mmb20](cases/turon2010CS_mmb20/reference_card.md) | numerical/analytical verification | Q1.1–Q1.3 |
| [turon2010CS_mmb50](cases/turon2010CS_mmb50/reference_card.md) | numerical/analytical verification | Q1.1–Q1.3 |
| [turon2010CS_mmb80](cases/turon2010CS_mmb80/reference_card.md) | numerical/analytical verification | Q1.1–Q1.3 |
| [krueger2012CR_mmb20](cases/krueger2012CR_mmb20/reference_card.md) | VCCT numerical benchmark | Q1.4–Q1.5 |
| [krueger2012CR_mmb50](cases/krueger2012CR_mmb50/reference_card.md) | VCCT numerical benchmark | Q1.4–Q1.5 |
| [krueger2012CR_mmb80](cases/krueger2012CR_mmb80/reference_card.md) | VCCT numerical benchmark | Q1.4–Q1.5 |
| [turon2006MM_exp_dcb](cases/turon2006MM_exp_dcb/reference_card.md) | provisional experimental validation target | Q1.6,Q1.9 |
| [turon2006MM_exp_mmb20](cases/turon2006MM_exp_mmb20/reference_card.md) | provisional experimental validation target | Q1.6,Q1.9 |
| [turon2006MM_exp_mmb50](cases/turon2006MM_exp_mmb50/reference_card.md) | provisional experimental validation target | Q1.6,Q1.9 |
| [turon2006MM_exp_mmb80](cases/turon2006MM_exp_mmb80/reference_card.md) | provisional experimental validation target | Q1.6,Q1.9 |
| [turon2006MM_exp_enf](cases/turon2006MM_exp_enf/reference_card.md) | provisional experimental validation target | Q1.6,Q1.9 |
| [turon2018CS_mmb20](cases/turon2018CS_mmb20/reference_card.md) | internal constitutive-formulation check;appendix at most | Q1.7 |
| [turon2018CS_mmb50](cases/turon2018CS_mmb50/reference_card.md) | internal constitutive-formulation check;appendix at most | Q1.7 |
| [turon2018CS_mmb70](cases/turon2018CS_mmb70/reference_card.md) | internal constitutive-formulation check;appendix at most | Q1.7 |
| [element](cases/element/reference_card.md) | constitutive verification, before DCB | Checkpoint 2 |

Shared source cards: [Turon2010](sources/turon2010CS.md), [Krueger2012](sources/krueger2012CR.md), [Turon2006](sources/turon2006MM.md), [Camanho2002](sources/camanho2002TM.md), [Turon2018](sources/turon2018CS.md). `parameters.json` files carry individual source classifications and null values for missing inputs. They are not approved model configurations.
