# Research continuation prompt

Tell the agent: **Read `CONTINUE_RESEARCH.md` and follow its instructions.**

Continue the CZE_UEL_ABQ research in this checkout.

First read `START_HERE.md`, its linked current handoff and decisions,
`Prompt0.md`, and `docs/research-handoff.md`. Inspect the actual files and
current machine state; do not assume historical queue status is live.

Preserve the completed campaign and canonical UEL. Continue the matched
2D/3D elastic fixed-crack study in `validation/elastic_dimensions_20261026`,
including the finite-width mesh sensitivity comparison. Follow any later
user decisions recorded in the current handoff rather than restarting work
already completed.

Before execution:

- Verify Abaqus, licence access, Python and local launch paths.
- Check for active solvers/controllers on this computer.
- Inspect actual run manifests and available outputs.
- Identify missing ignored artifacts or literature PDFs explicitly.
- Review the host-specific submission hold; do not copy another computer's
  process state or remove safeguards blindly.

Run one Abaqus solver at a time. Proceed autonomously with routine numerical
work under `Prompt0.md`. Preserve every completed or failed attempt; use fresh
folders for retries. Do not tune physical parameters to match reference curves
or change the constitutive law.

Compare compliance, VCCT and compliance-derivative energy release, element
formulation, dimensional state and mesh sensitivity. Distinguish solver
completion, numerical verification and physical validation.

Keep exact inputs, source hashes, execution commands, termination logs,
extracted CSV/JSON evidence, figures and assessments eligible for Git,
following `docs/research-handoff.md`. Keep large solver artifacts locally.

Update `START_HERE.md`, the current handoff and the study report with verified
results, limitations and the exact next step, so another agent on another
computer can review and continue the work.

Do not commit or push automatically. When finished, summarize the changes and
identify the files that need to be shared through Git.

Begin by briefly confirming the current state, missing prerequisites and the
next executable task, then proceed.
