# Sharing research between computers

Start each session with `START_HERE.md` and its linked decisions and latest
handoff. Git carries the reviewable research record; large solver files stay
on the computer that produced them or in a separate research archive.

## What belongs in Git

- Source code, builders, extraction/assessment scripts and dependency files.
- `START_HERE.md`, `Prompt0.md`, decisions, plans, handoffs and study reports.
- Reference cards, bibliographic provenance and digitized numerical data.
- Exact `.inp` decks and copied Fortran sources for executed attempts.
- Run/queue manifests with input/source hashes, parameters, commands, solver
  version, timestamps and actual termination state, including failed attempts.
- Extracted CSV/JSON evidence, assessments and generated figures/captions.
- Validation `.sta`, `.msg`, `.log`, execution stdout and extraction-attempt
  logs, so later review can inspect termination and convergence diagnostics.

Create new studies in `validation/<descriptive-study-name>/`. Nested `runs/`
there remain eligible for Git. The root `/runs/` is ignored scratch space.
Commit the scripts and actual evidence together when sharing a result; a
summary alone is insufficient. Keep prepared, running, completed and failed
attempts distinct. Never overwrite completed campaigns or silently retry a
failed run in its old folder. Update the current handoff and study report
before transferring work back.

## What stays outside Git

Solver databases (`.odb`, `.sim`), CAE models, restart/solver intermediates,
verbose `.dat` output, raw IPVALS/UNVALS dumps, VTK exports, binaries, caches,
environment dumps and local credentials remain ignored. Extract the quantities
needed to support conclusions into the versioned CSV/JSON record. If a failure
exists only in a `.dat` file, retain its relevant diagnostic text and provenance
in an execution-attempt text file or assessment JSON before sharing it.

Literature PDFs under `validation/references/` remain local. Generated research
figure PDFs elsewhere are eligible for Git. Transfer the original literature
and any required solver archives separately through an appropriate shared
location. A clone is sufficient for reviewing the retained record; it is not
a complete solver archive and cannot support fresh ODB extraction by itself.
Checks that inventory all original local artifacts may report missing files
in a clone; that is not proof the original archive was lost or modified.

## Transfer and review

Ignore rules do not upload anything. Changes must be committed and pushed on
the originating computer, then fetched/pulled on the other computer. Check
`git status --short` and review the proposed files before committing. Existing
tracked files are unaffected by new ignore rules. Avoid simultaneous edits to
the same study from both machines; use a separate branch/study for independent
work, then review its changes and evidence together.

For the initial transfer, this checkout has a substantial pre-existing public
layout migration and untracked research record. Review and commit that state
deliberately; changing `.gitignore` alone does not publish the project. Do not
reset or clean those files to make the transfer easier.

On the destination, verify Abaqus/licence/compiler availability and local
paths before executing. Check local solver/controller processes independently.
`external_queue_hold.json` is a host-specific safety hold and is ignored; do
not interpret its absence in a clone as permission to overlap a local queue.
Committed queue status and manifests are historical observations, not live
process checks. Absolute paths in old execution commands are provenance and
may need fresh launch commands on the new host. Preserve the recorded inputs,
source hashes and scientific assumptions when making those adjustments.

A future review should inspect the updated handoff, source/deck changes,
actual job statuses, diagnostic logs, extracted data and acceptance criteria.
Unavailable solver databases or papers must remain explicit evidence limits.
