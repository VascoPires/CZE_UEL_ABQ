# Execution notes — 2026-10-25

No Abaqus, CAE, compiler, datacheck or model-generation action was performed. Commands inspected files/platform state and created validation artifacts.

An initial metadata probe used the distribution name PIL instead of Pillow and failed after the baseline inventory and PDF text extraction had finished. Metadata was re-read correctly. No research input changed.

The first graphical extractor treated disconnected filled outlines as one stroke. Its geometry guard rejected the result before any CSV was written. After subpaths were separated, the guard exposed actual occlusion gaps. The final extractor retains visible bounded stroke samples without filling those gaps. An analytical checker started before extraction had completed and failed without creating output; it was rerun after valid CSVs existed. These were tooling failures, not failed UEL verification cases.

A possible initial-gap issue was investigated instead of reported as a bug. Builders collapse the interface surfaces; the first paired nodes in inspected DCB/MMB decks coincide. This does not verify every element in a future build.

The final integrity checks cover baseline file hashes, local Markdown links, JSON/Python/CSV parsing and reference extraction reproduced into fresh temporary folders. Physical verification remains NOT RUN; the DCB calculation is a source-equation transcription check only.
