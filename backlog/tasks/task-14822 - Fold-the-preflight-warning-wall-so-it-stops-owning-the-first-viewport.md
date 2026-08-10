---
id: TASK-14822
title: >-
  Fold the preflight warning wall so it stops owning the first viewport
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
labels:
  - library
  - ingest
  - ux
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 of the 2026-08-10 re-critique. `LibraryIngestPreflightSummary.compose` emits one `Static` per tooling warning (CSS double-spaces them) followed by one `Button` per distinct install command. A 21-file mixed folder rendered 11 warnings (~22 rows) plus 9 `Copy install command (…)` buttons — roughly 31 rows, the entire 52-row viewport, before the type breakdown, options, metadata or Start appear.

Four of the re-critique's six cognitive-load failures occur at this one block: it is not a single focus, it is not chunked (11 undifferentiated warnings, 9 stacked buttons), it flattens hierarchy (every warning shares `library-ingest-quiet-line` with the lines that actually matter), and it prevents seeing the preflight summary and the Start button together.

The emotional cost is the real damage: the honest reading of eleven amber warnings and nine install buttons is "this app is broken / I must install nine things," when the truth is "3 of your 21 files need optional extras." It also drowns the two lines that DO matter — `5 unsupported files will be skipped` and `1 empty file will fail` — at identical visual weight.

Related mechanical defects in the same block: the buttons are differentiated only by a raw snake_case packaging extra in the label (`Copy install command (mlx_whisper)`), and that suffix disappears entirely when there is exactly one button, so the same control has two label shapes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Tooling warnings collapse to a single summary line stating how many staged files are affected and what it means for the import, with the detail available behind a fold
- [ ] #2 With warnings present, the type breakdown and the Start affordance are reachable without scrolling past a wall of warnings at a supported terminal size
- [ ] #3 The unsupported-file and empty-file lines are visually distinguishable from tooling warnings rather than sharing their weight
- [ ] #4 Install commands remain recoverable, with one combined command available and per-extra commands inside the fold; button labels have one shape regardless of count
<!-- AC:END -->
