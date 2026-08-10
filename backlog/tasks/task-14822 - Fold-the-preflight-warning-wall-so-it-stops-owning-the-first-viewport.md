---
id: TASK-14822
title: Fold the preflight warning wall so it stops owning the first viewport
status: Done
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
updated_date: '2026-08-10 21:42'
labels:
  - library
  - ingest
  - ux
dependencies: []
priority: high
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
- [x] #1 Tooling warnings collapse to a single summary line stating how many staged files are affected and what it means for the import, with the detail available behind a fold
- [x] #2 With warnings present, the type breakdown and the Start affordance are reachable without scrolling past a wall of warnings at a supported terminal size
- [x] #3 The unsupported-file and empty-file lines are visually distinguishable from tooling warnings rather than sharing their weight
- [x] #4 Install commands remain recoverable, with one combined command available and per-extra commands inside the fold; button labels have one shape regardless of count
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the live evidence: 11 warning ``Static``s (CSS double-spaced) + 9 stacked copy buttons ~= 31 rows of a 52-row viewport, ahead of everything that matters.
2. RED first: mount the canvas with 11 warnings and assert (a) ONE canvas-level summary line, (b) the per-warning detail inside a collapsed ``Collapsible``, (c) the summary block's height is identical at 2 and at 11 warnings, (d) the type breakdown and Start are both inside the viewport at 80x52.
3. Fold: one ``⚠`` summary line + one combined-command copy button OUTSIDE the fold; warnings and per-extra copy buttons inside it.
4. Take the affected-FILE count from task-14820's single ``IngestForecast`` (``consent_affected``); degrade the wording rather than invent a count when no forecast is present.
5. Weight: move the unsupported/empty lines off ``library-ingest-quiet-line`` onto their own outcome class, and assert the painted style differs under the real stylesheet.
6. One label shape for the per-extra copy buttons at any count.
7. Mutation-check the fold (unfold it -> both geometry tests must fail).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Folded the tooling-warning wall into one line plus a ``Collapsible``.

**Shape.** ``LibraryIngestPreflightSummary.compose`` now emits, at canvas
level: ``#ingest-preflight-tooling-summary`` (one ``⚠`` sentence), one
``#ingest-preflight-copy-all-commands`` button, then a collapsed
``Collapsible#ingest-preflight-tooling-detail`` holding the per-warning
``Static``s (ids unchanged) and the per-extra copy buttons. Measured
under the real stylesheet at 80x52: the summary block is **14 rows at 2
warnings and 14 rows at 11** (it was ~31 at 11), the type breakdown lands
at y=17 and **Start at y=45 -- both in view**. Mutation check: setting the
fold ``collapsed=False`` fails both geometry tests.

**The count comes from ONE place.** ``ingest_tooling_summary_line`` reads
``state.forecast.consent_affected`` -- task-14820's single forecast -- and
nothing else. With no forecast it says "N optional components aren't
installed" rather than inventing a file count; a second independently
derived count is the exact P1 this arc exists to remove. *Open handoff:
``LibraryIngestCanvasState`` does not yet carry the forecast, so the
degraded wording is what renders today (see the report to task-14820).*

**Combined command.** ``combined_install_command`` folds the union of the
extras into one ``pip install -e ".[a,b,c]"``; anything not matching that
shape is chained with ``&&`` rather than silently rewritten. The per-extra
labels now always carry their ``(extra)`` suffix
(``install_command_button_label``) -- the suffix used to vanish at exactly
one command, giving one control two label shapes.

**Weight.** ``#ingest-unsupported-summary`` / ``#ingest-empty-summary``
moved to ``.library-ingest-outcome-line`` (bold, ``$ds-text-primary``);
the tooling summary keeps the muted weight. Asserted from the
compositor's own painted styles, not from declared CSS.

Files: ``Widgets/Library/library_ingest_canvas.py``,
``css/components/_agentic_terminal.tcss`` (+ rebuilt bundle),
``Tests/UI/test_library_ingest_canvas.py``,
``Tests/UI/test_library_ingest_structural.py``,
``Docs/User_Guide/library/import-and-export.md``.
<!-- SECTION:NOTES:END -->
