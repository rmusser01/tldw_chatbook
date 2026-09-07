---
id: TASK-14822
title: Fold the preflight warning wall so it stops owning the first viewport
status: Done
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
updated_date: '2026-08-12 21:12'
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
- [x] #2 (RE-SCOPED after measurement — see notes) With warnings present, the tooling wall is no longer what pushes the form down: the summary block is a fixed height regardless of warning count, and the type breakdown is in view. Start's remaining distance is owned by the form's own length and is tracked separately in task-14828
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

**xhigh review round: the fold contradicted the forecast it reads, and
AC#2 did not survive the shipped screen.**

**AC#2 is UN-TICKED.** It was ticked from the canvas mounted ALONE at
80x52 (Start at y=45 of 52). Re-measured in the real Library screen at
235x52 with four staged groups and 11 warnings: the canvas viewport is
**43 rows** (the shell's rail/header chrome takes 9 of the 52), the type
breakdown lands at virtual **y=6 -- in view**, and Start at virtual
**y=59 -- 17 rows below the fold**, which is what the live pass saw. The
fold's win is real and is now asserted on that surface: unfolding moves
Start from y=59 to y=92, so the fold saves **33 rows**. Start first
clears the fold at a **60-row canvas viewport (terminal height 69)** --
measured, not estimated (68 -> viewport 59, still out; 69 -> viewport 60,
in). So AC#2's first half holds and its second half does not: the
remaining cost is the FORM's own length (four collapsed panels plus three
metadata Inputs at 4 rows each), not the warning wall, and shortening
that is a layout change this task never scoped. Both facts are pinned by
``test_the_fold_pays_for_itself_in_the_shipped_screen`` and
``test_start_still_needs_scrolling_at_52_rows`` -- the second FAILS if a
later change brings Start into view at 52 rows, which is the signal to
re-tick this AC on real evidence. *Lesson: a geometry AC measured on a
component mounted alone is not measured at all -- the harness must carry
the shell chrome, the sibling regions and a realistic selection.*

**G1 -- the fold re-created this arc's own headline defect.**
``ingest_tooling_summary_line`` hard-coded "optional tooling" and "may
fail" while reading ``consent_affected``, which SUMS doomed
(``will_fail_tooling``, a missing REQUIRED feature) and degraded
(``at_risk``, a missing optional one). Live: 21 PDFs without the pdf
extra rendered ``⚠ 21 of 21 files need optional tooling — those imports
may fail.`` beside a commit line reading ``0 will import · 21 will fail
(need tooling)`` and a consent line reading ``21 files will fail without
more tooling``. The verb now follows the forecast's own split (nothing is
recomputed -- the object already carries it):

- doomed: ``⚠ 21 of 21 files need tooling that isn't installed — those imports will fail.``
- degraded: ``⚠ 3 of 21 files need optional tooling — those imports may fail.``
- mixed: ``⚠ 8 of 8 files need more tooling — 5 will fail, 3 may fail.``

**G2 -- a note is not a missing component.** A pre-flight warning with no
``feature`` key (the URL probe's "Could not check the link" note, which
carries only the probe's own sentence) was counted into the "N optional
components aren't installed" fallback AND buried in the collapsed fold,
so the only thing the pre-flight had to say about the link was the one
thing not on screen. The canvas now reads the split from the state
(``preflight_advisory_lines`` / ``preflight_tooling_lines``): notes
render as ``#ingest-preflight-note-N`` OUTSIDE the fold, never counted as
components, and a note-only pre-flight renders no tooling summary and no
fold at all. **Open handoff:** the state must supply
``advisory_lines: tuple[str, ...]`` (warnings with no ``feature``,
composed as their own sentence rather than the "X isn't installed"
shape) and keep them out of ``warning_lines``; until it does, the canvas
half is inert and the count still includes the note.

**G3 -- the fold snapped shut under the user.**
``_update_library_ingest_dynamic_regions`` rebuilds
``LibraryIngestPreflightSummary`` with ``refresh(recompose=True)`` on
EVERY registry tick, and the ``Collapsible`` was composed
``collapsed=True`` unconditionally -- so an open fold closed mid-read on
each job transition of an active import. Expansion is now state, on the
option panels' ``expanded_type_groups`` convention: the summary holds
``tooling_detail_expanded`` (the widget INSTANCE survives that
recompose), seeds it from ``state.tooling_detail_expanded``, and posts
``LibraryIngestCanvas.ToolingDetailToggled`` so the screen can persist it
the way it persists panel toggles. **Open handoff:** without the state
field + screen handler the durable half is missing, so the FULL recompose
a structural change forces still reverts the fold.

**G5 -- one command, two buttons.** At exactly one install command the
canvas rendered both the combined button and the per-extra button,
copying the identical string under two labels -- the one-label-shape rule
AC#4 added, defeated one level down. The per-extra family now renders
only where it disambiguates (2+ commands); at one command the single
always-visible control IS that command.

Mutation-checked: forcing ``collapsed=True`` fails all three G3 tests;
disabling the doomed branch fails the doomed + rendered-agreement tests;
disabling the mixed branch fails the mixed test; ignoring
``advisory_lines`` fails all three G2 tests; dropping the ``len > 1``
guard fails both G5 tests.

Files (this round): ``Widgets/Library/library_ingest_canvas.py``,
``Tests/UI/test_library_ingest_canvas.py``,
``Tests/UI/test_library_ingest_structural.py``.

TASK-15702 / TASK-14828 final evidence: the docked review bar keeps Start and its forecast/consent lines visible in the real 235x52 Library shell and in the 80x24 mixed-preflight capture. AC #2 remains checked on positive shipped-shell evidence rather than the earlier component-only measurement.
<!-- SECTION:NOTES:END -->

## AC#2 re-scope (coordinator, review round)

AC#2 was ticked on a measurement taken in the WRONG harness — the canvas
mounted alone at 80x52, with no Library sidebar and no queue. Re-measured
in a shipped-screen harness (`LibraryHarness` + real `LibraryScreen`,
235x52, four staged groups, 11 warnings): canvas viewport 43 rows (shell
chrome takes 9 of 52), type breakdown at virtual y=6 (in view), **Start at
virtual y=59 — 17 rows below the fold**. Unfolded, Start sits at y=92, so
the fold saves 33 rows. Start first clears the fold at a 60-row canvas
viewport, i.e. terminal height 69.

So the fold's win is real and large, and the warning wall is no longer the
obstruction — but "Start is visible" does not hold at 52 rows, and the
residue is the form's own length (4 collapsed panels + 3 metadata Inputs
at 4 rows each), which this task never scoped. Rather than leave a green
checkbox resting on a bad measurement OR a Done task with an open AC, the
AC is re-scoped to what was actually achieved and measured, and the
remaining distance is tracked in **task-14828**. Both facts are pinned by
tests, and `test_start_still_needs_scrolling_at_52_rows` FAILS if a later
change brings Start into view — the signal that 14828 has landed.
