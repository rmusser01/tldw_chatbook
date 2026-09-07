---
id: TASK-17166
title: Console text selection phase 1
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-15 04:39'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mouse text selection in Console transcript with stacked menu and Add to chat
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mouse drag selects text in plain rows
- [x] #2 Markdown rows selectable at line granularity
- [x] #3 Menu appears at release cell with Add to chat
- [x] #4 Add to chat inserts quote at composer caret
- [x] #5 Click vs drag disambiguated
- [x] #6 Streaming rows clamp selection
- [x] #7 Tests green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Per Docs/superpowers/plans/2026-08-14-console-selection-phase1.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADR: backlog/decisions/068-console-text-selection-and-annotations.md. Plan: Docs/superpowers/plans/2026-08-14-console-selection-phase1.md. Spec: Docs/superpowers/specs/2026-08-14-console-selection-annotations-design.md.

Phase 1 implemented across commits 99b204c64..(this task) on feat/console-selection-dev:

- **Selection core** (`tldw_chatbook/Widgets/Console/console_selection.py`): pure-logic `SelectionManager` (begin/extend/finish/cancel, single-row clamp), `TextSelection`/`SelectionState`, `cap_quote` (4000-char cap with truncation marker), `offset_for_cell`.
- **Plain rows** (`ConsoleTranscriptMessage`): four-method selection protocol over the plain body text (`get_display_text`/`get_selection_text`/`set_selection_range`/`clear_selection`), reverse-video rich span highlight on the body Static, clamp-on-sync for streaming bodies.
- **Drag wiring** (`ConsoleTranscript`): MouseDown/Move/Up handlers with mouse capture, wrap-aware cell→offset mapping (`_body_cell_to_offset`), drag-release click suppression, reconciliation cancel on row removal/rebuild.
- **Floating menu** (`console_selection_menu.py`): overlay-docked `ConsoleSelectionMenu` anchored at the release cell (jump-pill pattern), Escape/click-outside dismissal, `AddToChat` → `ConsoleSelectionQuoteRequested`; screen-level click-outside dismissal in `chat_screen.py`.
- **Composer** (`console_composer_bar.py`): `insert_quote` splices a `> `-prefixed block at the caret (end-of-draft when unfocused); ChatScreen routes `ConsoleSelectionQuoteRequested` into it.
- **Markdown rows, line granularity (this task)**: `ConsoleMarkdownMessage` implements the same protocol with offsets over the markdown SOURCE (`_body_text`): `set_selection_range` snaps outward to whole `'\n'`-delimited lines (`_snap_to_line_bounds`); the highlight is a reverse-video `Static` strip composed below the Markdown widget (`.console-markdown-selection-strip`, display-toggled, never mounted/removed at runtime) instead of restyling Markdown internals; clamp-on-sync mirrors the plain row in both streaming-append and body-replace paths. `ConsoleTranscript` accepts markdown rows in `_selection_row_for`, maps cells to source lines via `_markdown_cell_to_offset` (body-local y distributed evenly across source lines, nearest-line clamp — the Markdown renderer does not expose per-source-line layout; the last rendered row maps to the last source line so collapsed soft-wrapped paragraphs stay reachable; recorded phase-1 approximation in ADR-068), and `Add to chat`/clear/cancel paths handle both row kinds.

**Tests** (all green, 67 passed): `Tests/UI/test_console_selection_core.py`, `_rows.py` (plain + markdown protocol incl. line snap, strip, clamp), `_transcript.py` (drag wiring incl. markdown arm/suppress), `_menu.py` (anchoring, dismissal, markdown drag → Add to chat quoting whole lines), `_end_to_end.py` (quote routing + click-outside), and `_app_smoke.py` (real ChatScreen via the production `make_console_pilot` harness: plain drag → menu → Add to chat lands `"> ..."` at the composer caret; markdown drag quotes whole lines; plain click toggles selection with no menu).

**Baselines (pre-existing, verified unchanged before/after)**: `test_console_native_transcript.py` 3 failures (action-row speak tests), `test_console_native_chat_flow.py` 1 failure (inline image row), `test_console_transcript_markdown_widget.py` 4 failures (`ConsoleMessageHeader.renderable` API drift). Not touched by this branch.

**Live-terminal-only verification outstanding** (cannot be exercised under `App.run_test()`): shift-drag terminal-native copy coexisting with our mouse capture; Escape/click-outside dismissal feel in a real terminal; a selection surviving a real streaming reply's sync tick (covered by clamp-on-sync unit tests instead). These need the manual live spike per `backlog/docs/lessons-live-verification.md` before closing the phase.

**Deviations from plan**: the plan's task-G step suggested re-rendering selected lines inside the Markdown body; implemented as the strip below the body per the review-blessed reference approach (avoids fighting the Markdown renderer). Menu dismissal after Add to chat restores composer focus via the menu's unmount seam. Intended markdown semantics (recorded in ADR-068 Consequences): a streaming append that grows the source re-snaps the line range, so a selection touching the last line GROWS with the stream; plain rows hold the last stable range.
<!-- SECTION:NOTES:END -->

## Live spike result

Live spike PASSED 2026-08-15 (user-verified in kitty after three fix rounds): drag select, menu, all three actions, keyboard nav, dismissal.
