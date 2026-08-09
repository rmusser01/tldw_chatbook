---
id: TASK-3302
title: >-
  Ingest mode keyboard & focus: entry focus, Esc/i routes, F1/footer, focus visibility
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 19:30'
labels:
  - library
  - ingest
  - ux
  - accessibility
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Findings MI-03/04/05 (P1) of the 2026-08-07 Media Ingestion review. Three keyboard-first failures, all confirmed live: (1) entering Ingest mode leaves focus in the rail search box — typing a path runs a Library search (`_WORKBENCH_FOCUS_TARGETS` prefers a landing-only target); (2) there is no keyboard exit — Esc is bound only to skills-back, `i` is landing-scoped, the footer shows only generic hints, and F1 lists the screen's raw Skills BINDINGS (same BINDINGS-only-sourcing trap the Library P1 arc fixed for the footer — share the per-mode helper); (3) focus is structurally invisible on the path/title/author/keywords inputs and every compact button (`library-canvas-action`) — Tab produced byte-identical plain-text panes in two independent live walks, while panel fields show a heavy border. Color-only focus fails monochrome/low-contrast themes and the design system's three-signal dense-form convention.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Activating Ingest mode places focus on the path field; typing immediately edits the path (regression test)
- [x] #2 Esc in Ingest returns to the Library browse/hub surface; `i` enters Ingest from any Library canvas
- [x] #3 The Ingest footer and F1 help show the same per-mode ingest shortcuts (Enter start, Esc back, at minimum) from a shared source
- [x] #4 Focused top-level ingest fields and compact action buttons show a structural (glyph-level, not color-only) focus indicator, asserted via plain-text render diff in a CSS-true harness
- [x] #5 No focus/blur layout shift (dimensional stability preserved)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED tests: entry-focus assertion; Esc-in-ingest route; F1 content per mode; plain-text focus diff on a `library-canvas-action` button.
2. Add ingest-mode preferred focus target; per-mode escape handling; extend the footer/F1 shared shortcut helper with an ingest set.
3. Focus treatment: `outline: heavy $accent` (or heavy edge) on `.library-ingest-field:focus` and `.library-canvas-action:focus` in the app TCSS (rebuild bundle via build_css.py); keep DEFAULT_CSS parse-standalone.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD: 8 of 10 new tests (`Tests/UI/test_library_ingest_keyboard.py`) went
RED first on the exact live defects — "entering Ingest never focused
#library-ingest-path" (both entry routes), "#library-hub-action-import
never mounted" (Esc dead), "#library-ingest-path never mounted" (`i`
inert off-landing), footer text mismatch, `check_action` fall-through
`True`, and — for both the field and the button — "focus produced a
byte-identical plain-text pane", reproducing MI-05's live capture
byte-for-byte. All 10 green after the fix; three mutation checks
(field CSS rule → RED, button CSS rule → RED, entry-focus
`call_after_refresh` → RED) confirmed each guard bites.

Root causes found, not guessed:
- MI-05 fields: `.library-ingest-field:focus` (specificity 0,2,0) set
  `border: tall` color-swap + `outline: none`, silently BEATING the
  task-2014 `LibraryIngestCanvas Input:focus { outline: heavy $accent }`
  rule (0,1,2) — which is why the PANEL option inputs showed a heavy
  edge and the four top-level fields did not. Fix: the field's own
  `:focus` now carries `outline: heavy $ds-input-focus-accent`.
- MI-05 buttons: `Button:focus`/`Button:hover:focus` in
  `_buttons.tcss` pin `outline: none` with underline/reverse cues that
  vanish in a plain-text pane. Fix: `Button.library-canvas-action:focus
  { outline-left/right: heavy $ds-focus-accent }` — SIDE rails only,
  because a full `outline: heavy` overwrites a 1-row compact button's
  only row (the task-2041 label-eating trap, re-asserted by the test:
  the label must survive focus). Wins on (0,2,1) + later-in-bundle over
  both `_buttons.tcss` rules. No dimensional change either way
  (outline paints over the widget's own edge cells).
- MI-03: `_select_library_rail_row_after_source_admission` now
  `call_after_refresh`-focuses the path field on the INGEST_MEDIA row
  (the CREATE_PROMPT/CREATE_SKILL entry-focus seam); the
  `_WORKBENCH_FOCUS_TARGETS` canvas pane gained `library-ingest-path`
  as an ordered fallback so F6 stops skipping the Ingest canvas.
- MI-04: a seventh gated `escape` binding (`library_ingest_back`,
  `check_action` = row == INGEST_MEDIA, disjoint from the other six)
  routes back through `_select_library_rail_row("")` and focuses the
  hub's first action (not the search box, which would swallow the
  landing accelerators). `i` widened to any Library canvas (still
  guarded by the Input/TextArea early-return; `n` stays landing-scoped
  since it opens a create editor). New `LIBRARY_INGEST_SHORTCUTS`
  (`/`, F6, "enter start import", "esc back to hub") feeds footer AND
  F1 through the existing shared
  `_library_footer_shortcuts_for_current_state` helper (task-2858's
  single-source rule); the F1 test asserts identity (`is`), not
  equality, so the two surfaces cannot drift.

Files: `tldw_chatbook/UI/Screens/library_screen.py`,
`tldw_chatbook/css/components/_agentic_terminal.tcss` (+ regenerated
`tldw_chatbook/css/tldw_cli_modular.tcss` via build_css.py),
`Tests/UI/test_library_ingest_keyboard.py` (new, 10 tests; the render
assertions use `render_lines()` — the StylesCache layer where
border/outline glyphs exist; `render_line()` is content-only and would
miss them), `Docs/User_Guide/library/import-and-export.md` (Keyboard &
commands rewritten + task-3302 stamp).

Adjacent suites: guardrail modal + ingest flow 23 passed; footer hints
+ workbench help 21 passed; screen navigation 114 passed (includes the
BINDINGS-audit test the new binding must satisfy); footer context 10
passed. Pre-existing failures NOT touched by this task, verified
against unmodified files/lines: `test_landing_footer_advertises_the_
landing_keyboard_story` (stale vs the footer's global-cluster
rendering), 3 `test_library_screen.py` bare-fixture tests
(`object.__new__(LibraryScreen)` missing `__init__` attrs at
task-2043-era lines), `test_shared_form_and_native_inputs_use_thin_
non_semantic_focus` (stale vs TASK-2300's documented `Select:focus`
border removal).
<!-- SECTION:NOTES:END -->
