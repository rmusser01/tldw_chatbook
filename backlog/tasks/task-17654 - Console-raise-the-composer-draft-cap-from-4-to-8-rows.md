---
id: TASK-17654
title: 'Console: raise the composer draft cap from 4 to 8 rows'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-17'
labels:
  - console
  - ux
dependencies:
  - task-17651
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner follow-up from the 2026-08-17 bottom-stack decisions: after the composer flattens to 1-4 total rows (TASK-17651), raise the visible draft capacity to 8 rows for long prompts. This is deliberately separate because three things must move together: `MAX_DRAFT_ROWS`, the draft Static's `max-height`, and the viewport-window slicing logic (`_wrap_draft_line_slices`) that keeps the caret visible — changing only the constant would break the draft windowing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The draft area can display up to 8 rows and the composer's total height never exceeds 8 rows plus its (zero) chrome
- [x] #2 Draft windowing and caret visibility behave correctly at every draft height up to the new cap, including paste-collapse and ghost text
- [x] #3 Geometry pins are updated; growth stays demand-driven (the composer returns to 1 row as the draft shrinks), so a max-height draft borrows transcript space only while it exists
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: raise the growth pin (huge draft -> exactly 8) and the 1-8 bounds across the collapse/internals suites; watched fail at 4.
2. `MAX_DRAFT_ROWS` 4 -> 8; the two stylesheet caps (`#console-native-composer`, `#console-command-visible-text`) move with it; bundle rebuild.
3. The overflow/windowing suite gets a `_WINDOWED_BOUNDARY_TEXT` fixture (hard-newline pad rows keep the original fixture's exact wrap boundaries byte-identical) so its overflow premises exceed the larger cap; live probe.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The whole draft pipeline — growth clamp, caret windowing, tail bias, head elision — derives from the single `MAX_DRAFT_ROWS` constant, so the change is the constant plus the two stylesheet caps that mirror it (commented to move together). AC#3 was amended before implementing: the transcript-priority guard originally written there is unnecessary because growth is demand-driven — the composer borrows rows only while a long draft exists and returns to 1 as it empties (live-verified: region 31 -> 24 -> 31 at 150x44).

Test work: geometry pins 1-4 -> 1-8 across the collapse and internals suites (RED-first on the 8-row growth); the overflow suite's carefully built wrap-boundary fixtures no longer overflowed the cap, so windowing tests moved to a `_WINDOWED_BOUNDARY_TEXT` variant (five hard-newline pad rows — a newline restarts wrapping, keeping every downstream boundary of the original fixture byte-identical for the exact-width assertions that still use it); the ZWJ counterexample's lead tripled. 103 composer tests + 50 internals composer/paste tests green; head-elision (`... `) verified live at 8 rows.

Files: `tldw_chatbook/Widgets/Console/console_composer_bar.py`, `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ bundle), `Tests/UI/test_console_composer_collapse.py`, `Tests/UI/test_console_internals_decomposition.py`, `Tests/UI/test_console_composer_overflow.py`, `Docs/User_Guide/console.md`.
<!-- SECTION:NOTES:END -->
