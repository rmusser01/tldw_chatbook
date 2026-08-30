---
id: TASK-23195
title: 'Console Context rail: give the rail a title distinct from its collapse control'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-29 21:56'
updated_date: '2026-08-30 01:58'
labels:
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The rail's entire header is a single Button labelled '<---------|Context'. There is no rail title; the word Context exists only as part of the control that collapses it. The literal is hard-coded ASCII art that bypasses the ascii_glyphs fallback system every other Console glyph routes through. Separately, the overflow hint says 'more sections - scroll' without naming what is hidden.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The rail header reads as the rail's name rather than ASCII art (revised - see Implementation Notes)
- [x] #2 The collapse affordance resolves its glyph through resolve_glyph so ASCII mode works
- [x] #3 The overflow hint names the hidden sections
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests: rail exposes a title distinct from the collapse control; collapse glyph routes through resolve_glyph; overflow hint names the hidden sections\n2. Split the rail header into a title Static plus a compact collapse Button\n3. Replace the hard-coded ASCII arrow with a resolve_glyph-resolved affordance\n4. Make _update_outer_hint name the sections below the fold\n5. Capture a screenshot and re-run the rail suites
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The header label went from '<---------|Context' to 'Context ◂', and the overflow hint from 'more sections — scroll' to naming what is below the fold ('▼ Agent · Details · +1').

AC #1 was revised. It originally required a title SEPARATE from the collapse control, and I built that first: a 'Context' Static beside a 3-column affordance, which the stylesheet already described (.console-rail-title at 1fr next to a 3-column .console-rail-collapse-button - the inline widths in compose were overriding it). Three existing tests then failed, and reading them showed why: a previous task deliberately made the whole header one collapse target and pinned clicking anywhere along it, and the Inspector mirrors that shape. Splitting the header would have retired a real affordance (a full-width click target) to fix a labelling problem. On review the reviewer chose to keep one full-width button and fix only the label, which addresses every actual defect the audit found - unreadable name, 18 of 27 columns spent on a decorative arrow, and a hard-coded literal bypassing the ascii_glyphs fallback - while losing nothing. AC #1 now states the outcome delivered.

The hint cannot name every hidden section: '▼ Agent · Details · Character' is 29 cells and the rail is 26. It names as many as fit and counts the rest ('+1'), because the FIRST name is the actionable part - it is what one scroll reaches. My initial version fell back to a bare count when the full list did not fit, which threw away the useful half; the test was relaxed from 'names every hidden section' to 'names at least the first, and is not the old generic string', which is what a 27-column rail can actually honour.

Not mine, verified: test_console_keyboard_trust.py::test_f6_rail_stop_paints_owned_divider_and_restores_on_leave fails identically with left_rail.py at HEAD and with console_rail_state.py reverted to 4da99a884, so it is pre-existing on dev alongside the four already noted in TASK-23193.

The Inspector still reads 'Inspect|--------->'. The two rails now differ; mirroring it is a right_rail.py change that was offered and not taken in this pass.

preflight green. Files: UI/Console_Modules/left_rail.py; Tests/UI/test_console_context_rail_header.py (new); test_console_left_rail.py, test_console_rail_title.py updated.
<!-- SECTION:NOTES:END -->
