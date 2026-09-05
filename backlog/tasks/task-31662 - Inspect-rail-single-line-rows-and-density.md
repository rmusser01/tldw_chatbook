---
id: TASK-31662
title: >-
  Inspect rail density: single-line rows, right-aligned secondaries, redundancy removal
status: In Progress
assignee: []
created_date: '2026-09-05 07:00'
labels: [console, inspector, ux, critique-2026-09-05]
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique P1: ConsoleInspectorSectionRow hard-codes 2 lines per row, so at
80x24 the rail shows 8 pinned lines + a 3-line scroll body whose one
visible row restates the header (net-zero information); 25% of the
Environment section's row-lines are blank; a 2-file diff with two
expansions eats 20 lines at 235x52. Measured rail content width is 30
(80x24) to 36 (200x50) columns, not the 34 the summary budget assumes
(TASK-31629 #12/#13). Redundancies: header summary duplicates the branch
row and counts; Tasks header duplicates its only row in different words.
Owner ruling 2026-09-05: the Local row STAYS as designed (do not cut it).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Rows with empty secondary text occupy one line; rows whose secondary fits beside the primary render on one line (secondary right-aligned)
- [x] #2 At 80x24 the Environment section at rest shows its four top-level rows in five lines (header + four one-line rows, measured on the real Console; was eleven lines), with nothing scrolled off WITHIN the section — the rail's own eight-line pinned stack above its three-line scroll body still caps what the RAIL shows at 80x24 and is out of this task's scope (see Implementation Notes)
- [x] #3 The section header's summary is suppressed (or reduced) while the section is open, so open sections never duplicate their own first rows
- [x] #4 The Tasks counts row no longer restates the Tasks header verbatim
- [x] #5 Summary/title budgets derive from the rail's real content width (see TASK-31629 #12/#13) and the widget test pins a width the smallest supported terminal actually produces
<!-- AC:END -->

## Implementation Plan

1. MEASURE first: probe the real Console at 80x24 and 200x50 for the
   section/body/row widths and the rail's scroll geometry, rather than
   inheriting the "~34 columns" claim the current budget rests on.
2. Row geometry in `ConsoleInspectorSectionRow`: empty secondary -> one
   line (Static mounted but hidden, since consumers query it by id); a
   pair that fits the measured row budget -> one line, primary `1fr` +
   secondary `auto` (the header's own split); otherwise the existing
   two-line stack. Line shape joins the section's structural key.
3. Header-summary suppression while open, opt-in per section (the fleet
   section's aggregate summary is not a duplicate of its rows); both
   halves -- `compose` and `set_open`.
4. Tasks head row: keep the row (it is the list's only expand handle and
   an empty Tasks projection hides the whole card), drop the counts
   restatement.
5. Budgets derived from the measured width for both section titles;
   re-pin the widget test at the width the smallest terminal produces.
6. CSS in the component source only, regenerate, and sweep the status
   variants' child combinators for the new nesting.

## Implementation Notes

Rows in `ConsoleInspectorSection` now take one line unless they need two,
the two rail sections stop restating themselves in their own headers, and
every column budget is derived from the rail width that was MEASURED rather
than the one that was assumed.

**Measured first.** A throwaway probe drove the real Console at both sizes
before anything changed: the section box is 30 columns at 80x24 (body 29,
row content 27) and 36 at 200x50 — so the "~34 columns at every size"
comment the old budget rested on described a width no supported terminal
produces, and at the real 30 the title painted "Environm…". The same probe
recorded the shape this task is judged on: four top-level rows at two lines
each, section height 11, inside a rail whose scrollable body is **three**
lines under an eight-line pinned stack (rail header 1 + project-instruction
status 1 + send-authority summary 6) and above a one-line fold hint.

**Row geometry** (`console_inspector_section.py`). Three cases, chosen in
`compose` by `row_fits_one_line(primary, secondary)` against
`SINGLE_LINE_ROW_BUDGET` (27 = the measured row width): an empty secondary
renders one line (the Static stays mounted — consumers query it by id — but
`display: none`); a pair that fits shares one line as `primary 1fr +
secondary auto` inside a `Horizontal`, the header's own title/summary split,
so the secondary lands flush right; a pair that does not fit keeps the old
two-line stack. The fit test is a pure function of the TEXTS against the
narrowest supported rail, not of measured width: the shape has to be decided
before layout runs and has to survive a resize, and the conservative
direction never truncates a pair to force it onto one line. Line shape
joined the section's structural key, so a row that changes shape recomposes
instead of being patched into the wrong DOM.

**Both-halves catches.** (1) The status rules were
`.console-inspector-section-row-error > .console-inspector-section-row-primary`
— in the one-line form the primary is a grandchild, so the child combinator
silently dropped the status colour for exactly the rows that just became one
line. Now descendant selectors, pinned by a test that compares the painted
colour in both shapes and goes red when the `>` is put back. (2) An
empty→non-empty secondary keeps the same shape, so it stays an in-place
patch — `_apply_row_update` therefore has to un-hide the Static, or the new
text renders into a `display: none` widget.

**Header summary while open** is opt-in (`suppress_summary_when_open`), set
by the rail for Environment and Tasks. It is not a property of the grammar:
Environment's "branch ±counts" IS its first two rows, but the fleet
section's "2 working, 1 done" is an aggregate no row of it restates, and
suppressing that would have been a regression. Applied in `compose` and in
`set_open` — the second half is what keeps it correct after a collapse.

**Tasks head row** (AC#4). It read "3 in progress · 12 to do" under a header
reading "3 doing · 12 todo". The counts stay with the header (that is what a
collapsed section shows); the row became "Backlog / N tasks" — the handle
onto the list, saying what the list holds. The row is kept rather than
dropped because it is the list's only expand gesture and because an empty
Tasks projection hides the whole card (which
`test_poll_landing_that_hides_the_tasks_section_falls_back_to_environment_toggle`
depends on).

**Budgets** are now `RAIL_CONTENT_WIDTH_MIN - len(title) - toggle - 1`:
`ENV_SUMMARY_BUDGET` 18 → 15, and the Tasks summary gained the same
treatment at 21 (TASK-31629 #13 — "task-31450 · In Progress" is 24 columns
against a 5-column title). The widget test's harness moved from 34 columns
to 30 and loads the app sheets, since the padding that makes a row 27 wide
lives in the console-owned split sheet.

**Scope note on AC#2.** The section now shows its four rows in five lines
(section height 11 → 7). The rail still cannot show them all at 80x24,
because its scroll body is three lines: the eight pinned lines above it —
six of them the send-authority summary — are what the critique counted, and
nothing in this task's design touches them. That is the next density lever
and it belongs to whoever owns that block (TASK-31663 already has the
send-authority summary in its sights for a different reason).

**Files:** `tldw_chatbook/Widgets/Console/console_inspector_section.py`,
`tldw_chatbook/Chat/console_environment_state.py`,
`tldw_chatbook/UI/Console_Modules/right_rail.py`,
`tldw_chatbook/css/components/_agentic_terminal.tcss` (+ regenerated
`screen_agentic_console.tcss`), `Docs/User_Guide/console/context-and-rag.md`,
and the four test files
(`Tests/UI/test_console_inspector_section.py`,
`Tests/UI/test_console_environment_section.py`,
`Tests/UI/test_console_environment_wiring.py`,
`Tests/Chat/test_console_environment_state.py`).
