---
id: TASK-15110
title: >-
  Console context rail: Conversations starts below the fold with all sections open
status: Done
assignee: []
created_date: '2026-08-11 04:00'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured while repairing task-14920, and raised rather than guessed at.

TASK-14810 split the Console context rail into sections. The split itself is correct and the rows are genuinely reachable — but with all three sections expanded by default, `#console-left-rail-body` has a virtual height of **99 rows against a 29-row viewport** at 160x48: the Conversations section body starts at y=45 and its first row sits at y=70, roughly 20 rows below the fold. Reaching it needs a scroll the user has no cue to perform.

This surfaced because 12 tests began failing with `textual.pilot.OutOfBounds` — `pilot.click` addresses screen coordinates, so a target below the fold reports a coordinate error rather than the layout fact that caused it. Those tests were repaired by scrolling first (the honest test fix, since the rows do work once visible), which means **nothing now fails if the rail grows further**: the shipped tests for TASK-14810 assert section order and independent collapse, never on-screen reachability.

So this is a discoverability question for the owner, not a regression: should all three sections default to open when the third lands off-screen, should the rail remember collapse state, or should sections default collapsed below some height? Whatever is chosen, a test that pins reachability would stop the next growth from being invisible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 An owner ruling is recorded on the default expansion of the rail's sections at common terminal sizes
- [x] #2 The chosen behaviour is implemented, and the first row of every default-visible section is reachable without scrolling at a supported size
- [x] #3 A test pins on-screen reachability (not just section order and collapse), so future rail growth fails loudly instead of silently pushing content off the fold
<!-- AC:END -->

## Implementation Plan

1. Record the owner ruling (cap sections, scroll inside)
2. Cap each left-rail section body at a fixed viewport share with internal scroll
3. Add the missing reachability guard on the REAL CSS stack; mutation-check

## Implementation Notes

**Owner ruling (2026-08-15): cap sections, scroll inside** -- every section
header stays on-screen; each body gets a height budget with its own
scrollbar as the cue; a reachability guard pins it.

Implementation is one CSS rule (`#console-left-rail-body
.console-rail-section-body`): `max-height: 20%` + `overflow-y: auto` +
thin stable-gutter scrollbar, in `_agentic_terminal.tcss` with the bundle
regenerated (never hand-edited). Scoped to the LEFT rail -- the Inspector
and Library rails share the class but not the overflow problem. The rail
now has SEVEN sections (the task's "three" predates the agent/details/
character additions), so a fixed per-section share, not a computed split.

Two measured corrections along the way:

1. **The guard must run on the real CSS stack.** The bundle-less
   ConsoleHarness gave section headers `height: 1fr` (their `min-height: 2`
   lives in the bundle) -- capping the bodies made the fr headers collapse
   to 0 IN THE HARNESS ONLY. A geometry contract measured without the
   bundle is not measured (the 14822/15790 lesson again); the guard mounts
   the app's own three-file CSS_PATH stack.
2. **The reserved gutter must REPLACE the body's right padding, not add to
   it.** `scrollbar-gutter: stable` ate one column and re-clipped the
   Workspace value -- caught by test_session_rows_fit_inside_the_rail,
   whose 12+10+chrome contract spends the rail's full 30 columns. The
   gutter column now doubles as the right margin (`padding-right: 0`).

An earlier inline-styles variant was removed: with the bundle loaded it was
pure duplication, and the first mutation check proved it dead (reverting the
inline styles alone left the guard green). The shipped mutation check
removes the BUNDLE rule: guard goes red.

Guard: `test_console_left_rail_section_headers_all_visible_without_scrolling`
(160x48, the defect's own size): all four default-open headers inside the
rail viewport at scroll 0, and every displayed body within its 20%+1
budget. Collateral: internals + left_rail + rail_sections + width_budget +
shell_regions, 237 passed.

Modified: `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ bundle),
`Tests/UI/test_console_internals_decomposition.py`.
