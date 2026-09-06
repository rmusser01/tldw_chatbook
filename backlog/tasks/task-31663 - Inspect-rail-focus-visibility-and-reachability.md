---
id: TASK-31663
title: >-
  Inspect rail focus visibility and keyboard reachability
status: In Progress
assignee: []
created_date: '2026-09-05 07:00'
labels: [console, inspector, a11y, critique-2026-09-05]
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique P1, SGR-measured: all five focus-indicator styles in the rail
measure 1.03-1.79:1 against their unfocused state (below the 3:1 non-text
floor); five button stops show no plain-text change at all; one Tab stop
(stop 3 / summary block) has NO indication in either capture at both
sizes; Tab from the composer never reaches the rail (40 presses — a
hidden-but-focusable left-rail widget breaks the route); the section inner
scrollbar thumb renders fg==bg (1.00:1). Related: TASK-31624 (n/p ring)
now carries the trapdoor evidence. Prior art: TASK-24702 found a pure
background tint CANNOT clear 3:1 on this theme — the mechanism must be
shape/outline, not a stronger tint.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Every focusable rail widget has a visible focus indication in a plain-text capture (shape/glyph carrier, not tint-only), including buttons and chevrons
- [x] #2 No Tab stop in the rail is indication-free; the stop-3 gap is fixed or that widget is removed from the focus order
- [x] #3 The hidden-but-focusable left-rail widget is fixed so Tab routing from the composer is not silently absorbed (or the root cause is filed against the left rail with evidence)
- [x] #4 The section scrollbar thumb is visible against its track at both supported sizes
- [x] #5 (from 31662's AC#2 residue, review-required filing) The rail's pinned stack above the scroll body shrinks at small heights so Environment's four at-rest rows are visible at 80x24: 8 pinned lines today, 6 of them #console-send-authority-summary, over a 3-line scroll body (31662's measurement). Compress or make collapsible the send-authority block at constrained heights; keep its content reachable
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
### What shipped

**An accent edge for every BUTTON in the rail.** TASK-24702 had already proved
a tint cannot be rescued on this near-black theme, so the fix had to change
CELLS. `outline-left: thick $ds-action-focus` (which paints a solid `█`) is
applied by one rule, `#console-right-rail Button:focus`, covering the header
button, the project-instruction status button, section chevrons, "Refresh",
"Search Library", "Narrow…", the More toggle and the run-inspector actions.
The collapsed rail's handle (`#console-inspector-rail-open`) lives outside the
rail and needs its own rule — a RIGHT edge, because measured its column 0
carries the "<" of "<-Inspect" while its last column is blank on every row.

**Rows were already carrying a shape cue and are deliberately untouched.**
Round-1 review correction: an `outline-left` rule on
`.console-inspector-section-row:focus` shipped in the first cut and was DEAD
CODE. The app-wide `*:focus { outline: solid $ds-focus-accent }` in
`css/core/_reset.tcss` reaches rows (nothing opts them out the way
`Button:focus` opts buttons out with `outline: none`), and Textual takes a
line's left cell from the TOP edge's glyph set whenever that line is the
widget's top line — since TASK-31662 a row is one or two lines tall, so every
line is a top or bottom line and the thick left edge had no middle line to
paint on. Measured before and after: `┌Changes  +3 −1┐` either way. The rule
was removed and the real carrier documented in its place. The critique's "row
focus is shape-carried (good)" was right, and this task's first cut said
otherwise.

**Measurement first, both sizes, focus parked outside between stops.** Re-read
through the real compositor, seven stops painted byte-identical text focused and
unfocused at 80x24 AND 200x50: `console-inspector-rail-collapse`,
`console-project-instruction-status-button`, `…-environment-toggle`,
`…-environment-view-all`, `console-run-library-rag`,
`console-retrieval-scope-narrow`, `console-inspector-more-toggle` — plus
`#console-inspector-rail-body` at 80x24. Their whole indication was Textual's own
focus background swap, #1e1e1e → #1c4666, measured **1.68:1**, inside the
1.03–1.79:1 band the critique reported.

### Two things the measurement contradicted in the brief

1. **The "indication-free stop 3" is NOT the send-authority summary.** That block
   is the one stop that was already right: `outline: heavy $ds-action-focus`
   paints a `┏┓┗┛` box around it at both sizes. Its focusability and the Alt+I
   landing target are therefore unchanged — the 31450-arc flow and
   `test_console_inspector_keyboard_route.py`'s assertion still hold. The
   indication-free stops were the buttons, listed above.
2. **Nothing absorbs Tab.** Tab from the composer does not reach the rail because
   `ChatScreen.action_focus_next` scopes Tab to the focused widget's
   `CONSOLE_TAB_REGIONS` entry (TASK-2154.11 AC-02) and F6/Alt+I are the
   cross-pane routes. Measured: 45 Tab presses from the composer visit exactly the
   11 composer-region stops and cycle; the full 68-member focus chain contains no
   widget hidden by an ancestor's `display: none`. The widget the critique named
   ("Review changes", blank, hidden) is real but is
   `#console-prompt-improvement-review` in the COMPOSER's hidden
   prompt-improvement recovery row, not the left rail, and it is kept out of the
   ring only *incidentally* — by `disabled=True` plus the parent row's hidden
   display. That coupling is now explicit (`can_focus` follows the row's display,
   in `compose` and in `_sync_improvement_recovery`), and two standing guards pin
   the rest.

### AC#4 — the scrollbar

`scrollbar-color: $ds-grid-line` (#2d2d2d) on `scrollbar-background:
$ds-surface-panel` (#242f38): measured **1.01:1** through the compositor at both
sizes, matching the critique's 1.00:1. Nothing in the surface family clears 3:1
against that track ($ds-column-line 1.28:1, $panel-lighten-3 2.16:1), so the
thumb moved to `$ds-text-muted` — the app's other established scrollbar thumb
(seven Library rules already use it) and theme-polarity-generated.
`scrollbar-color-active` keeps the accent so dragging stays distinguishable.
Applied to both the rail's outer scroller and `.console-bounded-section-viewport`
(the section-inner bar the critique measured).

### AC#5 — the pinned stack

At 80x24 the rail's five children measured 1 / 1 / **6** / **3** / 1: eight pinned
rows over a three-row body, so the Environment section (seven rows at rest since
TASK-31662) could never show its four rows. `ConsoleSendAuthoritySummary` now has
a two-row compact density — heading + **Run** — driven by
`ConsoleInspectorRail._sync_authority_summary_density` from the rail's own height
against a DERIVED threshold (`INSPECTOR_FULL_DENSITY_MIN_ROWS` = full pinned stack
8 + hint 1 + a section at rest 7 = 16). At 80x24 the body is now **seven** rows and
all four Environment rows paint; at 200x50 nothing changes.

`Run` is the line that survives because it is not one fact among five but the
severity-ordered rollup of all of them. The other four stay complete and reachable:
the block's tooltip carries them, and F1 while it has focus still renders all five
(`action_show_workbench_help` reads the PROJECTION, not the mounted rows, so hiding
rows costs it nothing). All six Statics stay mounted, so `sync_state`'s in-place
patching and every id-based consumer are untouched.

The compaction **announces itself** (round-1 review): the compact heading is its
own copy, `If I send now? · +4 more`, not the full heading plus a suffix — the
block gets 29 content columns at 80x24 and the full heading already spends 27, so
`<heading> +4` (30) would have ellipsized the marker straight back off. The
`console-authority-compact` class is `console-`-prefixed so `build_css`'s owner
classifier files its rule with the Console screen sheet rather than leaving it in
the always-loaded boot bundle (verified: the rule moved out of
`tldw_cli_modular.tcss` into `screen_agentic_console.tcss`).

### The one non-obvious physics fact

`outline` on a CONTAINER is nearly invisible: the compositor paints children OVER
the container's own strips, so `#console-inspector-rail-body:focus`'s TASK-24702
edge showed only in the gaps between sections — and at 80x24, where all three of
its rows are covered, it showed nothing at all. The rule that fixes it targets the
innermost widget with a column no child occupies:
`#console-inspector-rail-body:focus .console-inspector-section-body`, whose
`padding-left: 1` reserves exactly one. The same physics is why the button edge is
lossless — every rail Button already paints a leading space (Textual's `padding: 0 1`,
or the centring of a fixed-width one), so the block lands on padding and no label
character is overwritten; a per-button test pins that so a future label filling its
own first column fails there rather than shipping as "█arrow…".

### Tests

New `Tests/UI/test_console_inspector_focus_carriers.py` (15 tests): the whole-ring
plain-text focus diff at both sizes (not a hand-listed subset — the defect was a
convention being invisible); the per-button lossless-edge pin; the row corner-bracket
pin; the collapsed-handle carrier; the explicit composer recovery coupling; the
Tab-region / F6-route pin; the scrollbar thumb-vs-track contrast measured from
painted cells at both sizes; the compact-density and compression-cue pins; and the
rail-level sibling of 31662's component test — Environment's four rows painted
through the real pinned stack at 80x24.

Two test-design corrections from round 1, both of which had let something ship
unverified. First, the ring walk ran against the DEFAULT fixture, whose Environment
section is UNBOUND and projects no rows — so "walk every stop" never touched a
`ConsoleInspectorSectionRow`, which is exactly how the dead row rule got through.
The walk now lands a canned OK-git snapshot first and asserts the ring actually
contains rows. Second, the lossless-edge check compared focused-vs-unfocused with
the carrier column already dropped from both sides, so it could not see the loss it
existed to catch; it now asserts the carrier column is blank BEFORE focus, which is
what caught that a left edge on the collapsed handle would paint "█-Inspect". A
third fixture fix: the Environment feed is frozen for the walk, because opening the
rail arms a background refresh whose landing replaces row widgets mid-walk (it
orphaned the "Refresh" button, region 0x0). Deleted rather than fixed: a
"no focus-chain member is hidden by an ancestor" test — Textual builds the chain
from `displayed_children`, so it asserted the framework and could not fail.

Regression sweep: every suite touching the changed surfaces was run against this
branch AND against its base commit (`21dcb743a4`) in a throwaway worktree. Failure
sets are byte-identical — 22 in workbench_contract/compact_access/run_inspector, 2
in inspector_navigation, 2 CSS-contract, 3 focus-contract, 3 geometry, 16 in
internals/live_work — all pre-existing. `./scripts/preflight.sh` green.

Round-1 addendum — an intermittent failure chased to ground rather than waved
through. The covering batch failed twice on this branch, on two DIFFERENT tests,
while the first two base runs were clean, which is the shape of a regression. It
is not one: `test_live_work_widget_swaps_cover_real_twenty_twenty_one_geometry`
also fails on the untouched base commit (branch 4 failures / 6 runs, base 1 / 5,
same test and parametrisation). Both tests wait on a WALL-CLOCK-bounded predicate
and then re-assert it after one more `pause()`; the failing tuple had every
geometry value correct and only `_outer_reconcile_scheduled` True — one extra
pause re-arming the loop. Two branch-local mechanisms were formed and both
disproven by direct measurement: the AC#5 density hook never flips at the sizes
involved (spied on every call), and deleting the new `outline-left` descendant
rule did not stop the failure (and that rule cannot apply there — the test never
focuses the rail body). Recorded in `backlog/docs/lessons-testing-evidence.md`;
the residual higher rate on this branch is flagged, not explained.

### Modified

`tldw_chatbook/css/components/_agentic_terminal.tcss` (+ regenerated
`screen_agentic_console.tcss`), `Widgets/Console/console_send_authority_summary.py`,
`Widgets/Console/console_composer_bar.py`, `UI/Console_Modules/right_rail.py`,
`Docs/User_Guide/console/sessions-tabs-workspaces.md`,
`Docs/User_Guide/console/context-and-rag.md`,
`Tests/UI/test_console_inspector_focus_carriers.py` (new).
<!-- SECTION:NOTES:END -->
