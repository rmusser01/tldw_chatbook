---
id: TASK-4020
title: Nav ghosting does not prevent mid-word tab cuts after the overflow-menu change
status: Done
assignee:
  - '@claude'
created_date: '2026-08-09 20:30'
updated_date: '2026-08-09 22:00'
labels:
  - navigation
  - regression
  - recritique-2026-08-09
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library re-critique 2026-08-09 (RC-02), measured at dev `4d0232358` by the mechanical arm.

task-3200's four-round arc existed to guarantee that a destination tab label is never rendered
mid-word-clipped. At dev tip that guarantee does not hold:

- 80 cols: `⌃6 Watc  More ▾`
- 120 cols: `⌃9 M  More ▾`
- scroll fragments observed: `‹ ts  ⌃5 Roleplay…`, `‹ oleplay…`, `‹ lists…`, `‹ edules…`

No ghosted tabs were observed at all — the bar scrolls with a `‹` indicator instead.

**The ghosting machinery is present** (10 references in `UI/Navigation/main_navigation.py`, 2 in
`css/components/_navigation.tcss`), so this is a failure of effect, not lost code. Leading
hypothesis: dev replaced the in-strip pager with `NavOverflowMenu` while task-3200 was in flight;
the polish batch's rebase (PR #1459) kept the ghosting mechanism and dropped the pager-specific
pieces, but the scroll/paging model the straddle detection was written against changed underneath
it. Re-root-cause against the current overflow model rather than re-patching the old assumptions.

Mitigating: no ghosted tab was clickable-while-invisible (a blank-cell click did not navigate), so
the round-1 interactivity hole stays closed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No destination tab label renders mid-word-clipped at 80, 100 or 120 columns, verified by rendered-geometry assertions and live capture
- [x] #2 The root cause is stated: why the existing ghost/straddle detection stopped producing its effect under the overflow-menu model
- [x] #3 The scroll-fragment renders (`‹ oleplay…`) are gone or are a deliberate, documented affordance
- [x] #4 Regression coverage runs against the CURRENT overflow model, and the now-obsolete assumptions in task-3200's tests are corrected rather than left passing vacuously
<!-- AC:END -->

## Implementation Plan

1. Instrument headlessly (bare-App + bundled-CSS probe) at 80/100/120 with several active
   destinations: dump `button.region` vs `strip.region`, straddle detection, ghost class, and
   disabled state. Confirm whether `_straddles_viewport`/`_ghost_clipped_buttons` still fire
   correctly under the current `NavOverflowMenu` model.
2. Live-verify against the real running app (tmux, scratch profile) at 80 and 120 cols to
   reproduce the recritique's exact captures (`Watc`, `M`, scroll fragments). Decode the STYLED
   ANSI (`capture-pane -e`), not just plain text, to determine whether the "clipped" fragments
   are genuinely rendered (fg != bg) or pixel-invisible (fg == bg, i.e. correctly ghosted but
   still present as buffer characters a colorless capture cannot filter out).
3. Based on findings, decide the fix: if ghosting is functioning correctly and the recritique's
   "no ghosting observed" conclusion is a measurement-methodology gap (colorless capture), the
   fix is test-coverage, not behavior -- upgrade task-3200's geometry/readable-text tests to run
   against the REAL bundled stylesheet (`_BUNDLED_CSS_PATH`), not just
   `MainNavigationBar.DEFAULT_CSS`, so a future regression in `_navigation.tcss`'s ghost-color
   override would actually be caught. If a genuine behavioral gap is found instead, fix
   `main_navigation.py`/`_navigation.tcss` directly.
4. Write a RED test that reproduces the recritique's observed effect exactly (naive/colorless
   text check against the bundled-CSS app showing the "clipped" fragment as present), then land
   the corrected/upgraded regression coverage (color- and geometry-aware, bundled CSS, at
   80/100/120, early + late active tab) as the permanent guard.
5. Address AC#3 (scroll-fragment affordance) with direct evidence from the
   instrumentation/live verification.
6. File a follow-up backlog task for any separate finding surfaced along the way that is out of
   this task's specific AC scope (e.g. a navigation-triggered scroll-settle issue unrelated to
   ghosting effectiveness).
7. Live-verify the final state at 80/100/120 with an early and a late active tab, update the
   task file (ACs, Implementation Notes), run targeted tests, commit.

## Implementation Notes

**The guarantee was never actually broken.** `_straddles_viewport`/`_ghost_clipped_buttons` and
`css/components/_navigation.tcss`'s `.nav-button.nav-button-clip-ghost:disabled` color-pin rule
are unaffected by the `NavOverflowMenu` rework and still function correctly under it. No behavior
change was needed in `main_navigation.py` or `_navigation.tcss`.

**Root cause (AC#2):** the re-critique's mechanical probe captured the nav bar with `tmux
capture-pane -p` (no `-e`/ANSI). Ghosting makes a straddling button's foreground EXACTLY equal to
its background (`color: $background` paired with the container's own `background: $background`)
so a real, color-aware terminal renders nothing legible -- but the underlying CHARACTERS are still
present in the compositor's cell buffer (ghosting is a color trick, not character removal). A
colorless capture reads those characters back regardless of color and therefore cannot distinguish
a genuinely-rendered mid-word-clipped label from a correctly-ghosted, pixel-invisible one. Direct
ANSI decoding of the real running app (`capture-pane -p -e`) reproduced every fragment the
re-critique quoted -- `⌃6 Watc` (80 cols), `⌃9 M` (120 cols), and both left-edge scroll-fragment
shapes (`‹  Library`, `‹ onsole`, `‹ Artifacts`) -- and in every case foreground RGB == background
RGB (typically `(18,18,18)` on `(18,18,18)`), i.e. pixel-invisible in a real terminal. The
re-critique's OWN corroborating check (a click on the "blank" cell did not navigate) is further
evidence FOR ghosting, not against it: an un-ghosted button is never `disabled`, so that click
would have navigated had ghosting truly not applied -- a self-contradiction in the original
finding that should have been the tell.

**Real, separate finding surfaced along the way:** navigating to Settings specifically (via the
overflow menu; reproduced headlessly too) leaves the nav bar's strip stuck at `scroll_x=0` for
15+ seconds -- the active destination's highlight is real (the `is-active` CSS class is correctly
applied) but never scrolled into view, so nothing shows it. This does NOT produce a mid-word-clipped
label (everything at the stuck position is either fully visible or correctly ghosted by the bar's
initial settle pass), so it is out of this task's AC scope. Filed as task-4024.

**AC#1/#4 fix (the actual code change):** every existing task-3200 geometry/readable-text test ran
under a bare `App()` with only `MainNavigationBar.DEFAULT_CSS` loaded -- never
`css/components/_navigation.tcss`'s separately-maintained color override, which is the copy that
actually wins live (`App.CSS_PATH` always outranks widget `DEFAULT_CSS`, `!important` or not). A
regression that broke ONLY the bundled-CSS-tier override (e.g. reverting `_navigation.tcss`, or a
bundle-rebuild dropping it) would have left every existing test green while the real app regressed
to legible-but-dim ghosted text -- the exact "test passes against broken code" failure mode AC#4
warns about, one CSS tier removed from where task-3200 originally looked. Added to
`Tests/UI/test_master_shell_navigation.py`:
- `test_naive_colorless_capture_false_positives_on_ghosted_labels` -- reproduces the re-critique's
  exact `"Watc"` finding via a colorless compositor dump under the bundled stylesheet, then
  contrasts it with the color-aware check, documenting the measurement-methodology gap directly in
  the suite (this is the RED-reproduction artifact: a throwaway inverted version of this assertion
  was run first and failed with the literal string `⌃6 Watc  More ▾`, confirmed against the exact
  fragment the re-critique quoted, before being corrected to the real invariant).
- `test_nav_strip_never_renders_a_partial_destination_label_under_bundled_css` -- the DEFAULT_CSS
  version's twin, now against `_BUNDLED_CSS_PATH`, parametrized over 80/100/120 with an early
  (`home`) and late (`settings`, `mcp`) active destination each, plus an explicit `"…"` -not-present
  assertion ruling out an ellipsis-truncation artifact as a second possible cause of AC#3's
  scroll-fragment concern.

**AC#3:** the `‹` indicator (`nav-overflow-hint-left`) is the pre-existing, deliberately-documented
affordance (see its own compose()-site comment); the word fragment immediately after it is, per the
above, correctly ghosted (invisible), not a readable truncation. The literal `…` glyphs quoted in
the re-critique's report were never reproduced live and are most likely the report's own
prose-elision marker, not a captured terminal character -- pinned as an explicit regression guard
regardless (see above).

**Files changed:** `Tests/UI/test_master_shell_navigation.py` only (+~180 lines: two new tests,
one new helper `_plain_nav_text`, a section docstring). No production code changed.

**Tests:** `Tests/UI/test_master_shell_navigation.py` -- 41 passed (33 pre-existing + 8
new/parametrized instances), 0 failed, run 3 of 4 times consistently green; one run hit an
unrelated pre-existing timing-sensitive test (`test_master_shell_navigation_keeps_active_
destination_visible_on_mount`) failing under load, which passed standalone immediately after and
in the next full-file run -- not attributable to this change. `Tests/UI --collect-only`: 9914
collected, 0 errors.

**Live verification:** 80/100/120 cols x early + late active tab (6/6 combinations), tmux scratch
profile, `capture-pane -p -e` ANSI-decoded -- every "clipped-looking" fragment confirmed
foreground==background (pixel-invisible). Details in `.superpowers/sdd/library-recritique-p1s/
task-1-report.md`.

**Follow-up filed:** task-4024 (Settings screen nav-bar scroll-settle race).
