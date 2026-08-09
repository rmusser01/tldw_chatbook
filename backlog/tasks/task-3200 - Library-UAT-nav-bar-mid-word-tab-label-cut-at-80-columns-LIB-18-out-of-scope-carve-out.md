---
id: TASK-3200
title: >-
  Library UAT nav-bar mid-word tab-label cut at 80 columns (LIB-18 out-of-scope
  carve-out)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 16:44'
updated_date: '2026-08-08 23:07'
labels:
  - library
  - ux
  - navigation
  - uat-2026-08-06
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 4 of the Library UAT P2 batch (task-2858) reproduced LIB-18's third finding live at 80x24: the shared MainNavigationBar's horizontally-scrolling destination strip clips the last partially-visible tab label mid-word (e.g. "Watchlists" -> "⌃6 Watc") right before the "More ›" overflow affordance, instead of hiding that partial button. This is shared app-wide chrome (tldw_chatbook/UI/Navigation/main_navigation.py), not Library-specific, and the fix is not a small one: the strip fills 1fr and lays out full-width buttons with CSS overflow-x: auto (main_navigation.py:104-113), so a button that only PARTIALLY fits at the viewport's right edge is visually clipped by the scroll container rather than being excluded -- the existing 'More ›' hint (main_navigation.py:226-237) already signals overflow exists but does not stop the strip's own clipping. A real fix needs the strip to measure whole-button widths and stop rendering (or scroll-clip) at a button boundary instead of relying on CSS overflow alone, touching shared navigation code exercised by every screen -- out of scope for the Library UAT P2 batch. Recorded per task-2858's Task 4 directive to record out-of-scope shared-chrome findings rather than force a fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The MainNavigationBar destination strip never clips a tab's label mid-word at narrow widths (e.g. 80 columns) -- either the partially-visible button is fully hidden until it can render whole, or its label degrades gracefully (e.g. abbreviates) without an ellipsis or hard cut inside a word
- [x] #2 Fix is verified live at 80x24 (and spot-checked at 100/120) with a fresh tmux session, confirming no destination label is cut mid-word
- [x] #3 A rendered-geometry or string-assertion test pins the fix (region widths or captured label text), matching the existing F-001 overflow-hint test coverage in Tests/UI/test_main_navigation*.py or equivalent
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify the defect live at 80 cols (tmux) and read the overflow-scroll logic in main_navigation.py (.main-nav CSS block, overflow_hint construction, _scroll_active_destination_into_view).
2. Attempt bounded scroll-geometry fixes iteratively, validating each against rendered-geometry tests before moving on:
   a. Hide (display=False) any button whose region straddles either viewport edge -- rejected: hiding a LEADING straddler reflows every button after it (display:none excludes it from layout), which can shift the active destination itself and expose a NEW straddler; for some active-destination/viewport-width combinations no scroll offset makes the active destination both fully visible and flush on a clean button boundary, so the hide-and-recheck cascade does not converge (traced analytically: it kept eating through the whole bar).
   b. Snap scroll_x to a clean button boundary via a pure-arithmetic model (cached natural widths, canonical scroll computation) -- rejected: Textual's own `scroll_to()`/`scroll_to_widget()` clamp to LIVE `max_scroll_x`, which changes the moment any button's `display` is toggled (deferred internally too), so a scroll target computed against the "nothing hidden" width was silently clamped to a different value than intended once hiding happened, reproducing the bug from a different angle.
3. Switch approach: visually blank (never geometrically hide) a straddling button via a CSS class that color-matches the bar's background (foreground/border/background all `$background`), leaving `display`/layout completely untouched. This sidesteps every geometry/cascade/clamp issue above since nothing about scroll math changes.
4. TDD: add rendered-geometry + actual-post-clip-rendered-text (compositor segment style, not raw `.label`) tests at 80/100 cols, two active-tab positions (early: no scroll needed: late: forces a scroll).
5. Fix an interval-driven oscillation (re-triggering the settle chain every 0.5s raced its own multi-hop `call_after_refresh` continuation against the next tick) by decoupling the periodic interval from the heavier settle logic.
6. Live-verify via tmux at 80/100/170 cols, both active-tab positions, plus paging via the "More >" affordance.
7. Verify against REAL screen-to-screen navigation (not just direct `MainNavigationBar(active=...)` construction) -- this surfaced a second bug: `on_resize` ghost-checked directly without first re-scrolling, so a screen swap's intermediate resize events could leave a stale ghost state uncorrected. Fix `on_resize` to route through the same re-scroll-then-ghost-check path as every other trigger.
8. Run the full targeted regression sweep (Tests/UI files referencing MainNavigationBar) plus a Tests/UI --collect-only sanity check; fix the one pre-existing test whose assertion encoded the rejected display=False approach.
9. Backlog hygiene + commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed via a CSS-only "ghost" treatment, not by hiding buttons. A straddling
destination button gets `.nav-button-clip-ghost` (main_navigation.py
DEFAULT_CSS): background/border/color all set to `$background`, matching
the bar's own background exactly, so whatever sliver renders reads as
empty space instead of a cut word. `display`/layout is never touched.

Two rejected approaches, both traced to root cause before abandoning:
- `display: none` on a straddling button: excludes it from the strip's
  virtual size, breaking the "More >" pager's reach to later destinations,
  and hiding a LEADING straddler reflows every button after it (including
  the active one), which can expose a NEW straddler -- for some
  active-destination/viewport-width pairs no scroll offset makes the
  active destination both fully visible and flush on a clean button
  boundary, so a hide-and-recheck loop never converges (analytically
  traced through the whole 13-button bar).
- Pure-arithmetic boundary-snapping (compute scroll_x from cached
  "nothing hidden" widths): Textual's own `scroll_to()`/`scroll_to_widget()`
  clamp to the LIVE `max_scroll_x`, which shrinks the instant any button's
  `display` changes (and the clamp/scroll itself is deferred internally),
  so a target computed against the wrong baseline silently landed
  somewhere else -- reproducing the bug from a different angle.

Two real bugs found and fixed along the way:
1. Re-running the full settle chain from the 0.5s interval raced its own
   `call_after_refresh` continuation against the next tick, so the ghosted
   set never stabilized. Fixed by decoupling the interval from the heavier
   settle path (interval keeps only the cheap scroll-pin).
2. `on_resize` used to ghost-check directly, without first re-scrolling
   for the current viewport. A real screen-to-screen navigation (not
   direct `MainNavigationBar(active=...)` construction) fires several
   resizes while content settles; the last one to land could ghost-check
   against a scroll position from an earlier layout, leaving a stale,
   wrong ghost state (live-reproduced: navigating Home -> Settings at 80
   cols left "Schedules" straddling and fully readable while an unrelated,
   fully off-screen "Watchlists" stayed ghosted for no reason). Fixed by
   routing on_resize through `_scroll_active_destination_into_view`
   (re-scroll, then ghost-check) like every other trigger.

Residual, ACCEPTED risk (documented, not hidden): even with fix #2, several
resize events during a screen swap can still interleave their
`call_after_refresh` chains, so the LAST ghost-check to physically execute
is not strictly guaranteed to be from the freshest chain -- a narrow race
the fix reduces but does not fully eliminate. This codebase's existing,
always-on 0.5s interval (unmodified, calls the same scroll-then-ghost pair
on its own schedule) is the intentional, pre-existing self-healing
mechanism for exactly this class of async layout race, and is what
resolves any residual race within a bounded, small window in practice.
Confirmed no reliable pytest timeout distinguishes buggy from fixed once
the interval is in play; see the lessons-file entry for the full trace of
that dead end and why a flaky "real navigation" test was written, then
DELETED rather than shipped once it was shown to false-pass 3/3 against
the still-buggy code.

Tests: Tests/UI/test_master_shell_navigation.py adds
test_nav_strip_never_renders_a_partial_destination_label (80/100 cols,
early + late active tab; asserts both geometry and the ACTUAL rendered
text via compositor segment style, since a ghosted segment's raw `.text`
still contains the real characters -- only its color matches the
background). Fixed one pre-existing test
(test_destination_visual_parity_correction.py) whose assertion encoded the
rejected display=False approach.

Live-verified via tmux at 80/100/170 cols, both an early (Home) and late
(Settings) active destination reached via the command palette (the
real-navigation path, not direct construction), including a resize-down
after reaching 170 cols and paging via "More >" -- confirmed via raw ANSI
capture (capture-pane -e) that every ghosted segment renders with
foreground color equal to background color, both before and after fix #2
(reproducing, then resolving, the on_resize bug live).

Files: tldw_chatbook/UI/Navigation/main_navigation.py,
Tests/UI/test_master_shell_navigation.py,
Tests/UI/test_destination_visual_parity_correction.py,
backlog/docs/lessons-testing-evidence.md.

### Review-round follow-up (fix round 1)

A review of the above found ghosted buttons stayed fully interactive (color-only
hiding, no `disabled`/focus change) and asked for an honest characterization of the
residual settle-chain race. Both closed; two real regressions were found and fixed
along the way (not anticipated by the review, caught by re-running the suite before
shipping):

- `_ghost_clipped_buttons` now also sets `button.disabled = should_ghost`.
  Textual's own `disabled` semantics do the rest for free: `Widget.focusable`
  excludes disabled widgets from the Tab focus chain, and `Button.press()` already
  no-ops when `self.disabled` is set. A new `on_descendant_focus` handler
  re-scrolls the strip to the newly-Tab-focused button (mirroring what
  `_scroll_active_destination_into_view` already does for the active destination)
  before re-running the ghost check, closing a staleness gap Tab-cycling opens that
  none of mount/resize/click ever did (Tab can move DOM focus without the app
  calling any of this class's own scroll methods).
- **Regression found and fixed**: disabling a button that had *just* received Tab
  focus (via the new `on_descendant_focus` re-scroll landing on a still-fractionally-
  straddling position) let Textual's `watch_disabled` blur it immediately, and the
  next Tab press wrapped all the way back to the first button in the bar instead of
  advancing — reproduced live via direct focus/disabled tracing, and via a broader
  regression sweep (`test_tab_order_reaches_visible_primary_action` regressed on
  5/9 cases that passed at HEAD). Fixed by exempting whichever button currently
  holds keyboard focus from ghosting, in `_ghost_clipped_buttons`, the same way the
  active destination is already exempt.
- **A second, pre-existing test in a different file**
  (`test_phase6_home_keyboard_focus_reaches_navigation_and_primary_action`,
  `Tests/UI/test_product_maturity_phase6_focus_visual_sweep.py`) asserted Tab
  reaches all 13 top-level destinations in exact canonical order, one press each —
  an invariant `disabled`-based ghosting cannot preserve by design (a straddling
  button is un-Tab-reachable exactly where it is straddling). Live-traced: the
  skipped destination recovers within a bounded number of extra presses (17 in the
  worst observed case, out of a ~14-stop bar), so it is not permanently
  unreachable — rewrote the test's per-destination advance to budget extra presses
  instead of assuming single-press lockstep, keeping the same final strict-order
  assertion.
- **Color-fidelity defect found and fixed**: even with `disabled` set, a ghosted
  button was NOT pixel-exact invisible in the real running app (only in a bare-
  widget pytest harness) — live tmux capture showed a faintly-readable foreground
  against the intended-matching background. Root cause: `tldw_chatbook/css/
  components/_buttons.tcss`'s app-wide `Button:disabled { opacity: 50%; }`, loaded
  via `App.CSS_PATH`, outranks ANY widget `DEFAULT_CSS` rule regardless of
  `!important` there (Textual gives `CSS_PATH` stylesheets priority over widget
  `DEFAULT_CSS` as a tier, not just by specificity) — a known, precedented gotcha in
  this codebase (`Tests/UI/test_mcp_inspector.py` documents and fixes the identical
  defect for the MCP inspector's action buttons the same way). Fixed by adding a
  targeted override to `tldw_chatbook/css/components/_navigation.tcss` (previously
  an empty stub), in the same `CSS_PATH` tier, where normal specificity resolves it
  (two classes + a pseudo-class beats one class + a pseudo-class) — no `!important`
  needed there. Verified via a real-`TldwCli`-app pytest repro
  (`get_visual_style()` went from `rgba(255,255,255,0.38) on rgb(15,15,15)` to
  `rgb(18,18,18) on rgb(18,18,18)`) and live in tmux (`38;2;18;18;18m` on
  `48;2;18;18;18m`, exact, at both the trailing and leading straddle edges).
- **Residual-race characterization, corrected**: the periodic 0.5s interval's own
  direct call is cheap, but it unconditionally chains into the SAME full
  straddle-scan (`_scroll_active_destination_into_view` → `call_after_refresh(_ghost_
  clipped_buttons)`) every other trigger uses — a prior report draft's "does NOT
  call a heavier settle function" framing was wrong and has been corrected, not
  reworded-and-hidden (see the task-5 report's Fix Report round-1 section for the
  full trace). Both directions of transient staleness are possible (a straddling
  button briefly staying visible+enabled, or a no-longer-straddling one briefly
  staying ghosted+disabled) until the next settle pass, bounded by one interval
  period (0.5s) plus a two-hop `call_after_refresh` chain — sub-second in practice.
  A generation-token hardening was considered (would only gate which chain's RESULT
  applies, not touch scroll/focus decisions like the two previously-reverted
  attempts above) and declined: the actual `call_after_refresh` timing here is
  exactly what the base round already showed cannot be pinned reliably via pytest,
  and shipping unverified new bookkeeping to close a sub-second, self-healing,
  cosmetic-only race is not a good trade for a review-fix round.
- Tests: `test_master_shell_navigation.py` 26/26 x5 runs; `test_destination_visual_
  parity_correction.py` 133/133 (5 pre-existing schedules/MCP failures unrelated,
  see task-2560); `test_product_maturity_phase6_focus_visual_sweep.py` 5/5; a
  13-file broader nav-referencing sweep 324 passed / 5 pre-existing failed / 1
  pre-existing skipped, unchanged before/after the CSS bundle fix.

Additional files this round: `tldw_chatbook/css/components/_navigation.tcss`
(new override rule), `tldw_chatbook/css/tldw_cli_modular.tcss` (regenerated),
`Tests/UI/test_product_maturity_phase6_focus_visual_sweep.py` (budget-tolerant Tab
advance), `backlog/tasks/task-2560 - ...md` (delta note),
`.superpowers/sdd/library-polish-batch/task-5-report.md` (Fix Report round 1 +
corrections to the base report's Important #2 framing and minors).
<!-- SECTION:NOTES:END -->
