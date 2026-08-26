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
  parity_correction.py` **108 passed, 5 failed** (5 pre-existing schedules/MCP
  failures unrelated, see task-2560 — CORRECTED in round 2's notes below: this
  number was originally misreported as "133/133" here and in the task-5 report;
  108/5 is the twice-verified figure); `test_product_maturity_phase6_focus_visual_
  sweep.py` 5/5; a 13-file broader nav-referencing sweep 324 passed / 5 pre-existing
  failed / 1 pre-existing skipped, unchanged before/after the CSS bundle fix.

Additional files this round: `tldw_chatbook/css/components/_navigation.tcss`
(new override rule), `tldw_chatbook/css/tldw_cli_modular.tcss` (regenerated),
`Tests/UI/test_product_maturity_phase6_focus_visual_sweep.py` (budget-tolerant Tab
advance), `backlog/tasks/task-2560 - ...md` (delta note),
`.superpowers/sdd/library-polish-batch/task-5-report.md` (Fix Report round 1 +
corrections to the base report's Important #2 framing and minors).

### Review round 2 follow-up

A scoped re-review confirmed round 1's findings (F1/F2/minors) ADDRESSED, but found
the round-1 focused-widget exemption itself reopened a live, deterministic mid-word
cut: the periodic 0.5s interval unconditionally recentered on the ACTIVE
destination every tick, indifferent to keyboard focus, dragging the strip back
whenever a Tab press had focused a DIFFERENT, far-away button -- leaving that
focused button straddling, un-ghosted (exempt), and `disabled=False` (Enter-
navigable). Deterministically reproduced with the reviewer's exact probe
(`active="schedules"`, Tab to `nav-settings`, `pilot.pause(0.9)`): `nav-settings`
measured genuinely straddling one interval tick later.

**Closed at the source**, not with another exemption: a new `_recenter_periodic`,
used ONLY by the interval's own trailing call, recenters on the DELIBERATELY-
focused button instead of active whenever they differ (`_scroll_active_destination_
into_view` itself stays unconditional, unchanged, for every other caller). "
Deliberate" specifically excludes Textual's own `AUTO_FOCUS = "*"` landing on the
first nav button the instant the bar mounts (empirically confirmed to fire strictly
before this bar's own first settle callback) -- a naive "any focus wins" version was
tried first and broke the mount-time "active is always visible" guarantee instead
(`test_master_shell_navigation_keeps_active_destination_visible_on_mount`); a new
`_mount_settled` flag + `_deliberate_focus_id` (both new instance state) cleanly
separate the two.

Two further regressions were found and fixed by my own re-verification before
shipping (neither from the reviewer): (1) a crash -- `_focused_strip_button`
accessed `self.screen.focused` without the `is_attached` guard every other
`self.screen` access in this class already uses, so a deferred callback reaching it
after a real screen swap had unmounted the bar raised `NoScreen` and took the whole
app down mid-test (every subsequent Tab press then silently did nothing because the
app had exited) -- fixed with the same guard pattern used everywhere else in the
file; (2) a synchronous "retry then re-measure" belt-and-braces guard for the
focused button (an initial attempt at satisfying the review's "drive scroll_to_
widget(focused) in the same chain" ask) could never see its own retry's effect,
because a `Region` only updates on the NEXT layout pass, not synchronously within
the same call -- it therefore sometimes ghosted+disabled the button Tab had JUST
landed on, reproducing round 1's original "Tab jumps back to the first button" bug
(caught via `test_tab_order_reaches_visible_primary_action` regressing from a clean
13-press traversal back to a 2-lap, 23-press one). Reverted to an UNCONDITIONAL
exemption for the deliberately-focused button, symmetric with active's, backed by
the two real (properly-deferred) mechanisms that already existed (`on_descendant_
focus`'s own re-scroll chain, `_recenter_periodic`'s interval-level backstop) rather
than a same-call re-measurement that cannot work.

Also corrected: the "133 passed" tally above (and in the task-5 report) was wrong,
carried forward unverified. Re-ran the file myself: **108 passed, 5 failed, 113
collected** -- twice-verified now, matching the re-reviewer's own count.

New regression test: `test_periodic_interval_does_not_drag_the_focused_button_out_
of_view` (`Tests/UI/test_master_shell_navigation.py`), the reviewer's exact probe.
Full suite re-verified: `test_master_shell_navigation.py` 27/27 x5 runs;
`test_product_maturity_phase6_focus_visual_sweep.py` 5/5 x3 runs (after the crash/
oscillation fixes); `test_destination_visual_parity_correction.py` 108/5 (corrected
tally, unchanged 5 pre-existing failures); the 13-file broader sweep re-run clean
after both fixes.

Files this round: `tldw_chatbook/UI/Navigation/main_navigation.py` (`_recenter_
periodic`, `_mount_settled`/`_deliberate_focus_id`/`_mark_mount_settled`,
`_focused_strip_button` crash fix, focused-guard reverted to unconditional
exemption), `Tests/UI/test_master_shell_navigation.py` (new regression test),
`.superpowers/sdd/library-polish-batch/task-5-report.md` (Fix Report round 2).

### Review round 3 follow-up

Round-3 re-review confirmed round 2's interval fix (`_recenter_periodic` + its
probe test) held, but found round 2's "closed at the source" framing OVERSTATED
coverage: the re-review live-reproduced the IDENTICAL stranding (a deliberately-
focused, non-active button left genuinely straddling, un-ghosted, enabled, still
focused) through THREE other active-only recenter triggers round 2 left
untouched -- `on_resize`, `restore_active`, and the "More ›" pager. The actual
source of the defect class was "every caller of an active-only, focus-indifferent
recenter," not "the periodic interval" specifically; round 2 had fixed exactly one
instance of it.

**Generalized, not patched a third and fourth time**: `_recenter_periodic` renamed
to `_recenter_strip` and made the ONE shared entry point every settle trigger
funnels through -- mount, the interval, `on_resize`, and `restore_active` now all
call it instead of the plain, active-only `_scroll_active_destination_into_view`
directly. Confirmed behavior-neutral for mount specifically: `_deliberate_focus_id`
cannot be set until AFTER `_mount_settled` becomes `True` (a later `call_after_
refresh` than the mount-time recenter call), so `_recenter_strip` is provably
equivalent to the old plain scroll throughout the entire mount-settle window,
regardless of Textual's own `AUTO_FOCUS`. `_activate_navigation_button` (a click)
deliberately kept calling the plain primitive -- documented reasoning in the code:
a click that just set `active_destination_id` typically also focuses that same
button via Textual's normal click handling, making the two targets identical in
the common case; the one way they could differ (a stale, unrelated deliberate
focus from an earlier Tab press still being the live-focused widget when a
DIFFERENT button gets clicked) is not covered by any of round 3's repros, and
migrating it there risks scrolling away from the button the user just activated.

**The pager got a genuinely different fix, not `_recenter_strip`**, because its
whole purpose is moving the viewport AWAY from wherever it sits -- scrolling back
to the focused button (what `_recenter_strip` does everywhere else) would make
"More ›" non-functional whenever a nav button holds focus (confirmed: the
generalized interval was already doing exactly this unwanted self-heal within
~0.5s, masking the pager's own defect by silently defeating the page). Decision,
matching standard paginated-control convention: move focus WITH the page --
specifically to the pager control itself (matching what a real mouse click on it
already does) -- but ONLY when the old focus target would actually be left
straddling by the new position (`_release_focus_if_left_straddling`). A second,
independently-found timing bug had to be fixed to make this reliable: Textual's
`Widget.scroll_to()` is itself internally deferred by one `call_after_refresh` hop
before the new position even begins applying, so a straddle check scheduled only
one hop after calling it can still read stale, pre-scroll geometry -- fixed by
chaining a second hop (`_defer_focus_release_check`) specifically for this
one-shot decision (unlike `_ghost_clipped_buttons`, which tolerates occasional
staleness because later triggers re-invoke it).

Four new regression tests, matching the re-review's specified repros exactly:
`test_resize_does_not_strand_the_focused_button`,
`test_restore_active_does_not_strand_the_focused_button`,
`test_pager_releases_focus_instead_of_stranding_it` (the actual straddling
scenario had to be found empirically -- `active="home"`, Tab to `nav-schedules`,
a single forward page; the `nav-settings`/wrap-to-0 scenario tried first never
straddles, it goes straight to fully-off-screen, which would have been a vacuous
test), and `test_recenter_strip_and_focused_strip_button_survive_a_detached_bar`
(the 5-line crash-guard: detach a mounted bar, call the focus-aware methods
directly, confirm none raise).

**"Closed at the source" is now actually true**: every active-recenter trigger
either shares `_recenter_strip` or has its own, differently-shaped but equally
deliberate fix (the pager). The remaining residual is a single in-flight layout
pass (the scroll-then-geometry lag documented above), not an unhandled trigger.

Tests: `test_master_shell_navigation.py` **31/31** (27 + 4 new) x5 runs;
`test_product_maturity_phase6_focus_visual_sweep.py` 5/5 x3 runs;
`test_destination_visual_parity_correction.py` 108 passed / 5 failed / 113
collected (unchanged); the 13-file broader sweep re-run once, clean.

Files this round: `tldw_chatbook/UI/Navigation/main_navigation.py`
(`_recenter_periodic` renamed `_recenter_strip` and wired into `on_resize`/
`restore_active`/`on_mount`; `_release_focus_if_left_straddling` +
`_defer_focus_release_check`, new, pager-specific), `Tests/UI/
test_master_shell_navigation.py` (four new regression tests),
`.superpowers/sdd/library-polish-batch/task-5-report.md` (Fix Report round 3 +
explicit correction of round 2's "closed at the source" overstatement).

### Round 3 takeover: verification + one more correction

Round 3 died mid-sweep (API limit) with its work uncommitted; a takeover verified
the inherited diff before committing. Independently confirmed:
`test_master_shell_navigation.py` **31/31 x7** (not just x5 — re-run twice more
during takeover verification, including once mid-mutation-test), `test_product_
maturity_phase6_focus_visual_sweep.py` **5/5 x3**, and the diff-stat matches the
inherited partial exactly (487 insertions / 79 deletions across the three files).

**The 13-file sweep tally needed a correction.** A fresh run found **328 passed, 6
failed, 1 skipped** (not the "324/5/1" pattern reported every prior round) — a
6th failure, `test_product_maturity_phase1_first_run.py::test_clean_first_run_
launches_home_and_exposes_setup_orientation`, outside the known 5 (which remain
exactly `test_destination_visual_parity_correction.py`'s schedules/MCP set,
task-2560 territory). A/B-bisected by swapping `main_navigation.py`'s content at
each commit (never `git checkout`, plain content swap + restore from a verified
backup): passes cleanly at base `451d95340` (~2.5s, reliable across 4 runs),
fails at round 1 `071a6c403` onward, unchanged by round 3's own diff (byte-for-
byte identical failure with round-2 vs round-3 content). **Not a round-3
regression** — it predates this round and was already broken at commit time for
round 1 and round 2, whose own reports' "5 known, unchanged" tallies for this
sweep were therefore never actually re-verified against this specific file, or
were run under conditions where it happened to pass. Root-caused via a temporary,
fully-reverted diagnostic (print instrumentation added and removed via `Edit`,
confirmed `git diff --quiet` after): round 1's "ghost ⇒ disabled" fix (intentional
and correct on its own terms) makes a genuinely off-screen `nav-settings`
non-interactive, and this pre-existing test presses `#nav-settings` by ID at 140
columns without paging/scrolling first — a premise round 1 legitimately broke,
since a real mouse click could never land on a button in that state anyway. Filed
as **task-3224** (root cause + fix options, not fixed here — out of round 3's
scope).

**A second correction, more material: "closed at the source" needs one more
caveat.** Spot-checking RED reconstructibility (revert the specific wiring, watch
the test fail for the right reason, restore via `Edit`) confirmed `restore_active`
cleanly: reverting its `_recenter_strip` call reproduces the exact reported
geometry (`nav-settings` at `Region(x=57, width=15)` vs `strip.region.right==70`)
and the test goes red for that reason, then green again once restored. **The
`on_resize` repro did not clear the same bar.** Reverting `on_resize`'s wiring
alone (interval left fixed) did not turn `test_resize_does_not_strand_the_
focused_button` red — the shipped scenario (resize 80→90) never produces a
straddle at all in this bare-widget harness within the test's own timing budget,
regardless of which method `on_resize` calls; `strip.region` simply never grows
past its pre-resize width in that direction inside the window tested. The test is
green, but vacuously — it is not exercising what its docstring claims. Building an
alternative scenario that DOES reproduce a genuine straddle (same Tab-to-
`nav-settings` setup, but SHRINKING 80→70 instead of growing) confirmed the
underlying code difference is real at short windows (reverted code straddles
immediately and stays that way; fixed code initially corrects it) — but also
surfaced something round 3's own review did not catch: the fixed code's
correction is only transient, drifting back to the same straddling geometry by
roughly 0.3s post-resize, still short of the interval's 0.5s tick (so the interval
cannot be the explanation), with focus/active/`_deliberate_focus_id` all
unchanged throughout. Root cause not established in the time available. Filed as
**task-3225** (root-cause the drift with direct on_resize call-count/timing
evidence, then either land a second corrective pass or a genuinely-discriminating
test).

Net: the generalization itself (every settle trigger funnels through one
focus-aware entry point, or the pager's principled alternative) is real and the
right shape, and `restore_active` and the periodic interval are confirmed correct
by mutation testing. `on_resize`'s coverage is the one leg not yet confirmed to
the same standard — "closed at the source" is accurate for the refactor's
structure, not yet proven end-to-end for every trigger's behavior under all
timings.

Tests (round-3-takeover, independently re-run): `test_master_shell_navigation.py`
31/31 x7; `test_product_maturity_phase6_focus_visual_sweep.py` 5/5 x3; 13-file
sweep 328 passed / 6 failed (5 known + 1 newly-characterized, task-3224) / 1
skipped x1 (RED/GREEN mutation-tested for `restore_active`; `on_resize`'s own
test found vacuous, follow-up task-3225 filed).

## Review round 4 (escalated): the drift-back root cause, and both open items closed

**Root cause of task-3225's drift-back: the ghost rule was not geometry-neutral.**
`MainNavigationBar.DEFAULT_CSS`'s `.nav-button-clip-ghost` rule declared
`border: solid $background !important`. Textual's own `Button.-style-default`
gives a nav button `border: none; border-top: tall ...; border-bottom: tall ...`
— i.e. ZERO horizontal border cells — so replacing that with a four-edge `solid`
border made a ghosted button **2 cells wider** than the same button un-ghosted
(directly measured, `#nav-workflows` 14 -> 16, by toggling the class/`disabled`
in isolation). Because the strip is a horizontal layout, that reflowed every
button after the ghosted one 2 cells to the right — which could push a
previously fully-visible button (including the deliberately-focused one) into a
straddling position AFTER the corrective scroll had already landed. Nothing
re-checks after a ghost pass, precisely because the entire reason this design
was chosen over `display: none` is that ghosting is supposed to leave geometry
untouched. Timeline instrumentation of the shrink repro (per-button
region/width dumps at 50ms intervals plus enter/exit logs on `on_resize`,
`_recenter_strip`, `_scroll_active_destination_into_view`,
`_ghost_clipped_buttons`, `_update_overflow_hints`): corrective scroll lands at
~40ms with `nav-settings` at x=45 (visible); the trailing ghost pass ghosts
`nav-workflows` (x=-2, straddling the left edge); one layout pass later
`nav-workflows` measures w16 and `nav-settings` has moved to x=47 — straddling.
Round 3's hypothesis (a second `on_resize` firing with stale geometry) was
wrong: `on_resize` fires exactly once.

**Fix:** the ghost rule now declares no box-model property at all — colors and
text style only — so the ghosted box is identical to the un-ghosted box in
whichever CSS tier is winning. `visibility: hidden` was tried first (it is the
Textual primitive for "invisible but still occupies space") and rejected with
evidence: `Widget.region` returns an EMPTY region for an invisible widget
(`outer_size` stays 14, `region.width` drops to 0), and `_ghost_clipped_buttons`
skips any button with `region.width <= 0` — a once-ghosted button could never be
measured again, so it could never be un-ghosted.

**Scope of the drift-back (task-3225 AC #3), stated honestly:** it is a
bare-widget-harness-only defect. Under the real bundled stylesheet the ghost's
`border` declaration never applied at all — `components/_buttons.tcss`'s
`Button { border: none; }` is in the `CSS_PATH` tier, which outranks widget
`DEFAULT_CSS` regardless of `!important` — measured both ways: ghosted width 14
before and after this fix with `CSS_PATH` loaded. So no shipped user ever saw
the drift-back. It is still fixed rather than documented-and-left, because (a)
the design's core invariant is geometry-neutrality and this rule silently broke
it, (b) it made the bare-widget harness — which is the entire deterministic
regression suite for this feature — model a different layout regime than
production, and (c) it survived only by accident of an unrelated app-wide rule.

**task-3225 AC #2 (a genuinely non-vacuous test).**
`test_resize_does_not_strand_the_focused_button` was rewritten, not tweaked. The
shipped version was vacuous twice over: growing 80 -> 90 with
`active="schedules"` never produced a straddle at all, and even in a scenario
that DOES strand, two mechanisms heal it before any wall-clock assertion can
see it — the focus-aware periodic interval, and `_ghost_clipped_buttons`'s
best-effort `scroll_to_widget(focused)` nudge (traced: with `on_resize` reverted,
`scroll_x` went 86 -> 75 -> 96 inside 40ms because the nudge read a stale region
that still measured as straddling). The rewrite (a) uses `active="home"` so a
recenter-on-active drags the focused button FULLY off-screen rather than into a
straddle — the one case the nudge cannot rescue, and strictly the worse
outcome — and (b) suppresses the interval backstop for the duration of the
resize via a test-local subclass, isolating the property actually under test:
`on_resize`'s OWN pass must leave the focused button fully visible without the
interval cleaning up after it. It asserts full visibility (not merely
"not straddling") at 8 checkpoints spanning 0.8s. Mutation-tested: reverting
`on_resize` to `_scroll_active_destination_into_view` fails 3/3 with
`Region(x=141) vs strip Region(x=3, width=77)`; restoring passes 3/3.

New `test_ghosting_a_button_never_reflows_the_strip` pins the root cause
directly and without timing: ghost one button by hand, require every button's
region and the strip's `virtual_size` to be unchanged. Mutation-tested: restoring
`border: solid $background !important` fails it 3/3 with the exact +2 reflow.

**task-3224 closed test-side.** `test_clean_first_run_launches_home_and_exposes_
setup_orientation` pressed `#nav-settings` by id at 140 cols with the strip at
its default scroll position; round 1's ghosting made that a no-op and the test
timed out (reproduced 4/5 at this HEAD). The test's contract is "every one of
these destinations is reachable from the nav bar and renders its copy", not "a
programmatic press works on an off-screen widget" — and a real mouse click could
never reach that button — so the fix is test-side: a `_click_nav_destination`
helper reveals the target with the product's own affordance ("More ›") and only
presses once the button is genuinely inside the strip's viewport. The ghosting
contract is untouched (production code unchanged for this item). Verified 5/5
passing at ~3.3s (was a 13s timeout), and instrumented to confirm the reveal
branch actually fires (exactly once per run — the helper is load-bearing, not
decorative).

Tests (round 4): `test_master_shell_navigation.py` **32/32 x5**;
`test_product_maturity_phase6_focus_visual_sweep.py` **5/5 x3**;
`test_clean_first_run_launches_home_and_exposes_setup_orientation` **5/5**;
13-file nav sweep **330 passed / 5 failed / 1 skipped** — the 5 are the same
known pre-existing schedules/MCP failures, and the 6th (task-3224) is gone;
`Tests/Library --collect-only -q` 1118 collected; `check_bundle_sync.py` clean
(no bundle change: the fix is widget `DEFAULT_CSS`).

Live (tmux, 80 -> 90 cols, `active=home`, `⌃9 MCP` deliberately Tab-focused):
the focused non-active button stayed fully spelled out with its focus ring
(`[4m` + fg 233;236;238 on bg 88;109;130) at +0.3s, +1.2s, +2.5s and +6.5s, and
again after shrinking back to 80; both straddlers (`ts`, the tail of
"⌃4 Artifacts", and ` F`, the head of "F7 Lab") rendered fg 18;18;18 on
bg 18;18;18 — pixel-exact invisible.
<!-- SECTION:NOTES:END -->
