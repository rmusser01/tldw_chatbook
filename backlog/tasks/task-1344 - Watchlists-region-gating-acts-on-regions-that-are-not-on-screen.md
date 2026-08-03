---
id: TASK-1344
title: Watchlists region gating acts on regions that are not on screen
status: In Progress
assignee: []
created_date: '2026-07-29 05:30'
labels:
  - watchlists
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The spec says "Only Read uses the three-pane split. Sources, Runs, Rules, and Artifacts take the
full centre width." Three related gaps remain after Phase D.

**FEEDS is always mounted** regardless of the active tab (unconditional in `_build_list_pane`),
the same violation Phase D fixed for CONTENT. Pre-existing from Phase C.

**The CONTENT gate collapses rather than unmounts.** `#wl-header-content` still measures
`height=1` on Sources at 160x42, so the region is not literally absent — it contributes a header
row. It no longer taxes the layout (that regression was fixed), but it is not "full centre width"
in the spec's sense.

**`Z` (solo) on CONTENT off the Read tab is ungated** (`watchlists_collections_screen.py:1578`).
Phase D gated the chevron and `z` so neither can persist a CONTENT collapse from a tab where the
region is invisible, and `collapsed_for_persistence()` returns the pre-solo baseline so solo cannot
corrupt persisted state. But solo still collapses FEEDS and ITEMS around a region the user cannot
see, leaving no expanded centre region — recoverable only by clicking a header.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FEEDS occupies the centre only on the Read tab, matching the CONTENT gating Phase D added
- [x] #2 Solo on a region that is not visible on the active tab is refused, with a notify, exactly as the chevron and z toggles now are
- [x] #3 A test asserts that no sequence of tab switches and region toggles leaves zero expanded centre regions
- [x] #4 A decision is recorded on whether gated regions should unmount or keep a one-row header, and the implementation matches it (UNMOUNT — see Implementation Notes)
<!-- AC:END -->

## Implementation Plan

1. Read the Phase D CONTENT-gating precedent (`_visible_region_layout`, `_refuse_content_toggle_off_read_tab`, `WatchlistsWorkbench._region_widget`) to mirror its shape for FEEDS.
2. Decide AC#4 (unmount vs one-row header) before writing the gate, since it determines the mechanism: a new orthogonal `hidden` concept on `WatchlistsWorkbench` (unmount) vs. extending the existing force-collapse-via-`RegionLayout.collapsed` derivation (header).
3. Discover and resolve the FEEDS-specific complication the CONTENT gate never had: `_build_list_pane` also builds the section tab strip (`#wl-tabs`) and the snapshot's loading/error/empty/summary markers, both of which many existing tests (and the app's own cross-tab visibility) depend on regardless of active tab. Extract both into an always-rendered header, tested against the full existing suite rather than assumed safe.
4. Implement `_hidden_centre_regions`/`_rendered_region_layout` (screen) and `hidden`/`header` (workbench), generalize the CONTENT-only refusal into `_refuse_region_gesture_off_read_tab`.
5. Update every existing test whose assertions encoded the pre-fix "FEEDS is always mounted" / "CONTENT collapses to a header" premises; add new tests for FEEDS gating, FEEDS solo/toggle refusal, and the AC#3 dead-end-layout sweep.
6. Mutation-test the FEEDS gate and the refusal gate by temporarily reverting each to RED, then restore.

## Implementation Notes

**AC#4 decision: UNMOUNT, not a one-row header.** Gated regions (FEEDS/CONTENT off the
Read tab) are now skipped entirely by `WatchlistsWorkbench.compose()` — no `#wl-header-*`,
no `#wl-region-*`. Reasoning: (a) it is what "full centre width" in the spec actually means,
vs. the header's `height=1` compromise the description called out; (b) the tab-switch path
already fully recomposes the whole screen (`watch_active_section`), so DOM focus is already
routinely rebuilt on every tab switch regardless of this change — unmounting a hidden region
introduces no NEW focus-loss mode; (c) persistence is unaffected either way, since it reads
only `self.region_layout` (the real, un-derived state) via `collapsed_for_persistence()`,
never anything this task's gating touches. The one place unmounting bites: a `focused_region`
left pointing at a region that is no longer mounted can't be re-focused by clicking (there is
no click target) — `_refuse_region_gesture_off_read_tab` is what makes that harmless (refuses
`z`/`Z` rather than silently corrupting the real layout), and it was needed under either AC#4
choice, not introduced because of this one.

**Design change from a straight port of the CONTENT mechanism.** The obvious mirror of Task
4's fix (`_visible_region_layout` force-collapsing CONTENT into the *same* `RegionLayout.
collapsed` set used for rendering) does not generalize to FEEDS cleanly, for two reasons
discovered while implementing, not anticipated in the task file:

1. `_build_list_pane` (FEEDS's factory) also builds the section tab strip (`#wl-tabs`) and the
   snapshot's own loading/error/empty/summary markers — genuinely cross-cutting chrome that
   many existing tests (and the shipped app) rely on being visible on *every* tab, not just
   Read. Gating FEEDS the same way CONTENT was gated (skip the factory entirely off Read)
   would have silently dropped both from every non-Read tab. Fix: extracted both into
   `_build_centre_status_header`, rendered by a new `WatchlistsWorkbench(header=...)` factory
   that mounts unconditionally above the (possibly-hidden) centre regions, wired only when
   `active_section != "items"` so it never coexists with FEEDS's own inline copy on Read.
2. Hiding FEEDS/CONTENT via a derived `RegionLayout.collapsed` union (as CONTENT's fix did)
   would leave ITEMS's own real collapse/solo bookkeeping (a Read-tab-only concept — e.g. a
   solo of CONTENT on Read collapses ITEMS too) leaking into non-Read tabs, where ITEMS is not
   a member of a three-pane split but the section's own always-shown pane. Fix: `hidden` is a
   new concept on `WatchlistsWorkbench`, orthogonal to `region_layout.collapsed` entirely
   (`_hidden_centre_regions`); `_rendered_region_layout` forces ITEMS out of `collapsed` on
   every non-Read tab regardless of what happened on Read. This also *simplifies* the old
   CONTENT-solo-pre-solo-baseline derivation `_visible_region_layout` needed, since hiding no
   longer touches `collapsed` at all.

**AC#2 generalization.** `_refuse_content_toggle_off_read_tab` renamed to
`_refuse_region_gesture_off_read_tab`, backed by one pure predicate
(`_region_hidden_on_active_section`, itself just `region in self._hidden_centre_regions()`).
`action_toggle_region`, `action_solo_region`, and `_on_region_toggled` all consult it —
the "one source of truth" the task asked for. Kept the task-1349 verb-naming discipline (the
notifying function is named as an action); the AST guard
(`Tests/Watchlists/test_no_side_effecting_predicates.py`) stayed green throughout.

**AC#3.** `Tests/UI/test_watchlists_destination_shell.py::
test_no_sequence_of_tab_switches_and_region_gestures_leaves_the_centre_empty` drives a
representative sequence (tab switches, `z`, `Z`, `[`/`]`) through the full production shell,
asserting after every step that at least one `#wl-region-*` centre region is actually mounted
— not just that `region_layout` looks sane. It specifically covers "solo CONTENT on Read, then
leave" (the PR #1091 review F2 report this task's description quotes) plus the newly-possible
"stale `focused_region` still pointing at a hidden region" and "ITEMS collapsed on Read, then
switch tabs" paths this task's own design surfaced.

**Test fallout.** FEEDS being gated broke the "tab strip position is a fixed row" assumption
one existing geometry test made (`test_watchlists_tab_strip_hit_regions_match_its_painted_labels`
captured `row`/`painted` once before looping over every tab-switch — valid pre-fix, since FEEDS
(and the strip nested in it) never moved; invalid now, since the strip's own container differs
structurally between Read (inside FEEDS's bordered body) and every other tab (the borderless
`#wl-centre-status` header) — fixed by recomputing both per iteration. A dozen more tests across
`test_watchlists_content_pane.py`, `test_watchlists_destination_shell.py`,
`test_destination_visual_parity_correction.py`, and `Tests/Watchlists/test_watchlists_
collections_screen.py`/`test_watchlists_artifacts_pane.py` needed either an added
`active_section = "items"` (FEEDS-content assertions that used to work at any tab) or an
updated assertion (CONTENT's collapsed-header checks, now unmount checks). The shared
`COMPACT_DESTINATION_CONTRACTS["watchlists_collections"]["object"]` selector (used by a
generic cross-destination compact-viewport test) was `#watchlists-list-pane` (FEEDS); changed
to `#wl-region-left_rail`, matching the rail-as-"object" convention "chat"/"library" already use.

**Mutation checks (both confirmed RED then restored, `git status --short` clean after):**
dropping the FEEDS half of `_hidden_centre_regions` (returning `{CONTENT}` only) reds
`test_feeds_region_is_gated_to_the_items_read_tab`; short-circuiting
`_refuse_region_gesture_off_read_tab` to always return `False` reds both
`test_solo_on_feeds_off_the_read_tab_is_refused` and the pre-existing
`test_solo_on_content_off_the_read_tab_is_refused`.

**Verification.** Full watchlists-relevant suite green: `Tests/Watchlists/` (467),
`Tests/UI/test_watchlists_content_pane.py` + `test_watchlists_destination_shell.py` +
`test_destination_visual_parity_correction.py` + `test_destination_shells.py` (689 combined,
1 pre-existing unrelated order-dependent flake in `test_watchlists_artifacts_pane.py::
test_a_claimed_watchlist_survives_an_artifacts_open` — passes in isolation and as part of its
own file both times it was run standalone; the test claims/releases a briefing via a module-level
`briefing_service._claim_briefing`, unrelated to region gating, and is not touched by this
change). `Tests/Watchlists/test_no_side_effecting_predicates.py` green throughout.

**Files touched:** `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py` (`hidden`/
`header` constructor params, `_is_sole_expanded_centre_region` hidden-aware),
`tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (`_hidden_centre_regions`,
`_rendered_region_layout` replacing `_visible_region_layout`, `_build_centre_status_header`,
`_watchlists_status_marker_widgets` extracted from `_build_list_pane`,
`_refuse_region_gesture_off_read_tab` replacing `_refuse_content_toggle_off_read_tab`),
`tldw_chatbook/css/features/_watchlists.tcss` (+`#wl-centre-status { height: auto; }`,
bundle regenerated via `build_css.py`), plus the test files listed above.

## Fix wave (whole-branch review, B1)

A whole-branch review of `fix/task-1344-region-gating` found AC#3 still violable: ITEMS is
never a member of `_hidden_centre_regions()` (it is force-shown off Read as the section's own
full-width pane, per `_rendered_region_layout`), so `_refuse_region_gesture_off_read_tab` never
refused a gesture aimed at it. A stale `focused_region == ITEMS` (set by `on_descendant_focus`
any time focus lands inside the section pane — simply using Sources/Runs/... off Read did it)
let `z`/`Z` reach `_apply_layout(region_layout.toggle(ITEMS))` against the REAL, persisted
layout with zero visible feedback (the render already forced ITEMS back out of `collapsed`).
Combined with FEEDS/CONTENT already collapsed on Read, this persisted a real ITEMS collapse to
disk, so returning to Read rendered three headers over an empty centre — the exact dead-end
AC#3 exists to rule out, surviving a restart, un-covered by the AC#3 test that shipped (which
never drove an off-Read ITEMS gesture).

**Fix:** `_refuse_region_gesture_off_read_tab` now refuses **any** `region in CENTRE_REGIONS`
off the Read tab (`active_section != "items"`), not only the ones `_hidden_centre_regions`
unmounts. `_region_hidden_on_active_section` is still consulted, but only to pick truthful
notify copy: FEEDS/CONTENT keep "only shown on the Read tab" (still true for them); ITEMS —
visible off Read, just not collapsible from there — gets "The pane layout can only be changed
on the Read tab." instead, since the old copy would be false for it. Both notify calls now pass
`markup=False`. `_rendered_region_layout`'s force-show behavior for ITEMS is unchanged (still
correct); only the gesture gate changed. Docstrings on `_refuse_region_gesture_off_read_tab`,
`_region_hidden_on_active_section`, `_rendered_region_layout`, and `action_solo_region` updated
to match — the old text asserted "a gesture aimed at [ITEMS] while off Read is never refused",
which was the bug, not a design fact.

**Tests.** Added `test_the_items_toggle_off_the_read_tab_neither_collapses_nor_persists` and
`test_solo_on_items_off_the_read_tab_is_refused` (direct units, mirroring the existing FEEDS
pair) plus `test_off_read_items_toggle_never_empties_the_read_centre_or_persists` (the missing
AC#3 leg: collapse FEEDS+CONTENT on Read, switch off Read, toggle/solo ITEMS there — both
refused, `region_layout`/`_last_persisted_collapsed` unchanged, and returning to Read still
shows an expanded centre) in `Tests/UI/test_watchlists_destination_shell.py`. Three pre-existing
tests (`test_collapsing_a_region_persists`, `test_focus_drives_which_region_z_collapses`,
`test_a_real_toggle_performs_exactly_one_write`) had exercised an ITEMS toggle from the default
`active_section == "overview"` — i.e., they were unwittingly relying on the B1 bug to pass;
updated each to switch to the Read tab (`active_section = "items"`) before the toggle they
actually measure, since region-layout gestures are a Read-tab-only concept now.

**Mutation checks (Edit-revert → RED → restore, `git status --short` clean after each):**
narrowing the gate back to `region in _hidden_centre_regions()` only reds all three new ITEMS
tests (`assert not True` / real-vs-expected `RegionLayout` mismatch / persisted-set mismatch);
separately, changing the gate's final `return True` to `return False` (notify still fires, but
the gesture is accepted anyway) reds the same three tests on their "must be refused" / "must not
touch the real layout" assertions. Both restored; `test_no_side_effecting_predicates.py` stayed
green throughout (the refusal is still named as an action and still owns the only `self.notify`
call in the pair).

**Verification.** `Tests/UI/test_watchlists_destination_shell.py` + `test_region_layout.py` +
`test_region_layout_store.py` + `test_watchlists_workbench.py` + `test_watchlists_content_pane.py`:
136 passed. `test_destination_visual_parity_correction.py` + `Tests/Watchlists/test_watchlists_
collections_screen.py` + `test_watchlists_artifacts_pane.py` + `test_no_side_effecting_
predicates.py`: 268 passed. `--collect-only Tests/UI Tests/Watchlists`: 8255 collected (+3 vs.
the review's 8252), no import errors.

**Files touched (this wave):** `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
(`_refuse_region_gesture_off_read_tab` gate + copy + docstrings), `Tests/UI/test_watchlists_
destination_shell.py` (3 new tests, 3 pre-existing tests updated to switch to Read before
their toggle).

## Fix wave 2 (Qodo correctness, two findings)

Two more Qodo findings against the same restructure (the always-rendered centre header
extracted for AC#1, `#wl-centre-status`/`#wl-tabs` via `WatchlistsWorkbench(header=...)`).

**Finding 1: stale off-Read header on a tree-scope change.** Off the Read tab,
`#wl-centre-status` renders the snapshot summary (`_watchlists_status_marker_widgets`), but
`watch_tree_scope` only ever called `_refresh_feeds_region_for_scope`, which rebuilds FEEDS --
unmounted off Read, so the call was a silent `NoMatches` no-op there. The header kept showing
the PREVIOUS scope's summary until some unrelated recompose came along. **Fix:** added
`WatchlistsWorkbench.refresh_header_content()`, mirroring `refresh_region_content` (rebuilds
`#wl-centre-status` in place from a fresh call to `self._header()`, never a full
`region_layout` recompose, for the same Inspector-preservation reason `watch_tree_scope`
already documents). `watch_tree_scope` now also calls a new worker,
`_refresh_centre_header_for_scope`, whenever `active_section != "items"` -- mirroring
`compose_content`'s own `header=` condition. Given its own exclusive worker GROUP
(`wc_header_scope_refresh`, distinct from `_refresh_feeds_region_for_scope`'s
`wc_feeds_scope_refresh`): both are called unconditionally-per-branch from the same watcher,
and a shared group would let whichever lands second cancel the other before it finishes.

**Finding 2: `z`/`Z` act on a stale region while focus is in the header.** The header/tab
strip (`#wl-centre-status`/`#wl-tabs`) is mounted directly under `#wl-centre`, outside every
`wl-region-*`/`wl-header-*` wrapper, so `on_descendant_focus` never updates `focused_region`
while focus sits there -- it keeps naming whatever region the user last actually visited. A
stale `focused_region == LEFT_RAIL` (from an earlier real focus), followed by tabbing into the
tab strip and pressing `z`, collapsed -- and PERSISTED -- the rail anyway:
`_refuse_region_gesture_off_read_tab` only gates `CENTRE_REGIONS`, so a rail's toggle was never
refused regardless of where real focus was. **Fix (chosen approach: on_descendant_focus sets a
sentinel, actions check it):** `on_descendant_focus` now also recognizes landing on
`#wl-centre-status` and sets a new flag, `_focus_in_centre_header` (cleared back to `False`
whenever focus lands in a real `wl-region-*`/`wl-header-*` match instead). `action_toggle_
region` and `action_solo_region` both check the flag first and silently no-op when it is set.
Chose the "flag on the widget" shape (option (a) from the review) over "actions re-derive
whether real focus is currently inside the named region" (option (b)): the latter would also
have had to treat every existing test that pokes `screen.focused_region = Region.X` directly
(without ever moving real focus) as "not live", which would have broken several pre-existing
AC#2/B1 tests that rely on exactly that pattern to reach `_refuse_region_gesture_off_read_tab`'s
own notify. The flag only changes behavior when a REAL `DescendantFocus` event lands in the
header, leaving every direct-`focused_region`-assignment test unaffected. `watch_active_section`
resets the flag to `False` unconditionally on every tab switch (a full recompose invalidates
whatever the old DOM's focus fact was; the header may not even exist on the new tab), so a
stale `True` can never wrongly refuse a legitimate gesture on a tab visited later.

Note: for `action_solo_region`, no scenario in the current codebase lets a stale
header-focused region actually MUTATE `region_layout` even without this guard --
`_refuse_region_gesture_off_read_tab` already refuses every `CENTRE_REGIONS` member off Read
unconditionally (task-1344 B1 above), and the header only exists off Read, so a centre-region
solo was always going to be refused anyway; a rail-focused solo was already blocked by solo's
own `focused_region not in CENTRE_REGIONS` check. The guard's only observable effect for solo
is suppressing an extraneous `self.notify(...)` keyed to a region the user is not looking at.
The toggle guard, by contrast, prevents a real, persisted mutation (rails are never gated by
`_refuse_region_gesture_off_read_tab` at all).

**Tests.** `Tests/Watchlists/test_watchlists_workbench.py`:
`test_refresh_header_content_rebuilds_the_header_in_place` and `test_refresh_header_content_
is_a_noop_without_a_header_factory` (workbench-level primitive, independent of the screen).
`Tests/Watchlists/test_watchlists_collections_screen.py`:
`test_centre_header_summary_follows_the_tree_scope_off_the_read_tab` (off Read, change
`tree_scope`, assert `#wc-watchlists-summary`'s text names the NEW scope, not the old).
`Tests/UI/test_watchlists_destination_shell.py`:
`test_z_with_focus_in_the_centre_header_does_not_toggle_a_stale_region` (focus the left rail,
then the tab strip, press `z` -- `region_layout` and the persisted collapse set are both
unchanged) and `test_capital_z_with_focus_in_the_centre_header_does_not_solo_a_stale_region`
(same shape for `Z`/ITEMS, asserting `notify` is never called).

**Mutation checks (Edit-revert -> RED -> restore, `git status --short` clean after each):**
(1) skipped the `_refresh_centre_header_for_scope()` call in `watch_tree_scope` ->
`test_centre_header_summary_follows_the_tree_scope_off_the_read_tab` REDs (summary still names
"All sources", the old scope). (2) short-circuited the `_focus_in_centre_header` check in
`action_toggle_region` to `False and ...` -> `test_z_with_focus_in_the_centre_header_does_not_
toggle_a_stale_region` REDs -- the log shows `collapsed_regions = ['left_rail']` actually
persisted. (3) same short-circuit in `action_solo_region` ->
`test_capital_z_with_focus_in_the_centre_header_does_not_solo_a_stale_region` REDs on
`notify.assert_not_called()` (one call: `"The pane layout can only be changed on the Read
tab."`) -- confirming the "notify-only" finding above, not a `region_layout` mutation. All
three restored; `git status --short` clean after each.

**Verification.** `Tests/UI/test_watchlists_destination_shell.py` + `test_region_layout.py` +
`test_region_layout_store.py` + `test_watchlists_workbench.py` + `test_no_side_effecting_
predicates.py` + `Tests/Watchlists/test_watchlists_collections_screen.py`: 129 passed (run
twice, once before and once after the mutation/revert cycles). `--collect-only Tests/UI
Tests/Watchlists`: 8260 collected (+5 vs. this wave's starting 8255), no import errors.
`test_no_side_effecting_predicates.py` green throughout -- `_focus_in_centre_header` is a plain
attribute, not a predicate-named function, and no new predicate-named function was added.

**Files touched (this wave):**
`tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py` (`refresh_header_content`),
`tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (`_refresh_centre_header_for_scope`,
`watch_tree_scope` calls it off Read, `_focus_in_centre_header` init + `on_descendant_focus` +
`watch_active_section` reset + `action_toggle_region`/`action_solo_region` guards),
`Tests/Watchlists/test_watchlists_workbench.py` (2 new tests),
`Tests/Watchlists/test_watchlists_collections_screen.py` (1 new test),
`Tests/UI/test_watchlists_destination_shell.py` (2 new tests).
