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
