---
id: TASK-2312
title: Stable tab strip and status line across Watchlists sections
status: Done
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: medium
---

## Description (the why)

UAT: the section tab strip changes position between tabs — outside the
content boxes on Overview/Sources, INSIDE the bordered Feeds region on
Items/Runs — so the navigation control visibly jumps as you use it. The
snapshot status line likewise wanders (top header line on Sources, buried
under the feed list on Items). The centre header also shows Sources-flavored
content ("No sources yet" + create CTAs) while the Overview tab is active.

UAT findings F2, F22, F23.

## Acceptance Criteria (the what)

- [x] The tab strip occupies the same visual position on every section.
- [x] The snapshot/status line has one consistent home.
- [x] Header content matches the active section (Overview header does not
      advertise Sources actions), or is explicitly section-agnostic in a way
      that reads as global status.
- [x] Existing region-gating and layout tests stay green.

## Implementation Plan (the how)

1. Trace exactly where the tab strip (`WatchlistsTabStrip`, `#wl-tabs`) and
   the snapshot status marker are constructed: `_build_centre_status_header`
   (used as the workbench's `header=` factory on every section EXCEPT
   "items") versus `_build_list_pane` (FEEDS's own content factory, which
   carried an independent inline copy of both, specifically for "items").
2. Map every test that could be affected by unifying these two call sites
   (tab-strip DOM position, header existence, focus tracking, empty-state
   copy) before touching production code, given the size of this screen.
3. Make `_build_centre_status_header` the ONE place either widget is ever
   built, unconditionally; strip the duplicate copy out of `_build_list_pane`.
4. Update every docstring/comment asserting the old "None on Read" shape.
5. Fix the resulting geometry-test fallout (the shared header is new
   `#wl-centre` content that existing row-budget tests did not account
   for) and add regression coverage for the behavioural changes nothing
   previously pinned (focus tracking on Read, header scope-refresh on Read).
6. Content-match fix for AC#3: reword the header's empty-state text to
   read as global status rather than borrowed Sources-tab copy.
7. Mutation verification; live verification in tmux across every section.

## Implementation Notes

**The bug was two independent construction sites for the same chrome, not
a CSS positioning issue.** No stylesheet rule anywhere positions
`#wl-tabs`/`#wl-centre-status` -- their screen position is 100% determined
by WHERE in the DOM they are mounted. `_build_centre_status_header` built
them as `#wl-centre-status`, the first child of `#wl-centre`, OUTSIDE
every bordered region box, wired as `WatchlistsWorkbench`'s `header=`
factory on 6 of 7 sections. `_build_list_pane` (FEEDS's own content,
Read-tab-only) built an independent SECOND copy of both, INSIDE its own
bordered `#watchlists-list-pane`. `compose_content` picked between them
with `header=(None if active_section == "items" else
_build_centre_status_header)` -- so on Read, the tab strip visibly jumped
into a bordered box, and jumped back out on every other tab.

**The fix: one factory, always wired.** `header=` is now unconditionally
`self._build_centre_status_header`; `_build_list_pane` keeps only what is
genuinely FEEDS-specific (the scope heading, the scoped source rows).
`watch_tree_scope`'s header refresh (`_refresh_centre_header_for_scope`)
is correspondingly unconditional -- it used to skip Read because the
header did not exist there.

**Geometry fallout, found by the existing test suite, not by guessing.**
Three tests encoded the OLD row budget, where `#wl-centre` on Read
contained ONLY the three centre regions (no separate header row):
`test_watchlists_soloed_feeds_fills_the_centre` and
`test_watchlists_feeds_cap_keeps_items_taller_when_it_actually_binds`
summed region heights against `#wl-centre`'s own height and came up 2
rows short (the header's own height -- 1 for the tab strip, 1 for the
one-line snapshot summary -- was real content of `#wl-centre` neither
test had ever had to account for). Both now include the header's height
in their sums; `expected_items` dropped from 18/17 to 16/15 accordingly,
matched empirically against the compositor, not assumed.

**Two behavioural changes nothing previously pinned, given a discriminating
test each** (mutation-verified both directions): focusing the tab strip on
Read now sets `_focus_in_centre_header` like every other section, instead
of matching the `wl-region-feeds` prefix first and setting `focused_region
= FEEDS` (a real change -- `z`/`Z` in the tab strip on Read now correctly
refuses via the header guard rather than acting on a stale FEEDS
reference); and a tree-scope change while on Read now repaints
`#wl-centre-status` the same way every other section already did.

**AC#3, scoped deliberately narrow.** The header's "no sources anywhere"
empty state ("No sources yet." + New source/Import OPML) is a genuinely
GLOBAL fact (there is no local Watchlists data at all), not a Sources-tab
one, but its OLD wording read as borrowed Sources-tab copy when seen on
Overview (UAT finding). Reworded to "No Watchlists sources yet." -- names
the app-level noun, satisfies the AC's "explicitly section-agnostic ...
reads as global status" branch -- without touching the New source/Import
OPML buttons themselves or the header/Overview-pane/Inspector triple-stack
redundancy task-2313 (empty-state sweep) owns explicitly as its own scope;
a bigger structural change here would have pre-empted that task's own
decision space.

**One bug found only by live verification, not by any test.** The
failure toast (task-2311) read "...API Key is required but not found..
Check Settings..." -- a visible double period, from unconditionally
appending ". Check Settings..." after a provider message that already
ends in its own period. Fixed with one `.rstrip(".")`; the existing
`_ExplodingChat` test fixture did not carry a trailing period so this
mutation-tested green before the fix and had to be corrected to actually
reproduce the trap (`Tests/Watchlists/test_watchlists_artifacts_pane.py`,
task-2311's own test file) before it could catch it.

### Verification

* New/extended tests: `Tests/UI/test_watchlists_destination_shell.py`
  (header exists + no duplicate tab strip on Read; the new focus-tracking
  pin), `Tests/Watchlists/test_watchlists_collections_screen.py` (scope
  refresh on Read; corrected the one test that read the summary line off
  `#watchlists-list-pane`, which no longer carries it), `Tests/UI/
  test_destination_shells.py` (reworded empty-state copy), `Tests/UI/
  test_destination_visual_parity_correction.py` (the two geometry fixes
  above, plus stale-comment corrections).
* Mutation-verified: 5 mutations (unconditional `header=`, the FEEDS
  duplicate-copy removal, the unconditional header scope-refresh, the
  AC#3 copy, the double-period fix), each reverted individually -> RED ->
  restored byte-exact (md5).
* Gates: `Tests/Watchlists/` + `Tests/UI/test_watchlists_destination_
  shell.py` + `Tests/UI/test_destination_shells.py` + `Tests/UI/
  test_destination_headers.py` + `Tests/UI/
  test_destination_visual_parity_correction.py` **733 passed, 1 skipped**,
  plus 7 pre-existing failures, none caused by this task: the 4 known
  schedules cases (task-2560), `test_stts_screen_composes_destination_
  header_in_the_lab_frame` and `test_main_navigation_overflow_hint_does_
  not_overlap_settings_at_default_size` (both zero-diff against
  `origin/dev` in their own screens, confirmed unrelated), and
  `test_subscriptions_alias_preserves_watchlists_navigation_context`
  (reproduces standalone against `origin/dev`'s own copy of this exact
  test file with this session's otherwise-untouched `app.py`, confirmed
  pre-existing).

### Live verification (235x52, fresh profile, real HN feed)

Tab strip + status line measured at the SAME row (row 8 of the captured
pane) on Overview, Sources, Items (Read), and Artifacts -- before this
fix, Read alone nested the strip inside FEEDS's bordered box. Header
content below the tab strip on Overview genuinely read as global status;
Sources/Items/Artifacts each show their own section content directly
below the same stable header. Live-caught and fixed the toast
double-period bug (see above) via a real Generate press against a
watchlist with a real ingested item and no provider key configured:
final toast read "Briefing generation failed using OpenAI: OpenAI API
Key is required but not found. Check Settings ▸ Providers & Models, then
press Generate again." -- single period, provider named, points at
Settings, all live-verified end to end (task-2311's own ACs, confirmed
live in this same session).

### Files

* `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` --
  `_build_centre_status_header`, `_build_list_pane`, `compose_content`,
  `watch_tree_scope`, `on_descendant_focus`/`watch_active_section`
  comments, the AC#3 copy, the double-period fix.
* `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py` --
  docstring corrections only (`__init__`'s `header` param,
  `refresh_header_content`); no behavioural change in this file.
* `Tests/UI/test_watchlists_destination_shell.py`, `Tests/Watchlists/
  test_watchlists_collections_screen.py`, `Tests/UI/
  test_destination_shells.py`, `Tests/UI/
  test_destination_visual_parity_correction.py`, `Tests/Watchlists/
  test_watchlists_artifacts_pane.py` (the double-period fix's test).
