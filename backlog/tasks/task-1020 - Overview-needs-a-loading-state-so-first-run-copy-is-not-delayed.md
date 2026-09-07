---
id: TASK-1020
title: >-
  Watchlists Overview needs a loading state so first-run guidance is not delayed
status: Done
assignee: []
created_date: '2026-07-28 00:30'
labels:
  - watchlists
  - ux
  - followup
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`OverviewPane.profile_is_empty` returns `False` while `overview_data` is missing `total_sources`, so first-run guidance cannot appear until the overview worker lands. On the **server** backend `get_overview_data` is a network call, so a brand-new user can sit looking at the non-first-run UI for as long as that request takes.

Raised by Qodo as finding #4 on PR #1017 and deliberately not fixed there, because the obvious alternative is worse: without that guard, `overview_data` starts `{}`, every key reads falsy, and **every** user — including one with hundreds of sources — gets a flash of first-run copy on every visit to the screen. The guard trades a one-time delay for a new user against a permanent flash for everyone.

The real fix is that the region has **three** states, not two: *loading*, *empty*, and *populated*. While the request is in flight the screen should show neither the cards nor the first-run copy. That changes the Overview contract and the Inspector text that keys off the same predicate, which is why it wants its own task rather than being folded into a UAT fix wave.

Note the failure path already behaves correctly: `_refresh_overview_data`'s `except` branch publishes `total_sources: 0`, so a failed or timed-out load resolves to first-run copy rather than sticking on the wrong UI. It is only the in-flight window that is wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 While the overview request is in flight the region shows a loading state, not the cards and not first-run copy
- [x] #2 A user with existing sources never sees first-run copy, not even for one frame, on any visit
- [x] #3 A brand-new user sees first-run guidance as soon as the load resolves, on both local and server backends
- [x] #4 The Inspector's first-run text follows the same three states, so the two regions never disagree
- [x] #5 A test covers the in-flight window with a deliberately slow backend, proven to fail against current code
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`OverviewPane.profile_is_empty` answered a two-valued question about a three-valued world, so `False` had to stand for both "loaded, and populated" and "nothing has answered yet". `OverviewPane.profile_state` now returns `LOADING`/`EMPTY`/`POPULATED` from the same `"total_sources" not in data` test that guarded the old predicate, and `profile_is_empty` is kept as a thin `== EMPTY` wrapper so both resolved answers are byte-for-byte what they were.

- **Overview region**: `LOADING` renders one muted line (`#overview-loading`) and returns — neither the seven cards nor the first-run panel, because both are claims about a profile nothing has reported on. A single line rather than a skeleton: the local backend resolves in milliseconds, and a shimmering placeholder would be the flash this task exists to remove.
- **Inspector (AC#4)**: `first_run: bool` became `profile_state: str`, seeded from `WatchlistsCollectionsScreen._watchlists_profile_state()` — the *same* call the Overview region makes, so the two regions cannot disagree by construction. During the in-flight window the rail no longer tells a brand-new user to "Select a source, run, item, rule, or notification", naming five things that do not exist.
- **AC#3** needs nothing new: `_refresh_overview_data`'s `except` branch already publishes `total_sources: 0`, so a failed or timed-out load resolves to `EMPTY` rather than sticking on `LOADING`. That is asserted by the existing `test_watchlists_first_run_replaces_empty_cards_with_guidance`, which still passes.

Tests use a scope service that holds every overview query open on an `asyncio.Event` — a deliberately slow backend at a real seam, which is what `server` looks like on a cold network. AC#2's guard is asserted by sampling **every frame** from mount until the cards appear and requiring first-run copy never to have been rendered; it passes before and after, which is the point — it is the regression the old guard existed to prevent and must survive the fix.

One existing test needed its fixture updated rather than its expectation: `test_data_recompose_releases_a_capture_that_lands_in_the_deferred_teardown_window` captured `#overview-total-sources` on a pane whose `data` was still `{}`, which is now the loading state and renders no cards. It seeds one payload first; the recompose it exercises is unchanged.

**Files:** `UI/Watchlists_Modules/overview_pane.py`, `UI/Watchlists_Modules/inspector_pane.py`, `UI/Screens/watchlists_collections_screen.py`, `css/features/_watchlists.tcss` (+ regenerated bundle), `Tests/UI/test_watchlists_overview_loading_state.py` (new), `Tests/Watchlists/test_watchlists_overview_pane.py`.
<!-- SECTION:NOTES:END -->
