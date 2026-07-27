---
id: TASK-1020
title: >-
  Watchlists Overview needs a loading state so first-run guidance is not delayed
status: To Do
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
- [ ] #1 While the overview request is in flight the region shows a loading state, not the cards and not first-run copy
- [ ] #2 A user with existing sources never sees first-run copy, not even for one frame, on any visit
- [ ] #3 A brand-new user sees first-run guidance as soon as the load resolves, on both local and server backends
- [ ] #4 The Inspector's first-run text follows the same three states, so the two regions never disagree
- [ ] #5 A test covers the in-flight window with a deliberately slow backend, proven to fail against current code
<!-- AC:END -->
