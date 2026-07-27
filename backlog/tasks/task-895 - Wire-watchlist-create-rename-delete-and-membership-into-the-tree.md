---
id: TASK-895
title: >-
  Wire watchlist create/rename/delete and membership editing into the tree
status: To Do
assignee: []
created_date: '2026-07-27 14:30'
labels:
  - watchlists
  - followup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Five methods on `WatchlistBundleService` have no production caller: `create`, `rename`, `delete`, `add_source` and `remove_source`. They are complete and tested; nothing reaches them.

Phase C shipped the read half of the watchlist tree — navigate, scope, count — and left the write half unbuilt. So a user can browse watchlists but cannot make one, and the only watchlists that can exist are ones seeded outside the app.

This is a milder form of what task-813 addressed. `migrate_folders` was orphaned *and* worthless by construction, so it was deleted. These five are orphaned but genuinely wanted: they are the tree's missing verbs. Filing them so the gap is tracked rather than rediscovered.

Note the server-backend constraint established during the spec: `SourceUpdateRequest` carries no `group_ids` and neither group request carries members, all with `extra="forbid"`. So watchlist creation and membership editing must be disabled on the server backend, not merely hidden — there is no wire path for them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user can create a watchlist from the tree, and it appears without a manual refresh
- [ ] #2 A user can rename and delete a watchlist, with delete explaining what happens to its sources
- [ ] #3 Deleting a watchlist never orphans a source into invisibility — the affected sources appear under the Unassigned root
- [ ] #4 A user can add a source to, and remove one from, a watchlist
- [ ] #5 All five actions are disabled with a stated reason on the server backend, since no wire path exists
- [ ] #6 Every method on `WatchlistBundleService` has a production caller
- [ ] #7 Names are escaped before rendering, and a name that is a duplicate or is empty is rejected with a visible reason
<!-- AC:END -->
