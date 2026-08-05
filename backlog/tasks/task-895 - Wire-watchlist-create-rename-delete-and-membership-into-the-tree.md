---
id: TASK-895
title: >-
  Wire watchlist create/rename/delete and membership editing into the tree
status: Done
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
- [x] #1 A user can create a watchlist from the tree, and it appears without a manual refresh
- [x] #2 A user can rename and delete a watchlist, with delete explaining what happens to its sources
- [x] #3 Deleting a watchlist never orphans a source into invisibility — the affected sources appear under the Unassigned root
- [x] #4 A user can add a source to, and remove one from, a watchlist
- [x] #5 All five actions are disabled with a stated reason on the server backend, since no wire path exists
- [x] #6 Every method on `WatchlistBundleService` has a production caller
- [x] #7 Names are escaped before rendering, and a name that is a duplicate or is empty is rejected with a visible reason
<!-- AC:END -->

## Implementation Plan

1. Give `WatchlistTree` a two-row action strip above the roots (New / Rename /
   Delete, Add source / Remove) plus a `write_disabled_reason` reactive, and
   five request messages for the screen to handle.
2. Add a name-entry dialog and a source-picker dialog next to the existing
   watchlists modals.
3. Wire five worker-backed flows on `WatchlistsCollectionsScreen` that own the
   dialogs, call the service, and reload the tree.
4. Style the strip and the dialogs in `features/_watchlists.tcss`; regenerate
   the bundle.
5. Cover the widget half, the end-to-end flows against the real service, and
   the rendering under the production stylesheet.

## Implementation Notes

Wired the tree's five missing verbs. `WatchlistTree` now composes a two-row
action strip above the roots and posts `CreateWatchlistRequested`,
`RenameWatchlistRequested`, `DeleteWatchlistRequested`,
`AddSourceToWatchlistRequested` and `RemoveSourceFromWatchlistRequested`;
`WatchlistsCollectionsScreen` handles each with a worker that awaits a modal
(`push_screen_wait`), calls `WatchlistBundleService`, and re-runs
`_load_tree_data()` so the rail updates with no manual refresh.

**Approach and decisions**

- *Enablement is derived from `tree_scope`, not from a second selection
  concept.* Rename/Delete/Add-source arm on a watchlist scope, Remove arms on
  a source scope, so the rail's verbs act on the node the user is already
  looking at. `tree_scope` keeps its single writer: the flows route scope
  changes through `_apply_tree_scope`.
- *One string carries the blocked reason.* `write_disabled_reason` is used
  verbatim as both the disabled buttons' tooltip and a visible note
  (`#wl-tree-actions-unavailable`), so hover copy and on-screen copy cannot
  drift. The server-backend text is built through `DestinationRecoveryState`
  (`WC_SERVER_WRITE_RECOVERY`) so the blocker is described in the same
  taxonomy as the screen's other unavailable actions;
  `visible_copy`'s six-line form does not fit a 26-column rail, so
  `disabled_tooltip` is what renders. The service-missing case reuses the
  screen's existing `WC_SERVICE_UNAVAILABLE_COPY`.
- *Delete states the consequence before it happens, then shows it.* The
  confirmation names the source count and says the sources move to
  Unassigned; afterwards the scope moves to Unassigned so the user lands on
  them rather than on an id that no longer resolves. `ConfirmationDialog`
  (title + message) rather than the screen's `ConfirmDeleteDialog`, which can
  only render "Delete {name}?".
- *Empty and duplicate names are rejected visibly.* `WatchlistNameDialog`
  keeps the prompt open with the reason in `#watchlist-name-error`. The
  duplicate check is a user-facing guard, not a reimplementation of
  `_unique_name` — that still resolves genuine collisions (imports, races) by
  suffixing; this exists so a user who types an existing name is told, rather
  than silently getting "Security (2)".
- *The source picker uses one button per candidate, not a `Select`.* Button
  ids are built from the integer subscription id (always a legal Textual id,
  unlike a free-text name), and a `Select` on this screen posts `Changed` on
  mount.
- *Found and fixed a staleness gap this change introduced.* Breadcrumb labels
  are resolved from `_tree_watchlists`; before this task nothing could change
  that list while a scope was in view. Creating scopes to an id not yet in the
  list (crumb read "Watchlist 3") and renaming left the old name standing.
  `_load_tree_data` now re-resolves `_breadcrumb_labels` after loading.
- *CSS: `min-width: 0` is load-bearing.* Textual pins `min-width: 16` on every
  Button and `compact=True` only drops the border, so three action buttons
  would claim 48 columns in a 26-column rail and render clipped — which also
  fails the existing `test_watchlists_left_rail_is_labelled_when_expanded`
  ellipsis guard.

**Verification**

Mutation-checked the load-bearing assertions: removing the server branch, the
name escaping, the post-delete Unassigned scope, the breadcrumb re-resolve and
the `min-width: 0` rule each turn a test red. The AC #6 guard was rewritten to
walk the AST (a text scan for `.create(` matches `completions.create(` in
`OCR_Backends` and `os.rename(` in `Chat_Functions`, so it passed vacuously);
it now resolves calls through `_watchlist_bundle_service()` /
`watchlist_bundle_service` and their locals, and fails when a real call is
removed. Also driven live in an isolated tmux profile
(`TLDW_CONFIG_PATH` scratch config, profile deleted afterwards): the strip
fits the rail, create/empty-name-rejection/delete-confirmation/cancel all
behave, and the server-backend note renders wrapped and unclipped.

Green: `Tests/Watchlists` (166), `Tests/Subscriptions` (107),
`Tests/UI/test_watchlists_destination_shell.py` (48),
`Tests/UI/test_destination_visual_parity_correction.py` (104),
`Tests/UI/test_watchlists_inspector.py` (16),
`Tests/UI/test_destination_shells.py` (103, 1 pre-existing skip).
`check_bundle_sync.py` reports the bundle reproduces from source.

**Files**

- `tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py` — action strip, five
  messages, `write_disabled_reason`.
- `tldw_chatbook/UI/Watchlists_Modules/opml_dialogs.py` —
  `WatchlistNameDialog`, `WatchlistSourcePickerDialog`.
- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` — handlers,
  flows, `_tree_write_disabled_reason`, breadcrumb re-resolve.
- `tldw_chatbook/css/features/_watchlists.tcss` (+ regenerated
  `tldw_cli_modular.tcss`).
- `Tests/Watchlists/test_watchlist_tree.py`,
  `Tests/Watchlists/test_watchlists_collections_screen.py`,
  `Tests/UI/test_destination_visual_parity_correction.py`.
