---
id: TASK-1631
title: "Watchlists test harness: _build_test_app misses local_watchlists_service's lazy db factory"
status: In Progress
assignee: []
created_date: '2026-07-31 19:03'
labels:
  - watchlists
  - tests
  - harness
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_screen_navigation.py::_build_test_app` patches
`tldw_chatbook.app.get_subscriptions_db_path` only for the duration of `TldwCli.__init__`. Inside
that init, `app._wire_watchlists_and_notifications_services` wires
`self.local_watchlists_service = LocalWatchlistsService(db_factory=lambda: SubscriptionsDB(get_subscriptions_db_path(), CLI_APP_CLIENT_ID))`
(`tldw_chatbook/app.py`) -- and `LocalWatchlistsService._db()` (`tldw_chatbook/Subscriptions/local_watchlists_service.py`)
calls `self.db_factory()` fresh on every access, not once at construction. So every call made
*after* `_build_test_app()` returns -- i.e. every call the running screen makes -- resolves
`get_subscriptions_db_path()` outside the patch, falling through to its real call-time fallback.
Meanwhile other init-time consumers (e.g. the `subscriptions_db`/`WatchlistProjection` built
directly inside `_wire_watchlists_and_notifications_services` while the patch is still live) keep
the patched path. The result is two `SubscriptionsDB` instances pointed at two different temp
files within the same test app -- a silent wrong-DB-instance split.

This is confined to pytest's HOME-redirected tmp directory (conftest's autouse
`isolate_test_environment` fixture), never real user data, so it is a test-harness correctness bug,
not a production one. But it has now forced three independent workarounds rather than one fix:
`Tests/UI/test_watchlists_inspector.py`'s `_seed_new_item` (seeds through
`app.local_watchlists_service._db()`, with a docstring explaining why), `Tests/UI/test_watchlists_read_status.py`
(same `app.local_watchlists_service._db()` seeding pattern, four call sites -- the identical need in
`Tests/UI/test_watchlists_item_actions.py` was resolved by moving that file's real-DB-write
assertion here rather than re-solving the split a second time), and Task 6's `db_factory`
monkeypatch (`monkeypatch.setattr(app.local_watchlists_service, "db_factory", lambda: db)`, two
occurrences) in `Tests/Watchlists/test_watchlists_artifacts_pane.py`. Each new suite that touches
watchlist item state through both paths has to rediscover and re-solve this rather than get it for
free from the harness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `_build_test_app()` gives every consumer of watchlist data -- `local_watchlists_service`, the eagerly-built `subscriptions_db`/`WatchlistProjection`, and any other `get_subscriptions_db_path()` caller -- the same on-disk database, whether the patch scope is widened or the factory's first resolution is cached
- [x] #2 The three documented workarounds (`test_watchlists_inspector.py`'s `_seed_new_item`, `test_watchlists_read_status.py`'s direct `_db()` seeding, `test_watchlists_artifacts_pane.py`'s `db_factory` monkeypatch) are removed or reduced to a one-line comment, since the split they route around no longer exists
- [x] #3 `Tests/Subscriptions/`, `Tests/Watchlists/`, `Tests/UI/test_watchlists_inspector.py`, and `Tests/UI/ -k watchlist` remain green after the harness change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fix in the harness (`Tests/UI/test_screen_navigation.py::_build_test_app`): make `get_subscriptions_db_path` resolve to the SAME temp path for the app's whole lifetime (widen the patch beyond __init__, or pre-resolve the path and patch with a constant-returning lambda held for the app fixture's duration), so the lazy `db_factory` and the eager init-time consumers agree. Prefer harness-side over caching in production `LocalWatchlistsService._db()` (production semantics unchanged).
2. Remove/reduce the three documented workarounds (inspector `_seed_new_item` re-route, read_status direct `_db()` seeding, artifacts_pane `db_factory` monkeypatch ×2) per AC #2.
3. AC #3 sweep: Tests/Subscriptions/, Tests/Watchlists/, test_watchlists_inspector.py, Tests/UI -k watchlist.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Harness fix** (`Tests/UI/app_factory.py::_build_test_app`): the
`get_subscriptions_db_path` patch is now `start()`ed independently of the
function's own `with ExitStack()` (which still owns every other init-time-only
patch) and tracked in a new module-level `_active_service_patches` list, so it
stays in effect for the app's whole life instead of reverting the instant
`_build_test_app()` returns. `Tests/conftest.py`'s existing
`drain_test_app_user_data_dirs` autouse fixture (already responsible for
draining `_created_dirs` after each test) now also calls the new
`drain_active_service_patches()` first, so the patch never leaks into the next
test. This makes the eager, init-time consumers (`subscriptions_db` /
`WatchlistProjection` / `watchlist_bundle_service`) and the lazy
`LocalWatchlistsService.db_factory` (re-resolved on every call) agree on one
on-disk file for as long as the built app is used -- closing every call site
named in AC #1, including background workers like
`_backfill_subscription_items_fts` that only run after a real `on_mount`.

**New harness test** (`Tests/UI/test_screen_navigation.py`):
`test_local_watchlists_service_db_factory_resolves_the_same_path_as_the_eager_subscriptions_db`
compares `app.watchlist_bundle_service.db.db_path` (the eager side) against
`app.local_watchlists_service.db_factory().db_path` (the lazy side, called
well after `_build_test_app()` returns) -- resolved paths, not object identity,
since `db_factory()` builds a fresh `SubscriptionsDB` every call by design.
Mutation-verified directly: temporarily reverted the patch back into the
`ExitStack` (the original bug) and confirmed this test reds (lazy path fell
through to the real, HOME-redirected fallback while the eager side kept the
temp path), then restored the fix and confirmed green again.

**AC #2 workarounds:**
- `test_watchlists_inspector.py::_seed_new_item` -- docstring's 14-line
  "two different temp files" explanation replaced with a 3-line note that
  task-1631 unified the paths; the seeding-through-`_db()` code and
  `_open_items_with_seeded_item`'s `watchlist_bundle_service._db = db`
  realignment are UNCHANGED (kept as a harmless belt-and-suspenders identity
  pin per the task's "still a convenient utility" allowance, since removing
  and re-verifying cross-connection SQLite visibility wasn't worth the risk
  for a low-priority harness task).
- `test_watchlists_artifacts_pane.py` -- both `db_factory` monkeypatch
  occurrences (with their ~18-line and ~5-line justifying comments) are
  DELETED outright (not just trimmed): they are now fully redundant, since
  `db_factory()` already resolves to the same file `watchlist_bundle_service.db`
  uses. Replaced with a 3-line comment pointing at task-1631. Verified by
  running the full `Tests/Watchlists/` suite after deletion (365 passed).
- `test_watchlists_read_status.py` -- inspected for the "two DBs" explanation
  named in the task description; none exists in this file (no docstring or
  comment mentions the split -- only the bare `app.local_watchlists_service.
  _db()` seeding pattern, four call sites). Left unchanged: there is nothing
  stale to trim, and the pattern is already the sanctioned "convenient utility
  routed through the now-unified path" case.

**AC #3 verification** (all via `.venv/bin/python -m pytest`, no `-q`):
- `Tests/Subscriptions/`: 535 passed
- `Tests/Watchlists/`: 365 passed (one apparent failure,
  `test_export_feed_press_survives_an_os_error_from_the_service`, turned out to
  be a resource-contention flake from accidentally running two concurrent full
  suites at once during verification, not a real regression -- confirmed by
  re-running the file alone, twice, both clean)
- `Tests/UI/test_watchlists_inspector.py`: 34 passed
- `Tests/UI/ -k watchlist` (excluding the pre-existing, unrelated
  `test_watchlists_source_create_form.py::test_a_source_can_be_created_end_to_end_through_the_form`,
  confirmed failing 2/2 in isolation with a Textual `Select` widget
  `NoMatches: No nodes match '#label'` error unrelated to this change): 253
  passed

**Files changed:** `Tests/UI/app_factory.py`, `Tests/conftest.py`,
`Tests/UI/test_screen_navigation.py` (new harness test),
`Tests/UI/test_watchlists_inspector.py`,
`Tests/Watchlists/test_watchlists_artifacts_pane.py`.

Left In Progress per instructions (not moved to Done in this pass).
<!-- SECTION:NOTES:END -->

Review correction (2026-08-02): the residual-risk audit line "no test builds two apps in one test"
was overstated — two same-test double-builds exist (`test_console_resize_reflow.py` converge test,
`test_console_fleet_discoverability.py` restart test); both are safe because the first app is fully
torn down before the second build, i.e. safe by test structure, not by harness guarantee.
