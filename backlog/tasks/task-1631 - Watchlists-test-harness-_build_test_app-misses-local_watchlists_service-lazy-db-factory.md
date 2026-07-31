---
id: TASK-1631
title: "Watchlists test harness: _build_test_app misses local_watchlists_service's lazy db factory"
status: To Do
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
- [ ] #1 `_build_test_app()` gives every consumer of watchlist data -- `local_watchlists_service`, the eagerly-built `subscriptions_db`/`WatchlistProjection`, and any other `get_subscriptions_db_path()` caller -- the same on-disk database, whether the patch scope is widened or the factory's first resolution is cached
- [ ] #2 The three documented workarounds (`test_watchlists_inspector.py`'s `_seed_new_item`, `test_watchlists_read_status.py`'s direct `_db()` seeding, `test_watchlists_artifacts_pane.py`'s `db_factory` monkeypatch) are removed or reduced to a one-line comment, since the split they route around no longer exists
- [ ] #3 `Tests/Subscriptions/`, `Tests/Watchlists/`, `Tests/UI/test_watchlists_inspector.py`, and `Tests/UI/ -k watchlist` remain green after the harness change
<!-- AC:END -->
