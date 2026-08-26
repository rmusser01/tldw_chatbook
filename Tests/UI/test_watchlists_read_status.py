"""Reading behaviour for the Watchlists content reader — Task 5.

Opening an item in the reader marks it read. Because marking read destroys
the unread list, there is an explicit toggle back (`Mark unread`,
`#content-mark-unread-button` in `ContentPane`). Both directions reuse the
existing item-status API — `SubscriptionsDB.mark_item_status`, reached
through `LocalWatchlistsService.update_item` / `_update_item_status` on the
screen, the same path the Inspector's `Ingest`/`Ignore` buttons already use
(see `Tests/UI/test_watchlists_item_actions.py`) — rather than any new
column. That API updates a single `subscription_items` row by its own id,
never by a (watchlist, item) pair, so status is global by construction: the
same article read from "All sources" reads as read in every watchlist whose
sources include it. `test_read_status_is_global_across_watchlists` pins that
down directly, so it is stated rather than later discovered as a bug.

Fix round 1 (coordinator review) added
`test_selecting_an_item_does_not_break_keyboard_navigation`: the auto
mark-read-on-open originally reused `_update_item_status`'s default refresh
path, which calls `_refresh_overview_data()` -- and `overview_data` is
`reactive({}, recompose=True)` on the screen, so every single item
*selection* (not just a deliberate button press) forced a full screen
recompose that replaced the mounted `ItemsPane`/`DataTable` wholesale and
dropped keyboard focus. `_mark_item_read_on_open` now calls
`_update_item_status(..., refresh=False, patch_item=item)` instead, patching
the same dict object already held by `ArticleListPane.items` in place rather
than reloading and recomposing. (TASK-3072 later swapped the pane itself:
`ArticleListPane` + `ListView` now sit where `ItemsPane` + `DataTable` did.)
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button, ListView

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane

pytestmark = pytest.mark.unit


def _seed_one_new_item(db, *, content_hash: str = "hash-read-status"):
    """Add one subscription with one "new" item, and return (source_id, item_id)."""
    source_id = db.add_subscription(
        name="Summit Route", type="rss", source="https://summitroute.com/blog/feed.xml"
    )
    with db.transaction() as conn:
        item_id = persist_subscription_item(
            conn,
            source_id,
            {
                "url": "https://summitroute.com/blog/2024/lightsail-part-2/",
                "title": "Lightsail object storage concerns - Part 2",
                "content_hash": content_hash,
                "status": "new",
            },
            run_id=None,
            now="2026-07-28T09:00:00+00:00",
        )
    return source_id, item_id


@pytest.mark.asyncio
async def test_opening_an_item_marks_it_read():
    """Selecting an item in the reader -- with no other action -- must move
    it from "new" to "reviewed" in the real database.

    Deliberately does NOT press any button: `Mark reviewed` already has its
    own test (`test_mark_reviewed_writes_the_new_status_to_the_database`).
    This one is about the *open* itself doing the marking.
    """
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _source_id, item_id = _seed_one_new_item(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen.active_section = "items"
        await pilot.pause(0.3)

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        for _ in range(40):
            await pilot.pause()
            if pane.items:
                break
        assert pane.items, "the seeded item must reach the Items pane"

        pane.select_item_by_id(str(pane.items[0]["id"]))
        for _ in range(60):
            await pilot.pause()
            if db.get_new_items(status="reviewed", limit=10):
                break

    rows = db.get_new_items(status="reviewed", limit=10)
    assert [row["id"] for row in rows] == [item_id], (
        "opening the item in the reader must mark it read (status -> "
        "'reviewed') without any further action"
    )
    # And it must actually have left the unread bucket.
    assert db.get_new_items(status="new", limit=10) == []


@pytest.mark.asyncio
async def test_the_unread_toggle_restores_unread():
    """Marking read destroys the unread list, so this must be reversible.

    Opens the item (marks it read), then presses the reader's explicit
    `Mark unread` button and confirms the item's status returns all the way
    to "new" -- back in the unread bucket, not stuck at "reviewed".
    """
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _source_id, item_id = _seed_one_new_item(db, content_hash="hash-toggle")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen.active_section = "items"
        await pilot.pause(0.3)

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        for _ in range(40):
            await pilot.pause()
            if pane.items:
                break
        assert pane.items, "the seeded item must reach the Items pane"

        pane.select_item_by_id(str(pane.items[0]["id"]))
        for _ in range(60):
            await pilot.pause()
            if db.get_new_items(status="reviewed", limit=10):
                break
        assert [row["id"] for row in db.get_new_items(status="reviewed", limit=10)] == [
            item_id
        ], "the open must have marked the item read before the toggle is exercised"

        content_pane = screen.query_one("#watchlists-content-pane", ContentPane)
        content_pane.query_one("#content-mark-unread-button", Button).press()
        for _ in range(60):
            await pilot.pause()
            if db.get_new_items(status="new", limit=10):
                break

    rows = db.get_new_items(status="new", limit=10)
    assert [row["id"] for row in rows] == [item_id], (
        "the explicit unread toggle must restore the item to 'new', "
        "reversing the automatic mark-read from opening it"
    )
    assert db.get_new_items(status="reviewed", limit=10) == []


@pytest.mark.asyncio
async def test_read_status_is_global_across_watchlists():
    """The same item read from 'All sources' is read everywhere.

    Asserted so the behaviour is pinned as intended rather than discovered as
    a bug later: one source is attached to TWO separate watchlists via
    `watchlist_sources`, and the item is opened through the reader with no
    watchlist scope selected (the Items pane's only fetch path today, "All
    sources" in effect). Both watchlists must resolve to the *same* row --
    proven by (a) exactly one `subscription_items` row for the item existing
    at all, and (b) both watchlists' own join to the shared source reporting
    the identical "reviewed" status -- not a per-watchlist copy that could
    independently disagree.
    """
    app = _build_test_app()
    db = app.local_watchlists_service._db()

    source_id = db.add_subscription(
        name="Summit Route", type="rss", source="https://summitroute.com/blog/feed.xml"
    )
    with db.transaction() as conn:
        conn.execute("INSERT INTO watchlists (name) VALUES ('Morning')")
        conn.execute("INSERT INTO watchlists (name) VALUES ('Security')")
        watchlist_ids = [
            row[0]
            for row in conn.execute("SELECT id FROM watchlists ORDER BY id").fetchall()
        ]
        for watchlist_id in watchlist_ids:
            conn.execute(
                "INSERT INTO watchlist_sources (watchlist_id, subscription_id) VALUES (?, ?)",
                (watchlist_id, source_id),
            )
        item_id = persist_subscription_item(
            conn,
            source_id,
            {
                "url": "https://summitroute.com/blog/2024/lightsail-part-2/",
                "title": "Lightsail object storage concerns - Part 2",
                "content_hash": "hash-global",
                "status": "new",
            },
            run_id=None,
            now="2026-07-28T09:00:00+00:00",
        )
    assert len(watchlist_ids) == 2, "fixture must attach the source to two distinct watchlists"

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen.active_section = "items"
        await pilot.pause(0.3)

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        for _ in range(40):
            await pilot.pause()
            if pane.items:
                break
        assert pane.items, "the seeded item must reach the Items pane"

        # No watchlist scope is applied to this fetch -- "All sources".
        pane.select_item_by_id(str(pane.items[0]["id"]))
        for _ in range(60):
            await pilot.pause()
            if db.get_new_items(status="reviewed", limit=10):
                break

    # Through `transaction()` like every other DB access in this repo
    # (CLAUDE.md), not `db.conn` directly: the context manager is what owns
    # commit/rollback on this connection, and a test that reaches around it
    # is quietly asserting that reaching around it is fine.
    with db.transaction() as conn:
        row_count = conn.execute(
            "SELECT COUNT(*) FROM subscription_items WHERE id = ?", (item_id,)
        ).fetchone()[0]
    assert row_count == 1, (
        "there must be exactly one canonical item row -- a per-watchlist "
        "copy would defeat the global-status guarantee even if both copies "
        "happened to agree right now"
    )

    for watchlist_id in watchlist_ids:
        with db.transaction() as conn:
            scoped = conn.execute(
                """
                SELECT si.status FROM subscription_items si
                JOIN watchlist_sources ws ON ws.subscription_id = si.subscription_id
                WHERE ws.watchlist_id = ? AND si.id = ?
                """,
                (watchlist_id, item_id),
            ).fetchone()
        assert scoped is not None, (
            f"watchlist {watchlist_id} shares this source and must still see the item"
        )
        assert scoped[0] == "reviewed", (
            f"watchlist {watchlist_id} must see the item as read too -- status is "
            "global, not scoped to whichever watchlist it was opened from"
        )


@pytest.mark.asyncio
async def test_selecting_an_item_does_not_break_keyboard_navigation():
    """Fix round 1, CRITICAL regression.

    Before this fix, `_mark_item_read_on_open` called `_update_item_status`
    with its default `refresh=True`, which ends with `_load_items()` +
    `_refresh_overview_data()` -- and `overview_data` is
    `reactive({}, recompose=True)` on the screen. So EVERY single item
    selection (not just a deliberate button press) forced a full screen
    recompose, which rebuilds every region through its factory and replaces
    the mounted `ItemsPane`/`DataTable` wholesale (today:
    `ArticleListPane`/`ListView`, TASK-3072). Proven live: with the old
    behaviour, one `down` press detached the pane, reset the cursor
    to row 0, cleared screen focus, and a SECOND `down` press did nothing at
    all -- the list became unusable by keyboard after the very first
    selection.

    Drives the real keyboard path (`pilot.press`, a focused `ListView`)
    rather than `select_item_by_id` directly, since the bug is specifically
    about what selecting a row does to the table hosting it, not about
    selection itself (already covered by the other tests in this file).

    ListView shape note (TASK-3072): the four seeded items share one
    `created_at`, so they render under a single date-group header -- child 0
    is the disabled header, the rows are children 1-4. `_ArticleListView`
    starts the cursor at the first enabled row, so the first `down` selects
    "Item 0" (index 1) and the second "Item 1" (index 2).
    """
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    source_id = db.add_subscription(
        name="Summit Route", type="rss", source="https://summitroute.com/blog/feed.xml"
    )
    with db.transaction() as conn:
        for index in range(4):
            persist_subscription_item(
                conn,
                source_id,
                {
                    "url": f"https://summitroute.com/blog/2024/nav-item-{index}/",
                    "title": f"Item {index}",
                    "content_hash": f"hash-nav-{index}",
                    "status": "new",
                },
                run_id=None,
                now="2026-07-28T09:00:00+00:00",
            )

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen.active_section = "items"
        await pilot.pause(0.3)

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        for _ in range(40):
            await pilot.pause()
            if len(pane.items) >= 4:
                break
        assert len(pane.items) == 4, "all four seeded items must reach the Items pane"

        list_view = pane.query_one("#items-table", ListView)
        list_view.focus()
        await pilot.pause(0.2)

        await pilot.press("down")
        # Let the silent, now non-refreshing mark-read worker run to
        # completion before checking anything -- the regression this test
        # guards against is specifically that worker's side effect.
        for _ in range(30):
            await pilot.pause()

        assert pane.is_attached, (
            "selecting a row must not detach the mounted ArticleListPane via a "
            "screen-level recompose"
        )
        assert screen.query_one("#watchlists-items-pane", ArticleListPane) is pane, (
            "the screen must still be hosting the SAME ArticleListPane instance, "
            "not one rebuilt from scratch"
        )
        current_list = pane.query_one("#items-table", ListView)
        assert current_list is list_view, "the ListView itself must survive too"
        assert current_list.has_focus, (
            "a recompose drops focus entirely -- the list must still be "
            "focused after a plain row selection"
        )
        assert current_list.index == 1, "the cursor must stay where the user put it"

        await pilot.press("down")
        await pilot.pause(0.3)
        assert current_list.index == 2, (
            "a SECOND arrow key must still move the cursor -- before the fix "
            "this did nothing at all once the list had been replaced"
        )
        assert pane.selected_item is not None
        assert pane.selected_item.get("title") == "Item 1"


@pytest.mark.asyncio
async def test_a_cancelled_mark_read_still_leaves_the_cached_dict_coherent():
    """TASK-1541, Qodo redesign -- desired-status coalescing, not cancellation.

    An earlier fix wave gave the read/unread pair a cross-item
    `exclusive=True` "supersede" worker group so a fast `j`/`k` run would not
    queue one write per keystroke, and patched a `CancelledError` handler to
    keep the cache coherent when that supersede cancelled an in-flight
    write. A later whole-branch re-review found that model unsound two
    independent ways once the write got a genuine `asyncio.to_thread`
    suspension point: (1) the superseded write's OS thread is not itself
    cancellable and keeps running, so it can commit AFTER its replacement --
    rapid opposing actions on ONE item could leave the DATABASE on the FIRST
    action while the UI showed the second; (2) `asyncio.to_thread` CAN be
    cancelled before the executor picks the work up at all, so the old
    handler's "cancelled implies durable" assumption could patch the cache
    to a status the database never reached.

    Desired-status coalescing (`_dispatch_item_status`/`_drain_item_status`)
    replaces cancellation entirely: nothing here is ever cancelled. A second
    dispatch for an item that already has one queued just overwrites the
    desired dict entry, and the per-item drainer always `await`s a write to
    genuine completion before looking at that entry again -- so the database
    (and, once a `refresh=True` write reloads it, the screen's own cache)
    always settles on whichever action was dispatched LAST, deterministically.

    Reproduces exactly that: Ingest, then -- before the (slowed) write can
    possibly land -- Ignore, both against the SAME item, mirroring the
    Inspector's own `IngestRequested`/`IgnoreRequested` gestures. The
    underlying write is slowed and counted via a spy on
    `WatchlistsBackendController.update_item_status`, so the coalescing bound
    (at most one queued write plus at most one in flight, however many
    actions were dispatched) is directly observable, not just inferred from
    the end state.

    Mutation: removing the "loop again if a newer desired entry appeared"
    re-check in `_drain_item_status` (i.e. exiting after the first write
    instead of re-popping the dict) reds the final DB-status assertion --
    the queued Ignore would simply never be drained, and the database would
    incorrectly settle on "ingested".
    """
    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import (
        IngestRequested,
        IgnoreRequested,
    )

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _source_id, item_id = _seed_one_new_item(db, content_hash="hash-opposing-actions")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen.active_section = "items"
        await pilot.pause(0.3)

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        for _ in range(40):
            await pilot.pause()
            if pane.items:
                break
        assert pane.items, "the seeded item must reach the Items pane"
        entity = dict(pane.items[0])
        assert entity["item_id"] == item_id

        write_calls: list[str] = []
        real_update_item_status = screen._controller.update_item_status

        async def _slow_and_counted(*, runtime_backend=None, item_id, status):
            write_calls.append(status)
            await asyncio.sleep(0.2)
            return await real_update_item_status(
                runtime_backend=runtime_backend, item_id=item_id, status=status
            )

        screen._controller.update_item_status = _slow_and_counted

        # Rapid opposing actions on the SAME item: Ingest, then -- while that
        # slowed write is still draining -- Ignore. Exactly the Inspector's
        # own gestures, fired back to back.
        screen.post_message(IngestRequested(entity))
        await pilot.pause(0.05)
        await pilot.pause(0.05)
        screen.post_message(IgnoreRequested(entity))

        for _ in range(80):
            await pilot.pause(0.05)
            if db.get_item_status(item_id) == "ignored":
                break
        assert db.get_item_status(item_id) == "ignored", (
            "the LAST dispatched action (Ignore) must be what the database "
            "settles on, deterministically -- not whichever write's OS "
            "thread happened to finish last"
        )
        assert len(write_calls) <= 2, (
            "coalescing must bound this item to at most one queued write "
            "plus at most one in-flight write, however many actions were "
            f"dispatched -- observed writes: {write_calls!r}"
        )

        # And the screen's own cache, reloaded by Ignore's `refresh=True`
        # tail, must end up coherent with that same final database state.
        #
        # TASK-2301 changed what "coherent" can be asserted as -- and made it
        # stronger. This used to assert the item was ABSENT from
        # `_loaded_items`, because `_load_items()` queries `status=None` and
        # `LocalWatchlistsService.list_items` collapsed that to
        # `status="new"`, so a triaged item fell out of the cache entirely
        # and "coherent" could only mean "no longer here". That collapse WAS
        # the defect (the Items tab could not show a triaged item at all), so
        # the cache then carried every status and this named the status the
        # item must hold: `ignored`.
        #
        # TASK-3072 changes the contract once more, deliberately: the
        # reader's "all" is `_READER_ALL_STATUSES` (new/reviewed/ingested) --
        # an item the user just told the reader to hide does not belong in
        # the article list, so the reloaded cache legitimately drops it, and
        # "coherent" is absence again. The DB assertion above is what pins
        # the write itself landing on `ignored` (the last dispatched action,
        # not the superseded Ingest's `ingested`, not the pre-write `new`).
        def _cached_status():
            return next(
                (
                    row.get("status")
                    for row in screen._loaded_items
                    if row.get("item_id") == item_id
                ),
                None,
            )

        for _ in range(40):
            await pilot.pause(0.05)
            if screen._loaded_items and _cached_status() is None:
                break
        assert _cached_status() is None, (
            "the reloaded reader cache must NOT carry the just-ignored item: "
            "TASK-3072's 'all' is the reader set (new/reviewed/ingested), "
            "and an item the user hid leaves the article list immediately -- "
            "the database assertion above is what pins the ignored write"
        )


@pytest.mark.asyncio
async def test_mark_read_on_open_does_not_overwrite_an_item_ingested_behind_the_cache():
    """TASK-1541, Qodo redesign -- the inside-drain gate re-check.

    `_mark_item_read_on_open`'s cheap pre-filter only ever declines against
    the CACHED dict ("new") -- and nothing patches that cache when an item
    is ingested through the Inspector's `Ingest` button (no `patch_item=`
    there, by design -- see `handle_ingest_requested`). So the real guard
    against overwriting an ingest has to be the backend re-check
    `_item_status_write_allowed` performs immediately before the write,
    INSIDE `_drain_item_status`'s loop -- not the pre-filter, and not a check
    done once at dispatch time.

    Reproduces the window directly: `_mark_item_read_on_open` is called
    first, which queues the desired "reviewed" entry and schedules (but does
    not run a single line of) the drainer worker -- it is a plain
    synchronous call, and `run_worker` only SCHEDULES a coroutine. The item
    is THEN ingested directly against the database, still with no `await` in
    between -- i.e. the item gets ingested WHILE the mark-read desired entry
    sits queued, waiting for its drainer to actually start. Only once the
    test yields to the event loop does the drainer run, re-ask the backend,
    and see "ingested" -- the gate must refuse the write then, not the
    cached "new" the pre-filter already let through.

    Mutation: removing the inside-drain gate re-check (the `intent.gate`
    branch in `_drain_item_status`, or `_item_status_write_allowed` itself)
    reds this on the final assertion -- the ingest gets overwritten with
    "reviewed".
    """
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _source_id, item_id = _seed_one_new_item(db, content_hash="hash-stale-cache-overwrite")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen.active_section = "items"
        await pilot.pause(0.3)

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        for _ in range(40):
            await pilot.pause()
            if pane.items:
                break
        assert pane.items, "the seeded item must reach the Items pane"

        item = pane.items[0]
        assert item["item_id"] == item_id
        assert item.get("status") == "new", "precondition: the cached dict starts at 'new'"

        # Dispatch mark-read-on-open first: queues the desired "reviewed"
        # entry and schedules the per-item drainer without running any of it
        # yet (no `await` has happened in this test).
        screen._mark_item_read_on_open(item)

        # Ingest it directly through the database -- still with no `await`
        # in between -- "while the desired entry waits", exactly as the
        # docstring above describes. The cached dict is a separate object
        # and is NOT touched by this.
        db.mark_item_status(item_id, "ingested")
        assert item.get("status") == "new", (
            "the cached dict must still read stale 'new' -- otherwise this "
            "test is not exercising a stale cache at all"
        )

        # Only now does control return to the event loop, letting the
        # drainer actually run its gate re-check.
        for _ in range(40):
            await pilot.pause(0.05)

        assert db.get_item_status(item_id) == "ingested", (
            "the item was ingested while the mark-read desired entry sat "
            "queued -- the inside-drain gate re-check must refuse the "
            "'reviewed' write when it finally runs, not overwrite the ingest"
        )


@pytest.mark.asyncio
async def test_a_failed_item_status_write_toasts_an_error_and_leaves_the_cache_untouched():
    """TASK-1541, Qodo redesign.

    `_drain_item_status` calls `_update_item_status` for each popped
    intent, and that method's `except Exception` branch is the only thing
    standing between a genuine DB failure and a cache silently claiming a
    status the write never reached. Pinned directly: a failed write (a
    deliberate Ingest, so `notify_toast=True`) must surface an error toast
    and must not move the item's status at all.
    """
    from unittest.mock import Mock

    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import IngestRequested

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _source_id, item_id = _seed_one_new_item(db, content_hash="hash-failed-write")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen.active_section = "items"
        await pilot.pause(0.3)

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        for _ in range(40):
            await pilot.pause()
            if pane.items:
                break
        assert pane.items, "the seeded item must reach the Items pane"
        entity = dict(pane.items[0])
        assert entity["item_id"] == item_id

        async def _raise(*, runtime_backend=None, item_id, status):
            raise RuntimeError("simulated DB failure")

        screen._controller.update_item_status = _raise
        app.notify = Mock()

        screen.post_message(IngestRequested(entity))
        for _ in range(40):
            await pilot.pause(0.05)

        assert db.get_item_status(item_id) == "new", (
            "a failed write must leave the item's status exactly as it was"
        )
        assert app.notify.called, (
            "a failed write must surface an error toast, not fail silently"
        )
        _args, kwargs = app.notify.call_args
        assert kwargs.get("severity") == "error"
