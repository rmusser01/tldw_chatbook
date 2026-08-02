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
the same dict object already held by `ItemsPane.items` in place rather than
reloading and recomposing.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button, DataTable

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemsPane

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

        pane = screen.query_one("#watchlists-items-pane", ItemsPane)
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

        pane = screen.query_one("#watchlists-items-pane", ItemsPane)
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

        pane = screen.query_one("#watchlists-items-pane", ItemsPane)
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
    the mounted `ItemsPane`/`DataTable` wholesale. Proven live: with the old
    behaviour, one `down` press detached the `ItemsPane`, reset the cursor
    to row 0, cleared screen focus, and a SECOND `down` press did nothing at
    all -- the list became unusable by keyboard after the very first
    selection.

    Drives the real keyboard path (`pilot.press`, a focused `DataTable`)
    rather than `select_item_by_id` directly, since the bug is specifically
    about what selecting a row does to the table hosting it, not about
    selection itself (already covered by the other tests in this file).
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

        pane = screen.query_one("#watchlists-items-pane", ItemsPane)
        for _ in range(40):
            await pilot.pause()
            if len(pane.items) >= 4:
                break
        assert len(pane.items) == 4, "all four seeded items must reach the Items pane"

        table = pane.query_one("#items-table", DataTable)
        table.focus()
        await pilot.pause(0.2)

        await pilot.press("down")
        # Let the silent, now non-refreshing mark-read worker run to
        # completion before checking anything -- the regression this test
        # guards against is specifically that worker's side effect.
        for _ in range(30):
            await pilot.pause()

        assert pane.is_attached, (
            "selecting a row must not detach the mounted ItemsPane via a "
            "screen-level recompose"
        )
        assert screen.query_one("#watchlists-items-pane", ItemsPane) is pane, (
            "the screen must still be hosting the SAME ItemsPane instance, "
            "not one rebuilt from scratch"
        )
        current_table = pane.query_one("#items-table", DataTable)
        assert current_table is table, "the DataTable itself must survive too"
        assert current_table.has_focus, (
            "a recompose drops focus entirely -- the table must still be "
            "focused after a plain row selection"
        )
        assert current_table.cursor_row == 1, "the cursor must stay where the user put it"

        await pilot.press("down")
        await pilot.pause(0.3)
        assert current_table.cursor_row == 2, (
            "a SECOND arrow key must still move the cursor -- before the fix "
            "this did nothing at all once the table had been replaced"
        )
        assert pane.selected_item is not None
        assert pane.selected_item.get("title") == "Item 2"


@pytest.mark.asyncio
async def test_a_cancelled_mark_read_still_leaves_the_cached_dict_coherent():
    """Fix wave, F2a (whole-branch review, Important).

    `_ITEM_STATUS_WORKER_GROUP` deliberately lets a repeat mark-read-on-open
    supersede its own in-flight sibling, so a fast `j`/`k` run does not queue
    one write per keystroke. Once TASK-1541 moved the write onto a genuine
    `asyncio.to_thread` suspension point, that same supersede became able to
    actually deliver `CancelledError` to a DIFFERENT item's write in flight
    -- but the OS thread underneath `asyncio.to_thread` is not itself
    cancellable, so item A's write still lands in the database regardless.

    Reproduces that directly: item A's write is slowed (monkeypatched
    `WatchlistsBackendController.update_item_status`, mirroring how the
    reviewer's scaffold widened a contended SQLite lock), A is opened
    (dispatching its mark-read-on-open worker), and -- before that slow
    write can possibly finish -- B is opened too, in the SAME worker group,
    simulating the fast `j`/`k` cross-item supersede. A's write is left to
    land, then the cached dict for A must read "reviewed", not the stale
    "new" that would leak through if `_update_item_status`'s continuation
    only patched the cache on the (never-reached, here) non-cancelled path.

    Mutation: removing the `except asyncio.CancelledError` patch block in
    `_update_item_status` reds this on the final assertion -- the database
    reaches "reviewed" but the cached dict stays stranded at "new".
    """
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    source_id, item_a_id = _seed_one_new_item(db, content_hash="hash-cancel-coherence-a")
    with db.transaction() as conn:
        item_b_id = persist_subscription_item(
            conn,
            source_id,
            {
                "url": "https://summitroute.com/blog/2024/cancel-coherence-b/",
                "title": "Item B",
                "content_hash": "hash-cancel-coherence-b",
                "status": "new",
            },
            run_id=None,
            now="2026-07-28T09:00:01+00:00",
        )

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen.active_section = "items"
        await pilot.pause(0.3)

        pane = screen.query_one("#watchlists-items-pane", ItemsPane)
        for _ in range(40):
            await pilot.pause()
            if len(pane.items) >= 2:
                break
        assert len(pane.items) == 2, "both seeded items must reach the Items pane"

        item_a = next(item for item in pane.items if item["item_id"] == item_a_id)
        item_b = next(item for item in pane.items if item["item_id"] == item_b_id)

        real_update_item_status = screen._controller.update_item_status

        async def _slow_for_a(*, runtime_backend=None, item_id, status):
            if item_id == item_a["id"]:
                await asyncio.sleep(0.4)
            return await real_update_item_status(
                runtime_backend=runtime_backend, item_id=item_id, status=status
            )

        screen._controller.update_item_status = _slow_for_a

        # Dispatch A's mark-read-on-open. Its write is now slowed to 0.4s, so
        # it is still suspended on the genuine `await asyncio.to_thread(...)`
        # boundary well past the couple of pauses below.
        screen._mark_item_read_on_open(item_a)
        await pilot.pause(0.05)
        await pilot.pause(0.05)

        # Simulate the fast `j`/`k` cross-item supersede: B's dispatch lands
        # in the SAME `_ITEM_STATUS_WORKER_GROUP`, `exclusive=True`, so it
        # cancels A's still-suspended worker.
        screen._mark_item_read_on_open(item_b)

        # Let A's slowed write actually land -- the OS thread cannot be
        # cancelled, so this must eventually succeed regardless of the
        # supersede above.
        for _ in range(60):
            await pilot.pause(0.05)
            if db.get_item_status(item_a_id) == "reviewed":
                break
        assert db.get_item_status(item_a_id) == "reviewed", (
            "the OS thread under asyncio.to_thread is not cancellable -- A's "
            "write must complete in the database even though its coroutine "
            "was cancelled by B's supersede"
        )

        for _ in range(40):
            await pilot.pause(0.05)
            if item_a.get("status") == "reviewed":
                break
        assert item_a.get("status") == "reviewed", (
            "the cached dict must be patched to match the database even "
            "though A's coroutine was cancelled mid-flight -- otherwise it "
            "is left reading a stale 'new' forever, diverged from a "
            "database the app itself just wrote"
        )


@pytest.mark.asyncio
async def test_mark_read_on_open_does_not_overwrite_an_item_ingested_behind_the_cache():
    """Fix wave, F2b (whole-branch review, Important) -- the other half of F2.

    `_mark_item_read_on_open` only ever declines its write when the CACHED
    dict already disagrees with "new" -- and nothing patches that cache when
    an item is ingested through the Inspector's `Ingest` button (no
    `patch_item=` there, by design -- see `handle_ingest_requested`). So a
    cache that goes stale for ANY reason (this test moves the database
    directly, "behind the cache's back", rather than reproducing the F2a
    cancellation race) must not let a subsequent open of that same item
    overwrite the real, backend-held status.

    Mutation: removing the `_blocking_status_for` backend gate in
    `_confirm_new_then_mark_item_read_on_open` (falling back to trusting the
    cached "new" alone, as `_mark_item_read_on_open` did before this fix)
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

        pane = screen.query_one("#watchlists-items-pane", ItemsPane)
        for _ in range(40):
            await pilot.pause()
            if pane.items:
                break
        assert pane.items, "the seeded item must reach the Items pane"

        item = pane.items[0]
        assert item["item_id"] == item_id
        assert item.get("status") == "new", "precondition: the cached dict starts at 'new'"

        # Ingest it directly through the database -- "behind the cache's
        # back" -- exactly what the real Ingest gesture also does, since
        # `handle_ingest_requested` passes no `patch_item=`. The cached
        # dict above is a separate object and is NOT touched by this.
        db.mark_item_status(item_id, "ingested")
        assert item.get("status") == "new", (
            "the cached dict must still read stale 'new' -- otherwise this "
            "test is not exercising a stale cache at all"
        )

        # Re-open the item: the cache still says "new", so without the
        # backend gate this fires the write unconditionally.
        screen._mark_item_read_on_open(item)
        for _ in range(40):
            await pilot.pause(0.05)

        assert db.get_item_status(item_id) == "ingested", (
            "opening an item whose cache is stale must not overwrite a real "
            "backend status the cache does not know about -- the ingest "
            "must survive"
        )
