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
