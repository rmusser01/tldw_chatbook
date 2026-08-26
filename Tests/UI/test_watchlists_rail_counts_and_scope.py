"""TASK-2304: the rail counts and the scoped Sources table tell the truth.

Two findings from the 2026-08-04 UAT, both of the same shape -- one screen
showing two numbers for what looked like the same fact:

* F15: the rail's counts sat on 0 across create -> assign -> Check now while
  the centre header read "(1 source)". They were never the same fact -- the
  rail counts UNREAD ITEMS (`SubscriptionsDB.get_watchlist_item_counts`), the
  header counts SOURCES in the tree scope -- but nothing on screen said so,
  and the rail genuinely was stale: `Check now` is the one gesture that
  manufactures items and it never reloaded the tree.
* F16: with the scope on a watchlist whose own header read "(0 sources)", the
  Sources table still listed an Unassigned source. `_load_sources` had no
  scope predicate at all.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
    ALL_SOURCES_BUCKET,
    UNASSIGNED_BUCKET,
    TreeScope,
    WatchlistTree,
)

pytestmark = pytest.mark.unit


def _seed_two_sources_one_assigned(app) -> tuple[int, int, int]:
    """One watchlist holding one source, plus one unassigned source.

    Returns:
        `(watchlist_id, assigned_source_id, unassigned_source_id)`.
    """
    db = app.local_watchlists_service._db()
    assigned = db.add_subscription(
        name="Assigned Feed", type="rss", source="https://assigned.test/feed.xml"
    )
    unassigned = db.add_subscription(
        name="Loose Feed", type="rss", source="https://loose.test/feed.xml"
    )
    bundle = app.watchlist_bundle_service
    watchlist = bundle.create("AI Research News")
    bundle.add_source(int(watchlist["id"]), assigned)
    return int(watchlist["id"]), assigned, unassigned


async def _mounted(host, pilot):
    """The mounted screen with its tree data and source list both loaded.

    `_load_sources` only runs for the Sources section, so it is awaited here
    explicitly -- these tests care about what the screen holds and what the
    rail paints, and half of them never open that tab.
    """
    await pilot.pause(0.3)
    screen = host.screen_stack[-1]
    for _ in range(40):
        await pilot.pause()
        if screen._tree_watchlists:
            break
    await screen._load_sources()
    await pilot.pause()
    return screen


def _rail_label(screen, node_id: str) -> str:
    return str(screen.query_one(f"#{node_id}", Button).label)


# --- AC#3: what the number counts ------------------------------------------


@pytest.mark.asyncio
async def test_the_rail_says_what_its_numbers_count():
    """AC#3. Legible without hovering, and again on hover."""
    app = _build_test_app()
    _seed_two_sources_one_assigned(app)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)

        legend = screen.query_one("#wl-tree-count-legend", Static)
        assert "unread" in str(legend.renderable).lower(), (
            "the rail must say what the number after each node counts"
        )
        assert "unread" in str(
            screen.query_one("#wl-tree-node-all", Button).tooltip
        ).lower()


def test_the_unread_phrase_covers_zero_one_and_many():
    """The zero case is the one the UAT could not read."""
    assert WatchlistTree._unread_phrase(0) == "No unread items"
    assert WatchlistTree._unread_phrase(1) == "1 unread item"
    assert WatchlistTree._unread_phrase(4) == "4 unread items"


# --- AC#1: the counts move --------------------------------------------------


@pytest.mark.asyncio
async def test_a_check_that_produces_items_updates_the_rail_counts():
    """AC#1. The UAT's exact sequence: assign, check, read the rail.

    The check itself is stubbed at the controller -- what is under test is
    whether the screen re-reads the counts after one, not whether the fetcher
    works -- but the ITEMS it would have produced are written to the real
    database first, so the number the rail ends up showing is a real query
    result and not an echo of anything this test told it.
    """
    app = _build_test_app()
    watchlist_id, assigned_id, _unassigned = _seed_two_sources_one_assigned(app)
    db = app.local_watchlists_service._db()

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)

        assert _rail_label(screen, f"wl-tree-node-watchlist-{watchlist_id}").endswith(
            "0"
        ), "precondition: nothing has been fetched yet"

        async def _check_now_that_fetched_two_items(*, runtime_backend=None, source_id):
            with db.transaction() as conn:
                for index in range(2):
                    persist_subscription_item(
                        conn,
                        assigned_id,
                        {
                            "url": f"https://assigned.test/post-{index}/",
                            "title": f"Post {index}",
                            "content_hash": f"hash-check-{index}",
                        },
                        run_id=None,
                        now=f"2026-08-04T09:00:0{index}+00:00",
                    )
            return {"status": "completed"}

        screen._controller.check_now = _check_now_that_fetched_two_items

        source = next(
            s for s in screen._loaded_sources if s.get("source_id") == assigned_id
        )
        screen.run_worker(screen._check_now_source(source))

        for _ in range(80):
            await pilot.pause(0.05)
            if screen._tree_counts.get(watchlist_id, {}).get("unread"):
                break

        assert screen._tree_counts[watchlist_id]["unread"] == 2
        assert screen._tree_counts[ALL_SOURCES_BUCKET]["unread"] == 2
        # And the rail the user is looking at, not just the screen's cache.
        for _ in range(40):
            await pilot.pause()
            if _rail_label(screen, f"wl-tree-node-watchlist-{watchlist_id}").endswith(
                "2"
            ):
                break
        assert _rail_label(
            screen, f"wl-tree-node-watchlist-{watchlist_id}"
        ).endswith("2"), "the rail node itself must show the new count"


@pytest.mark.asyncio
async def test_ingesting_an_item_updates_the_rail_counts():
    """AC#1. Triage moves items out of the unread bucket the rail counts."""
    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import IngestRequested
    from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane

    app = _build_test_app()
    watchlist_id, assigned_id, _unassigned = _seed_two_sources_one_assigned(app)
    db = app.local_watchlists_service._db()
    with db.transaction() as conn:
        persist_subscription_item(
            conn,
            assigned_id,
            {
                "url": "https://assigned.test/only/",
                "title": "The only item",
                "content_hash": "hash-only",
            },
            run_id=None,
            now="2026-08-04T09:00:00+00:00",
        )

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen.active_section = "items"
        await pilot.pause(0.3)
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        for _ in range(40):
            await pilot.pause()
            if pane.items:
                break
        assert screen._tree_counts[watchlist_id]["unread"] == 1

        screen.post_message(IngestRequested(dict(pane.items[0])))
        for _ in range(80):
            await pilot.pause(0.05)
            if screen._tree_counts.get(watchlist_id, {}).get("unread") == 0:
                break

        assert screen._tree_counts[watchlist_id]["unread"] == 0, (
            "ingesting the last unread item must be reflected in the rail "
            "without a tab switch"
        )


# --- AC#2: the table and the header agree -----------------------------------


@pytest.mark.asyncio
async def test_selecting_a_watchlist_scope_narrows_the_sources_table():
    """AC#2. The UAT's F16, driven through the real scope change."""
    app = _build_test_app()
    watchlist_id, assigned_id, unassigned_id = _seed_two_sources_one_assigned(app)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen.active_section = "sources"
        await pilot.pause(0.3)
        for _ in range(40):
            await pilot.pause()
            if screen.query("#watchlists-sources-pane"):
                break
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        assert {row["source_id"] for row in pane.sources} == {
            assigned_id,
            unassigned_id,
        }, "precondition: the default 'all' scope shows both"

        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        )
        await pilot.pause()
        await pilot.pause()

        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        assert {row["source_id"] for row in pane.sources} == {assigned_id}, (
            "the table must show exactly the scoped watchlist's sources"
        )
        # The header's own count, from the resolver both now share.
        assert len(screen.scoped_source_rows()) == len(pane.sources), (
            "the header count and the visible rows must be the same number"
        )


@pytest.mark.asyncio
async def test_the_unassigned_scope_excludes_assigned_sources():
    """AC#2's other direction, and the exact row the UAT saw leak through."""
    app = _build_test_app()
    _watchlist_id, assigned_id, unassigned_id = _seed_two_sources_one_assigned(app)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen.active_section = "sources"
        await pilot.pause(0.3)

        screen._apply_tree_scope(TreeScope(kind="unassigned"))
        await pilot.pause()
        await pilot.pause()

        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        assert {row["source_id"] for row in pane.sources} == {unassigned_id}
        assert assigned_id not in {row["source_id"] for row in pane.sources}
        assert len(screen.scoped_source_rows()) == len(pane.sources)


@pytest.mark.asyncio
async def test_a_source_reload_under_a_scope_stays_scoped():
    """AC#2. The reload path is the other way the table can re-widen.

    `_load_sources` runs after every create/delete/check, and it queries the
    backend unscoped -- so if it pushed its raw result the table would snap
    back to every source while the header still named one watchlist.
    """
    app = _build_test_app()
    watchlist_id, assigned_id, _unassigned = _seed_two_sources_one_assigned(app)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen.active_section = "sources"
        await pilot.pause(0.3)
        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        )
        await pilot.pause()

        await screen._load_sources()
        await pilot.pause()

        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        assert {row["source_id"] for row in pane.sources} == {assigned_id}
        # The unscoped mirror is deliberately left whole -- the Console
        # handoff and pane rebuilds read it.
        assert len(screen._loaded_sources) == 2


@pytest.mark.asyncio
async def test_assigning_a_source_fills_the_scoped_table_it_now_belongs_to():
    """AC#2, found in live verification and not by the rest of this suite.

    Membership is what decides the scope's contents, and the scope itself does
    NOT move when a source is added to the watchlist already in view -- so
    neither `watch_tree_scope` nor a source reload fires. Scoping the table
    without this left `Add source` writing a membership row while the table
    stayed empty under a header that had already updated to "(1 source)".
    """
    app = _build_test_app()
    watchlist_id, _assigned_id, unassigned_id = _seed_two_sources_one_assigned(app)
    bundle = app.watchlist_bundle_service

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen.active_section = "sources"
        await pilot.pause(0.3)
        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        )
        await pilot.pause()
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        assert unassigned_id not in {row["source_id"] for row in pane.sources}

        # The write the rail's `Add source` performs, followed by the reload
        # its flow performs -- the scope is untouched throughout.
        bundle.add_source(watchlist_id, unassigned_id)
        screen._load_tree_data()
        for _ in range(60):
            await pilot.pause(0.05)
            pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
            if unassigned_id in {row["source_id"] for row in pane.sources}:
                break

        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        assert unassigned_id in {row["source_id"] for row in pane.sources}, (
            "the newly assigned source must appear in the scoped table"
        )
        assert len(screen.scoped_source_rows()) == len(pane.sources), (
            "and the header count must still equal the visible rows"
        )


@pytest.mark.asyncio
async def test_a_workbench_rebuild_under_a_scope_stays_scoped():
    """AC#2. And the third way: a region toggle rebuilding the pane."""
    app = _build_test_app()
    watchlist_id, assigned_id, _unassigned = _seed_two_sources_one_assigned(app)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen.active_section = "sources"
        await pilot.pause(0.3)
        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        )
        await pilot.pause()

        # `[` toggles the left rail, which recomposes the whole workbench and
        # constructs a brand new SourcesPane from `_build_detail_pane`.
        await pilot.press("[")
        await pilot.pause()
        await pilot.pause()

        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        assert {row["source_id"] for row in pane.sources} == {assigned_id}


def test_the_all_scope_costs_no_extra_query():
    """`scoped_loaded_sources` must not pay for a scope that filters nothing.

    The default scope is `all`, every `_load_sources` and every pane rebuild
    calls this, and the resolver it would otherwise consult
    (`scoped_source_rows`) is a real database query.
    """
    app = _build_test_app()
    _seed_two_sources_one_assigned(app)

    from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
        WatchlistsCollectionsScreen,
    )

    screen = WatchlistsCollectionsScreen(app)
    screen._loaded_sources = [{"source_id": 1}, {"source_id": 2}]

    def _must_not_be_called():
        raise AssertionError("the 'all' scope must short-circuit")

    screen.scoped_source_rows = _must_not_be_called
    assert screen.scoped_loaded_sources() == screen._loaded_sources


def test_unassigned_bucket_ids_are_the_ones_the_rail_reads():
    """A guard on the two sentinels this suite asserts against."""
    assert ALL_SOURCES_BUCKET != UNASSIGNED_BUCKET


# --- review wave minors -----------------------------------------------------


@pytest.mark.asyncio
async def test_the_scoped_table_is_left_alone_on_the_server_backend():
    """Review wave, Minor 3. Scoping is a LOCAL-only fact, and says so.

    `scoped_source_rows()` resolves ids through the local bundle service
    (watchlists/watchlist_sources are local tables), while a server row's
    `source_id` is a server id from a different namespace. Intersecting them
    is empty for every non-`all` scope, so a scoped table under the server
    backend would render empty beneath a header claiming N sources -- the
    exact defect this fix exists to remove, produced by the fix for it.
    """
    from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
        WatchlistsCollectionsScreen,
    )

    app = _build_test_app()
    watchlist_id, _assigned, _unassigned = _seed_two_sources_one_assigned(app)

    screen = WatchlistsCollectionsScreen(app)
    # Server rows carry ids from the server's own namespace; nothing about
    # them can be matched against a local watchlist_sources row.
    screen._loaded_sources = [{"source_id": 9001}, {"source_id": 9002}]
    screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=watchlist_id)

    screen.runtime_backend = "local"
    assert screen.scoped_loaded_sources() == [], (
        "precondition: under the local backend these ids scope to nothing"
    )

    screen.runtime_backend = "server"
    assert screen.scoped_loaded_sources() == screen._loaded_sources, (
        "under the server backend the listing must be left unscoped rather "
        "than silently emptied"
    )


@pytest.mark.asyncio
async def test_a_source_scope_shows_exactly_that_source():
    """Review wave, Minor 5. `source` scope was untouched by any test.

    `scoped_loaded_sources()` applies to every non-`all` scope, so clicking a
    single source in the rail collapses the table to that one row. That is
    intended -- the header names the same one source, so the two still agree,
    which is the invariant this task is about -- but it is a behaviour change
    and it is now pinned.
    """
    app = _build_test_app()
    watchlist_id, assigned_id, _unassigned = _seed_two_sources_one_assigned(app)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen.active_section = "sources"
        await pilot.pause(0.3)

        screen._apply_tree_scope(
            TreeScope(
                kind="source", watchlist_id=watchlist_id, source_id=assigned_id
            )
        )
        await pilot.pause()
        await pilot.pause()

        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        assert [row["source_id"] for row in pane.sources] == [assigned_id]
        assert len(screen.scoped_source_rows()) == len(pane.sources), (
            "header and table must still agree under a source scope"
        )


@pytest.mark.asyncio
async def test_opening_items_refreshes_the_rail_once_the_user_pauses():
    """Review wave, Minor 6. The legend must not out-promise the number.

    Opening an item marks it read, which moves it out of the unread bucket the
    rail counts -- but that write is deliberately `refresh=False` (it fires on
    every arrow key). The counts therefore lagged by however many items had
    been opened, under a legend that says "Counts: unread items" flatly. The
    lag is removed with a debounce rather than the label weakened: a burst of
    opens costs one reload after the burst.
    """
    from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane

    app = _build_test_app()
    watchlist_id, assigned_id, _unassigned = _seed_two_sources_one_assigned(app)
    db = app.local_watchlists_service._db()
    with db.transaction() as conn:
        for index in range(3):
            persist_subscription_item(
                conn,
                assigned_id,
                {
                    "url": f"https://assigned.test/read-{index}/",
                    "title": f"Readable {index}",
                    "content_hash": f"hash-read-{index}",
                },
                run_id=None,
                now=f"2026-08-04T09:00:0{index}+00:00",
            )

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        screen.active_section = "items"
        await pilot.pause(0.3)
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        for _ in range(60):
            await pilot.pause()
            if len(pane.items) >= 3:
                break
        assert screen._tree_counts[watchlist_id]["unread"] == 3

        # Open one item; the write itself is silent and does not reload.
        pane.select_item_by_id(str(pane.items[0]["id"]))

        for _ in range(80):
            await pilot.pause(0.05)
            if screen._tree_counts.get(watchlist_id, {}).get("unread") == 2:
                break

        assert screen._tree_counts[watchlist_id]["unread"] == 2, (
            "the rail must catch up on silent mark-read writes once the user "
            "stops moving, so its legend stays true"
        )


@pytest.mark.asyncio
async def test_a_check_that_is_only_queued_does_not_re_read_the_counts():
    """Review wave, Minor 4. Don't report a number the action has not reached.

    `check_now` on the server backend delegates to `launch_run` and returns
    `queued`/`running` -- the toast is already careful about that distinction.
    Re-reading the rail's counts there would query for items the run has not
    produced yet, and present the answer with the same authority as a real
    one. The refresh now waits for a terminal status; a run that finishes
    later is picked up by the next refresh.
    """
    app = _build_test_app()
    _watchlist_id, assigned_id, _unassigned = _seed_two_sources_one_assigned(app)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        reloads: list[str] = []
        real_load_tree = screen._load_tree_data
        screen._load_tree_data = lambda: reloads.append("x")

        source = next(
            s for s in screen._loaded_sources if s.get("source_id") == assigned_id
        )

        async def _queued(*, runtime_backend=None, source_id):
            return {"status": "queued"}

        screen._controller.check_now = _queued
        await screen._check_now_source(source)
        # Let the source reload this method dispatches actually start, so it
        # is not left as an un-awaited coroutine at teardown.
        await pilot.pause()
        assert reloads == [], (
            "a queued run must not trigger an authoritative count re-read"
        )

        async def _completed(*, runtime_backend=None, source_id):
            return {"status": "completed"}

        screen._controller.check_now = _completed
        await screen._check_now_source(source)
        await pilot.pause()
        assert reloads == ["x"], "a finished run must refresh the counts"
        screen._load_tree_data = real_load_tree


# --- TASK-3072 plan task 6: the Starred root's badge -------------------------


@pytest.mark.asyncio
async def test_starred_root_badge_counts_flagged_items():
    """The Starred root's badge is `get_flagged_items_count` -- global and
    status-agnostic, refreshed through the same tree-data load as every
    other node. Two flagged items on different sources read as one "2".
    """
    app = _build_test_app()
    _watchlist_id, assigned_id, unassigned_id = _seed_two_sources_one_assigned(app)
    db = app.local_watchlists_service._db()
    with db.transaction() as conn:
        for index, source_id in enumerate((assigned_id, unassigned_id, unassigned_id)):
            persist_subscription_item(
                conn,
                source_id,
                {
                    "url": f"https://feed.test/post-{index}/",
                    "title": f"Post {index}",
                    "content_hash": f"hash-starred-{index}",
                },
                run_id=None,
                now=f"2026-08-04T09:00:0{index}+00:00",
            )
    item_ids = [
        row[0]
        for row in db.conn.execute("SELECT id FROM subscription_items ORDER BY id")
    ]
    db.set_item_flagged(item_ids[0], True)
    db.set_item_flagged(item_ids[1], True)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        for _ in range(40):
            await pilot.pause()
            if _rail_label(screen, "wl-tree-node-starred").endswith("2"):
                break
        assert _rail_label(screen, "wl-tree-node-starred") == "★ Starred  2"


@pytest.mark.asyncio
async def test_all_unread_and_today_badges_count_their_own_facts():
    """TASK-3791 plan task 4: All Unread reuses the All-sources unread count
    (one fact, two angles); Today counts unread items at/after local
    midnight only."""
    from datetime import datetime, timedelta, timezone

    app = _build_test_app()
    _watchlist_id, assigned_id, _unassigned = _seed_two_sources_one_assigned(app)
    db = app.local_watchlists_service._db()
    now = datetime.now(timezone.utc)
    with db.transaction() as conn:
        for index, published in enumerate(
            (now.isoformat(), now.isoformat(), (now - timedelta(hours=49)).isoformat())
        ):
            persist_subscription_item(
                conn,
                assigned_id,
                {
                    "url": f"https://assigned.test/today-{index}/",
                    "title": f"Today {index}",
                    "content_hash": f"hash-today-{index}",
                    "published_date": published,
                },
                run_id=None,
                now=now.isoformat(),
            )

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _mounted(host, pilot)
        for _ in range(40):
            await pilot.pause()
            if _rail_label(screen, "wl-tree-node-unread").endswith("3"):
                break
        assert _rail_label(screen, "wl-tree-node-unread") == "All Unread  3"
        assert _rail_label(screen, "wl-tree-node-today") == "Today  2", (
            "the two fresh items count; the 49-hour-old one does not"
        )
