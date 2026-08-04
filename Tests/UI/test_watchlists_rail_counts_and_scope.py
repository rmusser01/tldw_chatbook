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
    from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemsPane

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
        pane = screen.query_one("#watchlists-items-pane", ItemsPane)
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
