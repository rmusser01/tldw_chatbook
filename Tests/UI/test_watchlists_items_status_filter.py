"""TASK-2301: the Items list honours its filter, and triage is not deletion.

The 2026-08-04 UAT: the Items list behaved as "new items only" while its own
filter read "All statuses". Ingesting an item, ignoring one, or merely opening
one (which marks it read) made the row VANISH on the next reload, and with the
filter itself dead (TASK-2300) the item was then unreachable anywhere in the
tab. Acting on an item read as data loss.

The mechanism was two layers down, not in the pane: `SubscriptionsDB.
get_new_items` applied its status predicate unconditionally, and
`LocalWatchlistsService.list_items` collapsed `status=None` to `"new"`. So
`_load_items(status=None)` -- the screen's only item query -- was structurally
incapable of returning anything but `new` rows, and the pane's "All statuses"
option had nothing else to filter. The tests here run bottom-up: the two data
layers first, then the pane, then the user gesture.
"""

from __future__ import annotations

import pytest
from textual.widgets import DataTable

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import IngestRequested
from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemsPane

pytestmark = pytest.mark.unit

#: One item per status the local backend can hold, so "every status appears"
#: is asserted against the real vocabulary rather than a sample of it.
SEEDED = {
    "new": "Freshly fetched",
    "reviewed": "Already opened",
    "ingested": "Filed into the library",
    "ignored": "Deliberately dismissed",
    "error": "Failed to fetch",
}


def _seed_one_item_per_status(db) -> dict[str, int]:
    """Add one subscription carrying one item in each status.

    Returns:
        Mapping of status to the raw `subscription_items` row id.
    """
    source_id = db.add_subscription(
        name="Mixed Statuses", type="rss", source="https://example.invalid/feed.xml"
    )
    raw_ids: dict[str, int] = {}
    with db.transaction() as conn:
        for index, (status, title) in enumerate(SEEDED.items()):
            raw_ids[status] = persist_subscription_item(
                conn,
                source_id,
                {
                    "url": f"https://example.invalid/{status}",
                    "title": title,
                    "content_hash": f"hash-{status}",
                },
                run_id=None,
                now=f"2026-07-2{index}T09:00:00+00:00",
            )
    # `persist_subscription_item` always inserts `"new"` -- it is the ingest
    # path, and a fetched item is by definition unread. Triage is a separate
    # write, so the seed performs it the same way the UI does.
    for status, raw_id in raw_ids.items():
        if status != "new":
            db.mark_item_status(raw_id, status)
    return raw_ids


async def _settled_items_pane(screen, pilot, expected: int) -> ItemsPane:
    """The mounted `ItemsPane` once its loader has delivered `expected` rows."""
    screen.active_section = "items"
    await pilot.pause(0.3)
    pane = screen.query_one("#watchlists-items-pane", ItemsPane)
    for _ in range(60):
        await pilot.pause()
        pane = screen.query_one("#watchlists-items-pane", ItemsPane)
        if len(pane.items) >= expected:
            break
    return pane


def _status_column(pane: ItemsPane) -> list[str]:
    """The Status column as the LIVE table holds it, row by row.

    Read off the mounted `DataTable`, not off `pane.items`: the pane's
    reactive is what the loader handed it, while this is what the user is
    looking at, and TASK-2301's whole subject is those two disagreeing.
    """
    table = pane.query_one("#items-table", DataTable)
    status_key = pane._column_keys[2]
    return [
        str(table.get_cell(str(item["id"]), status_key))
        for item in pane.displayed_items()
    ]


# --- data layers -----------------------------------------------------------


def test_get_new_items_with_no_status_returns_every_status():
    """AC#1, at the layer that made it impossible."""
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_one_item_per_status(db)

    rows = db.get_new_items(status=None, limit=50)

    assert {row["status"] for row in rows} == set(SEEDED)


def test_get_new_items_still_defaults_to_new_only():
    """The default is load-bearing: `briefing_selection` and the smoke suite
    both call this with no status and mean "the unread bucket"."""
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_one_item_per_status(db)

    assert {row["status"] for row in db.get_new_items(limit=50)} == {"new"}


def test_get_new_items_scoped_to_one_subscription_still_filters_by_status():
    """Both predicates are independent; neither may swallow the other."""
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_one_item_per_status(db)
    subscription_id = db.get_all_subscriptions(limit=5)[0]["id"]

    scoped_all = db.get_new_items(subscription_id=subscription_id, status=None, limit=50)
    scoped_one = db.get_new_items(
        subscription_id=subscription_id, status="ingested", limit=50
    )

    assert {row["status"] for row in scoped_all} == set(SEEDED)
    assert {row["status"] for row in scoped_one} == {"ingested"}


@pytest.mark.asyncio
async def test_list_items_no_longer_collapses_none_to_new():
    """The service layer stopped rewriting the caller's intent."""
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_one_item_per_status(db)

    items = await app.local_watchlists_service.list_items(status=None, limit=50)

    assert {item["status"] for item in items} == set(SEEDED)


# --- the pane --------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_status_appears_in_the_list_and_is_distinguishable():
    """AC#1. All statuses present under "All statuses", each labelled."""
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_one_item_per_status(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane = await _settled_items_pane(screen, pilot, len(SEEDED))

        assert pane.status_filter == "all"
        assert len(pane.displayed_items()) == len(SEEDED)
        assert set(_status_column(pane)) == set(SEEDED), (
            "the Status column must tell the statuses apart on screen, not "
            "only in the pane's own data"
        )


@pytest.mark.asyncio
async def test_a_triaged_item_is_findable_under_its_own_status_filter():
    """AC#4. The regression test the task asks for, stated as the user's
    question: "where did my ingested item go?"."""
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_one_item_per_status(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane = await _settled_items_pane(screen, pilot, len(SEEDED))

        for status, title in SEEDED.items():
            pane.status_filter = status
            await pilot.pause()
            pane = screen.query_one("#watchlists-items-pane", ItemsPane)
            titles = [row["title"] for row in pane.displayed_items()]
            assert titles == [title], (
                f"filtering to {status!r} must show exactly that item; showed "
                f"{titles!r}"
            )


# --- the user gesture ------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_repaints_the_live_row_instead_of_removing_it():
    """AC#2 and AC#3, pinned on the immediate push rather than the reload.

    `_load_items` is stubbed to a no-op here on purpose. With the reload left
    in, an assertion made after it lands cannot tell the in-place repaint from
    the rebuild, so it would stay green with the repaint deleted -- and the
    repaint is the whole of "feedback beyond row removal", since the reload is
    asynchronous and, before this task, was what removed the row.
    """
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_one_item_per_status(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane = await _settled_items_pane(screen, pilot, len(SEEDED))

        target = next(
            item for item in pane.items if item["title"] == SEEDED["new"]
        )
        rows_before = len(pane.displayed_items())

        async def _no_reload() -> None:
            return None

        screen._load_items = _no_reload

        screen.post_message(IngestRequested(dict(target)))
        for _ in range(60):
            await pilot.pause()
            if db.get_item_status(int(target["item_id"])) == "ingested":
                break

        pane = screen.query_one("#watchlists-items-pane", ItemsPane)
        table = pane.query_one("#items-table", DataTable)
        assert len(pane.displayed_items()) == rows_before, (
            "ingesting must not remove the row from a view whose filter "
            "includes it"
        )
        assert (
            str(table.get_cell(str(target["id"]), pane._column_keys[2])) == "ingested"
        ), "the row the user acted on must show its new status immediately"


@pytest.mark.asyncio
async def test_ingest_feedback_toast_never_parses_markup():
    """AC#3's other half. The toast is app-authored today, but it is the
    Watchlists screen: every neighbouring toast carries feed- or item-derived
    text, and the convention here is to escape at the terminal step rather
    than to keep a mental list of which messages happen to be safe."""
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_one_item_per_status(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane = await _settled_items_pane(screen, pilot, len(SEEDED))

        toasts: list[tuple[str, dict]] = []
        screen.app_instance.notify = lambda message, **kwargs: toasts.append(
            (str(message), kwargs)
        )

        target = next(item for item in pane.items if item["title"] == SEEDED["new"])
        screen.post_message(IngestRequested(dict(target)))
        for _ in range(60):
            await pilot.pause()
            if db.get_item_status(int(target["item_id"])) == "ingested":
                break
        for _ in range(10):
            await pilot.pause()

        matching = [kwargs for message, kwargs in toasts if message == "Item marked ingested."]
        assert matching, f"Ingest must report what happened; saw {toasts!r}"
        assert all(kwargs.get("markup") is False for kwargs in matching)


@pytest.mark.asyncio
async def test_an_ingested_item_survives_the_reload_that_follows():
    """AC#2 end to end, with the real reload in place."""
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_one_item_per_status(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane = await _settled_items_pane(screen, pilot, len(SEEDED))

        target = next(item for item in pane.items if item["title"] == SEEDED["new"])
        screen.post_message(IngestRequested(dict(target)))
        for _ in range(80):
            await pilot.pause()
            if db.get_item_status(int(target["item_id"])) == "ingested":
                break
        for _ in range(40):
            await pilot.pause()

        pane = screen.query_one("#watchlists-items-pane", ItemsPane)
        titles = [row["title"] for row in pane.displayed_items()]
        assert SEEDED["new"] in titles, (
            "the ingested item must still be in the list after the reload"
        )
        assert set(_status_column(pane)) == set(SEEDED) - {"new"} | {"ingested"}
