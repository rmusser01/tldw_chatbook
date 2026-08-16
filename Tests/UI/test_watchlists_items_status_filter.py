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

TASK-3072 changed the filter's vocabulary: the five per-status options became
the reader's Unread/All pair. "Unread" is `status="new"` pushed into the
query; "All" is the reader set `new`/`reviewed`/`ingested` -- `ignored` stays
hidden (the user hid it on purpose) and `error` stays in Runs. Tests below
that predate the swap assert against the reader set, not the full status
vocabulary; the data-layer tests (`status=None` returns every status) are
untouched, because the narrowing lives in the screen's query, not the DB.
"""

from __future__ import annotations

import pytest
from textual.widgets import ListView

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.UI.Watchlists_Modules.article_list import (
    ArticleListPane,
    _ArticleRow,
)
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import IngestRequested

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

#: The subset of SEEDED the reader's "All" shows (TASK-3072): ignored stays
#: hidden, error stays in Runs. Every pane-level count expectation uses this.
READER_SEEDED = {
    status: title
    for status, title in SEEDED.items()
    if status in ArticleListPane._READER_STATUSES
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


async def _settled_items_pane(screen, pilot, expected: int) -> ArticleListPane:
    """The mounted `ArticleListPane` once its loader has delivered `expected` rows."""
    screen.active_section = "items"
    await pilot.pause(0.3)
    pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
    for _ in range(60):
        await pilot.pause()
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        if len(pane.items) >= expected:
            break
    return pane


def _row_texts_by_id(pane: ArticleListPane) -> dict[str, str]:
    """The rendered row text for each item, keyed by item id.

    Read off the mounted `ListView`, not off `pane.items`: the pane's
    reactive is what the loader handed it, while this is what the user is
    looking at, and TASK-2301's whole subject is those two disagreeing. The
    rows are `Text` objects built by appending, never markup-parsed
    (`_render_row`), so plain text comparison is exact.
    """
    list_view = pane.query_one("#items-table", ListView)
    out: dict[str, str] = {}
    for node in list_view.children:
        if isinstance(node, _ArticleRow):
            # task-15776: the row renders itself -- there is no inner Static.
            out[node.item_id_key] = node.render().plain
    return out


def test_the_filter_speaks_the_readers_vocabulary():
    """TASK-3072. The contract the rest of this file asserts against.

    Written out literally rather than derived, for the same vacuity reason
    the old STATUS_LABELS table documented: a test that asks production code
    what it displays cannot tell the display from the mapping.
    """
    assert ArticleListPane._FILTER_OPTIONS == [
        ("Unread", "unread"),
        ("All", "all"),
    ]
    assert ArticleListPane._READER_STATUSES == {"new", "reviewed", "ingested"}
    assert ArticleListPane._UNREAD_DOT == "●"
    assert ArticleListPane._STAR_GLYPH == "★"
    assert ArticleListPane._QUEUED_GLYPH == "◆"


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
async def test_every_reader_status_appears_in_the_list_and_is_distinguishable():
    """AC#1, reader vocabulary. The reader set is present under "All", each
    status visibly told apart: unread rows carry the dot, ingested rows the
    marker, reviewed rows neither."""
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_one_item_per_status(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane = await _settled_items_pane(screen, pilot, len(READER_SEEDED))

        assert pane.status_filter == "all"
        displayed = pane.displayed_items()
        assert {row["status"] for row in displayed} == set(READER_SEEDED)
        texts = _row_texts_by_id(pane)
        assert set(texts) == {str(row["id"]) for row in displayed}, (
            "every displayed item must have exactly one rendered row"
        )
        by_status = {
            row["status"]: texts[str(row["id"])] for row in displayed
        }
        assert by_status["new"].startswith("● "), (
            f"the unread row must lead with the unread dot; got {by_status['new']!r}"
        )
        assert "· ingested" in by_status["ingested"], (
            f"the ingested row must carry its marker; got {by_status['ingested']!r}"
        )
        reviewed = by_status["reviewed"]
        assert not reviewed.startswith("●") and "· ingested" not in reviewed, (
            f"the read row must carry neither mark; got {reviewed!r}"
        )


@pytest.mark.asyncio
async def test_a_triaged_item_is_findable_under_the_readers_filters():
    """AC#4. The regression test the task asks for, stated as the user's
    question: "where did my ingested item go?".

    TASK-3072 answer: under "All", right where it was -- ingested and read
    items stay listed. Unread narrows to the unread bucket. What deliberately
    does NOT come back is the ignored item (the user hid it) and the error
    row (a Runs-tab concern), and that exclusion is pinned here too.
    """
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_one_item_per_status(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane = await _settled_items_pane(screen, pilot, len(READER_SEEDED))

        titles = [row["title"] for row in pane.displayed_items()]
        assert set(titles) == set(READER_SEEDED.values()), (
            f"All must show exactly the reader set; showed {titles!r}"
        )

        pane.status_filter = "unread"
        for _ in range(60):
            await pilot.pause(0.05)
            pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
            if [row for row in pane.items if row.get("status") == "new"]:
                break
        titles = [row["title"] for row in pane.displayed_items()]
        assert titles == [SEEDED["new"]], (
            f"Unread must show exactly the unread item; showed {titles!r}"
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
        pane = await _settled_items_pane(screen, pilot, len(READER_SEEDED))

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

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert len(pane.displayed_items()) == rows_before, (
            "ingesting must not remove the row from a view whose filter "
            "includes it"
        )
        row_text = _row_texts_by_id(pane).get(str(target["id"]), "")
        assert "· ingested" in row_text, (
            "the row the user acted on must show its new status immediately; "
            f"rendered {row_text!r}"
        )


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
        pane = await _settled_items_pane(screen, pilot, len(READER_SEEDED))

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
        pane = await _settled_items_pane(screen, pilot, len(READER_SEEDED))

        target = next(item for item in pane.items if item["title"] == SEEDED["new"])
        screen.post_message(IngestRequested(dict(target)))
        for _ in range(80):
            await pilot.pause()
            if db.get_item_status(int(target["item_id"])) == "ingested":
                break
        for _ in range(40):
            await pilot.pause()

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        displayed = pane.displayed_items()
        titles = [row["title"] for row in displayed]
        assert SEEDED["new"] in titles, (
            "the ingested item must still be in the list after the reload"
        )
        assert {row["status"] for row in displayed} == {"reviewed", "ingested"}, (
            "the reader's All still shows exactly its set, with the acted-on "
            "item now ingested"
        )


# --- paging (review wave, I2) ----------------------------------------------


def _seed_many(db, *, ingested: int, new: int) -> tuple[int, list[str]]:
    """One source with `ingested` triaged items NEWER than `new` unread ones.

    The ordering is the whole fixture: `get_new_items` is `created_at DESC`, so
    the unread items sit past the end of any mixed page the screen fetches.

    Returns:
        `(source_id, titles_of_the_unread_items)`.
    """
    source_id = db.add_subscription(
        name="Busy", type="rss", source="https://busy.invalid/feed.xml"
    )
    stale_ids: list[int] = []
    unread_titles: list[str] = []
    with db.transaction() as conn:
        for index in range(new):
            title = f"Unread {index}"
            unread_titles.append(title)
            persist_subscription_item(
                conn,
                source_id,
                {
                    "url": f"https://busy.invalid/unread-{index}",
                    "title": title,
                    "content_hash": f"hash-unread-{index}",
                },
                run_id=None,
                now=f"2020-01-01T00:00:00.{index:04d}+00:00",
            )
        for index in range(ingested):
            stale_ids.append(
                persist_subscription_item(
                    conn,
                    source_id,
                    {
                        "url": f"https://busy.invalid/filed-{index}",
                        "title": f"Filed {index}",
                        "content_hash": f"hash-filed-{index}",
                    },
                    run_id=None,
                    now=f"2026-08-04T09:00:00.{index:04d}+00:00",
                )
            )
    db.bulk_update_items(stale_ids, "ingested")
    return source_id, unread_titles


@pytest.mark.asyncio
async def test_unread_items_past_the_newest_page_are_still_reachable():
    """Review wave, I2. The filter must narrow the QUERY, not just the page.

    TASK-2301 made `_load_items` ask for every status, which fixed "triaged
    items are unreachable" and quietly broke the other direction: the query
    pages at 100 and the pane's in-memory filter only filters what arrived, so
    the page went from "the newest 100 unread" to "the newest 100 of any
    status". With 120 triaged items newer than every unread one, picking
    "Unread" showed ZERO rows -- while the rail, which the same branch made
    accurate, honestly reported the unread count. Two numbers on one screen
    disagreeing about the same fact.

    The fixture is deliberately deeper than the 100-row page so nothing here
    can pass by accident of page size.
    """
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _source_id, unread_titles = _seed_many(db, ingested=120, new=5)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane = await _settled_items_pane(screen, pilot, 100)

        # The precondition that makes this a real test: under the reader's
        # "All" the page is entirely triaged, so an in-memory filter has
        # nothing unread to find.
        assert not [
            row for row in pane.items if row.get("status") == "new"
        ], "precondition: no unread item is inside the newest-100 page"

        pane.status_filter = "unread"
        for _ in range(80):
            await pilot.pause(0.05)
            pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
            if any(row.get("status") == "new" for row in pane.items):
                break

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        titles = [row["title"] for row in pane.displayed_items()]
        assert set(titles) == set(unread_titles), (
            "filtering to Unread must re-page against the unread bucket, not "
            f"filter the mixed page in memory; showed {titles!r}"
        )


@pytest.mark.asyncio
async def test_a_search_keystroke_does_not_re_page():
    """The reload is gated on the STATUS moving, not on any filter message.

    `ItemsFilterChanged` also fires per keystroke in the search box, which is
    a purely in-memory filter; a query per character would be a new defect
    paying for the fix above.
    """
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_one_item_per_status(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane = await _settled_items_pane(screen, pilot, len(READER_SEEDED))

        calls: list[str | None] = []
        real_list_items = screen._controller.list_items

        async def _counting(*, runtime_backend=None, status=None, **kwargs):
            calls.append(status)
            return await real_list_items(
                runtime_backend=runtime_backend, status=status, **kwargs
            )

        screen._controller.list_items = _counting

        pane.search_query = "fresh"
        await pilot.pause()
        await pilot.pause()
        assert calls == [], "a search keystroke must not re-query"

        pane.status_filter = "unread"
        for _ in range(40):
            await pilot.pause(0.05)
            if calls:
                break
        assert calls == ["new"], (
            "switching to Unread must re-page with status='new' pushed into "
            "the query"
        )


@pytest.mark.asyncio
async def test_the_delete_gesture_on_an_item_says_and_does_ignore():
    """Review wave, Minor 2. The affordance now matches the write.

    `d` over an item opened a dialog saying "Delete <title>?" and then wrote
    `status="ignored"`. Before TASK-2301 the row vanished on the next reload
    so it read as a delete; now the row stays, so the gesture looked like it
    had failed -- and on an already-ignored row it genuinely did nothing
    observable. It is routed to the Ignore vocabulary instead, through the
    same dispatch the Inspector's Ignore button uses -- which also puts it
    behind TASK-1541's per-item drain and terminal-status gate, where the old
    direct-to-controller write was not.
    """
    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import DeleteRequested

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    raw_ids = _seed_one_item_per_status(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane = await _settled_items_pane(screen, pilot, len(READER_SEEDED))

        toasts: list[str] = []
        screen.app_instance.notify = lambda message, **kwargs: toasts.append(str(message))
        dialogs: list[object] = []
        screen.app.push_screen = lambda *a, **k: dialogs.append(a)

        target = next(item for item in pane.items if item["title"] == SEEDED["new"])
        screen.handle_delete_requested(DeleteRequested(dict(target)))
        for _ in range(60):
            await pilot.pause(0.05)
            if db.get_item_status(raw_ids["new"]) == "ignored":
                break
        assert dialogs == [], "an item must not be offered a delete dialog"
        assert db.get_item_status(raw_ids["new"]) == "ignored"
        assert any("ignored" in message.lower() for message in toasts), (
            f"the gesture must report what it did; saw {toasts!r}"
        )


@pytest.mark.asyncio
async def test_the_open_item_survives_a_reload_under_a_narrow_filter():
    """Round 2, O2. The pin's guarantee must survive query-side filtering.

    The pane's `_filtered_items` pins the open item into the list whatever the
    filter says, because opening an item MARKS IT READ -- so under the
    "Unread" filter it drops out of its own list the instant it is opened,
    and `j`/`k` break for the rest of the session. That pin can only retain
    what the query returned. Pushing the status filter into the query (I2)
    meant a reload under `status="new"` came back without it, and the item
    the user was reading vanished.

    Driven through the real gesture: filter to Unread, open the only unread
    item (which marks it `reviewed`), then force the reload that any
    deliberate action would cause.
    """
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    raw_ids = _seed_one_item_per_status(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        pane = await _settled_items_pane(screen, pilot, len(READER_SEEDED))

        pane.status_filter = "unread"
        for _ in range(60):
            await pilot.pause(0.05)
            pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
            if [row for row in pane.items if row.get("status") == "new"]:
                break
        assert [row["title"] for row in pane.displayed_items()] == [SEEDED["new"]]

        target = pane.items[0]
        pane.select_item_by_id(str(target["id"]))
        for _ in range(60):
            await pilot.pause(0.05)
            if db.get_item_status(raw_ids["new"]) == "reviewed":
                break
        assert db.get_item_status(raw_ids["new"]) == "reviewed", (
            "precondition: opening the item marked it read, so the Unread "
            "filter no longer matches it"
        )

        # Any deliberate action reloads. Force exactly that.
        await screen._load_items()
        await pilot.pause()

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        titles = [row["title"] for row in pane.displayed_items()]
        assert SEEDED["new"] in titles, (
            "the item the reader has open must survive a reload whose filter "
            f"no longer matches it; the list showed {titles!r}"
        )
        assert pane.selected_item is not None
        assert str(pane.selected_item.get("id")) == str(target["id"]), (
            "and it must still be the selected item, or j/k walk a list the "
            "cursor is not in"
        )


@pytest.mark.asyncio
async def test_the_carried_open_item_keeps_the_pages_ordering():
    """Round 2, O2. Carried, not prepended.

    `j`/`k` walk this sequence, so an item jumping to the top of the list the
    moment its status changed would be its own small lie about recency.
    """
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_one_item_per_status(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        await _settled_items_pane(screen, pilot, len(READER_SEEDED))

        # An item older than every row of the page it will be carried into.
        screen._selected_content_item = {
            "id": "local:watchlist_item:99999",
            "title": "The oldest thing here",
            "status": "ignored",
            "created_at": "2000-01-01T00:00:00+00:00",
        }
        # The pre-reader per-status value, poked directly: TASK-3072's
        # `_normalize_items_status_filter` maps it to "unread", so this also
        # pins the legacy-value path -- the query below is `status="new"`.
        screen._items_status_filter = "new"
        screen._items_committed_page_key = screen._items_page_key(0)
        # Production selection records this provenance alongside the item.
        screen._selected_content_page_key = screen._items_committed_page_key
        await screen._load_items()

        titles = [row["title"] for row in screen._loaded_items]
        assert titles[-1] == "The oldest thing here", (
            f"the carried item must sort by created_at, not jump; got {titles!r}"
        )


def test_the_d_binding_names_both_verbs_it_performs():
    """Round 2, O3. The label stopped matching the action for one kind.

    `d` deletes a source/run/rule behind a confirmation, and IGNORES an item
    without one (review wave, Minor 2). A Textual binding description is
    static, so it has to name the pair rather than promise whichever verb the
    current selection is not.
    """
    from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
        WatchlistsCollectionsScreen,
    )

    binding = next(
        entry
        for entry in WatchlistsCollectionsScreen.BINDINGS
        if entry[0] == "d"
    )
    assert binding[1] == "delete_selected"
    label = binding[2].lower()
    assert "delete" in label and "ignore" in label, (
        f"the key performs both verbs; its label says {binding[2]!r}"
    )
    assert "after confirmation" not in (
        WatchlistsCollectionsScreen.action_delete_selected.__doc__ or ""
    ).split("\n")[0], "the summary line must not promise a dialog for every kind"
