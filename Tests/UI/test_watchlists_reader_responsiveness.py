"""Reader detail work must not own the Watchlists screen message queue."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static
from textual.worker import NoActiveWorker, WorkerCancelled, get_current_worker

from Tests.private_profile import private_profile_test


async def _settle_held_workers(state):
    """Join the captured old work before asserting absence of late effects."""
    assert state.held_workers, "held read must retain its screen-owned detail worker"
    for worker in state.held_workers:
        try:
            await worker.wait()
        except WorkerCancelled:
            pass


async def _until(predicate, *, seconds=8):
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.01)


@asynccontextmanager
async def _reader(
    monkeypatch, *, hold=True, fail=False, small_page=False, cancellation_survives=False
):
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import DestinationHarness
    from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
    from tldw_chatbook.UI.Screens import watchlists_collections_screen as module
    from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane

    if small_page:
        monkeypatch.setattr(module, "_ITEMS_PAGE_SIZE", 2)
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    source = db.add_subscription(
        name="Reader feed", type="rss", source="https://reader.invalid/feed"
    )
    ids = []
    with db.transaction() as cursor:
        for letter, day in [("A", "24"), ("B", "23"), ("C", "22")]:
            ids.append(
                persist_subscription_item(
                    cursor,
                    source,
                    {
                        "title": "Article " + letter,
                        "url": "https://reader.invalid/" + letter,
                        "content": "Body " + letter,
                        "content_hash": letter,
                        "content_kind": "article",
                        "content_format": "text",
                        "published_date": f"2026-10-{day}T10:00:00+00:00",
                    },
                    None,
                    "2026-10-25T10:00:00+00:00",
                )
            )
    entered, release, finished = (threading.Event() for _ in range(3))
    original = db.get_item_content
    calls = []
    held_workers = []
    held_once = False

    def read(item_id):
        nonlocal held_once
        calls.append((int(item_id), threading.get_ident()))
        if hold and int(item_id) == ids[1] and not held_once:
            held_once = True
            try:
                held_workers.append(get_current_worker())
            except NoActiveWorker:
                pass  # The unchanged handler has no worker (RED witness).
            entered.set()
            try:
                assert release.wait(20), "controlled read was never released"
                if fail:
                    raise RuntimeError("controlled obsolete body failure")
                return original(item_id)
            finally:
                finished.set()
        return original(item_id)

    monkeypatch.setattr(db, "get_item_content", read)
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        pending = []
        try:
            screen = host.screen_stack[-1]
            # Verify this interpreter executes the checked-out production sources.
            for kind in (type(screen), type(db), type(app.local_watchlists_service)):
                assert Path(sys.modules[kind.__module__].__file__).is_relative_to(
                    Path.cwd()
                )
            await _until(
                lambda: bool(screen.query("#watchlists-items-pane"))
                and len(
                    screen.query_one("#watchlists-items-pane", ArticleListPane).items
                )
                == (2 if small_page else 3)
            )
            await pilot.pause()
            pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
            assert all(row.get("content") is None for row in pane.items)
            content = screen.query_one("#watchlists-content-pane", ContentPane)
            publications = []
            host.watch(
                content,
                "item",
                lambda item: publications.append((item["title"], item.get("content")))
                if item
                else None,
                init=False,
            )
            if cancellation_survives:
                get_content = screen._controller.get_item_content

                async def complete_despite_cancel(**kwargs):
                    # Exercise publication authority independently of cancellation:
                    # the real file-DB thread may finish after its worker retires.
                    operation = asyncio.create_task(get_content(**kwargs))
                    try:
                        return await asyncio.shield(operation)
                    except asyncio.CancelledError:
                        return await operation

                monkeypatch.setattr(
                    screen._controller, "get_item_content", complete_despite_cancel
                )
            pane.focus_first_row_without_selecting()
            await pilot.press("enter")
            await _until(lambda: screen._selected_content_item is pane.items[0])
            await _until(lambda: bool(screen.query("#content-body")))
            assert "Body A" in str(screen.query_one("#content-body", Static).renderable)
            await _until(lambda: db.get_item_status(ids[0]) == "reviewed")
            state = SimpleNamespace(
                app=app,
                db=db,
                host=host,
                pilot=pilot,
                screen=screen,
                pane=pane,
                content=content,
                ids=ids,
                entered=entered,
                release=release,
                finished=finished,
                calls=calls,
                publications=publications,
                pending=pending,
                old_item=pane.items[1],
                held_workers=held_workers,
            )
            yield state
        finally:
            # Release inside run_test: waiting until context exit deadlocks teardown.
            release.set()
            if entered.is_set():
                await _until(finished.is_set)
                await _settle_held_workers(SimpleNamespace(held_workers=held_workers))
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await pilot.pause()


@pytest.mark.asyncio
@private_profile_test
async def test_ordinary_reader_publishes_one_hydrated_shared_row(
    request, tmp_path, monkeypatch
):
    async with _reader(monkeypatch, hold=False) as state:
        for index, letter in [(1, "B"), (2, "C")]:
            await state.pilot.press("down")
            await _until(
                lambda: state.screen._selected_content_item is state.pane.items[index]
            )
            assert state.content.item is state.pane.items[index]
            assert state.content.item is state.screen._loaded_items[index]
            assert state.content.item["content"] == "Body " + letter
            await _until(
                lambda: state.db.get_item_status(state.ids[index]) == "reviewed"
            )
        assert state.publications == [
            ("Article A", "Body A"),
            ("Article B", "Body B"),
            ("Article C", "Body C"),
        ]
        assert all(thread != threading.get_ident() for _, thread in state.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True], ids=["late-success", "late-failure"])
@private_profile_test
async def test_newer_selection_reads_before_older_body_finishes(
    request, tmp_path, monkeypatch, fail
):
    async with _reader(monkeypatch, fail=fail, cancellation_survives=True) as state:
        notices = []
        monkeypatch.setattr(
            state.app, "notify", lambda message, **kwargs: notices.append(message)
        )
        state.pending.append(asyncio.create_task(state.pilot.press("down")))
        await _until(state.entered.is_set)
        state.pane.select_item_by_id(str(state.pane.items[2]["id"]))
        try:
            await _until(
                lambda: state.screen._selected_content_item is state.pane.items[2],
                seconds=2,
            )
        except TimeoutError:
            pass
        assert (
            state.screen._selected_content_item is state.pane.items[2]
        ), "new selection stalled behind old body"
        assert not state.release.is_set()
        assert state.content.item["content"] == "Body C"
        state.release.set()
        await _until(state.finished.is_set)
        await _settle_held_workers(state)
        await state.pilot.pause()
        assert state.content.item is state.pane.items[2]
        assert state.old_item.get("content") is None
        assert state.db.get_item_status(state.ids[1]) == "new"
        assert not notices
        assert state.publications == [("Article A", "Body A"), ("Article C", "Body C")]


@pytest.mark.asyncio
@private_profile_test
async def test_enabled_section_navigation_runs_during_body_read(
    request, tmp_path, monkeypatch
):
    async with _reader(monkeypatch) as state:
        state.pending.append(asyncio.create_task(state.pilot.press("down")))
        await _until(state.entered.is_set)
        button = state.screen.query_one("#wl-tab-sources", Button)
        assert not button.disabled
        button.press()
        try:
            await _until(lambda: state.screen.active_section == "sources", seconds=2)
        except TimeoutError:
            pass
        assert (
            state.screen.active_section == "sources"
        ), "enabled section stalled behind old body"
        assert not state.release.is_set()
        state.release.set()
        await _until(state.finished.is_set)
        await _settle_held_workers(state)
        await state.pilot.pause()
        assert state.screen.active_section == "sources"
        assert state.screen._selected_content_item["title"] == "Article A"
        assert state.old_item.get("content") is None
        assert state.db.get_item_status(state.ids[1]) == "new"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transition",
    [
        "query-away-back",
        "page-away-back",
        "backend-away-back",
        "section-away-back",
        "unmount",
    ],
)
@pytest.mark.parametrize("fail", [False, True], ids=["late-success", "late-failure"])
@private_profile_test
async def test_retired_reader_context_cannot_mutate_or_report(
    request, tmp_path, monkeypatch, transition, fail
):
    async with _reader(
        monkeypatch,
        fail=fail,
        small_page=transition == "page-away-back",
        cancellation_survives=True,
    ) as state:
        screen = state.screen
        notices = []
        monkeypatch.setattr(
            state.app, "notify", lambda message, **kwargs: notices.append(message)
        )
        state.pane.select_item_by_id(str(state.old_item["id"]))
        await _until(state.entered.is_set)
        if transition == "query-away-back":
            screen._items_search_query = "Article C"
            screen._supersede_items_query_intent()
            assert await screen._replace_items_snapshot(reason="search")
            screen._items_search_query = ""
            screen._supersede_items_query_intent()
            assert await screen._replace_items_snapshot(reason="search")
        elif transition == "page-away-back":
            assert await screen._load_next_items_page()
            assert screen._items_page_index == 1
            assert await screen._present_cached_items_page(0)
        elif transition == "backend-away-back":
            screen.runtime_backend = "server"
            await _until(lambda: screen._read_recovery_active)
            screen.runtime_backend = "local"
            await _until(
                lambda: not screen._read_recovery_active
                and screen._items_snapshot is not None
                and not screen._items_page_loading
            )
        elif transition == "section-away-back":
            screen.query_one("#wl-tab-sources", Button).press()
            await _until(lambda: screen.active_section == "sources", seconds=2)
            screen.query_one("#wl-tab-items", Button).press()
            await _until(lambda: screen.active_section == "items", seconds=2)
        else:
            await state.host.pop_screen()
            await _until(lambda: not screen.is_attached)
            assert not screen.is_attached
        assert (
            not state.finished.is_set()
        ), "context transition waited for obsolete body"
        state.release.set()
        await _until(state.finished.is_set)
        await _settle_held_workers(state)
        await state.pilot.pause()
        assert state.old_item.get("content") is None
        assert state.db.get_item_status(state.ids[1]) == "new"
        assert not notices
        assert (
            screen._selected_content_item is None
            or screen._selected_content_item["title"] != "Article B"
        )


@pytest.mark.asyncio
@private_profile_test
async def test_current_body_failure_reports_and_later_selection_can_retry(
    request, tmp_path, monkeypatch
):
    async with _reader(monkeypatch, fail=True) as state:
        notices = []
        monkeypatch.setattr(
            state.app, "notify", lambda message, **kwargs: notices.append(message)
        )
        state.pane.select_item_by_id(str(state.old_item["id"]))
        await _until(state.entered.is_set)
        state.release.set()
        await _until(lambda: state.screen._selected_content_item is state.old_item)
        assert notices == ["Failed to load this item's full content."]
        assert state.content.item.get("content") is None
        state.pane.select_item_by_id(str(state.pane.items[2]["id"]))
        await _until(lambda: state.screen._selected_content_item is state.pane.items[2])
        state.pane.select_item_by_id(str(state.old_item["id"]))
        await _until(lambda: state.screen._selected_content_item is state.old_item)
        assert state.content.item["content"] == "Body B"
        assert len(notices) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["j", "k", "space"])
@private_profile_test
async def test_keyboard_navigation_follows_pending_selection(
    request, tmp_path, monkeypatch, key
):
    async with _reader(monkeypatch, cancellation_survives=True) as state:
        if key != "space":
            state.screen.set_focus(None)
        await state.pilot.press("space" if key == "space" else "j")
        await _until(state.entered.is_set)
        assert state.pane.selected_item is state.old_item
        assert state.screen._selected_content_item is state.pane.items[0]
        await state.pilot.press(key)
        expected = state.pane.items[0 if key == "k" else 2]
        assert (
            state.pane.selected_item is expected
        ), "navigation used published A, not pending B"
        await _until(lambda: state.content.item is expected)
        assert not state.release.is_set()
        assert state.content.item["content"] == ("Body A" if key == "k" else "Body C")
        state.release.set()
        await _until(state.finished.is_set)
        await _settle_held_workers(state)
        await state.pilot.pause()
        assert state.content.item is expected
        assert state.old_item.get("content") is None
        assert state.db.get_item_status(state.ids[1]) == "new"
        assert all(title != "Article B" for title, _ in state.publications)
