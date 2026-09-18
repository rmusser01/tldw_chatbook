"""Collections filter recovery, saved-search paging and reversible Archive."""

from dataclasses import replace

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_library_collection_reader_journeys import _open, _seed
from Tests.UI.test_library_prompt_collection_journeys import _focus
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _painted_text,
    _wait_for_condition,
    _wait_for_selector,
)
from Tests.UI.test_library_skill_editor_journeys import _activate
from tldw_chatbook.Library.collections_capture_models import (
    CapturePageRequest,
    CollectionsCaptureError,
    SavedCaptureSearch,
)


async def _show_pane(screen, host, pilot, name):
    shell = screen.query_one("#library-collections-reader-shell")
    if not getattr(shell.effective_layout, f"{name}_open"):
        await _activate(
            screen, host, pilot, f"#library-collections-{name}-grip", "--->"
        )


async def _saved_searches(scope, count=21):
    key = scope.active_authority.key
    return [
        await scope.save_saved_search(
            SavedCaptureSearch(
                key,
                "new",
                f"Research {n:02}",
                CapturePageRequest(key, search="Alpha", statuses=("saved",)),
                "",
                "",
                1,
            )
        )
        for n in range(count)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_clear_restores_results_and_valid_sort_within_scope(
    size, theme, monkeypatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    app, _scope, _ = await _seed()
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = await _open(host, pilot)
        await _show_pane(screen, host, pilot, "library")
        await _activate(
            screen, host, pilot, "#library-collections-scope-saved", "Saved"
        )
        engine = screen._library_collections_capture_controller
        # Seed a supported combined request; Clear must remove every narrowing
        # predicate while retaining the Saved rail scope and valid sorting.
        request = replace(
            engine.state.requested_scope,
            search="missing",
            sort="relevance",
            domain="absent.test",
            tags=("absent",),
            date_from="2001-01-01",
            date_to="2001-12-31",
        )
        await screen._collections_controller._apply_library_collection_capture_request(
            request
        )
        await pilot.pause()
        assert engine.state.page.total == 0
        await _show_pane(screen, host, pilot, "items")
        await _activate(screen, host, pilot, "#library-collections-filters", "Filters")
        await _activate(
            screen, host, pilot, "#library-collections-filters-clear", "Clear"
        )
        await _wait_for_condition(
            pilot,
            lambda: engine.state.page.total == 2,
            message="Clear did not restore the two saved captures",
        )
        current = engine.state.applied_scope
        assert current.search == ""
        assert current.sort == "saved_desc"
        assert current.statuses == ("saved",)
        assert current.tags == () and current.domain is None
        assert current.date_from is None and current.date_to is None
        assert current.page == 1
        assert screen.query_one("#library-collections-filter", Input).value == ""
        assert {item.title for item in engine.state.page.items} == {"Alpha", "Beta"}
        await _focus(screen, host, pilot, "#library-collections-filters-clear", "Clear")
        await _activate(screen, host, pilot, "#library-collections-sort", "Sort")
        assert engine.state.applied_scope.sort == "saved_asc"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_more_saved_searches_pages_and_applies_real_scope(
    size, theme, monkeypatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    app, scope, _ = await _seed()
    searches = await _saved_searches(scope)
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = await _open(host, pilot)
        await _show_pane(screen, host, pilot, "library")
        first_ids = {s.search_id for s in screen._collections_state.saved_searches}
        assert len(first_ids) == 20
        remaining = next(s for s in searches if s.search_id not in first_ids)
        engine = screen._library_collections_capture_controller
        before = engine.state.applied_scope
        await _activate(
            screen,
            host,
            pilot,
            "#library-collections-more-saved-searches",
            "More searches",
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                [s.search_id for s in screen._collections_state.saved_searches]
                == [remaining.search_id]
            ),
            message="More saved searches did not reach the remaining saved search",
        )
        assert engine.state.applied_scope == before
        selector = f"#library-collections-saved-search-{remaining.search_id}"
        await _focus(screen, host, pilot, selector, remaining.name)
        # Enter activates the naturally focused first row of the new window.
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: engine.state.page.total == 1,
            message="Saved-search selection did not filter captures",
        )
        assert engine.state.applied_scope.authority_key == scope.active_authority.key
        assert engine.state.applied_scope.search == "Alpha"
        assert engine.state.applied_scope.statuses == ("saved",)
        assert engine.state.page.items[0].title == "Alpha"
        selected_scope = screen._collections_state.active_scope
        await _activate(
            screen,
            host,
            pilot,
            "#library-collections-previous-saved-searches",
            "Previous",
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                {s.search_id for s in screen._collections_state.saved_searches}
                == first_ids
            ),
            message="Previous did not restore the first saved-search window",
        )
        assert screen._collections_state.active_scope == selected_scope
        assert engine.state.page.total == 1
        await _focus(
            screen,
            host,
            pilot,
            "#library-collections-more-saved-searches",
            "More searches",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_repeated_archive_keeps_original_undo_status(size, theme, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    app, scope, identities = await _seed()
    identity = identities[0]
    detail = (await scope.get_detail(identity)).capture
    await scope.update_capture(identity, detail.revision, {"status": "reading"})
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = await _open(host, pilot)
        await screen._select_library_collection_capture(identity)
        await pilot.pause()
        await _activate(
            screen, host, pilot, "#library-collections-archive", "Move to Archive"
        )
        await _wait_for_selector(screen, pilot, "#library-collections-archive-undo")
        archived = (await scope.get_detail(identity)).capture
        assert archived.status == "archived"
        button = screen.query_one("#library-collections-archive", Button)
        if not button.disabled:
            await _activate(
                screen, host, pilot, "#library-collections-archive", "Move to Archive"
            )
        assert (await scope.get_detail(identity)).capture.revision == archived.revision
        assert screen.query_one("#library-collections-archive", Button).disabled
        assert "Archived" in _painted_text(host, button.region)
        assert (
            "already"
            in str(
                screen.query_one("#library-collections-archive", Button).tooltip
            ).lower()
        )
        await _activate(
            screen, host, pilot, "#library-collections-archive-undo", "Undo"
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                not screen._library_collections_capture_controller.state.visible_archive_receipts
            ),
            message="Undo did not settle",
        )
        assert (await scope.get_detail(identity)).capture.status == "reading"


@pytest.mark.asyncio
async def test_failed_saved_search_page_keeps_rows_and_visible_retry(monkeypatch):
    app, scope, _ = await _seed()
    await _saved_searches(scope)
    host = LibraryProductionCSSHarness(app)
    host.theme = "textual-light"
    async with host.run_test(size=(80, 24)) as pilot:
        screen = await _open(host, pilot)
        await _show_pane(screen, host, pilot, "library")
        original = scope.list_saved_searches
        first = screen._collections_state.saved_searches

        async def fail(page, size=20):
            raise CollectionsCaptureError("unavailable")

        monkeypatch.setattr(scope, "list_saved_searches", fail)
        await _activate(
            screen,
            host,
            pilot,
            "#library-collections-more-saved-searches",
            "More searches",
        )
        await _focus(
            screen,
            host,
            pilot,
            "#library-collections-retry-saved-searches",
            "Retry searches",
        )
        assert screen._collections_state.saved_searches == first
        assert screen._collections_state.saved_searches_error
        monkeypatch.setattr(scope, "list_saved_searches", original)
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: len(screen._collections_state.saved_searches) == 1,
            message="Retry did not load the requested second page",
        )
        assert not screen._collections_state.saved_searches_error
        search = screen._collections_state.saved_searches[0]
        await _focus(
            screen,
            host,
            pilot,
            f"#library-collections-saved-search-{search.search_id}",
            search.name,
        )


@pytest.mark.asyncio
async def test_scope_rejects_repeated_archive_without_replacing_undo():
    _app, scope, identities = await _seed()
    identity = identities[0]
    detail = (await scope.get_detail(identity)).capture
    await scope.update_capture(identity, detail.revision, {"status": "reading"})
    detail = (await scope.get_detail(identity)).capture
    archived = await scope.archive(identity, detail.revision)
    with pytest.raises(CollectionsCaptureError, match="already_archived"):
        await scope.archive(identity, archived.revision)
    assert (await scope.get_detail(identity)).capture.revision == archived.revision
    restored = await scope.undo_archive(identity, archived.revision)
    assert restored.status == "reading"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "focus_target", ["saved", "outgoing-search", "replaced-opener"]
)
async def test_saved_search_page_preserves_newer_scope_focus(monkeypatch, focus_target):
    import asyncio

    app, scope, _ = await _seed()
    await _saved_searches(scope)
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)
        await _show_pane(screen, host, pilot, "library")
        entered, release = asyncio.Event(), asyncio.Event()
        list_searches = scope.list_saved_searches

        async def held_page(page, size=20):
            result = await list_searches(page, size)
            if page == 2:
                entered.set()
                await release.wait()
            return result

        monkeypatch.setattr(scope, "list_saved_searches", held_page)
        opener = screen.query_one("#library-collections-more-saved-searches", Button)
        opener.focus()
        await pilot.pause()
        pending = asyncio.create_task(
            screen._collections_controller._page_library_collection_saved_searches(
                2, opener
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            search = screen._collections_state.saved_searches[0]
            selector = (
                f"#library-collections-saved-search-{search.search_id}"
                if focus_target == "outgoing-search"
                else "#library-collections-scope-saved"
            )
            label = search.name if focus_target == "outgoing-search" else "Saved"
            await _activate(screen, host, pilot, selector, label)
            await _focus(screen, host, pilot, selector, label)
            if focus_target == "replaced-opener":
                replacement = screen.query_one(
                    "#library-collections-more-saved-searches"
                )
                assert replacement is not opener
                replacement.focus()
                await pilot.pause()
            active_scope = screen._collections_state.active_scope
            release.set()
            await pending
            if focus_target != "saved":
                search = screen._collections_state.saved_searches[0]
                selector = f"#library-collections-saved-search-{search.search_id}"
                label = search.name
            await _focus(screen, host, pilot, selector, label)
            assert screen._collections_state.active_scope == active_scope
            assert (
                screen._library_collections_capture_controller.state.applied_scope.statuses
                == ("saved",)
            )
        finally:
            release.set()
            await pending
