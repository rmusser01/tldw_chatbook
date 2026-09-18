"""Saved-search windows keep real rows through errors and stale responses."""

import asyncio

import pytest

from Tests.Library.test_collections_capture_scope_service import _local_service
from Tests.UI.test_library_collection_browse_journeys import _saved_searches
from Tests.UI.test_library_collection_reader_journeys import _seed
from tldw_chatbook.Library.collections_capture_models import CollectionsCaptureError
from tldw_chatbook.UI.Library_Modules.library_collections_saved_search_controller import (
    load_saved_search_page,
)
from tldw_chatbook.UI.Library_Modules.library_collections_state import (
    LibraryCollectionsState,
)


@pytest.mark.asyncio
async def test_failed_saved_search_continuation_keeps_window_for_retry(monkeypatch):
    _app, scope, _ = await _seed()
    searches = await _saved_searches(scope)
    state = LibraryCollectionsState()
    assert await load_saved_search_page(state, scope, 1)
    first = state.saved_searches
    list_searches = scope.list_saved_searches

    async def failed_page(page, size=20):
        if page == 2:
            raise CollectionsCaptureError("saved_search_transport_failed")
        return await list_searches(page, size)

    monkeypatch.setattr(scope, "list_saved_searches", failed_page)
    assert not await load_saved_search_page(state, scope, 2)
    assert state.saved_searches == first
    assert state.saved_searches_page == 1
    assert state.saved_searches_requested_page == 2
    assert state.saved_searches_total == 21
    assert "Retry" in state.saved_searches_error
    assert not state.saved_searches_loading
    monkeypatch.setattr(scope, "list_saved_searches", list_searches)
    assert await load_saved_search_page(state, scope, 2)
    assert len(state.saved_searches) == 1
    assert {s.search_id for s in (*first, *state.saved_searches)} == {
        s.search_id for s in searches
    }
    assert state.saved_searches_error == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("switch_authority", [False, True])
async def test_late_saved_search_window_cannot_replace_newer_window(
    monkeypatch, tmp_path, switch_authority
):
    _app, scope, _ = await _seed()
    await _saved_searches(scope)
    state = LibraryCollectionsState()
    assert await load_saved_search_page(state, scope, 1)
    entered, release = asyncio.Event(), asyncio.Event()
    list_searches = scope.list_saved_searches

    async def held_page(page, size=20):
        result = await list_searches(page, size)
        if page == 2:
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(scope, "list_saved_searches", held_page)
    pending = asyncio.create_task(load_saved_search_page(state, scope, 2))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if switch_authority:
            authority, _database, _repository, backend = _local_service(tmp_path)
            scope.activate(authority, backend)
            await _saved_searches(scope, count=1)
        assert await load_saved_search_page(state, scope, 1)
        latest = state.saved_searches
        release.set()
        assert not await pending
        assert state.saved_searches == latest
        assert state.saved_searches_page == 1
        assert state.saved_searches_authority == scope.active_authority.key
        assert all(s.authority_key == scope.active_authority.key for s in latest)
        assert len(latest) == (1 if switch_authority else 20)
    finally:
        release.set()
        await pending
