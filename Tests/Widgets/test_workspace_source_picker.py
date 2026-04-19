from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.tldw_api.notes_workspace_schemas import MediaSearchRequest
from tldw_chatbook.Widgets.Note_Widgets.workspace_source_picker import WorkspaceSourcePicker


class PickerTestApp(App[None]):
    def __init__(self, picker: WorkspaceSourcePicker):
        super().__init__()
        self._picker = picker

    def compose(self) -> ComposeResult:
        yield self._picker


def test_workspace_source_picker_returns_selected_media_id():
    picker = WorkspaceSourcePicker(results=[{"id": 42, "title": "Paper", "type": "pdf"}])
    picker.select_result(42)
    assert picker.selected_media_id == 42


def test_workspace_source_picker_select_result_persists_selection():
    picker = WorkspaceSourcePicker(results=[{"id": 9, "title": "Video", "type": "video"}])
    selected = picker.select_result(9)

    assert selected is True
    assert picker.selected_media_id == 9


@pytest.mark.asyncio
async def test_workspace_source_picker_loads_results_from_api_client_search():
    client = AsyncMock()
    client.search_media_items = AsyncMock(
        return_value={
            "items": [
                {"id": 11, "title": "Search Hit", "media_type": "pdf"},
            ]
        }
    )
    picker = WorkspaceSourcePicker(service=client)

    results = await picker.load_results("search hit")

    client.search_media_items.assert_awaited_once()
    args = client.search_media_items.await_args
    assert isinstance(args.kwargs["request_data"], MediaSearchRequest)
    assert args.kwargs["request_data"].query == "search hit"
    assert args.kwargs["page"] == 1
    assert args.kwargs["results_per_page"] == 10
    assert results[0]["id"] == 11
    assert picker.results[0]["title"] == "Search Hit"


@pytest.mark.asyncio
async def test_workspace_source_picker_loads_results_from_service_client_list():
    client = AsyncMock()
    client.list_media_items = AsyncMock(
        return_value={
            "items": [
                {"id": 7, "title": "Library Item", "type": "video"},
            ]
        }
    )
    service = Mock()
    service.client = client
    picker = WorkspaceSourcePicker(service=service)

    results = await picker.load_results()

    client.list_media_items.assert_awaited_once_with(
        page=1,
        results_per_page=10,
        include_keywords=False,
    )
    assert results[0]["id"] == 7
