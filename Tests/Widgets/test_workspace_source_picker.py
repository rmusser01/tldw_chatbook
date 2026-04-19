from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from textual.app import App, ComposeResult

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
async def test_workspace_source_picker_loads_results_from_service_search():
    service = AsyncMock()
    service.search_media = AsyncMock(
        return_value={
            "items": [
                {"id": 11, "title": "Search Hit", "media_type": "pdf"},
            ]
        }
    )
    picker = WorkspaceSourcePicker(service=service)

    results = await picker.load_results("search hit")

    service.search_media.assert_awaited_once_with("search hit")
    assert results[0]["id"] == 11
    assert picker.results[0]["title"] == "Search Hit"
