from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.UI.Mindmap_Viewer_Window import MindmapViewerWindow
from tldw_chatbook.Widgets.Evals.eval_additional_dialogs import RunSelectionDialog
from tldw_chatbook.Widgets.Evals.sample_browser_dialog import SampleBrowserDialog
from tldw_chatbook.Widgets.Note_Widgets.note_selection_dialog import NoteSelectionDialog
from tldw_chatbook.Widgets.collections_tag_window import CollectionsTagWindow
from tldw_chatbook.Widgets.multi_item_review_window import MultiItemReviewWindow


class _ScreenHost(App):
    def __init__(self, screen):
        super().__init__()
        self.screen_under_test = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.screen_under_test)


def _assert_button_tooltips(root, expected_tooltips: dict[str, str]) -> None:
    for button_id, expected_tooltip in expected_tooltips.items():
        button = root.query_one(f"#{button_id}", Button)
        assert str(button.tooltip) == expected_tooltip


@pytest.mark.asyncio
async def test_eval_run_selection_bulk_controls_have_tooltips():
    app = _ScreenHost(RunSelectionDialog(available_runs=[]))

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.screen_under_test,
            {
                "select-all-button": "Select every available evaluation run.",
                "clear-all-button": "Clear every selected evaluation run.",
            },
        )


@pytest.mark.asyncio
async def test_eval_sample_browser_bulk_controls_have_tooltips(monkeypatch):
    class _PlainTextArea(Static):
        def __init__(self, text: str = "", *args, id: str | None = None, **kwargs):
            super().__init__(text, id=id)

    monkeypatch.setattr(SampleBrowserDialog, "CSS", "", raising=False)
    monkeypatch.setattr(SampleBrowserDialog, "load_samples", lambda self: None)
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.Evals.sample_browser_dialog.TextArea",
        _PlainTextArea,
    )
    app = _ScreenHost(SampleBrowserDialog(dataset_path="/tmp/empty-dataset.json"))

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.screen_under_test,
            {
                "select-all-btn": "Select every sample on the current page.",
                "clear-btn": "Clear every selected dataset sample.",
            },
        )


@pytest.mark.asyncio
async def test_note_selection_bulk_controls_have_tooltips(monkeypatch):
    monkeypatch.setattr(NoteSelectionDialog, "CSS", "", raising=False)
    monkeypatch.setattr(NoteSelectionDialog, "load_notes", lambda self, notes: None)
    app = _ScreenHost(NoteSelectionDialog(notes=[]))

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.screen_under_test,
            {
                "select-all-btn": "Select every visible note for audio generation.",
                "clear-all-btn": "Clear every selected note.",
            },
        )


@pytest.mark.asyncio
async def test_tag_management_bulk_controls_have_tooltips():
    app_instance = SimpleNamespace(media_db=None, notify=Mock())

    class TagWindowApp(App):
        def compose(self) -> ComposeResult:
            yield CollectionsTagWindow(app_instance=app_instance)

    app = TagWindowApp()

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.query_one(CollectionsTagWindow),
            {
                "select-all-keywords": "Select every visible keyword or tag.",
                "clear-selection": "Clear every selected keyword or tag.",
            },
        )


@pytest.mark.asyncio
async def test_multi_item_review_bulk_controls_have_tooltips():
    app_instance = SimpleNamespace(media_db=None, notify=Mock())

    class MultiReviewApp(App):
        def compose(self) -> ComposeResult:
            yield MultiItemReviewWindow(app_instance=app_instance)

    app = MultiReviewApp()

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.query_one(MultiItemReviewWindow),
            {
                "select-all-items": "Select every visible media item for multi-item review.",
                "clear-all-items": "Clear every selected media item.",
            },
        )


@pytest.mark.asyncio
async def test_mindmap_source_selection_clear_control_has_tooltip():
    fake_db = SimpleNamespace(
        list_notes=lambda limit=100: [],
        get_conversations=lambda limit=50: [],
        get_all_characters=lambda: [],
    )
    app_instance = SimpleNamespace(chachanotes_db=fake_db)
    app = _ScreenHost(MindmapViewerWindow(app_instance=app_instance))

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.screen_under_test,
            {
                "clear-selection": "Clear selected source content before creating a mindmap.",
            },
        )
