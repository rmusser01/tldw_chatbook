from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

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
