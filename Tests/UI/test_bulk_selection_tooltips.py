import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button

from tldw_chatbook.Widgets.Note_Widgets.note_selection_dialog import NoteSelectionDialog


class _ScreenHost(ConsolidatedCSSApp):
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