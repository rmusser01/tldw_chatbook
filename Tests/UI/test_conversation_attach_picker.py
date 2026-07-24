import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, ListView

from tldw_chatbook.Widgets.Persona_Widgets.conversation_attach_picker import (
    ConversationAttachPicker,
)


class _PickerHost(App):
    def __init__(self, convs):
        super().__init__()
        self._convs = convs
        self.result = "unset"

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        # push_screen_wait requires an active worker (NoActiveWorker otherwise);
        # mirrors the pattern used by the merged sibling test
        # (Tests/UI/test_dictionary_attach_picker.py).
        self.run_worker(self._drive)

    async def _drive(self) -> None:
        self.result = await self.push_screen_wait(ConversationAttachPicker(self._convs))


@pytest.mark.asyncio
async def test_picker_select_returns_string_id():
    convs = [{"conversation_id": "c1", "title": "Alpha"},
             {"conversation_id": "c2", "title": "Beta"}]
    app = _PickerHost(convs)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen.query_one("#conversation-attach-list", ListView).index = 1
        await pilot.pause()
        await pilot.click("#conversation-attach-confirm")
        await pilot.pause()
    assert app.result == "c2"


@pytest.mark.asyncio
async def test_picker_filter_narrows_then_selects():
    # After filtering to "beta", index 0 must be Beta (c2) — proving the filter
    # rebuilt the row-id list. If the filter did nothing, index 0 would be c1.
    convs = [{"conversation_id": "c1", "title": "Alpha"},
             {"conversation_id": "c2", "title": "Beta"}]
    app = _PickerHost(convs)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen.query_one("#conversation-attach-search", Input).value = "beta"
        await pilot.pause()
        app.screen.query_one("#conversation-attach-list", ListView).index = 0
        await pilot.pause()
        await pilot.click("#conversation-attach-confirm")
        await pilot.pause()
    assert app.result == "c2"


@pytest.mark.asyncio
async def test_picker_cancel_returns_none():
    convs = [{"conversation_id": "c1", "title": "Alpha"}]
    app = _PickerHost(convs)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#conversation-attach-cancel")
        await pilot.pause()
    assert app.result is None
