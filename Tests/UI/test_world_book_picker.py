import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, ListView

from tldw_chatbook.Widgets.Persona_Widgets.world_book_picker import WorldBookPicker


class _Host(App):
    def __init__(self, books):
        super().__init__()
        self._books = books
        self.result = "unset"

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.run_worker(self._drive)

    async def _drive(self) -> None:
        self.result = await self.push_screen_wait(WorldBookPicker(self._books))


@pytest.mark.asyncio
async def test_pick_returns_int_id():
    books = [{"world_book_id": 10, "name": "Alpha"}, {"world_book_id": 20, "name": "Beta"}]
    app = _Host(books)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen.query_one("#worldbook-pick-list", ListView).index = 1
        await pilot.pause()
        await pilot.click("#worldbook-pick-confirm")
        await pilot.pause()
    assert app.result == 20


@pytest.mark.asyncio
async def test_filter_then_select():
    books = [{"world_book_id": 10, "name": "Alpha"}, {"world_book_id": 20, "name": "Beta"}]
    app = _Host(books)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen.query_one("#worldbook-pick-search", Input).value = "beta"
        await pilot.pause()
        app.screen.query_one("#worldbook-pick-list", ListView).index = 0
        await pilot.pause()
        await pilot.click("#worldbook-pick-confirm")
        await pilot.pause()
    assert app.result == 20


@pytest.mark.asyncio
async def test_cancel_returns_none():
    app = _Host([{"world_book_id": 10, "name": "Alpha"}])
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#worldbook-pick-cancel")
        await pilot.pause()
    assert app.result is None
