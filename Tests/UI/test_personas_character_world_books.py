import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_character_world_books import (
    PersonasCharacterWorldBooksWidget,
    CharacterWorldBookAttachRequested,
    CharacterWorldBookDetachRequested,
)


class _Host(App):
    def __init__(self):
        super().__init__()
        self.attach_posts = []
        self.detach_posts = []

    def compose(self) -> ComposeResult:
        yield PersonasCharacterWorldBooksWidget()

    def on_character_world_book_attach_requested(self, message) -> None:
        self.attach_posts.append(message)

    def on_character_world_book_detach_requested(self, message) -> None:
        self.detach_posts.append(message.name)


@pytest.mark.asyncio
async def test_empty_then_render():
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        w = app.query_one(PersonasCharacterWorldBooksWidget)
        w.load_world_books([])
        await pilot.pause()
        assert app.query_one("#personas-char-worldbooks-empty", Static).display is True
        assert app.query_one("#personas-char-worldbooks-table", DataTable).row_count == 0
        w.load_world_books([{"name": "Lore", "entry_count": 3, "enabled": True}])
        await pilot.pause()
        assert app.query_one("#personas-char-worldbooks-empty", Static).display is False
        assert app.query_one("#personas-char-worldbooks-table", DataTable).row_count == 1


@pytest.mark.asyncio
async def test_duplicate_names_do_not_crash():
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        w = app.query_one(PersonasCharacterWorldBooksWidget)
        dup = {"name": "Dup", "entry_count": 1, "enabled": True}
        w.load_world_books([dup, dup])  # would DuplicateKey without the guard
        await pilot.pause()
        assert app.query_one("#personas-char-worldbooks-table", DataTable).row_count == 1


@pytest.mark.asyncio
async def test_attach_button_posts():
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.click("#personas-char-worldbooks-add")
        await pilot.pause()
        assert len(app.attach_posts) == 1


@pytest.mark.asyncio
async def test_detach_posts_selected_name():
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        w = app.query_one(PersonasCharacterWorldBooksWidget)
        w.load_world_books([{"name": "Lore", "entry_count": 1, "enabled": True}])
        await pilot.pause()
        app.query_one("#personas-char-worldbooks-table", DataTable).move_cursor(row=0)
        await pilot.pause()
        await pilot.click("#personas-char-worldbooks-detach")
        await pilot.pause()
        assert app.detach_posts == ["Lore"]
