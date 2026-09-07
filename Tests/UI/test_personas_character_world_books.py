import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, DataTable, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_character_world_books import (
    PersonasCharacterWorldBooksWidget,
)


class _Host(ConsolidatedCSSApp):
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
        # The section starts collapsed (task-2231): expand to reach Attach.
        await pilot.click("#personas-char-worldbooks-toggle")
        await pilot.pause()
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
        # The section starts collapsed (task-2231): expand to reach Detach.
        await pilot.click("#personas-char-worldbooks-toggle")
        await pilot.pause()
        app.query_one("#personas-char-worldbooks-table", DataTable).move_cursor(row=0)
        await pilot.pause()
        await pilot.click("#personas-char-worldbooks-detach")
        await pilot.pause()
        assert app.detach_posts == ["Lore"]


# ===================================================================
# task-2231: the panel is a collapsible section - one line when empty,
# count in the header, and a collapse state that data reloads never reset.
# ===================================================================


@pytest.mark.asyncio
async def test_collapsed_by_default_renders_one_line():
    """AC#4: an empty section costs exactly one line, not 16."""
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        w = app.query_one(PersonasCharacterWorldBooksWidget)
        await pilot.pause()
        assert w.query_one("#personas-char-worldbooks-body").display is False
        assert (
            str(w.query_one("#personas-char-worldbooks-toggle", Button).label)
            == "▸ World Books (0)"
        )
        assert w.size.height == 1


@pytest.mark.asyncio
async def test_toggle_expands_and_collapses_in_place():
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        w = app.query_one(PersonasCharacterWorldBooksWidget)
        await pilot.pause()
        toggle = w.query_one("#personas-char-worldbooks-toggle", Button)

        await pilot.click("#personas-char-worldbooks-toggle")
        await pilot.pause()
        assert w.query_one("#personas-char-worldbooks-body").display is True
        assert str(toggle.label) == "▾ World Books (0)"

        # Button presses are debounced for the -active animation window
        # (Textual ignores a re-click while it lasts); wait it out like a
        # real user's second click would.
        await pilot.pause(0.4)
        await pilot.click("#personas-char-worldbooks-toggle")
        await pilot.pause()
        assert w.query_one("#personas-char-worldbooks-body").display is False
        assert str(toggle.label) == "▸ World Books (0)"


@pytest.mark.asyncio
async def test_header_count_tracks_rows_without_resetting_collapse_state():
    """AC#3: attach/detach refreshes update the count but keep the user's
    expand/collapse choice (session persistence)."""
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        w = app.query_one(PersonasCharacterWorldBooksWidget)
        await pilot.pause()
        toggle = w.query_one("#personas-char-worldbooks-toggle", Button)

        w.load_world_books([{"name": "Lore", "entry_count": 3, "enabled": True}])
        await pilot.pause()
        assert str(toggle.label) == "▸ World Books (1)"
        assert w.query_one("#personas-char-worldbooks-body").display is False

        await pilot.click("#personas-char-worldbooks-toggle")
        await pilot.pause()
        w.load_world_books([])
        await pilot.pause()
        assert str(toggle.label) == "▾ World Books (0)"
        assert w.query_one("#personas-char-worldbooks-body").display is True
