"""P2a: PersonasLoreDetailWidget — Entries + Settings tabs, I/O-free."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable

from tldw_chatbook.Widgets.Persona_Widgets.personas_lore_detail import (
    PersonasLoreDetailWidget,
    LoreEntryAddRequested,
)


class _DetailHost(App):
    def __init__(self):
        super().__init__()
        self.posted = []

    def compose(self) -> ComposeResult:
        yield PersonasLoreDetailWidget(id="personas-lore-detail")

    def on_lore_entry_add_requested(self, message: LoreEntryAddRequested) -> None:
        self.posted.append(message.payload)


@pytest.mark.asyncio
async def test_detail_loads_book_and_lists_entries():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "Blackreach", "description": "",
                          "scan_depth": 3, "token_budget": 500,
                          "recursive_scanning": False, "enabled": True})
        widget.update_entries([
            {"id": 7, "keys": ["Warden"], "content": "grim jailer",
             "position": "before_char", "enabled": True, "insertion_order": 0},
        ])
        await pilot.pause()
        table = app.query_one("#personas-lore-entries-table", DataTable)
        assert table.row_count == 1
        assert widget.settings_payload()["name"] == "Blackreach"


@pytest.mark.asyncio
async def test_add_entry_posts_payload_from_form():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        # fill the form
        app.query_one("#personas-lore-entry-keys").value = "Warden, Jailer"
        app.query_one("#personas-lore-entry-content").text = "grim jailer"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        assert app.posted, "add must post LoreEntryAddRequested"
        payload = app.posted[-1]
        assert payload["keys"] == ["Warden", "Jailer"] and payload["content"] == "grim jailer"


@pytest.mark.asyncio
async def test_reorder_posts_full_id_list():
    app = _DetailHost()
    posted = []
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        widget.update_entries([
            {"id": 1, "keys": ["a"], "content": "x", "position": "before_char",
             "enabled": True, "insertion_order": 0},
            {"id": 2, "keys": ["b"], "content": "y", "position": "before_char",
             "enabled": True, "insertion_order": 1},
        ])
        await pilot.pause()
        assert widget.entry_ids_in_order() == ["1", "2"]
