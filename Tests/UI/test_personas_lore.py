"""P2a: PersonasLoreDetailWidget — Entries + Settings tabs, I/O-free.
Also: PersonasLoreTryItWidget — injection preview + diagnostics story, I/O-free."""

import pytest
from textual.app import App, ComposeResult
from textual.coordinate import Coordinate
from textual.widgets import DataTable, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_lore_detail import (
    PersonasLoreDetailWidget,
    LoreEntryAddRequested,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_lore_tryit import PersonasLoreTryItWidget


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


@pytest.mark.asyncio
async def test_bracket_text_in_entries_not_backslash_escaped():
    """Keys/content containing bracket-tag-like text (e.g. "[note]") must render
    raw in the DataTable — cells are rich Text objects (not markup), so escaping
    ahead of Text() would inject a literal backslash. Regression for the P2a
    Task-4 review finding."""
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        widget.update_entries([
            {"id": 7, "keys": ["[note]", "warden"], "content": "He says [aside] quietly.",
             "position": "before_char", "enabled": True, "insertion_order": 0},
        ])
        await pilot.pause()
        table = app.query_one("#personas-lore-entries-table", DataTable)
        keys_cell = table.get_cell_at(Coordinate(0, 0))
        content_cell = table.get_cell_at(Coordinate(0, 1))
        assert "\\" not in keys_cell.plain and "[note]" in keys_cell.plain
        assert "\\" not in content_cell.plain and "[aside]" in content_cell.plain


class _TryItHost(App):
    def compose(self) -> ComposeResult:
        yield PersonasLoreTryItWidget(id="personas-lore-tryit")


@pytest.mark.asyncio
async def test_tryit_renders_injections_by_position_and_diagnostics():
    app = _TryItHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreTryItWidget)
        injections = {"before_char": ["grim jailer"], "after_char": [],
                      "at_start": [], "at_end": []}
        diagnostics = {
            "entries": [
                {"entry_id": 1, "keys": ["Warden"], "activation_reason": "matched key 'Warden'",
                 "status": "fired", "token_cost": 3, "injection_order": 0,
                 "position": "before_char", "content_preview": "grim jailer", "depth_level": 0},
                {"entry_id": 2, "keys": ["Ghost"], "activation_reason": "disabled (key 'Ghost' matched)",
                 "status": "skipped:disabled", "token_cost": 0, "injection_order": None,
                 "position": "before_char", "content_preview": "pale", "depth_level": 0},
            ],
            "matched": 2, "fired": 1, "skipped": 1, "tokens_used": 3,
            "token_budget": 500, "budget_exceeded": False, "books_scanned": 1,
        }
        widget.render_result(injections, diagnostics)
        await pilot.pause()
        summary = app.query_one("#personas-lore-tryit-summary", Static)
        assert "1 fired" in str(summary.renderable)
        # injection preview shows the fired content under before_char
        preview = app.query_one("#personas-lore-tryit-injections", Static)
        assert "grim jailer" in str(preview.renderable)


@pytest.mark.asyncio
async def test_tryit_degrades_on_bad_diagnostics():
    app = _TryItHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreTryItWidget)
        widget.render_result({"before_char": [], "after_char": [], "at_start": [], "at_end": []}, None)
        await pilot.pause()  # must not raise


@pytest.mark.asyncio
async def test_tryit_bracket_tag_content_not_backslash_escaped():
    """Fired-entry keys/content_preview containing bracket-tag-like text
    (e.g. "[aside]") must render raw in the fired-list Static — the sink is a
    Static rendering a rich.text.Text object (not markup), so escaping ahead
    of Text() would inject a literal backslash. Regression for the P2a
    Task-4 review finding, mirrored here for the Try-it diagnostics story."""
    app = _TryItHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreTryItWidget)
        injections = {"before_char": [], "after_char": [], "at_start": [], "at_end": []}
        diagnostics = {
            "entries": [
                {"entry_id": 1, "keys": ["[note]", "Warden"], "activation_reason": "matched key 'Warden'",
                 "status": "fired", "token_cost": 3, "injection_order": 0,
                 "position": "before_char", "content_preview": "He says [aside] quietly.",
                 "depth_level": 0},
            ],
            "matched": 1, "fired": 1, "skipped": 0, "tokens_used": 3,
            "token_budget": 500, "budget_exceeded": False, "books_scanned": 1,
        }
        widget.render_result(injections, diagnostics)
        await pilot.pause()  # must not raise / not corrupt render
        fired = app.query_one("#personas-lore-tryit-fired", Static)
        plain = str(fired.renderable)
        assert "[aside]" in plain and "[note]" in plain
        assert "\\" not in plain
