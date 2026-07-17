"""P2a: PersonasLoreDetailWidget — Entries + Settings tabs, I/O-free.
Also: PersonasLoreTryItWidget — injection preview + diagnostics story, I/O-free.
Also (Task 6): PersonasScreen wiring — mounted integration against a REAL
CharactersRAGDB seeded through WorldBookManager (mirrors
test_personas_dictionaries.py's PersonasTestApp harness)."""

import pytest
from textual.app import App, ComposeResult
from textual.coordinate import Coordinate
from textual.widgets import Button, DataTable, Input, ListView, Static, TextArea

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


# ===================================================================
# Task 6: PersonasScreen wiring — mounted integration, REAL CharactersRAGDB
# ===================================================================
#
# Unlike test_personas_dictionaries.py (which fakes the dictionaries scope
# service), Lore has no scope-service layer yet: the screen talks to
# WorldBookManager directly. A fake manager could not prove the screen wires
# create/select/entry-CRUD/Try-it against the real DB, so these tests seed a
# real CharactersRAGDB (tmp_path) through WorldBookManager and mount the real
# PersonasScreen against it.

from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import PersonaActionRequested


@pytest.fixture
def lore_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "personas_lore.db", "test-client")
    yield db
    db.close_connection()


@pytest.fixture
def seeded_lore_book(lore_db):
    """One world book + one keyword entry, seeded via the real WorldBookManager."""
    manager = WorldBookManager(lore_db)
    book_id = manager.create_world_book(
        "Blackreach", description="Dark elf city beneath the world."
    )
    entry_id = manager.create_world_book_entry(
        book_id,
        keys=["Warden"],
        content="The Warden is the grim jailer of Blackreach.",
    )
    return {"book_id": book_id, "entry_id": entry_id}


@pytest.fixture
def stub_characters_lore(monkeypatch):
    """Empty character library so Characters-mode code paths don't explode on mount."""
    import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module

    monkeypatch.setattr(character_handler_module, "fetch_all_characters", lambda: [])
    monkeypatch.setattr(character_handler_module, "fetch_character_by_id", lambda character_id: None)


class LorePersonasTestApp(App):
    """Delegating App harness — mirrors PersonasTestApp in test_personas_dictionaries.py."""

    def __init__(self, mock_app_instance):
        super().__init__()
        self._mock = mock_app_instance
        self.character_persona_scope_service = mock_app_instance.character_persona_scope_service

    _NON_DELEGATED_PREFIXES = ("_", "watch_", "compute_", "validate_", "action_", "key_", "on_")

    def __getattr__(self, name):
        if name.startswith(self._NON_DELEGATED_PREFIXES):
            raise AttributeError(name)
        return getattr(self.__dict__["_mock"], name)

    def compose(self) -> ComposeResult:
        yield AppFooterStatus(id="app-footer-status")

    def on_mount(self) -> None:
        self.push_screen(PersonasScreen(self))


async def _mounted_lore(pilot):
    await pilot.pause()
    return pilot.app.screen


async def _enter_lore(pilot):
    screen = await _mounted_lore(pilot)
    await pilot.click("#personas-mode-lore")
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return screen


async def _select_first_lore(pilot, screen):
    rows = screen.query_one("#personas-library-rows", ListView)
    rows.index = 0
    rows.action_select_cursor()
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


@pytest.mark.asyncio
class TestLoreModeIntegration:
    """Load-bearing wiring assertions (Task 6, brief Step 1)."""

    async def test_entering_lore_mode_shows_detail_and_library_row(
        self, mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
    ):
        """Entering Lore mode shows the real detail widget (not the coming-soon
        placeholder), and the seeded book appears in the library rail."""
        mock_app_instance.chachanotes_db = lore_db
        app = LorePersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_lore(pilot)
            assert screen.query_one("#personas-mode-placeholder", Static).display is False
            # The detail widget is mounted but stays hidden until a book is selected.
            assert screen.query_one(PersonasLoreDetailWidget).display is False
            rows = screen.query_one("#personas-library-rows", ListView).children
            texts = [str(s.renderable) for r in rows for s in r.query(Static).results()]
            assert "Blackreach" in texts

    async def test_selecting_book_loads_entries_into_detail(
        self, mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
    ):
        """Selecting the book loads its settings + entries into the detail widget."""
        mock_app_instance.chachanotes_db = lore_db
        app = LorePersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_lore(pilot)
            await _select_first_lore(pilot, screen)
            detail = screen.query_one(PersonasLoreDetailWidget)
            assert detail.display is True
            table = screen.query_one("#personas-lore-entries-table", DataTable)
            assert table.row_count == 1
            assert detail.settings_payload()["name"] == "Blackreach"

    async def test_tryit_run_renders_fired_row(
        self, mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
    ):
        """Running Try-it with sample text containing the entry's key fires it -
        the screen builds a WorldInfoProcessor and calls
        process_messages_with_diagnostics against the real (in-memory) entries."""
        mock_app_instance.chachanotes_db = lore_db
        app = LorePersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_lore(pilot)
            await _select_first_lore(pilot, screen)
            tryit = screen.query_one(PersonasLoreTryItWidget)
            assert tryit.display is True
            run = screen.query_one("#personas-lore-tryit-run", Button)
            assert run.disabled is False
            screen.query_one("#personas-lore-tryit-sample", TextArea).text = "Tell me about the Warden."
            await pilot.click("#personas-lore-tryit-run")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            summary = screen.query_one("#personas-lore-tryit-summary", Static)
            assert "1 fired" in str(summary.renderable)
            fired = screen.query_one("#personas-lore-tryit-fired", Static)
            assert "Warden" in str(fired.renderable)


@pytest.mark.asyncio
class TestLoreModeCrudRoundTrip:
    """Wiring for book create/duplicate/delete and entry CRUD, round-tripped
    through the real DB (WorldBookManager is synchronous — asyncio.to_thread)."""

    async def test_create_new_lore_book_persists_and_lands_in_settings(
        self, mock_app_instance, stub_characters_lore, lore_db
    ):
        mock_app_instance.chachanotes_db = lore_db
        app = LorePersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_lore(pilot)
            await pilot.click("#personas-library-new")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            manager = WorldBookManager(lore_db)
            books = manager.list_world_books(True)
            assert any(b["name"] == "Untitled world book" for b in books)
            # Settings tab is focused for an immediate rename.
            from textual.widgets import TabbedContent

            assert (
                screen.query_one("#personas-lore-tabs", TabbedContent).active
                == "personas-lore-tab-settings"
            )

    async def test_duplicate_selected_lore_book_copies_entries(
        self, mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
    ):
        """Duplicate is wired via the generic PersonaActionRequested seam (the
        library-pane Duplicate button is gated to Dictionaries mode only; the
        library-rail gate for Lore is a follow-up UI-polish item outside this
        task's file scope), so this drives the handler by posting the message
        directly - proving the screen-side plumbing that button will call."""
        mock_app_instance.chachanotes_db = lore_db
        app = LorePersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_lore(pilot)
            await _select_first_lore(pilot, screen)
            screen.post_message(PersonaActionRequested(action="duplicate"))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            manager = WorldBookManager(lore_db)
            books = manager.list_world_books(True)
            assert len(books) == 2
            copy = next(b for b in books if b["name"] != "Blackreach")
            assert copy["name"] == "Blackreach (copy)"
            entries = manager.get_world_book_entries(copy["id"])
            assert len(entries) == 1 and entries[0]["keys"] == ["Warden"]

    async def test_add_entry_via_form_persists_to_real_db(
        self, mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
    ):
        mock_app_instance.chachanotes_db = lore_db
        app = LorePersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_lore(pilot)
            await _select_first_lore(pilot, screen)
            screen.query_one("#personas-lore-entry-keys", Input).value = "Ghost"
            screen.query_one("#personas-lore-entry-content", TextArea).text = "A pale spirit."
            await pilot.pause()
            await pilot.click("#personas-lore-entry-add")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            table = screen.query_one("#personas-lore-entries-table", DataTable)
            assert table.row_count == 2
            manager = WorldBookManager(lore_db)
            entries = manager.get_world_book_entries(seeded_lore_book["book_id"])
            assert len(entries) == 2

    async def test_delete_book_confirms_and_removes_row(
        self, mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book, monkeypatch
    ):
        mock_app_instance.chachanotes_db = lore_db
        app = LorePersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_lore(pilot)
            await _select_first_lore(pilot, screen)

            async def _fake_confirm(name):
                return True

            monkeypatch.setattr(screen, "_confirm_delete", _fake_confirm)
            await pilot.click("#personas-delete")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            manager = WorldBookManager(lore_db)
            assert manager.list_world_books(True) == []
            assert screen.query_one(PersonasLoreDetailWidget).display is False
            assert screen.state.selected_entity_id is None
