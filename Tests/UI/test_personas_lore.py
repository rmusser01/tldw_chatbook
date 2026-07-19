"""P2a: PersonasLoreDetailWidget — Entries + Settings tabs, I/O-free.
Also: PersonasLoreTryItWidget — injection preview + diagnostics story, I/O-free.
Also (Task 6): PersonasScreen wiring — mounted integration against a REAL
CharactersRAGDB seeded through WorldBookManager (mirrors
test_personas_dictionaries.py's PersonasTestApp harness)."""


import pytest
from textual.app import App, ComposeResult
from textual.coordinate import Coordinate
from textual.widgets import Button, DataTable, Input, ListView, Static, Switch, TextArea

from tldw_chatbook.Widgets.Persona_Widgets.personas_lore_detail import (
    PersonasLoreDetailWidget,
    LoreBookEnableToggled,
    LoreBookExportRequested,
    LoreBookSettingsSaveRequested,
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


@pytest.mark.asyncio
async def test_entry_priority_round_trips_through_form():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        app.query_one("#personas-lore-entry-keys", Input).value = "Warden"
        app.query_one("#personas-lore-entry-content", TextArea).text = "grim jailer"
        app.query_one("#personas-lore-entry-priority", Input).value = "80"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        assert app.posted[-1]["priority"] == 80


@pytest.mark.asyncio
async def test_matching_controls_round_trip_through_form():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        app.query_one("#personas-lore-entry-keys", Input).value = "Warden"
        app.query_one("#personas-lore-entry-content", TextArea).text = "grim jailer"
        app.query_one("#personas-lore-entry-case-sensitive", Switch).value = True
        app.query_one("#personas-lore-entry-selective", Switch).value = True
        app.query_one("#personas-lore-entry-secondary-keys", Input).value = " sword , shield ,"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        payload = app.posted[-1]
        assert payload["case_sensitive"] is True
        assert payload["selective"] is True
        assert payload["secondary_keys"] == ["sword", "shield"]  # trimmed, blank dropped


@pytest.mark.asyncio
async def test_blank_secondary_keys_is_empty_list():
    """A blank secondary-keys field yields [] (never raises)."""
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        app.query_one("#personas-lore-entry-keys", Input).value = "Warden"
        app.query_one("#personas-lore-entry-content", TextArea).text = "grim jailer"
        # secondary-keys left blank
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        assert app.posted[-1]["secondary_keys"] == []


@pytest.mark.asyncio
async def test_secondary_keys_stored_even_when_not_selective():
    """Data fidelity: secondary keys are stored regardless of the Selective
    switch, so toggling Selective off does not erase them."""
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        app.query_one("#personas-lore-entry-keys", Input).value = "Warden"
        app.query_one("#personas-lore-entry-content", TextArea).text = "grim jailer"
        app.query_one("#personas-lore-entry-selective", Switch).value = False
        app.query_one("#personas-lore-entry-secondary-keys", Input).value = "sword"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        payload = app.posted[-1]
        assert payload["selective"] is False
        assert payload["secondary_keys"] == ["sword"]


@pytest.mark.asyncio
async def test_fill_form_populates_matching_controls():
    """Selecting a row fills the three controls; an entry with selective=False
    but stored secondary keys keeps its keys (fidelity) and shows them disabled."""
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        widget.update_entries([
            {"id": 7, "keys": ["Warden"], "content": "grim jailer",
             "position": "before_char", "enabled": True, "insertion_order": 0,
             "case_sensitive": True, "selective": False,
             "secondary_keys": ["alpha", "beta"]},
        ])
        await pilot.pause()
        table = app.query_one("#personas-lore-entries-table", DataTable)
        table.move_cursor(row=0)
        await pilot.pause()
        assert app.query_one("#personas-lore-entry-case-sensitive", Switch).value is True
        assert app.query_one("#personas-lore-entry-selective", Switch).value is False
        sec = app.query_one("#personas-lore-entry-secondary-keys", Input)
        assert sec.value == "alpha, beta"   # preserved even though selective is False
        assert sec.disabled is True          # selective off → disabled hint


@pytest.mark.asyncio
async def test_secondary_keys_disabled_hint_tracks_selective():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        await pilot.pause()
        sec = app.query_one("#personas-lore-entry-secondary-keys", Input)
        sel = app.query_one("#personas-lore-entry-selective", Switch)
        assert sec.disabled is True          # selective defaults off → disabled on mount
        sec.value = "kept"
        sel.value = True
        await pilot.pause()
        assert sec.disabled is False         # selective on → enabled
        assert sec.value == "kept"
        sel.value = False
        await pilot.pause()
        assert sec.disabled is True          # selective off → disabled again
        assert sec.value == "kept"           # value survives the toggle (fidelity)


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
from tldw_chatbook.Character_Chat.world_info_processor import WorldInfoProcessor
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


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

    async def test_duplicate_button_visible_and_duplicates_selected_lore_book(
        self, mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
    ):
        """The library-pane Duplicate button is reachable in Lore mode (gated to
        dictionaries+lore) and clicking it duplicates the selected book (entries
        copied) through the real WorldBookManager export+import seam."""
        from textual.widgets import Button

        mock_app_instance.chachanotes_db = lore_db
        app = LorePersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_lore(pilot)
            await _select_first_lore(pilot, screen)
            # Reachable from the toolbar in Lore mode (not gated to dictionaries).
            assert screen.query_one("#personas-library-duplicate", Button).display is True
            await pilot.click("#personas-library-duplicate")
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

    async def test_enable_toggle_then_settings_save_no_stale_version_conflict(
        self, mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
    ):
        """Toggling Enabled bumps world_books.version; a subsequent Settings save
        in the SAME selection must NOT hit a stale-expected_version ConflictError
        (regression for the whole-branch-review Important finding)."""
        mock_app_instance.chachanotes_db = lore_db
        app = LorePersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_lore(pilot)
            await _select_first_lore(pilot, screen)
            # Toggle enabled off — persists and bumps the book version to 2.
            screen.post_message(LoreBookEnableToggled(enabled=False))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # Now edit + save settings in the same selection.
            screen.post_message(LoreBookSettingsSaveRequested({
                "name": "Blackreach Renamed", "description": "",
                "scan_depth": 5, "token_budget": 750,
                "recursive_scanning": True, "enabled": False,
            }))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            status = str(screen.query_one("#personas-lore-status", Static).renderable)
            assert "changed since it was loaded" not in status
            manager = WorldBookManager(lore_db)
            book = manager.get_world_book(seeded_lore_book["book_id"])
            assert book["name"] == "Blackreach Renamed"
            assert book["enabled"] is False and book["scan_depth"] == 5


@pytest.mark.asyncio
async def test_new_entry_appends_after_max_insertion_order():
    """Adding a new entry appends AFTER the current max insertion_order (not at
    len()), so non-contiguous/imported books keep injection order (Qodo #673)."""
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        widget.update_entries([
            {"id": 1, "keys": ["a"], "content": "x", "position": "before_char",
             "enabled": True, "insertion_order": 0},
            {"id": 2, "keys": ["b"], "content": "y", "position": "before_char",
             "enabled": True, "insertion_order": 10},
        ])
        await pilot.pause()
        app.query_one("#personas-lore-entry-keys", Input).value = "c"
        app.query_one("#personas-lore-entry-content", TextArea).text = "z"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        assert app.posted, "add must post LoreEntryAddRequested"
        # max(0, 10) + 1 == 11 — NOT len()==2 nor the selected row's order.
        assert app.posted[-1]["insertion_order"] == 11


@pytest.mark.asyncio
async def test_add_entry_persists_priority_through_real_screen_handler(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
):
    """The real _handle_lore_entry_add must forward priority to the DB (not just
    the widget post) — regression for the P2c whole-branch-review Important."""
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        screen.query_one("#personas-lore-entry-keys", Input).value = "Ghost"
        screen.query_one("#personas-lore-entry-content", TextArea).text = "a pale spirit"
        screen.query_one("#personas-lore-entry-priority", Input).value = "80"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        entries = WorldBookManager(lore_db).get_world_book_entries(seeded_lore_book["book_id"])
        ghost = next(e for e in entries if e["keys"] == ["Ghost"])
        assert ghost["priority"] == 80


@pytest.mark.asyncio
async def test_add_entry_persists_matching_fields_through_real_screen_handler(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
):
    """The real _handle_lore_entry_add must forward selective/secondary_keys/
    case_sensitive to the DB — regression against the explicit-kwarg drop that
    bit `priority` in P2c."""
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        screen.query_one("#personas-lore-entry-keys", Input).value = "Ghost"
        screen.query_one("#personas-lore-entry-content", TextArea).text = "a pale spirit"
        screen.query_one("#personas-lore-entry-case-sensitive", Switch).value = True
        screen.query_one("#personas-lore-entry-selective", Switch).value = True
        screen.query_one("#personas-lore-entry-secondary-keys", Input).value = "sword"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        entries = WorldBookManager(lore_db).get_world_book_entries(seeded_lore_book["book_id"])
        ghost = next(e for e in entries if e["keys"] == ["Ghost"])
        assert ghost["case_sensitive"] is True
        assert ghost["selective"] is True
        assert ghost["secondary_keys"] == ["sword"]


@pytest.mark.asyncio
async def test_update_entry_persists_matching_fields_through_real_screen_handler(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
):
    """_handle_lore_entry_update forwards the three fields via **payload."""
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        table = screen.query_one("#personas-lore-entries-table", DataTable)
        table.move_cursor(row=0)          # select the seeded "Warden" entry → fills form
        await pilot.pause()
        screen.query_one("#personas-lore-entry-case-sensitive", Switch).value = True
        screen.query_one("#personas-lore-entry-selective", Switch).value = True
        screen.query_one("#personas-lore-entry-secondary-keys", Input).value = "prison"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-update")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        entries = WorldBookManager(lore_db).get_world_book_entries(seeded_lore_book["book_id"])
        warden = next(e for e in entries if e["keys"] == ["Warden"])
        assert warden["case_sensitive"] is True
        assert warden["selective"] is True
        assert warden["secondary_keys"] == ["prison"]


@pytest.mark.asyncio
async def test_selective_entry_created_via_editor_gates_matching(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
):
    """A selective entry created through the real editor/add-handler path gates
    matching in WorldInfoProcessor: it fires only when a secondary key is present
    in the scan text. Proves editor config → DB → matcher end to end."""
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        screen.query_one("#personas-lore-entry-keys", Input).value = "hero"
        screen.query_one("#personas-lore-entry-content", TextArea).text = "the brave hero"
        screen.query_one("#personas-lore-entry-selective", Switch).value = True
        screen.query_one("#personas-lore-entry-secondary-keys", Input).value = "sword"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    manager = WorldBookManager(lore_db)
    book = manager.get_world_book(seeded_lore_book["book_id"])
    entries = manager.get_world_book_entries(seeded_lore_book["book_id"])
    world_book = {**book, "entries": entries}
    proc = WorldInfoProcessor(world_books=[world_book])

    # primary key "hero" present but secondary "sword" absent → selective entry does NOT fire
    r1 = proc.process_messages("the hero walks alone", [])
    assert all("brave hero" not in c for c in r1["injections"]["before_char"])
    # primary + secondary present → fires
    r2 = proc.process_messages("the hero draws a sword", [])
    assert any("brave hero" in c for c in r2["injections"]["before_char"])


@pytest.mark.asyncio
async def test_export_selected_lore_book_writes_json_file(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book, tmp_path, monkeypatch
):
    """Exporting the selected lore book writes a JSON file that parses back to the
    book's export payload (name + entries)."""
    import json as _json
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        target = tmp_path / "exported.json"

        async def _fake_save(picker):
            return target

        monkeypatch.setattr(app, "push_screen_wait", _fake_save)
        screen.post_message(LoreBookExportRequested())
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert target.exists()
        payload = _json.loads(target.read_text("utf-8"))
        assert payload["name"] == "Blackreach"
        assert any(e["keys"] == ["Warden"] for e in payload["entries"])


@pytest.mark.asyncio
async def test_import_world_book_from_file_creates_book_and_entries(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book, tmp_path
):
    """_import_world_book_from_path imports a tldw-shaped file, preserving priority
    and matching fields."""
    import json as _json
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    payload = {"name": "Imported Realm", "description": "", "scan_depth": 3,
               "token_budget": 500, "recursive_scanning": False,
               "entries": [{"keys": ["Sword"], "content": "a blade", "priority": 55,
                            "selective": True, "secondary_keys": ["hilt"],
                            "case_sensitive": True, "insertion_order": 0,
                            "position": "before_char", "enabled": True}]}
    f = tmp_path / "realm.json"
    f.write_text(_json.dumps(payload), "utf-8")
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        await screen._import_world_book_from_path(str(f))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        manager = WorldBookManager(lore_db)
        books = manager.list_world_books(True)
        realm = next(b for b in books if b["name"] == "Imported Realm")
        entries = manager.get_world_book_entries(realm["id"])
        assert len(entries) == 1
        e = entries[0]
        assert e["keys"] == ["Sword"] and e["priority"] == 55
        assert e["selective"] is True and e["secondary_keys"] == ["hilt"] and e["case_sensitive"] is True


@pytest.mark.asyncio
async def test_import_sillytavern_world_info_object_form(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book, tmp_path
):
    """A SillyTavern World Info object-form file imports with fields remapped."""
    import json as _json
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    payload = {"name": "ST Book",
               "entries": {"0": {"key": ["Dragon"], "keysecondary": [], "content": "a wyrm",
                                 "order": 3, "position": 0, "disable": False}}}
    f = tmp_path / "st.json"
    f.write_text(_json.dumps(payload), "utf-8")
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        await screen._import_world_book_from_path(str(f))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        manager = WorldBookManager(lore_db)
        book = next(b for b in manager.list_world_books(True) if b["name"] == "ST Book")
        e = manager.get_world_book_entries(book["id"])[0]
        assert e["keys"] == ["Dragon"] and e["content"] == "a wyrm" and e["insertion_order"] == 3


@pytest.mark.asyncio
async def test_import_malformed_world_book_creates_no_book(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book, tmp_path
):
    """A file whose entry has no keys is rejected up front — no partial book."""
    import json as _json
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    payload = {"name": "Bad Book", "entries": [{"keys": [], "content": "x"}]}
    f = tmp_path / "bad.json"
    f.write_text(_json.dumps(payload), "utf-8")
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        await screen._import_world_book_from_path(str(f))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        manager = WorldBookManager(lore_db)
        assert all(b["name"] != "Bad Book" for b in manager.list_world_books(True))


@pytest.mark.asyncio
async def test_import_name_collision_renames(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book, tmp_path
):
    """Importing a book whose name clashes with an existing one imports under a
    unique name."""
    import json as _json
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    payload = {"name": "Blackreach",  # same as the seeded book
               "entries": [{"keys": ["Echo"], "content": "a sound"}]}
    f = tmp_path / "dup.json"
    f.write_text(_json.dumps(payload), "utf-8")
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        await screen._import_world_book_from_path(str(f))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        manager = WorldBookManager(lore_db)
        names = [b["name"] for b in manager.list_world_books(True)]
        assert "Blackreach" in names
        assert any(n != "Blackreach" and n.startswith("Blackreach") for n in names)


@pytest.mark.asyncio
async def test_library_import_button_visible_in_lore_mode(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
):
    """The library Import button is displayed in lore mode (un-gated)."""
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        assert screen.query_one("#personas-library-import", Button).display is True


@pytest.mark.asyncio
async def test_export_then_import_round_trip_preserves_entries(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book, tmp_path, monkeypatch
):
    """Export the seeded book to a file, then import that file — the new book's
    entry matches the original (keys/content preserved through export→import)."""
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    manager = WorldBookManager(lore_db)
    # Give the seeded entry non-default matching fields to prove they survive.
    entry = manager.get_world_book_entries(seeded_lore_book["book_id"])[0]
    manager.update_world_book_entry(entry["id"], priority=70, selective=True,
                                    secondary_keys=["oath"], case_sensitive=True)
    target = tmp_path / "roundtrip.json"
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)

        async def _fake_save(picker):
            return target

        monkeypatch.setattr(app, "push_screen_wait", _fake_save)
        screen.post_message(LoreBookExportRequested())
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert target.exists()
        await screen._import_world_book_from_path(str(target))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
    books = manager.list_world_books(True)
    imported = next(b for b in books if b["name"] != "Blackreach" and b["name"].startswith("Blackreach"))
    e = manager.get_world_book_entries(imported["id"])[0]
    assert e["keys"] == ["Warden"] and e["priority"] == 70
    assert e["selective"] is True and e["secondary_keys"] == ["oath"] and e["case_sensitive"] is True
