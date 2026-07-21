"""Roleplay P2f Task 6: PersonasScreen character world-books panel — mounted
integration against a REAL CharactersRAGDB.

Mirrors two existing harnesses rather than inventing a third:

- ``test_personas_character_attach.py`` for character-selection plumbing
  (``PersonasTestApp``, a ``stub_characters``-style monkeypatch of
  ``ccp_character_handler.fetch_all_characters`` /
  ``fetch_character_by_id``, and clicking a
  ``#personas-library-row-character-<id>`` row).
- ``test_personas_lore.py``'s Task 6 section for the real-``chachanotes_db``
  pattern (``mock_app_instance.chachanotes_db = <real CharactersRAGDB>``),
  since ``WorldBookManager.get_world_books_for_character`` (the panel's real
  data source, consumed via ``PersonasScreen._lore_manager()``) reads
  straight from the DB with no scope-service layer to fake.

The stubbed character list is reconciled with the real DB by seeding the
character through ``db.add_character_card`` first and then feeding that
SAME id back into the ``fetch_all_characters`` / ``fetch_character_by_id``
stubs - the character the UI selects is the one
``WorldBookManager.get_world_books_for_character`` reads via
``db.get_character_card_by_id``.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_card_widget import (
    PersonasCharacterCardWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_dictionaries import (
    PersonasCharacterDictionariesWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_world_books import (
    PersonasCharacterWorldBooksWidget,
)

from Tests.UI.test_personas_dictionaries import PersonasTestApp

pytestmark = pytest.mark.asyncio


# ===================================================================
# Real-DB harness: a character seeded in CharactersRAGDB, with the SAME id
# fed back into the stubbed character-handler module functions.
# ===================================================================


@pytest.fixture
def worldbooks_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "personas_char_worldbooks.db", "test-client")
    yield db
    db.close_connection()


@pytest.fixture
def seeded_character_with_worldbook(worldbooks_db):
    """A real character with one embedded world-book snapshot attached."""
    char_id = worldbooks_db.add_character_card({"name": "Hero"})
    manager = WorldBookManager(worldbooks_db)
    book_id = manager.create_world_book("Blackreach", description="Dark elf city.")
    manager.create_world_book_entry(
        book_id, keys=["castle"], content="A castle looms over the district."
    )
    manager.attach_world_book_to_character(book_id, char_id)
    return {"char_id": char_id, "book_id": book_id}


@pytest.fixture
def stub_characters_for_worldbooks(monkeypatch, seeded_character_with_worldbook):
    """Feed the screen's character list/loader the SAME id as the real-db record."""
    char_id = seeded_character_with_worldbook["char_id"]
    record = {
        "id": char_id,
        "name": "Hero",
        "description": "",
        "first_message": "Hello.",
        "version": 1,
    }
    monkeypatch.setattr(
        character_handler_module, "fetch_all_characters", lambda: [dict(record)]
    )
    monkeypatch.setattr(
        character_handler_module,
        "fetch_character_by_id",
        lambda character_id: (
            dict(record) if str(character_id) == str(char_id) else None
        ),
    )


async def _mounted(pilot):
    await pilot.pause()
    return pilot.app.screen


async def _select_seeded_character(pilot, char_id):
    screen = await _mounted(pilot)
    await pilot.click(f"#personas-library-row-character-{char_id}")
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return screen


class TestCharacterWorldBooksScreenWiring:
    async def test_selecting_character_shows_attached_world_books(
        self,
        mock_app_instance,
        worldbooks_db,
        seeded_character_with_worldbook,
        stub_characters_for_worldbooks,
    ):
        """Selecting a character with an embedded world-book snapshot feeds the
        panel through the REAL WorldBookManager against the REAL db (proves
        ``_refresh_character_worldbooks`` is actually wired, not just mounted)."""
        mock_app_instance.chachanotes_db = worldbooks_db
        # No dictionaries scope service configured: keep this suite's
        # assertions scoped to world books; _refresh_character_dictionaries
        # degrades to an empty panel when the service is None.
        mock_app_instance.chat_dictionary_scope_service = None
        char_id = seeded_character_with_worldbook["char_id"]

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _select_seeded_character(pilot, char_id)
            assert screen.state.selected_entity_kind == "character"
            assert screen.state.selected_entity_id == str(char_id)

            panel = screen.query_one(PersonasCharacterWorldBooksWidget)
            table = panel.query_one("#personas-char-worldbooks-table", DataTable)
            assert table.row_count == 1

    async def test_character_without_worldbooks_shows_empty_panel(
        self,
        mock_app_instance,
        worldbooks_db,
        monkeypatch,
    ):
        """A character with no embedded world books yields an empty (not
        stale/crashed) panel - the refresh runs the real manager either way."""
        char_id = worldbooks_db.add_character_card({"name": "Plain"})
        record = {
            "id": char_id,
            "name": "Plain",
            "description": "",
            "first_message": "Hi.",
            "version": 1,
        }
        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", lambda: [dict(record)]
        )
        monkeypatch.setattr(
            character_handler_module,
            "fetch_character_by_id",
            lambda character_id: (
                dict(record) if str(character_id) == str(char_id) else None
            ),
        )
        mock_app_instance.chachanotes_db = worldbooks_db
        mock_app_instance.chat_dictionary_scope_service = None

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _select_seeded_character(pilot, char_id)
            panel = screen.query_one(PersonasCharacterWorldBooksWidget)
            table = panel.query_one("#personas-char-worldbooks-table", DataTable)
            assert table.row_count == 0


# ===================================================================
# Editor coherence (mirrors test_personas_character_editor_sync.py's
# sync_attached_dictionaries coverage, for the world-books sibling method).
# ===================================================================


class _EditorHost(App):
    def compose(self) -> ComposeResult:
        yield PersonasCharacterEditorWidget()


class TestEditorSyncWorldBooks:
    async def test_editor_sync_patches_base_without_conflict(self):
        async with _EditorHost().run_test(size=(120, 40)) as pilot:
            editor = pilot.app.query_one(PersonasCharacterEditorWidget)
            editor.load_character(
                {"id": 5, "name": "Noir", "version": 1, "extensions": {}}
            )
            await pilot.pause()

            blocks = [{"name": "Blackreach", "enabled": True, "entries": []}]
            editor.sync_attached_world_books(blocks, new_version=2)

            data = editor.get_character_data()
            assert data["version"] == 2
            assert data["extensions"]["character_world_books"] == blocks

    async def test_sync_is_noop_without_a_loaded_character(self):
        async with _EditorHost().run_test(size=(120, 40)) as pilot:
            editor = pilot.app.query_one(PersonasCharacterEditorWidget)
            # no load_character call
            editor.sync_attached_world_books(
                [{"name": "X", "entries": []}], new_version=9
            )
            assert editor._character_data == {}


# ===================================================================
# Geometry: two stacked bottom-docked panels (dictionaries + world books)
# must not squeeze the character card to zero height.
# ===================================================================


class TestTwoDockedPanelsGeometry:
    @pytest.mark.parametrize("size", [(100, 30), (160, 50)])
    async def test_two_docked_panels_do_not_clip_card(
        self,
        mock_app_instance,
        worldbooks_db,
        seeded_character_with_worldbook,
        stub_characters_for_worldbooks,
        size,
    ):
        mock_app_instance.chachanotes_db = worldbooks_db
        mock_app_instance.chat_dictionary_scope_service = None
        char_id = seeded_character_with_worldbook["char_id"]

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=size) as pilot:
            screen = await _select_seeded_character(pilot, char_id)

            card = screen.query_one(PersonasCharacterCardWidget)
            wb = screen.query_one(PersonasCharacterWorldBooksWidget)
            dicts = screen.query_one(PersonasCharacterDictionariesWidget)

            assert card.size.height > 0, f"card clipped at size={size}"
            assert wb.size.height > 0, f"world-books panel clipped at size={size}"
            assert dicts.size.height > 0, f"dictionaries panel clipped at size={size}"


# ===================================================================
# Regression (code review, Task 6 follow-up): the docked wrapper holding
# BOTH character-attachment panels must be gated by _show_center's
# per-center-view condition, not just the coarse "mode == characters"
# toggle _apply_mode used to set. Before the fix, moving the dict panel
# into #personas-character-attachments meant the world-books panel was no
# longer covered by _show_center's per-view gate at all, so it stayed
# docked/visible (stale data) once the center view swapped away from the
# character card/editor within Characters mode (e.g. the conversation
# transcript), and showed as an empty docked panel at initial mount before
# any character was selected.
# ===================================================================


class TestCharacterAttachmentsWrapperGating:
    async def test_initial_mount_hides_wrapper_before_any_selection(
        self,
        mock_app_instance,
        worldbooks_db,
        seeded_character_with_worldbook,
        stub_characters_for_worldbooks,
    ):
        mock_app_instance.chachanotes_db = worldbooks_db
        mock_app_instance.chat_dictionary_scope_service = None

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _mounted(pilot)
            wrapper = screen.query_one("#personas-character-attachments")
            assert wrapper.display is False, (
                "the world-books/dictionaries wrapper must not be visible "
                "before any character is selected"
            )

    async def test_leaving_the_card_view_hides_the_wrapper(
        self,
        mock_app_instance,
        worldbooks_db,
        seeded_character_with_worldbook,
        stub_characters_for_worldbooks,
    ):
        """Switching the center view away from the character card/editor
        (e.g. opening the conversation transcript, which calls
        ``screen._show_center(_CONVERSATION_VIEW_ID)`` per
        ``personas_conversations_controller.open_conversation``) while still
        in Characters mode must hide the docked attachments wrapper too, so
        the world-books panel does not leak stale data over the transcript.
        """
        mock_app_instance.chachanotes_db = worldbooks_db
        mock_app_instance.chat_dictionary_scope_service = None
        char_id = seeded_character_with_worldbook["char_id"]

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _select_seeded_character(pilot, char_id)
            wrapper = screen.query_one("#personas-character-attachments")
            assert wrapper.display is True, (
                "sanity: selecting a character shows the wrapper"
            )

            # Same center-view swap the conversation-transcript open path
            # performs, without needing to stand up the full conversations
            # list/transcript-loading harness.
            screen._show_center("#personas-conversation-transcript-view")
            await pilot.pause()

            assert wrapper.display is False, (
                "the attachments wrapper must be hidden once the center "
                "view swaps away from the character card/editor, even "
                "within Characters mode"
            )
