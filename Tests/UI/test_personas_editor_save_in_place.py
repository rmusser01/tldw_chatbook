"""Roleplay P3b Task 1: save-in-place + dirty re-arm.

After Save, both editors now stay open (instead of flipping to the read-only
card) with dirty-tracking re-armed against the just-persisted record. Two
layers of coverage:

- Widget-level: ``mark_saved`` re-baselines the dirty snapshot from the
  CURRENT form (already showing the saved values) without repopulating it,
  and adopts the saved record's ``version`` for the next round trip.
- Screen-level: the character finisher (``_after_character_save``) re-reads
  the just-persisted record over a REAL ``CharactersRAGDB`` before deciding
  whether to stay in the editor, so the test proves the optimistic-lock
  ``version`` genuinely comes from the DB, not a stub.

Mirrors ``Tests/UI/test_personas_character_widgets.py`` /
``test_persona_profile_widgets.py`` for the bare-widget harness, and
``Tests/UI/test_personas_character_world_books_screen.py`` /
``test_personas_dictionaries.py`` for the real-DB screen harness
(``PersonasTestApp`` + routing ``ccp_character_handler._default_character_db``
at a real db - the same seam ``create_character``/``update_character``/
``fetch_character_by_id``/the library-paging loader all read from).
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_editor_widget import (
    PersonaProfileEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    EditorContentChanged,
)

from Tests.UI.test_personas_dictionaries import PersonasTestApp

pytestmark = pytest.mark.asyncio


# ===================================================================
# Widget-level: mark_saved re-arms dirty tracking without repopulating.
# ===================================================================


class _CharHost(App):
    def __init__(self):
        super().__init__()
        self.dirty = 0

    def compose(self) -> ComposeResult:
        yield PersonasCharacterEditorWidget()

    def on_editor_content_changed(self, message: EditorContentChanged) -> None:
        self.dirty += 1


class _PersonaHost(App):
    def __init__(self):
        super().__init__()
        self.dirty = 0

    def compose(self) -> ComposeResult:
        yield PersonaProfileEditorWidget()

    def on_editor_content_changed(self, message: EditorContentChanged) -> None:
        self.dirty += 1


async def test_character_mark_saved_rearms_dirty():
    app = _CharHost()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"id": 5, "name": "A", "version": 1})
        await pilot.pause()
        ed.query_one("#personas-char-editor-name", Input).value = "A2"  # first edit
        await pilot.pause()
        assert app.dirty == 1

        # Simulate a save landing: new version, re-arm. mark_saved must NOT
        # repopulate the form - the user's saved edit ("A2") stays on screen.
        ed.mark_saved({"id": 5, "name": "A2", "version": 2})
        await pilot.pause()
        assert ed._dirty_posted is False
        assert ed.query_one("#personas-char-editor-name", Input).value == "A2"

        ed.query_one("#personas-char-editor-name", Input).value = "A3"  # second edit
        await pilot.pause()
        assert app.dirty == 2  # re-armed -> re-posted
        assert ed.get_character_data()["version"] == 2  # new optimistic-lock version


async def test_persona_mark_saved_rearms_dirty():
    app = _PersonaHost()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonaProfileEditorWidget)
        ed.load_persona({"id": "p1", "name": "B", "version": 1})
        await pilot.pause()
        ed.query_one("#personas-editor-name", Input).value = "B2"  # first edit
        await pilot.pause()
        assert app.dirty == 1

        ed.mark_saved({"id": "p1", "name": "B", "version": 2})
        await pilot.pause()
        assert ed._dirty_posted is False
        assert ed.query_one("#personas-editor-name", Input).value == "B2"

        ed.query_one("#personas-editor-name", Input).value = "B3"  # second edit
        await pilot.pause()
        assert app.dirty == 2  # re-armed -> re-posted
        assert ed.collect()["version"] == 2  # new optimistic-lock version


# ===================================================================
# Screen-level: character Save stays in the editor, carrying the fresh
# optimistic-lock version from a REAL CharactersRAGDB.
# ===================================================================


@pytest.fixture
def real_char_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "personas_save_in_place.db", "test-client")
    yield db
    db.close_connection()


@pytest.fixture
def route_character_db(monkeypatch, real_char_db):
    """Route character CRUD + library paging (all read via
    ``ccp_character_handler._default_character_db``) at a real DB, so a
    round-tripped Save carries the DB's genuine incremented ``version``."""
    monkeypatch.setattr(
        character_handler_module, "_default_character_db", lambda: real_char_db
    )
    return real_char_db


async def _mounted(pilot):
    await pilot.pause()
    return pilot.app.screen


class TestCharacterSaveInPlace:
    async def test_create_save_stays_in_editor_and_rearms_dirty(
        self, mock_app_instance, route_character_db
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.press("ctrl+n")
            await pilot.pause()
            assert screen._edit_mode == "create"

            screen.query_one("#personas-char-editor-name", Input).value = "New Hero"
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            # (a) the editor center is still displayed - NOT the card.
            assert screen.query_one("#ccp-character-editor-view").display is True
            assert screen.query_one("#ccp-character-card-view").display is False
            # (b) the just-completed save clears the unsaved flag.
            assert screen.state.has_unsaved_changes is False
            # (c) create -> edit (a real row now backs the open session).
            assert screen._edit_mode == "edit"
            # The persisted row carries a real, DB-assigned optimistic-lock
            # version (proves the round trip hit the real db, not a stub).
            saved_id = screen.state.selected_entity_id
            assert saved_id is not None
            record = route_character_db.get_character_card_by_id(int(saved_id))
            assert record is not None
            assert record["version"] == 1

            # (d) a further field edit re-flags has_unsaved_changes.
            screen.query_one("#personas-char-editor-name", Input).value = "New Hero 2"
            await pilot.pause()
            assert screen.state.has_unsaved_changes is True
