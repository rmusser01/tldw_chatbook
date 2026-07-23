"""Roleplay P3d-1 Task 4: character expression authoring slots
(thinking/speaking/error) in the character editor.

Mirrors ``test_personas_character_editor_avatar.py``'s screen-level harness
(real ``CharactersRAGDB``, a character seeded via ``add_character_card``, its
id fed back into the stubbed ``ccp_character_handler`` module functions, then
driven through the real ``PersonasScreen``) - wrapped in a
``pytest_asyncio.fixture`` (the ``console_screen_with_db``-style pattern in
``test_console_character_avatar.py``) so each test starts with the editor
already open for a saved character, rather than repeating the mount/select/
open-editor boilerplate per test.
"""

from io import BytesIO

import pytest
import pytest_asyncio
from PIL import Image

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    EditCharacterRequested,
)

from Tests.UI.test_personas_dictionaries import PersonasTestApp, patch_character_paging


@pytest.fixture
def expr_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "personas_char_expression.db", "test-client")
    yield db
    db.close_connection()


async def _select_character(pilot, char_id):
    await pilot.pause()
    await pilot.click(f"#personas-library-row-character-{char_id}")
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return pilot.app.screen


async def _open_editor_for(pilot, screen, char_id):
    screen.post_message(EditCharacterRequested(str(char_id)))
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


@pytest_asyncio.fixture
async def personas_editor_with_saved_character(mock_app_instance, monkeypatch, expr_db):
    """Mounted ``PersonasScreen`` with the character editor open for a saved
    character, wired to a real file-backed ``CharactersRAGDB``."""
    mock_app_instance.chachanotes_db = expr_db
    mock_app_instance.chat_dictionary_scope_service = None
    char_id = expr_db.add_character_card({"name": "Expressive"})

    record = {
        "id": char_id,
        "name": "Expressive",
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
    patch_character_paging(monkeypatch)

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _select_character(pilot, char_id)
        await _open_editor_for(pilot, screen, char_id)
        yield app, screen, expr_db, char_id


@pytest.mark.asyncio
async def test_expression_slots_present_for_saved_character(
    personas_editor_with_saved_character,
):
    app, screen, db, char_id = personas_editor_with_saved_character
    for state in ("thinking", "speaking", "error"):
        assert screen.query_one(f"#char-expression-slot-{state}") is not None


@pytest.mark.asyncio
async def test_upload_writes_expression_row(personas_editor_with_saved_character):
    app, screen, db, char_id = personas_editor_with_saved_character
    buf = BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="PNG")
    await screen._apply_expression_upload(
        char_id, "speaking", buf.getvalue(), "image/png"
    )
    assert db.get_character_expression_image(char_id, "speaking") is not None


@pytest.mark.asyncio
async def test_clear_soft_deletes_expression_row(personas_editor_with_saved_character):
    app, screen, db, char_id = personas_editor_with_saved_character
    db.set_character_expression_image(char_id, "error", b"x")
    await screen._clear_expression_slot(char_id, "error")
    assert db.get_character_expression_image(char_id, "error") is None
