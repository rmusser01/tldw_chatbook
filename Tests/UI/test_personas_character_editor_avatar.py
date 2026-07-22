"""Roleplay P3b Task 2: character editor avatar thumbnail preview + Remove.

Widget-level coverage mirrors ``test_personas_character_editor_sync.py``'s
bare-``PersonasCharacterEditorWidget`` host harness; the screen-level coverage
mirrors ``test_personas_character_world_books_screen.py``'s real-DB
``PersonasTestApp`` harness (seed a character via ``add_character_card``,
feed the same id back into the stubbed ``ccp_character_handler`` module
functions, then drive the editor through the real screen).
"""

import pytest
from PIL import Image
from rich_pixels import Pixels
from textual.app import App, ComposeResult
from textual.widgets import Button

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)

from Tests.UI.test_personas_dictionaries import PersonasTestApp, patch_character_paging

pytestmark = pytest.mark.asyncio


# ===================================================================
# Widget-level: set_avatar_thumbnail mount/clear, current_avatar_bytes,
# Remove button posting the new message.
# ===================================================================


class _Host(App):
    def __init__(self):
        super().__init__()
        self.removed = 0

    def compose(self) -> ComposeResult:
        yield PersonasCharacterEditorWidget()

    def on_character_image_remove_requested(self, m):
        self.removed += 1


async def test_thumbnail_mounts_and_text_fallback():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A", "image": b"x"})
        await pilot.pause()
        # None -> text fallback present, no image widget
        ed.set_avatar_thumbnail(None)
        await pilot.pause()
        assert len(ed.query("#personas-char-editor-avatar-thumb > *")) == 0
        # A pixels renderable -> a widget is mounted in the avatar row
        px = Pixels.from_image(Image.new("RGBA", (8, 8)))
        ed.set_avatar_thumbnail(px)
        await pilot.pause()
        assert len(ed.query("#personas-char-editor-avatar-thumb > *")) == 1


async def test_current_avatar_bytes_and_remove():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A", "image": b"abc"})
        await pilot.pause()
        assert ed.current_avatar_bytes() == b"abc"
        # .press() (not pilot.click) - the avatar row sits below the fold of
        # the scrollable editor body at default pilot size, so a coordinate
        # click can miss it (see test_upload_button_posts_image_upload_request
        # in test_personas_character_widgets.py for the same convention).
        ed.query_one("#personas-char-editor-avatar-remove", Button).press()
        await pilot.pause()
        assert app.removed == 1


async def test_current_avatar_bytes_none_without_image():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A"})
        await pilot.pause()
        assert ed.current_avatar_bytes() is None


# ===================================================================
# Screen-level: real PersonasScreen renders the thumbnail off-thread when
# the editor opens for a character with an embedded image, and Remove
# clears it + marks the session unsaved.
# ===================================================================


@pytest.fixture
def avatar_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "personas_char_avatar.db", "test-client")
    yield db
    db.close_connection()


@pytest.fixture
def avatar_image_bytes():
    """Real PNG-encoded avatar image bytes for a character record."""
    from io import BytesIO

    buf = BytesIO()
    Image.new("RGB", (4, 4), color=(200, 40, 40)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def seeded_character_with_avatar(avatar_db, avatar_image_bytes):
    """A real character row, with the SAME id fed back into the stubbed
    ``ccp_character_handler`` module functions (mirrors
    ``test_personas_character_world_books_screen.py``'s
    ``seeded_character_with_worldbook``)."""
    char_id = avatar_db.add_character_card({"name": "Portraitless"})
    return {"char_id": char_id, "image": avatar_image_bytes}


@pytest.fixture
def stub_character_with_avatar(monkeypatch, seeded_character_with_avatar):
    """Feed the screen a character record carrying embedded image bytes."""
    char_id = seeded_character_with_avatar["char_id"]
    record = {
        "id": char_id,
        "name": "Portraitless",
        "description": "",
        "first_message": "Hi.",
        "version": 1,
        "image": seeded_character_with_avatar["image"],
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


async def _select_character(pilot, char_id):
    await pilot.pause()
    await pilot.click(f"#personas-library-row-character-{char_id}")
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return pilot.app.screen


class TestCharacterEditorAvatarThumbnailScreen:
    async def test_editor_with_image_renders_thumbnail(
        self,
        mock_app_instance,
        avatar_db,
        seeded_character_with_avatar,
        stub_character_with_avatar,
    ):
        mock_app_instance.chachanotes_db = avatar_db
        mock_app_instance.chat_dictionary_scope_service = None
        char_id = seeded_character_with_avatar["char_id"]

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _select_character(pilot, char_id)
            # Open the editor for this character (edit action).
            from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
                EditCharacterRequested,
            )

            screen.post_message(EditCharacterRequested(str(char_id)))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            editor = screen.query_one(PersonasCharacterEditorWidget)
            assert editor.current_avatar_bytes() is not None
            assert len(editor.query("#personas-char-editor-avatar-thumb > *")) == 1

            editor.query_one("#personas-char-editor-avatar-remove", Button).press()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert editor.current_avatar_bytes() is None
            assert len(editor.query("#personas-char-editor-avatar-thumb > *")) == 0
            assert screen.state.has_unsaved_changes is True
