"""Roleplay P3b Task 2: character editor avatar thumbnail preview + Remove.

Widget-level coverage mirrors ``test_personas_character_editor_sync.py``'s
bare-``PersonasCharacterEditorWidget`` host harness; the screen-level coverage
mirrors ``test_personas_character_world_books_screen.py``'s real-DB
``PersonasTestApp`` harness (seed a character via ``add_character_card``,
feed the same id back into the stubbed ``ccp_character_handler`` module
functions, then drive the editor through the real screen).
"""

import rich_pixels
import pytest
from PIL import Image
from rich_pixels import Pixels
from textual.app import App, ComposeResult
from textual.widgets import Button

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.personas_screen import (
    AVATAR_THUMB_COLS,
    AVATAR_THUMB_LINES,
    PersonasScreen,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)

from Tests.UI.test_personas_dictionaries import PersonasTestApp, patch_character_paging

# The async tests below carry an explicit per-function `@pytest.mark.asyncio`
# decorator rather than a module-level `pytestmark`: this module also has
# plain sync unit tests (the `_fit_avatar_cell_size` coverage) that
# pytest-asyncio warns about under a module-level marker, and the explicit
# per-test decorator keeps the async tests collectable even when the run's
# rootdir isn't Tests/UI (so Tests/UI/pytest.ini's `asyncio_mode = auto`
# doesn't apply) — e.g. a run mixing Tests/UI and Tests/Character_Chat.


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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_current_avatar_bytes_none_without_image():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A"})
        await pilot.pause()
        assert ed.current_avatar_bytes() is None


# ===================================================================
# Unit-level: PersonasScreen._fit_avatar_cell_size (graphics-mode fitting).
#
# Cells are ~2x taller than wide in pixels, so the aspect ratio is compared
# in "cell units" (pixel_width : pixel_height/2) before fitting into the
# AVATAR_THUMB_COLS x AVATAR_THUMB_LINES box. Both returned dimensions must
# stay explicit ints >= 1 - that's what the graphics-mode ``ValueError:
# height/width must be > 0`` crash this rendering path works around
# depends on.
# ===================================================================


def test_fit_avatar_cell_size_wide_image_fits_width():
    # 200x100px -> cell aspect 200:(100/2) = 4.0, wider than the 24:10 = 2.4
    # box aspect, so it's width-limited.
    w, h = PersonasScreen._fit_avatar_cell_size(200, 100)
    assert w == AVATAR_THUMB_COLS
    assert 1 <= h < AVATAR_THUMB_LINES
    assert isinstance(w, int) and isinstance(h, int)


def test_fit_avatar_cell_size_tall_image_fits_height():
    # 100x400px -> cell aspect 100:(400/2) = 0.5, narrower than the box
    # aspect, so it's height-limited.
    w, h = PersonasScreen._fit_avatar_cell_size(100, 400)
    assert h == AVATAR_THUMB_LINES
    assert 1 <= w < AVATAR_THUMB_COLS
    assert isinstance(w, int) and isinstance(h, int)


def test_fit_avatar_cell_size_square_image_stays_in_box():
    w, h = PersonasScreen._fit_avatar_cell_size(512, 512)
    assert 1 <= w <= AVATAR_THUMB_COLS
    assert 1 <= h <= AVATAR_THUMB_LINES
    # Aspect preserved: a square image is 2:1 in cell units (cells are
    # ~2x taller than wide), so width should be roughly double height.
    assert w == pytest.approx(2 * h, abs=1)


def test_fit_avatar_cell_size_tiny_image_still_positive():
    """A tiny (e.g. 4x4) source must still yield a renderable >=1x1 box."""
    w, h = PersonasScreen._fit_avatar_cell_size(4, 4)
    assert w >= 1 and h >= 1
    assert w <= AVATAR_THUMB_COLS and h <= AVATAR_THUMB_LINES


def test_fit_avatar_cell_size_degenerate_dimensions_fall_back_to_box():
    w, h = PersonasScreen._fit_avatar_cell_size(0, 0)
    assert (w, h) == (AVATAR_THUMB_COLS, AVATAR_THUMB_LINES)


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


@pytest.fixture
def wide_avatar_image_bytes():
    """A realistic, non-square (2:1) avatar image - wider than tall."""
    from io import BytesIO

    buf = BytesIO()
    Image.new("RGBA", (200, 100), color=(10, 120, 200, 255)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def seeded_character_with_wide_avatar(avatar_db, wide_avatar_image_bytes):
    """Same pattern as ``seeded_character_with_avatar``, but with a
    realistic non-square source image so the fit-not-clip regression test
    below can't pass by coincidence on a square image."""
    char_id = avatar_db.add_character_card({"name": "WidePortrait"})
    return {"char_id": char_id, "image": wide_avatar_image_bytes}


@pytest.fixture
def stub_character_with_wide_avatar(monkeypatch, seeded_character_with_wide_avatar):
    char_id = seeded_character_with_wide_avatar["char_id"]
    record = {
        "id": char_id,
        "name": "WidePortrait",
        "description": "",
        "first_message": "Hi.",
        "version": 1,
        "image": seeded_character_with_wide_avatar["image"],
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


async def _open_editor_for(pilot, screen, char_id):
    from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
        EditCharacterRequested,
    )

    screen.post_message(EditCharacterRequested(str(char_id)))
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


class TestCharacterEditorAvatarThumbnailScreen:
    @pytest.mark.asyncio
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


# ===================================================================
# Screen-level regression (P3b PR fix - Qodo #3): Remove must not be
# visually undone by a stale in-flight render.
#
# ``_render_character_editor_avatar`` drops its result when its captured
# ``token`` no longer matches ``_character_editor_generation``. Remove used
# to leave the generation untouched, so an earlier in-flight render (queued
# before Remove was clicked, same token) could complete AFTER Remove and
# re-mount the old avatar bytes. The fix bumps the generation in
# ``_handle_character_image_remove`` before dispatching its own render, so
# any prior-token render in flight is dropped instead of winning the race.
#
# The exact interleaving (an in-flight decode finishing after Remove) is not
# reliably forceable in a unit test without invasively faking the asyncio
# scheduler, so this instead asserts the mechanism the fix relies on: the
# generation token strictly increases across Remove, which is sufficient to
# guarantee any earlier-token render is dropped by the guard in
# ``_render_character_editor_avatar`` regardless of completion order.
# ===================================================================


class TestCharacterEditorAvatarRemoveGenerationBump:
    @pytest.mark.asyncio
    async def test_remove_bumps_generation_past_open_time_token(
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
            await _open_editor_for(pilot, screen, char_id)

            generation_at_open = screen._character_editor_generation

            editor = screen.query_one(PersonasCharacterEditorWidget)
            editor.query_one("#personas-char-editor-avatar-remove", Button).press()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            # The generation strictly advanced past the open-time token, so
            # a render still carrying that older token (queued before Remove
            # was clicked) is dropped by _render_character_editor_avatar's
            # token check even if it completes after Remove's own render.
            assert screen._character_editor_generation > generation_at_open
            # And Remove's own effect actually landed: no stale re-mount.
            assert editor.current_avatar_bytes() is None
            assert len(editor.query("#personas-char-editor-avatar-thumb > *")) == 0


# ===================================================================
# Screen-level regression: pixels-mode thumbnail fits the avatar box
# instead of clipping.
#
# ``ConsoleImageRenderCache.get_pixels`` thumbnails to the 80x40
# chat-transcript box; reusing it for the avatar preview produced an
# oversized Pixels grid that Rich does not reflow, so the small 24x10
# thumb container just cropped it to a top-left sliver. These tests drive
# the real screen render in forced pixels mode with a realistic,
# non-square avatar and inspect the PIL image actually handed to
# ``Pixels.from_image`` - proving it was rescaled to the avatar box (not
# the transcript box) and kept its aspect ratio, rather than merely
# asserting a widget got mounted (which the clipped version also did).
# ===================================================================


class TestCharacterEditorAvatarThumbnailFit:
    @pytest.mark.asyncio
    async def test_wide_avatar_pixels_thumbnail_fits_box_not_clipped(
        self,
        monkeypatch,
        mock_app_instance,
        avatar_db,
        seeded_character_with_wide_avatar,
        stub_character_with_wide_avatar,
    ):
        mock_app_instance.chachanotes_db = avatar_db
        mock_app_instance.chat_dictionary_scope_service = None
        # Force pixels mode regardless of terminal auto-detection so this
        # test is deterministic in CI.
        mock_app_instance.app_config = {
            "chat": {"images": {"default_render_mode": "pixels"}}
        }
        char_id = seeded_character_with_wide_avatar["char_id"]

        captured: list[Image.Image] = []
        original_from_image = rich_pixels.Pixels.from_image

        def _spy_from_image(image, *args, **kwargs):
            captured.append(image.copy())
            return original_from_image(image, *args, **kwargs)

        monkeypatch.setattr(rich_pixels.Pixels, "from_image", _spy_from_image)

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _select_character(pilot, char_id)
            await _open_editor_for(pilot, screen, char_id)

            editor = screen.query_one(PersonasCharacterEditorWidget)
            assert editor.current_avatar_bytes() is not None
            assert len(editor.query("#personas-char-editor-avatar-thumb > *")) == 1

        assert len(captured) == 1, (
            "Pixels.from_image should be called exactly once building the "
            "avatar thumbnail"
        )
        thumb = captured[0]
        # Fits the avatar box, in half-block "pixel" units (COLS wide,
        # LINES*2 tall) - NOT the 80x40 chat-transcript box that
        # get_pixels() thumbnails to.
        assert thumb.width <= AVATAR_THUMB_COLS
        assert thumb.height <= AVATAR_THUMB_LINES * 2
        # It actually used (not shrunk far below) the box's width - the old
        # bug's symptom was a sliver, not merely "some image smaller than
        # the transcript box".
        assert thumb.width == AVATAR_THUMB_COLS
        # Aspect preserved (not stretched/cropped to fill the box): the
        # 200x100 source is 2:1.
        source_aspect = 200 / 100
        thumb_aspect = thumb.width / thumb.height
        assert thumb_aspect == pytest.approx(source_aspect, rel=0.15)

    @pytest.mark.asyncio
    async def test_tiny_avatar_still_produces_pixels_thumbnail(
        self,
        monkeypatch,
        mock_app_instance,
        avatar_db,
        seeded_character_with_avatar,
        stub_character_with_avatar,
    ):
        """A tiny (4x4) avatar must still render a mounted thumbnail."""
        mock_app_instance.chachanotes_db = avatar_db
        mock_app_instance.chat_dictionary_scope_service = None
        mock_app_instance.app_config = {
            "chat": {"images": {"default_render_mode": "pixels"}}
        }
        char_id = seeded_character_with_avatar["char_id"]

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _select_character(pilot, char_id)
            await _open_editor_for(pilot, screen, char_id)

            editor = screen.query_one(PersonasCharacterEditorWidget)
            assert editor.current_avatar_bytes() is not None
            assert len(editor.query("#personas-char-editor-avatar-thumb > *")) == 1
