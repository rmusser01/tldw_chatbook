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
from pathlib import Path

import pytest
import pytest_asyncio
from PIL import Image
from textual.app import App, ComposeResult
from textual.widgets import Button

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Utils.paths import get_user_data_dir
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
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


@pytest.mark.asyncio
async def test_apply_expression_set_stages_idle_and_writes_three(personas_editor_with_saved_character):
    app, screen, db, char_id = personas_editor_with_saved_character
    import io as _io
    from PIL import Image as _Img
    def _png(c=(1, 2, 3)):
        b = _io.BytesIO(); _Img.new("RGB", (8, 8), c).save(b, format="PNG"); return b.getvalue()

    result = await screen._apply_expression_set(
        char_id, {"idle": _png((9, 9, 9)), "speaking": _png(), "thinking": _png()}
    )
    # idle STAGED in the editor (not the table); three -> table
    from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import PersonasCharacterEditorWidget
    editor = screen.query_one(PersonasCharacterEditorWidget)
    assert editor.current_avatar_bytes() == _png((9, 9, 9))      # idle staged
    assert db.get_character_expression_image(char_id, "speaking") is not None
    assert db.get_character_expression_image(char_id, "idle") is None
    assert set(result.applied) >= {"idle", "speaking", "thinking"}


# ===== Roleplay P3d-2 Task 4: import/export expression-set buttons + workers =====


@pytest.mark.asyncio
async def test_import_expression_set_from_zip_path(personas_editor_with_saved_character, tmp_path):
    app, screen, db, char_id = personas_editor_with_saved_character
    from tldw_chatbook.Character_Chat.expression_set_io import build_expression_set_zip
    import io as _io
    from PIL import Image as _Img
    def _png(): b=_io.BytesIO(); _Img.new("RGB",(8,8)).save(b,format="PNG"); return b.getvalue()
    z = tmp_path / "set.zip"
    z.write_bytes(build_expression_set_zip("Ada", {"idle": _png(), "speaking": _png()}))

    await screen._import_expression_set_from_path(char_id, str(z))

    assert db.get_character_expression_image(char_id, "speaking") is not None
    from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import PersonasCharacterEditorWidget
    assert screen.query_one(PersonasCharacterEditorWidget).current_avatar_bytes() is not None  # idle staged


@pytest.mark.asyncio
async def test_import_vpack_from_path(personas_editor_with_saved_character, tmp_path):
    app, screen, db, char_id = personas_editor_with_saved_character
    from Tests.Character_Chat.test_expression_set_io import simple_vpack, _png
    z = tmp_path / "pack.tldw-persona-vpack"
    z.write_bytes(simple_vpack({"idle": _png(), "speaking": _png(), "thinking": _png()}))

    await screen._import_expression_set_from_path(char_id, str(z))

    assert db.get_character_expression_image(char_id, "speaking") is not None
    assert db.get_character_expression_image(char_id, "thinking") is not None
    from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
        PersonasCharacterEditorWidget,
    )
    assert screen.query_one(PersonasCharacterEditorWidget).current_avatar_bytes() is not None  # idle staged


@pytest.mark.asyncio
async def test_export_expression_set_writes_a_zip(personas_editor_with_saved_character):
    app, screen, db, char_id = personas_editor_with_saved_character
    import io as _io
    from PIL import Image as _Img
    def _png(): b=_io.BytesIO(); _Img.new("RGB",(8,8)).save(b,format="PNG"); return b.getvalue()
    db.set_character_expression_image(char_id, "speaking", _png())
    target = await screen._export_expression_set(char_id, "Ada")
    assert target is not None
    from pathlib import Path
    import zipfile
    assert zipfile.is_zipfile(Path(target))
    assert "speaking.png" in zipfile.ZipFile(target).namelist()


@pytest.mark.asyncio
async def test_import_export_buttons_present_for_saved_character(
    personas_editor_with_saved_character,
):
    app, screen, db, char_id = personas_editor_with_saved_character
    assert screen.query_one("#personas-char-editor-expr-import") is not None
    assert screen.query_one("#personas-char-editor-expr-export") is not None


# ===== Review fix 1: _export_expression_set cleans up its temp file on
# failure (mirrors _dictionary_export_worker's try/except OSError +
# temp.unlink(missing_ok=True) idiom, which the initial implementation
# omitted). =====


@pytest.mark.asyncio
async def test_export_expression_set_cleans_up_temp_on_replace_failure(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character

    def _png():
        buf = BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        return buf.getvalue()

    db.set_character_expression_image(char_id, "speaking", _png())

    def _boom(self, target):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", _boom)

    # Diff before/after rather than asserting the dir is empty: it's the
    # shared test-home exports dir, which other test runs may have already
    # left debris in.
    exports_dir = get_user_data_dir() / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    before = set(exports_dir.glob("*.tmp"))

    with pytest.raises(OSError):
        await screen._export_expression_set(char_id, "Ada")

    after = set(exports_dir.glob("*.tmp"))
    assert after - before == set()


# ===== Review fix 2: the Import set…/Export set… buttons must be disabled
# for an unsaved character, same as the per-slot Upload/Clear buttons
# (_sync_expression_slots_enabled). =====


class _UnsavedEditorHost(App):
    def compose(self) -> ComposeResult:
        yield PersonasCharacterEditorWidget()


@pytest.mark.asyncio
async def test_import_export_buttons_disabled_for_unsaved_character():
    app = _UnsavedEditorHost()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A"})  # no "id" key -> unsaved
        await pilot.pause()
        assert ed.expression_character_id() is None
        assert ed.query_one("#personas-char-editor-expr-import", Button).disabled is True
        assert ed.query_one("#personas-char-editor-expr-export", Button).disabled is True


# ===== Qodo review fix 1: _import_expression_set_from_path validates the
# picked path at the screen boundary (mirrors _read_avatar_image_bytes'
# use of validate_path_simple), instead of handing an unvalidated path
# straight to the pure resolver. =====


@pytest.mark.asyncio
async def test_import_expression_set_nonexistent_path_notifies_and_does_not_crash(
    personas_editor_with_saved_character,
):
    app, screen, db, char_id = personas_editor_with_saved_character
    # Shadow the delegating test App's notify (like TestDictionaryImport's
    # _capture_notifications) rather than mock_app_instance.notify --
    # screen.app_instance is the PersonasTestApp, and its real (Textual App)
    # notify() shadows the mock's via normal attribute lookup.
    captured: list[tuple[str, str]] = []
    app.notify = lambda message, severity="information", **kwargs: captured.append(
        (str(message), severity)
    )

    # Must not raise -- the invalid path is rejected before it ever reaches
    # resolve_local_expression_set.
    await screen._import_expression_set_from_path(char_id, "/no/such/path/set.zip")

    assert captured, "expected a notification for a rejected path"
    assert captured[-1][1] == "error"


# ===== Qodo review fix 7: the export handler must honor the same
# _io_dialog_active gate as the import handler and _dictionary_export_worker,
# so a queued export cannot race a second worker onto the same filename. =====


@pytest.mark.asyncio
async def test_export_handler_blocked_while_io_dialog_active(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
        CharacterExpressionSetExportRequested,
    )

    calls: list[int] = []
    monkeypatch.setattr(screen, "run_worker", lambda *a, **k: calls.append(1))
    screen._io_dialog_active = True

    screen._handle_expression_set_export_requested(
        CharacterExpressionSetExportRequested()
    )

    assert calls == []  # gate blocked a second worker from starting


@pytest.mark.asyncio
async def test_export_handler_starts_worker_and_sets_gate_when_clear(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
        CharacterExpressionSetExportRequested,
    )

    calls: list[int] = []

    def _fake_run_worker(coro, *a, **k):
        calls.append(1)
        coro.close()  # avoid a "coroutine was never awaited" warning

    monkeypatch.setattr(screen, "run_worker", _fake_run_worker)
    screen._io_dialog_active = False

    screen._handle_expression_set_export_requested(
        CharacterExpressionSetExportRequested()
    )

    assert calls == [1]
    assert screen._io_dialog_active is True  # gate set before the worker starts


@pytest.mark.asyncio
async def test_import_export_buttons_enabled_for_saved_character(
    personas_editor_with_saved_character,
):
    app, screen, db, char_id = personas_editor_with_saved_character
    assert (
        screen.query_one("#personas-char-editor-expr-import", Button).disabled is False
    )
    assert (
        screen.query_one("#personas-char-editor-expr-export", Button).disabled is False
    )
