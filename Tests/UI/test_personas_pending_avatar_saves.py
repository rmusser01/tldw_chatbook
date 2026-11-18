"""Character Save acknowledges its avatar submission without losing later uploads."""

import asyncio

import pytest
from PIL import Image
from rich.console import Console
from textual.widgets import Button, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_personas_dictionaries import PersonasTestApp
from Tests.UI import test_personas_pending_save_drafts as pending
from tldw_chatbook.UI.Persona_Modules.roleplay_frame_state import (
    roleplay_has_unsaved_work,
)
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen

pytestmark = pytest.mark.asyncio
character_store = pending.character_store


@pytest.fixture
def avatar_files(tmp_path):
    paths = [tmp_path / "submitted.png", tmp_path / "later.png"]
    for path, color in zip(paths, [(210, 30, 40), (30, 40, 210)], strict=True):
        Image.new("RGB", (8, 8), color).save(path, format="PNG")
    return paths


async def _upload(pilot, screen, editor, path):
    editor.query_one("#personas-char-editor-avatar-upload", Button).press()
    await pending._until(
        lambda: isinstance(pilot.app.screen, EnhancedFileOpen)
        and bool(pilot.app.screen.query("EnhancedDirectoryNavigation"))
    )
    await pilot.pause()
    # The actual mounted picker result; file validation/read and staging stay real.
    pilot.app.screen.dismiss(path)
    await pending._until(
        lambda: pilot.app.screen is screen and not screen._io_dialog_active
    )
    await pilot.pause()
    assert editor.current_avatar_bytes() == path.read_bytes()


def _assert_later_preview(editor, later_bytes):
    assert editor.current_avatar_bytes() == later_bytes
    assert str(
        editor.query_one("#personas-char-editor-avatar-status", Static).content
    ) == ("Avatar: embedded")
    thumb = editor.query_one("#personas-char-editor-avatar-thumb > Static", Static)
    console = Console(record=True, force_terminal=True, color_system="truecolor")
    console.print(thumb.content)
    assert "30;40;210" in console.export_text(styles=True)


async def _start(pilot, avatar_files, *, update):
    screen, editor, name = await pending._open(pilot, "character")
    name.focus()
    await pilot.press(*"Avatar owner")
    await _upload(pilot, screen, editor, avatar_files[0])
    if update:
        await pending._save(pilot, screen, "character")
        assert not roleplay_has_unsaved_work(
            screen._aggregate_roleplay_draft_snapshot()
        )
    return screen, editor, name


@pytest.mark.parametrize("update", [False, True], ids=["create", "update"])
@private_profile_test
async def test_pending_save_retains_later_upload_and_same_record_next_save(
    request, mock_app_instance, character_store, monkeypatch, avatar_files, update
):
    """Replacing the live attachment on acknowledgement loses the blue draft."""
    mock_app_instance.app_config = {
        "chat": {"images": {"default_render_mode": "pixels"}}
    }
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(160, 50)) as pilot:
        screen, editor, name = await _start(pilot, avatar_files, update=update)
        prior_id = screen.state.selected_entity_id
        entered, release = pending._hold_save(monkeypatch, "character", None, update)
        saving = asyncio.create_task(pilot.press("ctrl+s"))
        try:
            await pending._until(entered.is_set)
            await _upload(pilot, screen, editor, avatar_files[1])
            assert screen._character_save_inflight
            assert editor.has_unsaved_attachment()
        finally:
            release.set()
        await saving
        await screen._character_save_worker_handle.wait()
        await pilot.pause()
        saved_id = screen.state.selected_entity_id
        saved = character_store.get_character_card_by_id(int(saved_id))
        assert saved["image"] == avatar_files[0].read_bytes()
        assert saved["version"] == (2 if update else 1)
        if update:
            assert saved_id == prior_id
        live = editor.get_character_data()
        assert str(live["id"]) == saved_id
        assert live["version"] == saved["version"]
        _assert_later_preview(editor, avatar_files[1].read_bytes())
        assert editor.has_unsaved_attachment()
        assert screen.state.has_unsaved_changes
        assert roleplay_has_unsaved_work(screen._aggregate_roleplay_draft_snapshot())
        await pending._stay_through_guards(pilot, screen, name)
        _assert_later_preview(editor, avatar_files[1].read_bytes())
        assert editor.has_unsaved_attachment()

        await pending._save(pilot, screen, "character")
        updated = character_store.get_character_card_by_id(int(saved_id))
        assert screen.state.selected_entity_id == saved_id
        assert updated["image"] == avatar_files[1].read_bytes()
        assert updated["version"] == saved["version"] + 1
        assert editor.get_character_data()["version"] == updated["version"]
        assert not editor.has_unsaved_attachment()
        assert not screen.state.has_unsaved_changes
        assert not roleplay_has_unsaved_work(
            screen._aggregate_roleplay_draft_snapshot()
        )


@private_profile_test
async def test_upload_before_save_persists_normally(
    request, mock_app_instance, character_store, avatar_files
):
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(160, 50)) as pilot:
        screen, editor, _name = await _start(pilot, avatar_files, update=True)
        saved_id = screen.state.selected_entity_id
        await _upload(pilot, screen, editor, avatar_files[1])
        await pending._save(pilot, screen, "character")
        stored = character_store.get_character_card_by_id(int(saved_id))
        assert stored["image"] == avatar_files[1].read_bytes()
        assert stored["version"] == 2
        assert screen.state.selected_entity_id == saved_id
        assert editor.current_avatar_bytes() == stored["image"]
        assert not editor.has_unsaved_attachment()
        assert not roleplay_has_unsaved_work(
            screen._aggregate_roleplay_draft_snapshot()
        )


@private_profile_test
async def test_failed_pending_save_retains_upload_for_same_record_retry(
    request, mock_app_instance, character_store, monkeypatch, avatar_files
):
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(160, 50)) as pilot:
        screen, editor, _name = await _start(pilot, avatar_files, update=True)
        saved_id = screen.state.selected_entity_id
        with monkeypatch.context() as failure:
            entered, release = pending._hold_save(
                failure, "character", None, True, fail=True
            )
            saving = asyncio.create_task(pilot.press("ctrl+s"))
            try:
                await pending._until(entered.is_set)
                await _upload(pilot, screen, editor, avatar_files[1])
            finally:
                release.set()
            await saving
            await screen._character_save_worker_handle.wait()
            await pilot.pause()
        stored = character_store.get_character_card_by_id(int(saved_id))
        assert stored["image"] == avatar_files[0].read_bytes()
        assert stored["version"] == editor.get_character_data()["version"] == 1
        assert screen.state.selected_entity_id == saved_id
        assert editor.current_avatar_bytes() == avatar_files[1].read_bytes()
        assert editor.has_unsaved_attachment()
        assert roleplay_has_unsaved_work(screen._aggregate_roleplay_draft_snapshot())
        await pending._save(pilot, screen, "character")
        stored = character_store.get_character_card_by_id(int(saved_id))
        assert screen.state.selected_entity_id == saved_id
        assert stored["image"] == avatar_files[1].read_bytes()
        assert stored["version"] == editor.get_character_data()["version"] == 2
        assert not editor.has_unsaved_attachment()
        assert not roleplay_has_unsaved_work(
            screen._aggregate_roleplay_draft_snapshot()
        )
