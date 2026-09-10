"""Mounted Buddy conversion review and cancellation contracts."""

from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Widgets.Persona_Widgets.buddy_character_review import (
    BuddyCharacterReviewDialog,
)


@pytest.mark.asyncio
async def test_review_keeps_edits_when_mapping_collision_is_reported(monkeypatch):
    snapshot = SimpleNamespace(
        title="Buddy", source_sha256="a" * 64, artwork=None, is_current=lambda: True
    )
    rows = [
        SimpleNamespace(
            source_state="idle", expression_key="neutral", fallback=False, frame_count=2
        ),
        SimpleNamespace(
            source_state="thinking",
            expression_key="thinking",
            fallback=True,
            frame_count=2,
        ),
    ]
    dialog = BuddyCharacterReviewDialog(
        snapshot,
        rows,
        db=object(),
        local_service=object(),
        profile_root=None,
        authority_guard=lambda: True,
        config={},
    )
    app = ConsolidatedCSSApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(dialog)
        await pilot.pause()
        assert dialog.query_one("#buddy-create", Button).disabled
        dialog.query_one("#buddy-mapping-1", Input).value = "neutral"
        assert dialog.collect_mappings() == {"idle": "neutral", "thinking": "neutral"}
        dialog.query_one("#buddy-mapping-1", Input).value = ""
        assert dialog.collect_mappings()["thinking"] is None
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not dialog


@pytest.mark.asyncio
async def test_saved_entry_is_disabled_for_dirty_draft_and_archive_posts_own_source():
    from Tests.UI.test_personas_persona_visual_pack import PackApp, _inventory
    from tldw_chatbook.Widgets.Persona_Widgets.personas_persona_visual_pack_widget import (
        PersonasPersonaVisualPackWidget,
    )

    class CaptureApp(PackApp):
        def __init__(self):
            super().__init__()
            self.requests = []

        def on_buddy_character_create_requested(self, message):
            self.requests.append(message.archive)

    app = CaptureApp()
    async with app.run_test(size=(90, 40)) as pilot:
        widget = app.query_one(PersonasPersonaVisualPackWidget)
        widget.show_inventory(_inventory(), dirty=True)
        await pilot.pause()
        assert widget.query_one(
            "#personas-persona-visual-create-character", Button
        ).disabled
        await pilot.click("#personas-persona-visual-character-archive")
        widget.show_inventory(_inventory(), dirty=False)
        await pilot.pause()
        await pilot.click("#personas-persona-visual-create-character")
        assert app.requests == [True, False]


@pytest.mark.asyncio
async def test_real_conversion_collision_static_preview_portrait_and_creation(tmp_path):
    from textual.widgets import Checkbox, Static, TextArea

    from Tests.Actor_Packs.test_actor_pack_attribution import services
    from Tests.Character_Chat.test_buddy_conversion import snapshot
    from tldw_chatbook.Character_Chat.buddy_conversion import suggest_buddy_mappings
    from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
    from tldw_chatbook.Widgets.Console.character_expression_avatar import (
        CharacterExpressionAvatar,
    )

    source = snapshot()
    rows = suggest_buddy_mappings(source)
    db, importer, _, _ = services(tmp_path / "profile")
    app = ConsolidatedCSSApp()
    app.console_character = "existing"
    app.buddy_enabled = True
    results = []
    dialog = BuddyCharacterReviewDialog(
        source,
        rows,
        db=db,
        local_service=importer._local_service,
        profile_root=tmp_path / "profile",
        authority_guard=lambda: True,
        config={"appearance": {"reduce_motion": True}},
    )
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(dialog, results.append)
            await pilot.pause()
            for index, row in enumerate(rows):
                dialog.query_one(f"#buddy-mapping-{index}", Input).value = (
                    "neutral" if row.source_state in {"idle", "speaking"} else ""
                )
            await pilot.click("#buddy-prepare")
            await app.workers.wait_for_complete()
            assert "collision" in str(
                dialog.query_one("#buddy-status", Static).renderable
            )
            assert dialog.query_one("#buddy-create", Button).disabled
            for index, row in enumerate(rows):
                if row.source_state == "speaking":
                    dialog.query_one(
                        f"#buddy-mapping-{index}", Input
                    ).value = "custom:speaking"
            dialog.query_one("#buddy-name", Input).value = "Independent Buddy"
            dialog.query_one("#buddy-personality", TextArea).text = "Patient companion."
            dialog.query_one("#buddy-greeting", TextArea).text = "Hello."
            dialog.query_one("#buddy-portrait-frame", Input).value = "2"
            await pilot.pause(0.6)
            await pilot.click("#buddy-prepare")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert dialog._conversion is not None, str(
                dialog.query_one("#buddy-status", Static).renderable
            )
            preview = dialog.query_one(
                "#buddy-expression-preview", CharacterExpressionAvatar
            )
            assert preview._animate is False  # Global reduce-motion wins over Dynamic.
            portrait = dialog.query_one(
                "#buddy-portrait-preview", CharacterExpressionAvatar
            )
            assert portrait._animate is False
            assert dialog._conversion.expressions[0].metadata["is_animated"]
            dialog.query_one("#buddy-warnings-accepted", Checkbox).value = True
            await pilot.pause()
            await pilot.click("#buddy-create")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert dialog._result is not None
            card = db.get_character_card_by_id(int(dialog._result.local_actor_id))
            assert card["name"] == "Independent Buddy"
            assert card["personality"] == "Patient companion."
            assert card["first_message"] == "Hello."
            assert VisualIdentityRepository(db).get_active_actor_pack(
                "character", dialog._result.local_actor_id
            )
            assert app.console_character == "existing" and app.buddy_enabled
            assert not results
            await pilot.click("#buddy-open")
            await pilot.pause()
            assert results[0].open_console is True
            assert preview._disposed and portrait._disposed
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_stale_destination_after_preview_cannot_publish(tmp_path):
    from textual.widgets import Static

    from Tests.Actor_Packs.test_actor_pack_attribution import services
    from Tests.Character_Chat.test_buddy_conversion import snapshot
    from tldw_chatbook.Character_Chat.buddy_conversion import suggest_buddy_mappings

    source = snapshot(same=True)
    db, importer, _, _ = services(tmp_path / "profile")
    current = [True]
    before = db.list_character_cards()
    dialog = BuddyCharacterReviewDialog(
        source,
        suggest_buddy_mappings(source),
        db=db,
        local_service=importer._local_service,
        profile_root=tmp_path / "profile",
        authority_guard=lambda: current[0],
        config={},
    )
    app = ConsolidatedCSSApp()
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(dialog)
            await pilot.click("#buddy-prepare")
            await app.workers.wait_for_complete()
            current[0] = False
            await pilot.click("#buddy-create")
            await app.workers.wait_for_complete()
            assert "Destination changed" in str(
                dialog.query_one("#buddy-status", Static).renderable
            )
            assert db.list_character_cards() == before
            await pilot.press("escape")
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "archive_source,stale_picker", [(True, False), (False, False), (True, True)]
)
async def test_entry_opens_created_character_through_personas_console_action(
    tmp_path, monkeypatch, mock_app_instance, archive_source, stale_picker
):
    from unittest.mock import AsyncMock, Mock

    from textual.screen import ModalScreen

    from Tests.Actor_Packs.test_actor_pack_attribution import services
    from Tests.Persona_Visual.test_persona_visual_importer import _write_archive
    from Tests.UI.test_personas_dictionaries import patch_character_paging
    from Tests.UI.test_personas_workbench import PROFILE, PersonasTestApp, _mounted
    from tldw_chatbook.UI.CCP_Modules import ccp_character_handler
    from tldw_chatbook.UI.Persona_Modules import buddy_conversion as workflow
    from tldw_chatbook.Widgets import enhanced_file_picker
    from tldw_chatbook.Widgets.Persona_Widgets.personas_persona_visual_pack_widget import (
        BuddyCharacterCreateRequested,
    )

    root = tmp_path / "profile"
    db, importer, _, _ = services(root)
    archive = _write_archive(tmp_path / "buddy.tldw-persona-vpack")
    scope = Mock()
    scope.local_service = importer._local_service
    record = {**PROFILE, "version": 2, "is_active": True, "deleted": False}
    scope.list_persona_profiles = AsyncMock(
        return_value={"items": [record], "total": 1}
    )
    scope.get_persona_profile = AsyncMock(return_value=record)
    mock_app_instance.character_persona_scope_service = scope
    mock_app_instance.chachanotes_db = db
    monkeypatch.setattr(workflow, "get_user_data_dir", lambda: root)
    monkeypatch.setattr(
        ccp_character_handler, "fetch_all_characters", db.list_character_cards
    )
    monkeypatch.setattr(
        ccp_character_handler,
        "fetch_character_by_id",
        lambda actor_id: db.get_character_card_by_id(int(actor_id)),
    )
    patch_character_paging(monkeypatch)

    class ChosenArchive(ModalScreen):
        def __init__(self, **kwargs):
            super().__init__()

        def on_mount(self):
            def choose():
                if stale_picker:
                    mock_app_instance.chachanotes_db = object()
                self.dismiss(str(archive))

            self.call_after_refresh(choose)

    monkeypatch.setattr(enhanced_file_picker, "EnhancedFileOpen", ChosenArchive)
    if not archive_source:
        import json

        import tldw_chatbook.UI.Screens.personas_screen as personas_module
        from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository
        from tldw_chatbook.Persona_Visual.snapshot import read_buddy_archive

        source = read_buddy_archive(archive)
        asset = source.assets[0]
        (root / "idle.png").write_bytes(asset.data)
        PersonaVisualRepository(db).activate_new_pack(
            persona_id="p-1",
            title="Saved Buddy",
            description="",
            source_kind="manual",
            source_context={},
            manifest=json.loads(source.manifest_json),
            manifest_storage_relpath="manifest.json",
            assets=[
                {
                    "asset_key": asset.metadata.asset_key,
                    "role": "frame",
                    "storage_relpath": "idle.png",
                    "mime_type": "image/png",
                    "bytes": len(asset.data),
                    "sha256": asset.metadata.sha256,
                    "width": 4,
                    "height": 5,
                    "frame_count": 1,
                    "duration_ms": None,
                }
            ],
            expected_persona_revision=2,
            authority_guard=lambda: True,
        )
        monkeypatch.setattr(personas_module, "get_user_data_dir", lambda: root)
    app = PersonasTestApp(mock_app_instance)
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            screen = await _mounted(pilot)
            if not archive_source:
                from Tests.UI.test_personas_persona_visual_authoring import _open_editor

                screen = await _open_editor(pilot)
                assert screen._persona_visual_authoring is not None
            monkeypatch.setattr(screen, "_provider_send_block_reason", lambda: None)
            original_selection = screen.state.selected_entity_id
            if archive_source:
                # Reach the native source from Characters, with no Persona selected.
                assert screen.state.active_mode == "characters"
                await pilot.click("#personas-library-import")
                if stale_picker:
                    await app.workers.wait_for_complete()
                    assert app.screen is screen
                    assert not screen._io_dialog_active
                    assert not any(
                        card["name"] == "Archive character"
                        for card in db.list_character_cards()
                    )
                    mock_app_instance.open_chat_with_handoff.assert_not_called()
                    return
            else:
                screen.post_message(BuddyCharacterCreateRequested())
            for _ in range(100):
                await pilot.pause(0.02)
                if isinstance(app.screen, BuddyCharacterReviewDialog):
                    break
            assert isinstance(app.screen, BuddyCharacterReviewDialog)
            dialog = app.screen
            await pilot.pause()
            dialog.query_one("#buddy-name", Input).value = "Archive character"
            assert await pilot.click("#buddy-prepare")
            while dialog._busy or dialog._conversion is None:
                await pilot.pause(0.02)
                if not dialog._busy:
                    break
            assert dialog._conversion is not None, str(
                dialog.query_one("#buddy-status").renderable
            )
            await pilot.click("#buddy-create")
            for _ in range(100):
                await pilot.pause(0.02)
                if dialog._result is not None:
                    break
            assert dialog._result is not None
            assert screen.state.selected_entity_id == original_selection
            mock_app_instance.open_chat_with_handoff.assert_not_called()
            created_id = dialog._result.local_actor_id
            await pilot.click("#buddy-open")
            for _ in range(100):
                await pilot.pause(0.02)
                if mock_app_instance.open_chat_with_handoff.called:
                    break
            assert screen.state.selected_entity_id == created_id
            mock_app_instance.open_chat_with_handoff.assert_called_once()
            payload = mock_app_instance.open_chat_with_handoff.call_args.args[0]
            assert payload.source_id == created_id
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_cancel_during_conversion_discards_late_result(tmp_path, monkeypatch):
    import threading

    from Tests.Actor_Packs.test_actor_pack_attribution import services
    from Tests.Character_Chat.test_buddy_conversion import snapshot
    from tldw_chatbook.Character_Chat import buddy_conversion

    source = snapshot()
    started = threading.Event()
    release = threading.Event()
    convert = buddy_conversion.convert_buddy

    def delayed_convert(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return convert(*args, **kwargs)

    monkeypatch.setattr(buddy_conversion, "convert_buddy", delayed_convert)
    db, importer, _, _ = services(tmp_path / "profile")
    before = db.list_character_cards()
    dialog = BuddyCharacterReviewDialog(
        source,
        buddy_conversion.suggest_buddy_mappings(source),
        db=db,
        local_service=importer._local_service,
        profile_root=tmp_path / "profile",
        authority_guard=lambda: True,
        config={},
    )
    app = ConsolidatedCSSApp()
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(dialog)
            await pilot.click("#buddy-prepare")
            for _ in range(50):
                if started.is_set():
                    break
                await pilot.pause(0.02)
            assert started.is_set()
            await pilot.press("escape")
            assert app.screen is not dialog
            release.set()
            await app.workers.wait_for_complete()
            assert db.list_character_cards() == before
            assert dialog._conversion is None
    finally:
        release.set()
        db.close_connection()


@pytest.mark.asyncio
async def test_all_conversion_warnings_are_scrollable_and_require_acknowledgment(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from textual.containers import VerticalScroll
    from textual.widgets import Checkbox, Static

    from Tests.Character_Chat.test_buddy_conversion import snapshot
    from tldw_chatbook.Character_Chat import buddy_conversion

    source = snapshot(same=True)
    converted = buddy_conversion.convert_buddy(source)
    warnings = tuple(
        f"State {index}: animated encoding unavailable; the static result uses the first source frame."
        for index in range(12)
    )
    converted = replace(converted, warnings=warnings)
    monkeypatch.setattr(
        buddy_conversion, "convert_buddy", lambda *args, **kwargs: converted
    )
    dialog = BuddyCharacterReviewDialog(
        source,
        buddy_conversion.suggest_buddy_mappings(source),
        db=object(),
        local_service=object(),
        profile_root=tmp_path,
        authority_guard=lambda: True,
        config={},
    )
    app = ConsolidatedCSSApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(dialog)
        await pilot.click("#buddy-prepare")
        await app.workers.wait_for_complete()
        notice = dialog.query_one("#buddy-warnings", Static)
        assert isinstance(notice.parent, VerticalScroll)
        assert str(notice.renderable) == "\n".join(warnings)
        assert notice.styles.height.is_auto
        assert dialog.query_one("#buddy-create", Button).disabled
        notice.scroll_visible(animate=False)
        await pilot.pause()
        dialog.query_one("#buddy-warnings-accepted", Checkbox).value = True
        await pilot.pause()
        assert not dialog.query_one("#buddy-create", Button).disabled
        await pilot.press("escape")


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_picker", [False, True])
async def test_unavailable_destination_releases_import_dialog_slot(
    monkeypatch, shared_picker
):
    from unittest.mock import Mock

    from tldw_chatbook.UI.Persona_Modules import buddy_conversion as workflow
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    def unavailable_root():
        raise OSError("profile unavailable")

    monkeypatch.setattr(workflow, "get_user_data_dir", unavailable_root)
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(),
        _io_dialog_active=True,
        _local_character_actions_allowed=lambda: True,
        _notify=Mock(),
    )
    if shared_picker:
        with pytest.raises(OSError, match="profile unavailable"):
            await PersonasScreen._import_dialog_worker(screen)
    else:
        await workflow.review_buddy_character(screen, archive=True)
        screen._notify.assert_called_once()
    assert not screen._io_dialog_active
