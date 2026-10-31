"""Pending saves acknowledge submitted forms while retaining later raw drafts."""

import asyncio
import threading

import pytest
from textual.widgets import Button, Input, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.test_personas_dictionaries import PersonasTestApp
from Tests.UI import test_personas_editor_save_in_place as save_in_place
from Tests.UI.test_personas_editor_save_in_place import _enter_personas_mode
from Tests.UI.roleplay_frame_harness import start_new
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.CCP_Modules import ccp_character_handler as character_handler
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    RoleplayDraftNavigationDialog,
)
from tldw_chatbook.UI.Persona_Modules.roleplay_frame_state import (
    roleplay_has_unsaved_work,
)
from tldw_chatbook.Widgets.confirmation_dialog import UnsavedChangesDialog
from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_editor_widget import (
    PersonaProfileEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)

pytestmark = pytest.mark.asyncio

real_persona_scope_service = save_in_place.real_persona_scope_service


@pytest.fixture
def character_store(tmp_path, monkeypatch):
    """Borrow one real SQLite owner and retire all of its thread handles."""
    database = CharactersRAGDB(tmp_path / "pending-save.db", "pending-save-test")
    monkeypatch.setattr(character_handler, "_default_character_db", lambda: database)
    yield database
    with database.quiesce_connections(timeout_seconds=5):
        pass


async def _until(predicate):
    async with asyncio.timeout(15):
        while not predicate():
            await asyncio.sleep(0.01)


async def _open(pilot, kind):
    if kind == "persona":
        screen = await _enter_personas_mode(pilot)
        await start_new(pilot, via="press")
    else:
        await pilot.pause()
        screen = pilot.app.screen
        await start_new(pilot, via="ctrl+n")
    await pilot.pause()
    editor_type = (
        PersonaProfileEditorWidget
        if kind == "persona"
        else PersonasCharacterEditorWidget
    )
    editor = screen.query_one(editor_type)
    selector = (
        "#personas-editor-name" if kind == "persona" else "#personas-char-editor-name"
    )
    return screen, editor, editor.query_one(selector, Input)


async def _save(pilot, screen, kind):
    await pilot.press("ctrl+s")
    await pilot.pause()
    if kind == "character":
        await screen._character_save_worker_handle.wait()
    else:
        await _until(lambda: not screen._profile_save_operation_inflight)
    await pilot.pause()


def _record(kind, screen, database, service):
    saved_id = screen.state.selected_entity_id
    if kind == "character":
        return database.get_character_card_by_id(int(saved_id))
    # A fresh file-backed service reads the durable store, independent of UI caches.
    from tldw_chatbook.Character_Chat.local_character_persona_service import (
        LocalCharacterPersonaService,
    )

    return LocalCharacterPersonaService(
        None, persona_store_path=service.local_service.persona_store_path
    ).get_persona_profile(str(saved_id))


def _hold_save(monkeypatch, kind, service, update, *, fail=False):
    """Delay the original persistence entry; successful releases delegate to it."""
    entered = threading.Event()
    release = threading.Event()
    if kind == "character":
        method = "update_character" if update else "create_character"
        original = getattr(character_handler, method)

        def held(*args, **kwargs):
            entered.set()
            assert release.wait(20), "test did not release the pending save"
            if fail:
                raise OSError("deliberate persistence failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(character_handler, method, held)
    else:
        method = "update_persona_profile" if update else "create_persona_profile"
        original = getattr(service, method)

        async def held(*args, **kwargs):
            entered.set()
            await _until(release.is_set)
            if fail:
                raise OSError("deliberate persistence failure")
            return await original(*args, **kwargs)

        monkeypatch.setattr(service, method, held)
    return entered, release


async def _stay_through_guards(pilot, screen, name):
    app = pilot.app
    draft = name.value
    # Actual New action must use the incumbent unsaved-changes modal.
    await start_new(pilot, via="press")
    await _until(lambda: isinstance(app.screen, UnsavedChangesDialog))
    await _until(lambda: bool(app.screen.query("#cancel-button")))
    app.screen.query_one("#cancel-button", Button).press()
    await _until(lambda: app.screen is screen and not screen._guard_active)
    assert name.value == draft
    for hook in (screen.confirm_navigation, screen.confirm_quit):
        worker = screen.run_worker(hook(), group="pending-save-guard-check")
        await _until(lambda: isinstance(app.screen, RoleplayDraftNavigationDialog))
        await _until(lambda: bool(app.screen.query("#roleplay-draft-stay")))
        app.screen.query_one("#roleplay-draft-stay", Button).press()
        assert await worker.wait() is False
        assert name.value == draft
        assert screen.state.has_unsaved_changes


@pytest.mark.parametrize("kind", ["character", "persona"])
@pytest.mark.parametrize("update", [False, True], ids=["create", "update"])
@private_profile_test
async def test_pending_save_retains_later_raw_draft_and_same_record_retry(
    request,
    mock_app_instance,
    character_store,
    real_persona_scope_service,
    monkeypatch,
    kind,
    update,
):
    mock_app_instance.character_persona_scope_service = real_persona_scope_service
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(160, 50)) as pilot:
        screen, editor, name = await _open(pilot, kind)
        name.focus()
        await pilot.press(*"Submitted")
        await pilot.pause()
        if update:
            await _save(pilot, screen, kind)
            assert not screen.state.has_unsaved_changes
            prior_id = screen.state.selected_entity_id
            await pilot.press("end", "-")
            await pilot.pause()
        else:
            prior_id = None
        submitted = name.value
        entered, release = _hold_save(
            monkeypatch, kind, real_persona_scope_service, update
        )
        saving = asyncio.create_task(pilot.press("ctrl+s"))
        try:
            await _until(entered.is_set)
            # Raw whitespace is retained even though collect() strips names.
            typing = asyncio.create_task(pilot.press("end", "space", "space"))
            await _until(lambda: name.value == submitted + "  ")
            assert roleplay_has_unsaved_work(
                screen._aggregate_roleplay_draft_snapshot()
            )
            if kind == "character":
                editor.query_one(
                    "#personas-char-editor-greeting-edit", TextArea
                ).text = "Later greeting"
                editor.query_one("#personas-char-editor-greeting-add", Button).press()
                await _until(
                    lambda: editor.get_character_data()["alternate_greetings"]
                    == ["Later greeting"]
                )
        finally:
            release.set()
        await asyncio.gather(saving, typing)
        if kind == "character":
            await screen._character_save_worker_handle.wait()
        else:
            await _until(lambda: not screen._profile_save_operation_inflight)
        await pilot.pause()
        saved = _record(kind, screen, character_store, real_persona_scope_service)
        assert saved["name"] == submitted.strip()
        assert saved["version"] == (2 if update else 1)
        saved_id = screen.state.selected_entity_id
        if prior_id is not None:
            assert saved_id == prior_id
        data = editor.collect() if kind == "persona" else editor.get_character_data()
        assert str(data["id"]) == str(saved_id)
        assert data["version"] == saved["version"]
        if kind == "character":
            assert saved["alternate_greetings"] == []
            assert data["alternate_greetings"] == ["Later greeting"]
        assert name.value == submitted + "  "
        assert (
            screen.state.has_unsaved_changes
        ), "later raw edits were falsely acknowledged"
        assert roleplay_has_unsaved_work(screen._aggregate_roleplay_draft_snapshot())
        assert editor._form_snapshot() != editor._loaded_snapshot
        await _stay_through_guards(pilot, screen, name)

        # Turn the retained draft into a distinct durable value and save it.
        name.focus()
        await pilot.press("end", "n", "e", "w")
        await pilot.pause()
        await _save(pilot, screen, kind)
        updated = _record(kind, screen, character_store, real_persona_scope_service)
        assert screen.state.selected_entity_id == saved_id
        assert updated["name"] == submitted + "  new"
        assert updated["version"] == saved["version"] + 1
        if kind == "character":
            assert updated["alternate_greetings"] == ["Later greeting"]
        assert not screen.state.has_unsaved_changes
        assert editor._form_snapshot() == editor._loaded_snapshot
        assert not roleplay_has_unsaved_work(
            screen._aggregate_roleplay_draft_snapshot()
        )


@pytest.mark.parametrize("kind", ["character", "persona"])
@pytest.mark.parametrize("update", [False, True], ids=["create", "update"])
@private_profile_test
async def test_failed_pending_save_preserves_draft_and_persisted_version(
    request,
    mock_app_instance,
    character_store,
    real_persona_scope_service,
    monkeypatch,
    kind,
    update,
):
    mock_app_instance.character_persona_scope_service = real_persona_scope_service
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(160, 50)) as pilot:
        screen, editor, name = await _open(pilot, kind)
        name.focus()
        await pilot.press(*"Original")
        await pilot.pause()
        if update:
            await _save(pilot, screen, kind)
            saved_id = screen.state.selected_entity_id
        else:
            saved_id = None
        name.focus()
        await pilot.press("end", "-")
        await pilot.pause()
        entered, release = _hold_save(
            monkeypatch, kind, real_persona_scope_service, update, fail=True
        )
        saving = asyncio.create_task(pilot.press("ctrl+s"))
        try:
            await _until(entered.is_set)
            typing = asyncio.create_task(pilot.press("end", "n", "e", "w"))
            await _until(lambda: name.value == "Original-new")
        finally:
            release.set()
        await asyncio.gather(saving, typing)
        if kind == "character":
            await screen._character_save_worker_handle.wait()
        else:
            await _until(lambda: not screen._profile_save_operation_inflight)
        await pilot.pause()
        assert name.value == "Original-new"
        assert screen.state.has_unsaved_changes
        assert roleplay_has_unsaved_work(screen._aggregate_roleplay_draft_snapshot())
        assert screen.state.selected_entity_id == saved_id
        if update:
            record = _record(kind, screen, character_store, real_persona_scope_service)
            assert record["name"] == "Original"
            assert record["version"] == 1
        else:
            assert screen._edit_mode == "create"
            assert character_store.get_character_card_by_name("Original-") is None
            assert not real_persona_scope_service.local_service.list_persona_profiles()


@pytest.mark.parametrize("kind", ["character", "persona"])
@private_profile_test
async def test_queued_save_acknowledges_widget_submission_before_handler_runs(
    request,
    mock_app_instance,
    character_store,
    real_persona_scope_service,
    monkeypatch,
    kind,
):
    """A delayed screen handler cannot capture a newer raw form as submitted."""
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
        CharacterSaveRequested,
        PersonaProfileSaveRequested,
    )

    message_type = (
        CharacterSaveRequested if kind == "character" else PersonaProfileSaveRequested
    )
    original = PersonasScreen._on_message
    entered = asyncio.Event()
    release = asyncio.Event()

    async def held_handler(self, message):
        if isinstance(message, message_type):
            entered.set()
            await asyncio.wait_for(release.wait(), 15)
        await original(self, message)

    monkeypatch.setattr(PersonasScreen, "_on_message", held_handler)
    mock_app_instance.character_persona_scope_service = real_persona_scope_service
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(160, 50)) as pilot:
        screen, editor, name = await _open(pilot, kind)
        name.focus()
        await pilot.press(*"Queued")
        await pilot.pause()
        saving = asyncio.create_task(pilot.press("ctrl+s"))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            typing = asyncio.create_task(pilot.press("end", "-", "n", "e", "w"))
            await _until(lambda: name.value == "Queued-new")
        finally:
            release.set()
        await asyncio.gather(saving, typing)
        if kind == "character":
            await screen._character_save_worker_handle.wait()
        else:
            await _until(lambda: not screen._profile_save_operation_inflight)
        await pilot.pause()
        saved = _record(kind, screen, character_store, real_persona_scope_service)
        assert saved["name"] == "Queued"
        assert saved["version"] == 1
        assert name.value == "Queued-new"
        assert screen.state.has_unsaved_changes, "handler acknowledged a later form"
        assert editor._form_snapshot() != editor._loaded_snapshot


@private_profile_test
async def test_persona_refresh_never_temporarily_acknowledges_later_draft(
    request, mock_app_instance, character_store, real_persona_scope_service, monkeypatch
):
    mock_app_instance.character_persona_scope_service = real_persona_scope_service
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(160, 50)) as pilot:
        screen, editor, name = await _open(pilot, "persona")
        name.focus()
        await pilot.press(*"Refreshing")
        await pilot.pause()
        entered, release = _hold_save(
            monkeypatch, "persona", real_persona_scope_service, False
        )
        refreshing = asyncio.Event()
        finish_refresh = asyncio.Event()
        original = screen._render_profile_rows

        async def held_refresh(*args, **kwargs):
            if screen._profile_save_operation_inflight and screen._edit_mode == "edit":
                refreshing.set()
                await asyncio.wait_for(finish_refresh.wait(), 15)
            return await original(*args, **kwargs)

        monkeypatch.setattr(screen, "_render_profile_rows", held_refresh)
        saving = asyncio.create_task(pilot.press("ctrl+s"))
        typing = None
        try:
            await _until(entered.is_set)
            typing = asyncio.create_task(pilot.press("end", "-", "n", "e", "w"))
            await _until(lambda: name.value == "Refreshing-new")
            release.set()
            await asyncio.wait_for(refreshing.wait(), 10)
            saved = _record(
                "persona", screen, character_store, real_persona_scope_service
            )
            assert saved["name"] == "Refreshing"
            assert (
                screen.state.has_unsaved_changes
            ), "refresh exposed a falsely clean draft"
        finally:
            release.set()
            finish_refresh.set()
            await saving
            if typing is not None:
                await typing
            await _until(lambda: not screen._profile_save_operation_inflight)
        assert name.value == "Refreshing-new"
        assert editor._form_snapshot() != editor._loaded_snapshot
