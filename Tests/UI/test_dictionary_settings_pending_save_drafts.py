"""Dictionary saves acknowledge submitted settings, preserving newer raw drafts."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from textual import events
from textual.widgets import (
    Button,
    Input,
    Select,
    Static,
    Switch,
    TabbedContent,
    TextArea,
)

from Tests.private_profile import private_profile_test
from Tests.UI.roleplay_frame_harness import (
    PersonasTestApp,
    select_roleplay_item,
    start_new,
    visible_item_ids,
)
from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import (
    ChatDictionaryScopeService,
)
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import (
    LocalChatDictionaryService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.CCP_Modules import ccp_character_handler
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    RoleplayDraftNavigationDialog,
)
from tldw_chatbook.UI.Persona_Modules.roleplay_frame_state import (
    roleplay_has_unsaved_work,
)
from tldw_chatbook.Widgets.confirmation_dialog import UnsavedChangesDialog
from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_detail import (
    PersonasDictionaryDetailWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
    PersonasLibraryPane,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(90)]


async def _until(predicate):
    async with asyncio.timeout(20):
        while not predicate():
            await asyncio.sleep(0.01)


@asynccontextmanager
async def _dictionary_app(mock_app_instance, tmp_path, monkeypatch):
    db = CharactersRAGDB(tmp_path / "dictionary.db", "pending-dictionary-settings")
    monkeypatch.setattr(ccp_character_handler, "_default_character_db", lambda: db)
    local = LocalChatDictionaryService(db, history_store_path=tmp_path / "history.json")
    service = ChatDictionaryScopeService(local_service=local, server_service=None)
    record = await service.create_dictionary(
        {"name": "Initial", "description": "base"}, mode="local"
    )
    mock_app_instance.chat_dictionary_scope_service = service
    mock_app_instance.character_persona_scope_service = None
    app = PersonasTestApp(mock_app_instance)
    try:
        async with app.run_test(size=(160, 60)) as pilot:
            await pilot.pause()
            screen = app.screen
            await pilot.click("#personas-mode-dictionaries")
            await _until(lambda: screen.state.active_mode == "dictionaries")
            library = screen.query_one(PersonasLibraryPane)
            option_id = f"dictionary:{record['id']}"
            # The previous mode's options can remain while dictionaries load.
            await _until(lambda: option_id in visible_item_ids(library))
            await select_roleplay_item(pilot, library, option_id)
            await _until(
                lambda: screen.state.selected_entity_kind == "dictionary"
                and screen.state.selected_entity_id == str(record["id"])
                and bool(screen.query(PersonasDictionaryDetailWidget))
            )
            await pilot.pause()
            detail = screen.query_one(PersonasDictionaryDetailWidget)
            detail.query_one(
                "#personas-dict-tabs", TabbedContent
            ).active = "personas-dict-tab-settings"
            await pilot.pause()
            yield pilot, screen, detail, service, record
    finally:
        with db.quiesce_connections(timeout_seconds=5):
            pass


def _dirty(screen):
    return roleplay_has_unsaved_work(screen._aggregate_roleplay_draft_snapshot())


async def _submitted(pilot, detail):
    name = detail.query_one("#personas-dict-name", Input)
    name.focus()
    name.action_select_all()
    await pilot.press(*"Submitted")
    await pilot.pause()
    assert name.value == "Submitted"


def _hold(monkeypatch, service, *, fail=False):
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = service.update_dictionary
    calls = []

    async def held(*args, **kwargs):
        calls.append((args, kwargs))
        entered.set()
        await release.wait()
        try:
            if fail:
                raise OSError("deliberate dictionary write failure")
            return await original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(service, "update_dictionary", held)
    return entered, release, finished, calls, original


async def _later_input(pilot, detail, field):
    """Witness child delivery without requiring the blocked screen pump to settle."""
    if field in ("name", "raw-name", "reverted-name"):
        widget = detail.query_one("#personas-dict-name", Input)
        widget.focus()
        if field == "reverted-name":
            widget.action_select_all()
            task = asyncio.create_task(pilot.press(*"Initial"))
            expected = "Initial"
        else:
            task = asyncio.create_task(
                pilot.press("end", *("-later" if field == "name" else "  "))
            )
            expected = "Submitted-later" if field == "name" else "Submitted  "
        await _until(lambda: widget.value == expected)
        return task, lambda: widget.value, expected
    if field in ("budget", "valid-budget"):
        widget = detail.query_one("#personas-dict-max-tokens", Input)
        if field == "valid-budget":
            widget.action_select_all()
            for character in "2048":
                widget.post_message(events.Key(character, character))
            expected = "2048"
        else:
            widget.cursor_position = len(widget.value)
            widget.post_message(events.Key("x", "x"))
            expected = "1000x"
        await _until(lambda: widget.value == expected)
        return None, lambda: widget.value, expected
    if field == "description":
        widget = detail.query_one("#personas-dict-description", TextArea)
        widget.move_cursor((0, len(widget.text)))
        for character in " later":
            widget.post_message(
                events.Key("space" if character == " " else character, character)
            )
        await _until(lambda: widget.text == "base later")
        return None, lambda: widget.text, "base later"
    if field == "enabled":
        widget = detail.query_one("#personas-dict-enabled", Switch)
        widget.action_toggle_switch()
        await _until(lambda: widget.value is False)
        return None, lambda: widget.value, False
    widget = detail.query_one("#personas-dict-strategy", Select)
    widget.action_show_overlay()
    overlay = widget.query_one("SelectOverlay")
    overlay.action_cursor_down()
    overlay.action_select()
    await _until(lambda: widget.value == "character_lore_first")
    return None, lambda: widget.value, "character_lore_first"


async def _stay_through_guards(pilot, screen, current, expected):
    await start_new(pilot, via="press")
    await _until(lambda: isinstance(pilot.app.screen, UnsavedChangesDialog))
    await _until(lambda: bool(pilot.app.screen.query("#cancel-button")))
    pilot.app.screen.query_one("#cancel-button", Button).press()
    await _until(lambda: pilot.app.screen is screen and not screen._guard_active)
    assert current() == expected
    for hook in (screen.confirm_navigation, screen.confirm_quit):
        worker = screen.run_worker(hook(), group="dictionary-pending-guard")
        await _until(
            lambda: isinstance(pilot.app.screen, RoleplayDraftNavigationDialog)
        )
        await _until(lambda: bool(pilot.app.screen.query("#roleplay-draft-stay")))
        pilot.app.screen.query_one("#roleplay-draft-stay", Button).press()
        assert await worker.wait() is False
        assert current() == expected
        assert screen.state.has_unsaved_changes and _dirty(screen)


@pytest.mark.parametrize(
    "field",
    [
        "name",
        "raw-name",
        "description",
        "budget",
        "strategy",
        "enabled",
        "queued-name",
        "valid-budget",
        "reverted-name",
    ],
)
@private_profile_test
async def test_pending_dictionary_save_retains_raw_child_input_and_next_save(
    request, mock_app_instance, tmp_path, monkeypatch, field
):
    async with _dictionary_app(mock_app_instance, tmp_path, monkeypatch) as (
        pilot,
        screen,
        detail,
        service,
        record,
    ):
        await _submitted(pilot, detail)
        entered, release, finished, calls, original = _hold(monkeypatch, service)
        changed_queued, deliver_changed = asyncio.Event(), asyncio.Event()
        if field == "queued-name":
            original_on_message = detail._on_message

            async def hold_changed(message):
                if (
                    isinstance(message, Input.Changed)
                    and message.input.id == "personas-dict-name"
                    and message.value == "Submitted-later"
                ):
                    changed_queued.set()
                    await deliver_changed.wait()
                await original_on_message(message)

            monkeypatch.setattr(detail, "_on_message", hold_changed)
        detail.query_one("#personas-dict-settings-save", Button).press()
        typing = None
        try:
            await _until(entered.is_set)
            typing, current, expected = await _later_input(
                pilot, detail, "name" if field == "queued-name" else field
            )
            # This observation occurs while the parent awaits persistence. The
            # child's pilot task may still await pump settlement; do not join it.
            if field == "queued-name":
                await _until(changed_queued.is_set)
            if field == "reverted-name":
                await _until(lambda: detail._last_dirty_sent is False)
            assert not release.is_set() and not finished.is_set()
            assert current() == expected
            print(
                f"CHILD_ACCEPTED field={field} value={expected!r} save_pending=True",
                flush=True,
            )
        finally:
            release.set()
        await _until(finished.is_set)
        if field == "queued-name":
            await _until(lambda: screen._selected_dictionary_version == 2)
            assert not deliver_changed.is_set()
            deliver_changed.set()
        if typing is not None:
            await typing
        await pilot.pause()
        stored = await service.get_dictionary(record["id"], mode="local")
        assert stored["name"] == "Submitted" and stored["version"] == 2
        assert stored["description"] == "base" and stored["max_tokens"] == 1000
        assert stored["strategy"] == "sorted_evenly" and stored["enabled"] is True
        assert (
            current() == expected
        ), "earlier Save overwrote later raw Dictionary Settings input"
        assert screen._selected_dictionary_version == 2
        assert screen.state.has_unsaved_changes and _dirty(screen)
        if field == "raw-name":
            await _stay_through_guards(pilot, screen, current, expected)
        detail.query_one("#personas-dict-settings-save", Button).press()
        await pilot.pause()
        await _until(lambda: not screen.state.has_unsaved_changes)
        stored = await service.get_dictionary(record["id"], mode="local")
        assert (
            stored["version"] == 3
            and str(stored["id"]) == screen.state.selected_entity_id
        )
        assert screen._selected_dictionary_version == 3
        assert stored["name"] == (
            "Submitted-later"
            if field in ("name", "queued-name")
            else "Initial"
            if field == "reverted-name"
            else "Submitted"
        )
        assert stored["description"] == (
            "base later" if field == "description" else "base"
        )
        assert stored["strategy"] == (
            "character_lore_first" if field == "strategy" else "sorted_evenly"
        )
        assert stored["enabled"] is (field != "enabled")
        assert stored["max_tokens"] == (2048 if field == "valid-budget" else 1000)
        assert not _dirty(screen)
        assert [call[1]["expected_version"] for call in calls] == [1, 2]
        assert [call[0][0] for call in calls] == [record["id"], record["id"]]


@private_profile_test
async def test_dictionary_input_before_save_is_persisted_and_clean(
    request, mock_app_instance, tmp_path, monkeypatch
):
    async with _dictionary_app(mock_app_instance, tmp_path, monkeypatch) as (
        pilot,
        screen,
        detail,
        service,
        record,
    ):
        await _submitted(pilot, detail)
        typing, current, expected = await _later_input(pilot, detail, "name")
        await typing
        await pilot.pause()
        assert screen.state.has_unsaved_changes
        detail.query_one("#personas-dict-settings-save", Button).press()
        await pilot.pause()
        await _until(lambda: not screen.state.has_unsaved_changes)
        stored = await service.get_dictionary(record["id"], mode="local")
        assert stored["name"] == expected and stored["version"] == 2
        assert current() == expected and screen._selected_dictionary_version == 2
        assert not _dirty(screen)


@private_profile_test
async def test_dictionary_definite_failure_preserves_live_draft_for_retry(
    request, mock_app_instance, tmp_path, monkeypatch
):
    async with _dictionary_app(mock_app_instance, tmp_path, monkeypatch) as (
        pilot,
        screen,
        detail,
        service,
        record,
    ):
        await _submitted(pilot, detail)
        entered, release, finished, calls, original = _hold(
            monkeypatch, service, fail=True
        )
        detail.query_one("#personas-dict-settings-save", Button).press()
        typing = None
        try:
            await _until(entered.is_set)
            typing, current, expected = await _later_input(pilot, detail, "raw-name")
            assert not finished.is_set()
        finally:
            release.set()
        await _until(finished.is_set)
        if typing is not None:
            await typing
        await pilot.pause()
        assert "Save failed:" in str(
            detail.query_one("#personas-dict-status", Static).render()
        )
        stored = await service.get_dictionary(record["id"], mode="local")
        assert stored["name"] == "Initial" and stored["version"] == 1
        assert current() == expected and screen._selected_dictionary_version == 1
        assert screen.state.has_unsaved_changes and _dirty(screen)
        monkeypatch.setattr(service, "update_dictionary", original)
        detail.query_one("#personas-dict-settings-save", Button).press()
        await pilot.pause()
        await _until(lambda: not screen.state.has_unsaved_changes)
        stored = await service.get_dictionary(record["id"], mode="local")
        assert stored["name"] == "Submitted" and stored["version"] == 2
        assert screen._selected_dictionary_version == 2 and not _dirty(screen)


@private_profile_test
async def test_pending_dictionary_save_without_later_edit_becomes_clean(
    request, mock_app_instance, tmp_path, monkeypatch
):
    async with _dictionary_app(mock_app_instance, tmp_path, monkeypatch) as (
        pilot,
        screen,
        detail,
        service,
        record,
    ):
        await _submitted(pilot, detail)
        entered, release, finished, calls, original = _hold(monkeypatch, service)
        detail.query_one("#personas-dict-settings-save", Button).press()
        try:
            await _until(entered.is_set)
            assert screen.state.has_unsaved_changes and not finished.is_set()
        finally:
            release.set()
        await _until(finished.is_set)
        await pilot.pause()
        stored = await service.get_dictionary(record["id"], mode="local")
        assert stored["name"] == "Submitted" and stored["version"] == 2
        assert screen._selected_dictionary_version == 2
        assert detail.query_one("#personas-dict-name", Input).value == "Submitted"
        assert not screen.state.has_unsaved_changes and not _dirty(screen)


@private_profile_test
async def test_dictionary_save_survives_library_pane_removed_during_completion(
    request, mock_app_instance, tmp_path, monkeypatch
):
    """A committed Save must not kill the screen when its list pane disappears."""
    from tldw_chatbook.UI.Persona_Modules import persona_save_completion

    async with _dictionary_app(mock_app_instance, tmp_path, monkeypatch) as (
        pilot,
        screen,
        detail,
        service,
        record,
    ):
        await _submitted(pilot, detail)
        entered, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original_render = screen._render_dictionary_rows
        original_completion = persona_save_completion.finish_dictionary_settings_save

        async def held_render(*args, **kwargs):
            await original_render(*args, **kwargs)
            if screen._selected_dictionary_version == 2 and not entered.is_set():
                entered.set()
                await release.wait()

        async def observed_completion(*args, **kwargs):
            try:
                return await original_completion(*args, **kwargs)
            finally:
                completed.set()

        monkeypatch.setattr(screen, "_render_dictionary_rows", held_render)
        monkeypatch.setattr(
            persona_save_completion,
            "finish_dictionary_settings_save",
            observed_completion,
        )
        detail.query_one("#personas-dict-settings-save", Button).press()
        try:
            await _until(entered.is_set)
            stored = await service.get_dictionary(record["id"], mode="local")
            assert stored["name"] == "Submitted" and stored["version"] == 2
            library = screen.query_one(PersonasLibraryPane)
            removal = library.remove()
            await _until(lambda: not screen.query(PersonasLibraryPane))
            assert not completed.is_set()
            print(
                "DICTIONARY_COMMITTED_V2_LIBRARY_PANE_REMOVED_COMPLETION_HELD",
                flush=True,
            )
        finally:
            release.set()
        await _until(completed.is_set)
        await removal
        await pilot.pause()
        assert screen.is_mounted and detail.is_mounted
        name = detail.query_one("#personas-dict-name", Input)
        name.focus()
        await pilot.press("end", "!")
        await _until(lambda: name.value == "Submitted!")
        assert screen.state.has_unsaved_changes and _dirty(screen)
        stored = await service.get_dictionary(record["id"], mode="local")
        assert stored["name"] == "Submitted" and stored["version"] == 2
