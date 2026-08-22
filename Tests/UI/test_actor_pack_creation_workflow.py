"""Mounted Workbench coverage for pack-ready actor creation."""

# Imported pytest fixtures are intentionally rebound as test parameters.
# ruff: noqa: F811

from __future__ import annotations

import asyncio
import hashlib
from io import BytesIO
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from textual.widgets import Button, Input, Select, Static, TextArea
from textual.widgets._select import SelectCurrent

from Tests.UI.test_personas_workbench import (
    PersonasTestApp,
    StyledPersonasTestApp,
    _mounted,
    stub_characters as _stub_characters,  # noqa: F401 - registers fixture
    stub_scope_service as _stub_scope_service,  # noqa: F401 - registers fixture
)
from tldw_chatbook.Actor_Packs.creation import (
    ActorPackCreationError,
    ActorPackCreationResult,
)
from tldw_chatbook.UI.Screens import personas_screen as personas_screen_module
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)


pytestmark = pytest.mark.asyncio


def _portrait_bytes(color: tuple[int, int, int, int] = (20, 40, 60, 255)) -> bytes:
    portrait = BytesIO()
    Image.new("RGBA", (2, 2), color).save(portrait, format="PNG")
    return portrait.getvalue()


def _character_rows() -> list[dict[str, object]]:
    return [
        {
            "id": 7,
            "name": "Portrait Host",
            "image": _portrait_bytes(),
            "version": 3,
            "deleted": 0,
        },
        {"id": 8, "name": "No Portrait", "image": None, "version": 1},
    ]


async def test_new_actor_pack_is_distinct_and_reuses_character_editor(
    mock_app_instance, _stub_characters
) -> None:
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.click("#personas-library-new-actor-pack")
        await pilot.pause()

        assert screen.query_one("#ccp-character-editor-view").display is True
        assert "New Character Actor Pack" in str(
            screen.query_one("#personas-char-editor-title", Static).renderable
        )
        portrait = screen.query_one("#personas-char-editor-pack-status", Static)
        assert portrait.display is True
        assert "Portrait required" in str(portrait.renderable)
        assert pilot.app.focused is not None
        assert pilot.app.focused.id == "personas-char-editor-name"

        await screen._begin_create_character()
        assert "Character Editor" in str(
            screen.query_one("#personas-char-editor-title", Static).renderable
        )
        assert portrait.display is False


async def test_persona_actor_pack_uses_labelled_local_portrait_selector(
    mock_app_instance, _stub_characters, _stub_scope_service
) -> None:
    mock_app_instance.chachanotes_db = SimpleNamespace(
        list_character_cards=Mock(return_value=_character_rows())
    )
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.click("#personas-mode-personas")
        await pilot.pause()
        await pilot.click("#personas-library-new-actor-pack")
        await pilot.pause()

        assert "New Persona Actor Pack" in str(
            screen.query_one("#personas-editor-title", Static).renderable
        )
        portrait = screen.query_one("#personas-editor-pack-portrait")
        assert portrait.display is True
        selector = screen.query_one("#personas-editor-character-portrait", Select)
        assert selector.value == 7
        assert "Portrait Host" in str(
            selector.query_one(SelectCurrent).query_one("#label", Static).renderable
        )
        assert pilot.app.focused is not None
        assert pilot.app.focused.id == "personas-editor-name"

        await screen._begin_create_profile()
        assert portrait.display is False
        assert selector.value is Select.NULL
        assert (
            "character_card_id"
            not in screen.query_one("#ccp-persona-editor-view").collect()
        )


async def test_server_persona_actor_pack_explains_local_copy_requirement(
    mock_app_instance, _stub_characters, _stub_scope_service
) -> None:
    app = PersonasTestApp(mock_app_instance)
    notifications: list[tuple[str, str, float | None]] = []
    app.notify = lambda message, severity="information", timeout=None, **_: (
        notifications.append((str(message), severity, timeout))
    )
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.click("#personas-mode-personas")
        await screen.handle_runtime_backend_changed("server")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert screen.query_one("#personas-library-new-actor-pack", Button).display
        await screen._begin_create_actor_pack()

        assert notifications[-1] == ("Save a local copy first", "warning", None)
        assert screen.query_one("#ccp-persona-editor-view").display is False


@pytest.mark.parametrize("size", [(80, 24), (160, 48)])
async def test_actor_pack_action_is_keyboard_focusable_inside_supported_layouts(
    mock_app_instance, _stub_characters, size: tuple[int, int]
) -> None:
    app = StyledPersonasTestApp(mock_app_instance)
    async with app.run_test(size=size) as pilot:
        screen = await _mounted(pilot)
        button = screen.query_one("#personas-library-new-actor-pack", Button)
        assert button.region.width > 0 and button.region.height > 0
        assert button.region.right <= size[0]
        button.focus()
        await pilot.pause()
        assert pilot.app.focused is button


async def test_character_actor_pack_save_uses_atomic_creation_service_once(
    mock_app_instance, _stub_characters, monkeypatch
) -> None:
    service = Mock()
    service.create_character.return_value = ActorPackCreationResult(
        "character", "77", "123e4567-e89b-42d3-a456-426614174000"
    )
    mock_app_instance.actor_pack_creation_service = service
    monkeypatch.setattr(
        personas_screen_module.ccp_character_handler,
        "fetch_character_by_id",
        lambda actor_id: {
            "id": int(actor_id),
            "name": "Pack Hero",
            "version": 1,
            "image": _character_rows()[0]["image"],
        },
    )
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.click("#personas-library-new-actor-pack")
        editor = screen.query_one(PersonasCharacterEditorWidget)
        screen.query_one("#personas-char-editor-name", Input).value = "Pack Hero"
        screen.query_one(
            "#personas-char-editor-first-message", TextArea
        ).text = "Welcome"
        editor.set_avatar_image(_character_rows()[0]["image"])
        await pilot.click("#personas-char-editor-save")
        await app.workers.wait_for_complete()
        await pilot.pause()

        service.create_character.assert_called_once()
        kwargs = service.create_character.call_args.kwargs
        assert kwargs["portrait_name"] == "portrait.png"
        assert kwargs["portrait_bytes"] == _character_rows()[0]["image"]
        request = service.create_character.call_args.args[0]
        assert "image" not in request
        assert request["first_message"] == "Welcome"
        assert "first_mes" not in request
        assert "Portable UUID:" in str(
            screen.query_one("#personas-char-editor-pack-status", Static).renderable
        )


async def test_persona_actor_pack_save_pins_selected_portrait_identity(
    mock_app_instance, _stub_characters, _stub_scope_service
) -> None:
    rows = _character_rows()
    mock_app_instance.chachanotes_db = SimpleNamespace(
        list_character_cards=Mock(return_value=rows)
    )
    service = Mock()
    service.create_persona.return_value = ActorPackCreationResult(
        "persona", "local-pack", "223e4567-e89b-42d3-a456-426614174000"
    )
    mock_app_instance.actor_pack_creation_service = service
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.click("#personas-mode-personas")
        await pilot.click("#personas-library-new-actor-pack")
        screen.query_one("#personas-editor-name", Input).value = "Pack Guide"
        await pilot.click("#personas-editor-save")
        await app.workers.wait_for_complete()
        await pilot.pause()

        service.create_persona.assert_called_once()
        args, kwargs = service.create_persona.call_args
        assert args[0]["character_card_id"] == 7
        assert kwargs["source"] == "local"
        assert kwargs["expected_portrait_revision"] == 3
        assert (
            kwargs["expected_portrait_sha256"]
            == hashlib.sha256(rows[0]["image"]).hexdigest()
        )
        assert "Portable UUID:" in str(
            screen.query_one("#personas-editor-pack-status", Static).renderable
        )


async def test_navigation_signals_and_drains_pack_creation_before_continuing(
    mock_app_instance, _stub_characters
) -> None:
    started = threading.Event()
    finished = threading.Event()
    cancel_observed = threading.Event()
    calls = 0

    def create_character(*args, cancel_requested, **kwargs):
        nonlocal calls
        calls += 1
        started.set()
        deadline = time.monotonic() + 0.75
        while not cancel_requested() and time.monotonic() < deadline:
            threading.Event().wait(0.005)
        finished.set()
        if cancel_requested():
            cancel_observed.set()
        raise ActorPackCreationError("actor_pack_creation_cancelled")

    mock_app_instance.actor_pack_creation_service = SimpleNamespace(
        create_character=create_character
    )
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.click("#personas-library-new-actor-pack")
        editor = screen.query_one(PersonasCharacterEditorWidget)
        screen.query_one("#personas-char-editor-name", Input).value = "Cancelled"
        editor.set_avatar_image(_character_rows()[0]["image"])
        await pilot.click("#personas-char-editor-save")
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        await pilot.click("#personas-char-editor-save")
        for _ in range(20):
            if calls > 1:
                break
            await asyncio.sleep(0.01)

        screen.state.has_unsaved_changes = False
        continued: list[bool] = []

        async def continuation() -> None:
            assert finished.is_set()
            continued.append(True)

        await screen._run_guarded(continuation)
        assert calls == 1
        assert cancel_observed.is_set()
        assert continued == [True]
        assert screen._actor_pack_operation_task is None


async def test_declined_dirty_navigation_preserves_actor_pack_draft(
    mock_app_instance, _stub_characters
) -> None:
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.click("#personas-library-new-actor-pack")
        session = screen._actor_pack_session
        screen.state.has_unsaved_changes = True

        async def decline() -> bool:
            return False

        screen._confirm_discard_unsaved = decline
        continued: list[bool] = []

        async def continuation() -> None:
            continued.append(True)

        await screen._run_guarded(continuation)
        await app.workers.wait_for_complete()
        assert continued == []
        assert screen._actor_pack_session is session


async def test_stale_editor_session_is_rejected_before_pack_service_call(
    mock_app_instance, _stub_characters, _stub_scope_service
) -> None:
    rows = _character_rows()
    mock_app_instance.chachanotes_db = SimpleNamespace(
        list_character_cards=Mock(return_value=rows)
    )
    service = Mock()
    mock_app_instance.actor_pack_creation_service = service
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.click("#personas-mode-personas")
        await pilot.click("#personas-library-new-actor-pack")
        editor = screen.query_one("#ccp-persona-editor-view")
        editor.load_persona({}, runtime_source="local")
        screen.query_one("#personas-editor-name", Input).value = "Stale"
        screen.post_message(
            personas_screen_module.PersonaProfileSaveRequested(
                {"name": "Stale", "character_card_id": 7}
            )
        )
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert service.create_persona.call_count == 0


async def test_stale_pack_result_does_not_reconcile_into_newer_editor_authority(
    mock_app_instance, _stub_characters, monkeypatch
) -> None:
    started = threading.Event()
    release = threading.Event()

    def create_character(*args, **kwargs):
        started.set()
        release.wait(timeout=3)
        return ActorPackCreationResult(
            "character", "88", "323e4567-e89b-42d3-a456-426614174000"
        )

    mock_app_instance.actor_pack_creation_service = SimpleNamespace(
        create_character=create_character
    )
    monkeypatch.setattr(
        personas_screen_module.ccp_character_handler,
        "fetch_character_by_id",
        lambda actor_id: {
            "id": int(actor_id),
            "name": "Stale result",
            "version": 1,
            "image": _character_rows()[0]["image"],
        },
    )
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.click("#personas-library-new-actor-pack")
        editor = screen.query_one(PersonasCharacterEditorWidget)
        screen.query_one("#personas-char-editor-name", Input).value = "Stale result"
        editor.set_avatar_image(_character_rows()[0]["image"])
        await pilot.click("#personas-char-editor-save")
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        screen._actor_pack_generation += 1
        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert "Portable UUID:" not in str(
            screen.query_one("#personas-char-editor-pack-status", Static).renderable
        )


async def test_runtime_source_change_fences_inflight_portrait_inventory(
    mock_app_instance, _stub_characters, _stub_scope_service
) -> None:
    started = threading.Event()
    release = threading.Event()

    def list_cards(**kwargs):
        started.set()
        release.wait(timeout=3)
        return _character_rows()

    mock_app_instance.chachanotes_db = SimpleNamespace(list_character_cards=list_cards)
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.click("#personas-mode-personas")
        opening = asyncio.create_task(screen._begin_create_actor_pack())
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        await screen.handle_runtime_backend_changed("server")
        release.set()
        await opening

        assert screen._actor_pack_session is None
        assert screen.query_one("#ccp-persona-editor-view").display is False


async def test_cancel_signals_and_drains_owned_pack_operation(
    mock_app_instance, _stub_characters
) -> None:
    started = threading.Event()
    cancelled = threading.Event()

    def create_character(*args, cancel_requested, **kwargs):
        started.set()
        while not cancel_requested():
            threading.Event().wait(0.005)
        cancelled.set()
        raise ActorPackCreationError("actor_pack_creation_cancelled")

    mock_app_instance.actor_pack_creation_service = SimpleNamespace(
        create_character=create_character
    )
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.click("#personas-library-new-actor-pack")
        editor = screen.query_one(PersonasCharacterEditorWidget)
        screen.query_one("#personas-char-editor-name", Input).value = "Cancel me"
        editor.set_avatar_image(_character_rows()[0]["image"])
        await pilot.click("#personas-char-editor-save")
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()

        async def confirm() -> bool:
            return True

        screen._confirm_discard_unsaved = confirm
        await pilot.click("#personas-char-editor-cancel")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert cancelled.is_set()
        assert screen._actor_pack_operation_task is None
        assert screen._actor_pack_session is None
        assert screen.query_one("#ccp-character-editor-view").display is False


async def test_pack_error_copy_is_plain_text_and_path_free(
    mock_app_instance, _stub_characters
) -> None:
    service = Mock()
    service.create_character.side_effect = ActorPackCreationError(
        "actor_pack_creation_failed", user_message="/Users/private/portrait.png"
    )
    mock_app_instance.actor_pack_creation_service = service
    app = StyledPersonasTestApp(mock_app_instance)
    notifications: list[str] = []
    app.notify = lambda message, **_: notifications.append(str(message))
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _mounted(pilot)
        await pilot.click("#personas-library-new-actor-pack")
        editor = screen.query_one(PersonasCharacterEditorWidget)
        screen.query_one("#personas-char-editor-name", Input).value = "Private"
        editor.set_avatar_image(_character_rows()[0]["image"])
        status = screen.query_one("#personas-char-editor-pack-status", Static)
        assert status.region.width > 0 and status.region.height > 0
        await pilot.click("#personas-char-editor-save")
        await app.workers.wait_for_complete()

        assert notifications[-1] == "Actor Pack creation failed."
        assert "/Users" not in notifications[-1]


async def test_portrait_change_during_commit_signals_authority_and_skips_reconcile(
    mock_app_instance, _stub_characters, tmp_path
) -> None:
    started = threading.Event()
    authority_lost = threading.Event()

    def create_character(*args, cancel_requested, authority_guard, **kwargs):
        started.set()
        deadline = time.monotonic() + 1
        while (
            authority_guard() and not cancel_requested() and time.monotonic() < deadline
        ):
            threading.Event().wait(0.005)
        if not authority_guard() or cancel_requested():
            authority_lost.set()
        raise ActorPackCreationError("actor_pack_creation_authority_changed")

    mock_app_instance.actor_pack_creation_service = SimpleNamespace(
        create_character=create_character
    )
    replacement = tmp_path / "replacement.png"
    replacement.write_bytes(_portrait_bytes((80, 10, 20, 255)))
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.click("#personas-library-new-actor-pack")
        editor = screen.query_one(PersonasCharacterEditorWidget)
        screen.query_one("#personas-char-editor-name", Input).value = "Changing"
        editor.set_avatar_image(_portrait_bytes())
        await pilot.click("#personas-char-editor-save")
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()

        await screen._stage_character_avatar_from_path(str(replacement))
        await app.workers.wait_for_complete()
        assert authority_lost.is_set()
        assert "Portable UUID:" not in str(
            screen.query_one("#personas-char-editor-pack-status", Static).renderable
        )
