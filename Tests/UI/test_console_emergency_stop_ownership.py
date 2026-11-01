"""Stop writes retain their captured input and physical mutation ownership."""

from __future__ import annotations

import asyncio
import contextlib
import threading

import pytest

import tldw_chatbook.app  # noqa: F401
from Tests.private_profile import private_profile_test
from Tests.UI.test_console_command_origin_chat import (
    _commands_idle,
    _load,
    _record_toasts,
    _switch_to_new_chat,
    _type_into,
)
from Tests.UI.test_console_emergency_stop_view import _Screen
from Tests.UI.test_console_composer_run_controls import _mounted
from Tests.UI.test_console_hook_review_send_freeze import _key, _until
from Tests.UI.test_console_send_transition_layout_requests import _console_host
from tldw_chatbook.Chat.console_command_grammar import KIND_COMMAND, CommandParse
from tldw_chatbook.emergency_stop import (
    EmergencyStopState,
    read_emergency_stop,
    set_emergency_stop,
)
from tldw_chatbook.UI.Console_Modules import emergency_stop_hold as stop
from tldw_chatbook.UI.Console_Modules.command_handoff import run_console_command

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@contextlib.asynccontextmanager
async def _console(tmp_path):
    """Mount the real ChatScreen without unrelated whole-app startup workers."""
    _, host = _console_host(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        console, _ = await _mounted(host, pilot)
        yield host, console


def _handoff(console, active: bool) -> None:
    composer = console._console_composer_or_none()
    run_console_command(
        console,
        CommandParse(KIND_COMMAND, "emergency-stop", "freeze" if active else "clear"),
        console._ensure_console_chat_store().active_session_id,
        composer.capture_draft_for_send(),
    )


def _hold_write(monkeypatch, active: bool):
    entered, release = threading.Event(), threading.Event()
    name = "set_emergency_stop" if active else "clear_emergency_stop"
    original = getattr(stop, name)

    def write(*args, **kwargs):
        entered.set()
        assert release.wait(30), "test did not release its physical stop write"
        return original(*args, **kwargs)

    monkeypatch.setattr(stop, name, write)
    return entered, release


@pytest.mark.parametrize("active", [True, False], ids=["set", "clear"])
@pytest.mark.parametrize("edit", ["unchanged", "replaced", "retyped", "chat-switch"])
@private_profile_test
async def test_completed_command_consumes_only_its_capture(
    monkeypatch, request, tmp_path, active, edit
):
    path = tmp_path / "stop.json"
    if not active:
        set_emergency_stop(path, reason="previous hold")
    monkeypatch.setattr(stop, "default_emergency_stop_path", lambda: path)
    entered, release = _hold_write(monkeypatch, active)
    draft = "/emergency-stop freeze" if active else "/emergency-stop clear"
    async with _console(tmp_path) as (app, console):
        console._console_emergency_stop_path = path
        await _load(console, draft)
        composer = console._console_composer_or_none()
        captured = composer.capture_draft_for_send()
        _handoff(console, active)
        try:
            assert await _until(entered.is_set, 10), "command did not reach its writer"
            expected = ""
            if edit == "chat-switch":
                await _switch_to_new_chat(console)
                expected = "the other chat draft"
                await _type_into(app, composer, expected)
            elif edit != "unchanged":
                _key(app, "ctrl+u")
                assert await _until(lambda: composer.draft_text() == "", 5)
                expected = draft if edit == "retyped" else "newer unsent text"
                await _type_into(app, composer, expected)
                assert composer.edit_serial != captured.edit_serial
            release.set()
            assert await _until(lambda: _commands_idle(console), 10)
            assert read_emergency_stop(path).active is active
            assert (
                composer.draft_text() == expected
            ), "completed command erased newer input"
        finally:
            release.set()
            await _until(lambda: _commands_idle(console), 10)


@pytest.mark.parametrize(
    ("first_active", "second_route"),
    [(False, "command"), (True, "command"), (True, "chip")],
)
@private_profile_test
async def test_competing_command_and_chip_mutations_are_refused(
    monkeypatch, request, tmp_path, first_active, second_route
):
    path = tmp_path / "stop.json"
    if not first_active:
        set_emergency_stop(path, reason="previous hold")
    monkeypatch.setattr(stop, "default_emergency_stop_path", lambda: path)
    entered, release = _hold_write(monkeypatch, first_active)
    first = "/emergency-stop freeze" if first_active else "/emergency-stop clear"
    second = "/emergency-stop clear" if first_active else "/emergency-stop freeze"
    async with _console(tmp_path) as (app, console):
        console._console_emergency_stop_path = path
        notices = _record_toasts(monkeypatch, console.app_instance)
        await _load(console, first)
        _handoff(console, first_active)
        try:
            assert await _until(entered.is_set, 10)
            await _load(console, second)
            if second_route == "chip":
                await stop.clear_from_strip(console)
            else:
                _handoff(console, not first_active)
            assert await _until(lambda: bool(notices), 5)
            assert any("already being changed" in text for text, _ in notices), notices
            assert not any(
                text in (stop.ACTIVATED_NOTICE, stop.CLEARED_NOTICE)
                for text, _ in notices
            ), "a concurrent mutation claimed success"
            release.set()
            assert await _until(lambda: _commands_idle(console), 10)
            assert read_emergency_stop(path).active is first_active
            assert stop.held(console) is first_active
            assert console._console_composer_or_none().draft_text() == second
        finally:
            release.set()
            await _until(lambda: _commands_idle(console), 10)


async def test_repeated_cancel_retains_physical_write_across_screen_instances(
    tmp_path, monkeypatch
):
    path = tmp_path / "stop.json"
    set_emergency_stop(path, reason="previous hold")
    entered, release = _hold_write(monkeypatch, False)
    first, second = _Screen(), _Screen()
    second.app_instance = first.app_instance
    for screen in (first, second):
        screen._console_emergency_stop_path = path
        screen._console_emergency_stop_copy = stop.HELD_COPY
    write = asyncio.create_task(stop.clear_from_strip(first))
    try:
        assert await _until(entered.is_set, 5)
        for _ in range(2):
            write.cancel()
            await asyncio.sleep(0)
        assert not write.done(), "cancellation retired a still-running native write"
        result = await stop._write_off_loop(second, EmergencyStopState(active=True))
        assert result is False
        assert any("already being changed" in text for text in first.notices)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await write
        assert not read_emergency_stop(path).active
        assert not stop.held(first)
        assert stop.CLEARED_NOTICE not in first.notices
        assert await stop._write_off_loop(second, EmergencyStopState(active=True))
        assert read_emergency_stop(path).active
        assert stop.held(second)
    finally:
        release.set()
        await asyncio.gather(write, return_exceptions=True)
