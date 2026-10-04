"""A slash command acts only on the Console chat it was sent from.

TASK-33622.16 review. Every slash command now runs in a Console worker
(``UI/Console_Modules/command_handoff.py``), so the Console keeps taking input
while a command awaits a prompt search, a health check or a URL resolution:
the user can switch chats or type a new draft before the command acts. A
command written when the send parked the pump re-read "the active chat" and
"the live draft" after that await, so it acted on whatever was showing then:

* ``/system <name>`` applied the saved prompt to the chat switched to, and
  cleared that chat's draft -- or, without a switch, wiped text typed after
  the keypress;
* ``/doctor``, ``/skills``, ``/fewer-permission-prompts`` and
  ``/stream-video`` posted their answer into the chat switched to;
* a command whose worker started after a switch ran against the new chat.

Each test parks the command at its own await seam with a gate, does what the
user can now do, then releases it. Keys are delivered the way the terminal
driver does (``_key``), as in ``test_console_video_send_freeze``.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from types import SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_hook_review_send_freeze import _key, _until
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_command_grammar import KIND_COMMAND, CommandParse
from tldw_chatbook.Media_Playback import stream_resolve
from tldw_chatbook.UI.Console_Modules.command_handoff import (
    COMMAND_WORKER_GROUP,
    run_console_command,
)
from tldw_chatbook.Utils import doctor

pytestmark = pytest.mark.bootstrap_profile

SYSTEM_DRAFT = "/system Terse"
TERSE = {
    "name": "Terse",
    "system_prompt": "Answer tersely.",
    "user_prompt": "",
    "artifact_type": "prompt",
}
TYPED = "what about a sailboat"
ROUTES = ["enter", "send-button"]


@contextlib.asynccontextmanager
async def _console():
    """Run the real ``TldwCli`` on its Console and yield ``(app, console)``."""
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    async with app.run_test(size=(140, 44)) as pilot:
        assert await _until(
            lambda: (
                type(app.screen).__name__ == "ChatScreen"
                and bool(app.screen.query("#console-native-composer"))
            ),
            15,
        ), "the Console composer never mounted"
        await pilot.pause(0.2)
        yield app, app.screen


def _record_toasts(monkeypatch, app) -> list[tuple[str, str | None]]:
    toasts: list[tuple[str, str | None]] = []
    original = app.notify

    def _notify(message, *args, **kwargs):
        toasts.append((str(message), kwargs.get("severity")))
        return original(message, *args, **kwargs)

    monkeypatch.setattr(app, "notify", _notify)
    return toasts


async def _load(console, draft: str) -> None:
    composer = console._console_composer_or_none()
    composer.load_draft(draft)
    composer.focus()
    await asyncio.sleep(0.1)
    assert composer.draft_text() == draft


def _send(route: str, app, console) -> None:
    if route == "send-button":
        console.query_one("#console-send-message", Button).press()
    else:
        _key(app, "enter", "\r")


def _type(app, text: str) -> None:
    for char in text:
        _key(app, "space" if char == " " else char, char)


async def _type_into(app, composer, text: str) -> None:
    composer.focus()
    await asyncio.sleep(0.1)
    _type(app, text)
    assert await _until(lambda: composer.draft_text() == text, 5), (
        f"typing never reached the composer: {composer.draft_text()!r}"
    )


async def _switch_to_new_chat(console) -> str:
    other = console._ensure_console_chat_controller().new_session().id
    await console._sync_native_console_chat_ui()
    assert await _until(
        lambda: console._console_visible_draft_session_id == other, 5
    ), "the Console never switched to the new chat"
    return other


def _commands_idle(console) -> bool:
    return not [
        worker
        for worker in console.workers
        if worker.group == COMMAND_WORKER_GROUP and not worker.is_finished
    ]


def _system_rows(store, session_id: str) -> list[str]:
    return [
        message.content
        for message in store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.SYSTEM
    ]


def _system_prompt(store, session_id: str):
    settings = store.session_settings(session_id)
    return getattr(settings, "system_prompt", None)


class _AsyncGate:
    """Parks an awaited seam until the test releases it."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.entered = threading.Event()
        self.released = threading.Event()
        self._result = result
        self._error = error

    async def __call__(self, *_args, **_kwargs):
        self.entered.set()
        while not self.released.is_set():
            await asyncio.sleep(0.02)
        if self._error is not None:
            raise self._error
        return self._result


class _ThreadGate:
    """Parks a seam that runs in ``asyncio.to_thread``."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.entered = threading.Event()
        self.released = threading.Event()
        self._result = result
        self._error = error

    def __call__(self, *_args, **_kwargs):
        self.entered.set()
        self.released.wait(30)
        if self._error is not None:
            raise self._error
        return self._result


def _gate_prompt_search(monkeypatch, console) -> _AsyncGate:
    gate = _AsyncGate(result=[TERSE])
    monkeypatch.setattr(console._prompts, "_console_prompt_search", gate)
    return gate


# -- /system <name> -----------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
async def test_a_named_system_prompt_never_lands_on_the_chat_switched_to(
    route, monkeypatch
):
    """Review (Qodo #1/#2): ``/system <name>`` applied its prompt to the chat
    active once the search returned, and cleared that chat's draft. Neither
    chat changes, the other chat's draft stays, the command stays in its own
    chat's draft, and the refusal says so."""
    async with _console() as (app, console):
        toasts = _record_toasts(monkeypatch, app)
        gate = _gate_prompt_search(monkeypatch, console)
        try:
            store = console._ensure_console_chat_store()
            await _load(console, SYSTEM_DRAFT)
            _send(route, app, console)
            assert await _until(gate.entered.is_set, 10), "/system never searched"
            origin = store.active_session_id
            origin_before = _system_prompt(store, origin)
            composer = console._console_composer_or_none()
            other = await _switch_to_new_chat(console)
            other_before = _system_prompt(store, other)
            await _type_into(app, composer, TYPED)
            gate.released.set()
            assert await _until(lambda: _commands_idle(console), 10)
            await asyncio.sleep(0.3)
            assert _system_prompt(store, other) == other_before, (
                f"{route}: /system applied to the chat switched to"
            )
            assert _system_prompt(store, origin) == origin_before
            assert composer.draft_text() == TYPED, (
                f"{route}: /system cleared the other chat's draft: "
                f"{composer.draft_text()!r}"
            )
            assert store.session_draft(origin) == SYSTEM_DRAFT
            assert any(
                "/system" in message and severity == "warning"
                for message, severity in toasts
            ), f"{route}: the refusal was silent: {toasts}"
        finally:
            gate.released.set()


@pytest.mark.parametrize("edit", ["appended", "replaced"])
async def test_a_named_system_prompt_takes_only_the_draft_its_send_captured(
    edit, monkeypatch
):
    """Review (Qodo #2): with no switch, the prompt applies -- but its success
    path cleared the live composer, wiping what was typed after Enter. Only
    the captured ``/system Terse`` goes; later text stays."""
    async with _console() as (app, console):
        gate = _gate_prompt_search(monkeypatch, console)
        try:
            store = console._ensure_console_chat_store()
            await _load(console, SYSTEM_DRAFT)
            _send("enter", app, console)
            assert await _until(gate.entered.is_set, 10), "/system never searched"
            origin = store.active_session_id
            composer = console._console_composer_or_none()
            if edit == "appended":
                _type(app, " " + TYPED)
                expected = " " + TYPED
                assert await _until(
                    lambda: composer.draft_text() == SYSTEM_DRAFT + expected, 5
                )
            else:
                _key(app, "ctrl+u")
                assert await _until(lambda: composer.draft_text() == "", 5)
                await _type_into(app, composer, TYPED)
                expected = TYPED
            gate.released.set()
            assert await _until(lambda: _commands_idle(console), 10)
            await asyncio.sleep(0.3)
            assert _system_prompt(store, origin) == "Answer tersely."
            assert composer.draft_text() == expected, (
                f"{edit}: /system took text typed after its send: "
                f"{composer.draft_text()!r}"
            )
        finally:
            gate.released.set()


# -- output rows ----------------------------------------------------------------


def _doctor(monkeypatch, _app, _console):
    gate = _ThreadGate(result=[])
    monkeypatch.setattr(doctor, "run_doctor", gate)
    return gate


def _skills(monkeypatch, _app, console):
    gate = _AsyncGate(result={})
    monkeypatch.setattr(console._skill, "_fetch_console_skill_context", gate)
    return gate


def _fewer_permission_prompts(monkeypatch, app, _console):
    gate = _AsyncGate(error=RuntimeError("recommendations unavailable"))
    monkeypatch.setattr(
        app,
        "unified_mcp_service",
        SimpleNamespace(permission_prompt_recommendations=gate),
        raising=False,
    )
    return gate


def _stream_video(monkeypatch, _app, _console):
    gate = _ThreadGate(error=stream_resolve.StreamResolutionError("not a stream"))
    monkeypatch.setattr(stream_resolve, "resolve_stream_url", gate)
    return gate


OUTPUT_COMMANDS = [
    pytest.param("/doctor", _doctor, id="doctor"),
    pytest.param("/skills", _skills, id="skills"),
    pytest.param(
        "/fewer-permission-prompts",
        _fewer_permission_prompts,
        id="fewer-permission-prompts",
    ),
    pytest.param(
        "/stream-video https://example.com/clip.mp4", _stream_video, id="stream-video"
    ),
]


@pytest.mark.parametrize(("draft", "install_gate"), OUTPUT_COMMANDS)
async def test_a_command_answers_in_the_chat_it_was_sent_from(
    draft, install_gate, monkeypatch
):
    """A command that awaits before it answers posted its row into whichever
    chat was showing when the await returned. It lands in the chat it was
    sent from; the chat switched to gets nothing."""
    async with _console() as (app, console):
        gate = install_gate(monkeypatch, app, console)
        try:
            store = console._ensure_console_chat_store()
            await _load(console, draft)
            # A bare command has the popup open, where Enter accepts the
            # completion instead of sending; the Send button always sends.
            _send("send-button", app, console)
            assert await _until(gate.entered.is_set, 10), f"{draft} never ran"
            origin = store.active_session_id
            origin_rows = len(_system_rows(store, origin))
            other = await _switch_to_new_chat(console)
            other_rows = _system_rows(store, other)
            gate.released.set()
            assert await _until(lambda: _commands_idle(console), 10)
            # The answer is posted after the worker returns; wait for the row
            # itself rather than a fixed settle (flaked under load).
            await _until(lambda: len(_system_rows(store, origin)) > origin_rows, 10)
            await asyncio.sleep(0.3)
            assert _system_rows(store, other) == other_rows, (
                f"{draft} answered in the chat switched to: "
                f"{_system_rows(store, other)}"
            )
            assert len(_system_rows(store, origin)) == origin_rows + 1, (
                f"{draft} never answered in its own chat: {_system_rows(store, origin)}"
            )
        finally:
            gate.released.set()


# -- the hand-off's own origin check ------------------------------------------


async def test_a_command_never_starts_in_a_chat_it_was_not_sent_from(monkeypatch):
    """The hand-off runs a command on a later loop step than its send. If the
    chat it came from is no longer the one showing by then, it does not run
    -- in either chat -- and says so; from the chat showing, it runs."""
    async with _console() as (app, console):
        toasts = _record_toasts(monkeypatch, app)
        dispatched: list[str] = []

        async def _record(parse) -> None:
            dispatched.append(parse.name)

        monkeypatch.setattr(console, "_dispatch_console_command", _record)
        origin = console._console_visible_send_session_id()
        other = await _switch_to_new_chat(console)
        parse = CommandParse(kind=KIND_COMMAND, name="help", args="")

        run_console_command(console, parse, origin, "/help")
        assert await _until(lambda: _commands_idle(console), 5)
        await asyncio.sleep(0.2)
        assert dispatched == [], "a command ran in a chat it was not sent from"
        assert any(
            "/help" in message and severity == "warning"
            for message, severity in toasts
        ), f"the refusal was silent: {toasts}"

        run_console_command(console, parse, other, "/help")
        assert await _until(lambda: dispatched == ["help"], 5), (
            "a command sent from the chat showing never ran"
        )
