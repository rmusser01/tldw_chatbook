"""Qualified speculative-voice selection and view-lifecycle wiring."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from Tests.UI.test_console_dictation import FakeDictationSession, _mounted_console
from Tests.UI.test_console_hands_free_wiring import _ready_host
from Tests.UI.test_console_native_chat_flow import _ReadyResolutionGateway

from tldw_chatbook.Chat.console_hands_free import HandsFreeController
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.console_voice_controls import ControlKind
from tldw_chatbook.Chat import console_speculative_voice_session as session_module
from tldw_chatbook.UI.Console_Modules import dictation as dictation_module
from tldw_chatbook.UI.Console_Modules import hands_free as hands_free_module
from tldw_chatbook.Widgets.Console import (
    ConsoleTranscript,
    VoicePreviewProjection,
)
from textual.widgets import Switch


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["toggle", "navigation", "suspend", "unmount"])
async def test_visible_default_factory_selects_attested_private_process(
    monkeypatch, tmp_path, exit_kind
):
    from Tests.Chat.test_console_voice_process import fake_popen, cleanup
    from tldw_chatbook.Chat import console_voice_process as process_module
    from tldw_chatbook.Chat import console_voice_input
    from tldw_chatbook.Audio.voice_process_lifetime import source_identity

    launches = []
    failures = []
    created = []
    original_factory = session_module.create_console_speculative_voice_session

    def production_factory(**kwargs):
        try:
            return original_factory(**kwargs)
        except Exception as error:
            failures.append(error)
            raise

    monkeypatch.setattr(
        session_module, "create_console_speculative_voice_session", production_factory
    )
    original_init = process_module.ConsoleVoiceProcess.__init__

    def private_child(self, *args, **kwargs):
        kwargs["popen"] = fake_popen("normal", tmp_path / "events", launches)
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(process_module.ConsoleVoiceProcess, "__init__", private_child)
    monkeypatch.setattr(hands_free_module, "speculative_voice_qualified", lambda: True)
    monkeypatch.setattr(
        console_voice_input,
        "resolve",
        lambda: SimpleNamespace(provider="faster-whisper", model=None, language="en"),
    )
    app, host = _ready_host()
    monkeypatch.delattr(app, "console_speculative_voice_session_factory", raising=False)
    engine = None
    try:
        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            assert await pilot.click("#console-hands-free-switch")
            async with asyncio.timeout(5):
                while console._console_hands_free is None:
                    assert not failures, failures
                    assert not created or created[0]._failure is None, created[
                        0
                    ]._failure
                    await asyncio.sleep(0.01)
            selected = console._console_hands_free
            assert isinstance(
                selected, hands_free_module.ConsoleSpeculativeHandsFreeSession
            )
            engine = selected.engine
            assert isinstance(engine, process_module.ConsoleVoiceProcess)
            assert not isinstance(selected.controller, HandsFreeController)
            async with asyncio.timeout(8):
                while not engine.ready:
                    await asyncio.sleep(0.01)
            assert engine.process.pid != os.getpid()
            assert engine.hello_verified and launches == [True]
            assert engine._gate.identity == source_identity()
            assert (
                Path(engine._gate.identity.root) == Path(__file__).resolve().parents[2]
            )
            if exit_kind == "toggle":
                assert await pilot.click("#console-hands-free-switch")
            elif exit_kind == "navigation":
                assert await console.confirm_navigation()
            elif exit_kind == "suspend":
                console.on_screen_suspend()
            else:
                await console.remove()
            await engine.close()
            await engine.wait_effects_closed()
            assert engine.snapshot.phase == "off"
            assert engine.snapshot.fenced
            if exit_kind != "unmount":
                assert not console.query_one("#console-hands-free-switch", Switch).value
            if exit_kind == "toggle":
                original = engine
                await pilot.pause()
                assert await pilot.click("#console-hands-free-switch")
                async with asyncio.timeout(8):
                    while len(created) != 2 or not created[-1].ready:
                        await asyncio.sleep(0.01)
                engine = created[-1]
                assert (
                    engine is not original
                    and engine.process.pid != original.process.pid
                )
                assert engine.device_lease is original.device_lease
                assert await pilot.click("#console-hands-free-switch")
                await engine.close()
                await engine.wait_effects_closed()
    finally:
        if isinstance(engine, process_module.ConsoleVoiceProcess):
            await cleanup(engine)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["starting", "recording"])
async def test_qualified_entry_refuses_existing_legacy_capture(monkeypatch, state):
    monkeypatch.setattr(hands_free_module, "speculative_voice_qualified", lambda: True)
    app, host = _ready_host()
    launches = []
    app.console_speculative_voice_session_factory = lambda **kw: (
        launches.append(kw) or _QualifiedSession()
    )
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console._console_dictation_state = state
        await pilot.click("#console-hands-free-switch")
        await pilot.pause()
        assert launches == []
        assert console._console_dictation_state == state
        assert console._console_hands_free is None
        assert not console.query_one("#console-hands-free-switch", Switch).value


@pytest.fixture(autouse=True)
def ready_voice_provider(monkeypatch):
    """Wiring tests use a resolved provider; no availability network is needed."""
    monkeypatch.setattr(
        ConsoleProviderGateway,
        "resolve_for_send",
        _ReadyResolutionGateway.resolve_for_send,
    )


class _QualifiedSession:
    """Factory result whose close edge fences synchronously."""

    def __init__(self) -> None:
        self.entered_with: list[bool] = []
        self.close_reasons: list[ControlKind] = []

    def enter(self, *, capture_live: bool) -> None:
        self.entered_with.append(capture_live)

    def fence_and_close(self, reason: ControlKind) -> None:
        self.close_reasons.append(reason)


def _preview(*, epoch: int, text: str) -> VoicePreviewProjection:
    return VoicePreviewProjection(
        turn_id="qualified-turn",
        attempt_epoch=epoch,
        user_text=text,
        assistant_text="",
        status="listening",
    )


@pytest.mark.asyncio
async def test_hard_off_gate_keeps_legacy_hands_free_controller(monkeypatch):
    fake_dictation = FakeDictationSession()
    factory_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake_dictation,
    )
    monkeypatch.setattr(
        hands_free_module,
        "speculative_voice_qualified",
        lambda: False,
    )
    app, host = _ready_host()
    app.console_speculative_voice_session_factory = lambda **kwargs: (
        factory_calls.append(kwargs)
    )

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console.action_toggle_console_hands_free()
        await pilot.pause()

        assert console._console_hands_free is not None
        assert isinstance(
            console._console_hands_free.controller,
            HandsFreeController,
        )
        assert factory_calls == []


@pytest.mark.asyncio
async def test_qualified_path_uses_one_view_session_and_runtime_owners(monkeypatch):
    factory_calls: list[dict[str, Any]] = []
    session = _QualifiedSession()
    monkeypatch.setattr(
        hands_free_module,
        "speculative_voice_qualified",
        lambda: True,
    )
    app, host = _ready_host()

    def _factory(**kwargs: Any) -> _QualifiedSession:
        factory_calls.append(kwargs)
        return session

    app.console_speculative_voice_session_factory = _factory

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        runtime = console._console_runtime()
        promotion_owner = runtime.voice_promotion_owner
        dispatch_supervisor = runtime.voice_dispatch_supervisor

        console.action_toggle_console_hands_free()
        await pilot.pause()
        console._enter_console_hands_free_loop(capture_live=True)

        assert len(factory_calls) == 1
        assert factory_calls[0]["promotion_owner"] is promotion_owner
        assert factory_calls[0]["dispatch_supervisor"] is dispatch_supervisor
        assert callable(factory_calls[0]["project_preview"])
        assert callable(factory_calls[0]["clear_preview"])
        assert session.entered_with == [False, True]
        assert console._console_hands_free is not None
        assert not isinstance(
            console._console_hands_free.controller,
            HandsFreeController,
        )

        console.action_exit_console_hands_free()

        assert session.close_reasons == [ControlKind.ESCAPE]
        assert console._console_hands_free is None
        assert runtime.voice_promotion_owner is promotion_owner
        assert runtime.voice_dispatch_supervisor is dispatch_supervisor

        console.action_toggle_console_hands_free()
        await pilot.pause()
        console.action_toggle_console_hands_free()
        assert session.close_reasons[-1] is ControlKind.HANDS_FREE_EXIT

        console.action_toggle_console_hands_free()
        await pilot.pause()
        console._dictation._handle_console_dictation_button()
        assert session.close_reasons[-1] is ControlKind.MICROPHONE_DISABLED

        console.action_toggle_console_hands_free()
        await pilot.pause()
        console._console_hands_free.controller.on_stop_request()
        assert session.close_reasons[-1] is ControlKind.STOP


@pytest.mark.asyncio
async def test_qualified_path_uses_production_factory_when_no_test_seam_is_injected(
    monkeypatch,
):
    factory_calls: list[dict[str, Any]] = []
    session = _QualifiedSession()
    monkeypatch.setattr(
        hands_free_module,
        "speculative_voice_qualified",
        lambda: True,
    )
    monkeypatch.setattr(
        session_module,
        "create_console_speculative_voice_session",
        lambda **kwargs: factory_calls.append(kwargs) or session,
    )
    app, host = _ready_host()
    monkeypatch.delattr(
        app,
        "console_speculative_voice_session_factory",
        raising=False,
    )

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        runtime = console._console_runtime()

        console.action_toggle_console_hands_free()
        await pilot.pause()

        assert len(factory_calls) == 1
        assert factory_calls[0]["promotion_owner"] is runtime.voice_promotion_owner
        assert (
            factory_calls[0]["dispatch_supervisor"] is runtime.voice_dispatch_supervisor
        )
        assert session.entered_with == [False]


@pytest.mark.asyncio
async def test_cold_default_factory_load_keeps_visible_toggle_responsive(monkeypatch):
    session = _QualifiedSession()
    load_started = threading.Event()
    release_load = threading.Event()
    monkeypatch.setattr(
        hands_free_module,
        "speculative_voice_qualified",
        lambda: True,
    )
    monkeypatch.setattr(
        hands_free_module,
        "_load_default_speculative_voice_factory",
        lambda: (
            load_started.set(),
            release_load.wait(1),
            lambda **_kwargs: session,
        )[-1],
        raising=False,
    )
    monkeypatch.setattr(
        session_module,
        "create_console_speculative_voice_session",
        lambda **_kwargs: session,
    )
    app, host = _ready_host()
    monkeypatch.delattr(
        app,
        "console_speculative_voice_session_factory",
        raising=False,
    )

    try:
        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            console.action_toggle_console_hands_free()
            assert await asyncio.to_thread(load_started.wait, 1)
            await pilot.pause()

            switch = console.query_one("#console-hands-free-switch", Switch)
            assert switch.value is True
            assert console._console_hands_free is None
            startup_preview = console.query_one(ConsoleTranscript).query_one(
                "#console-voice-preview"
            )
            assert startup_preview.projection is not None
            assert startup_preview.projection.status == "preparing"

            release_load.set()
            async with asyncio.timeout(1):
                while console._console_hands_free is None:
                    await pilot.pause()

            assert session.entered_with == [False]
            assert startup_preview.projection is not None
            assert startup_preview.projection.status == "listening"
    finally:
        release_load.set()


@pytest.mark.asyncio
async def test_duplicate_visible_state_requests_do_not_spawn_replacement_sessions(
    monkeypatch,
):
    sessions: list[_QualifiedSession] = []
    monkeypatch.setattr(
        hands_free_module,
        "speculative_voice_qualified",
        lambda: True,
    )
    app, host = _ready_host()

    def _factory(**_kwargs: Any) -> _QualifiedSession:
        session = _QualifiedSession()
        sessions.append(session)
        return session

    app.console_speculative_voice_session_factory = _factory

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console._hands_free.request_console_hands_free_state(True)
        console._hands_free.request_console_hands_free_state(True)
        await pilot.pause()

        assert len(sessions) == 1
        assert sessions[0].entered_with == [False]

        console._hands_free.request_console_hands_free_state(False)
        console._hands_free.request_console_hands_free_state(False)

        assert sessions[0].close_reasons == [ControlKind.HANDS_FREE_EXIT]


@pytest.mark.asyncio
async def test_qualified_unmount_fences_view_session_not_runtime_owners(monkeypatch):
    session = _QualifiedSession()
    monkeypatch.setattr(
        hands_free_module,
        "speculative_voice_qualified",
        lambda: True,
    )
    app, host = _ready_host()
    app.console_speculative_voice_session_factory = lambda **_kwargs: session
    runtime = None
    promotion_owner = None
    dispatch_supervisor = None

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        runtime = console._console_runtime()
        promotion_owner = runtime.voice_promotion_owner
        dispatch_supervisor = runtime.voice_dispatch_supervisor
        console.action_toggle_console_hands_free()
        await pilot.pause()

    assert session.close_reasons == [ControlKind.TEARDOWN]
    assert runtime is not None
    assert runtime.voice_promotion_owner is promotion_owner
    assert runtime.voice_dispatch_supervisor is dispatch_supervisor


@pytest.mark.asyncio
async def test_replaced_qualified_session_cannot_repaint_or_clear_new_preview(
    monkeypatch,
):
    factory_calls: list[dict[str, Any]] = []
    sessions: list[_QualifiedSession] = []
    monkeypatch.setattr(
        hands_free_module,
        "speculative_voice_qualified",
        lambda: True,
    )
    app, host = _ready_host()

    def _factory(**kwargs: Any) -> _QualifiedSession:
        session = _QualifiedSession()
        sessions.append(session)
        factory_calls.append(kwargs)
        return session

    app.console_speculative_voice_session_factory = _factory

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        transcript = console.query_one(ConsoleTranscript)

        console.action_toggle_console_hands_free()
        await pilot.pause()
        first_callbacks = factory_calls[0]
        first_callbacks["project_preview"](_preview(epoch=1, text="first"))
        await pilot.pause()

        console.action_exit_console_hands_free()
        console.action_toggle_console_hands_free()
        await pilot.pause()
        second_callbacks = factory_calls[1]
        winning = _preview(epoch=2, text="second")
        second_callbacks["project_preview"](winning)

        first_callbacks["project_preview"](_preview(epoch=3, text="stale"))
        first_callbacks["clear_preview"]()
        await pilot.pause()

        assert transcript.query_one("#console-voice-preview").projection is winning
        assert sessions[0].close_reasons == [ControlKind.ESCAPE]


@pytest.mark.asyncio
async def test_qualified_enter_failure_clears_session_and_allows_retry(monkeypatch):
    sessions: list[_QualifiedSession] = []
    monkeypatch.setattr(
        hands_free_module,
        "speculative_voice_qualified",
        lambda: True,
    )
    app, host = _ready_host()

    class _BrokenEntry(_QualifiedSession):
        def enter(self, *, capture_live: bool) -> None:
            raise RuntimeError("startup failed")

    def _factory(**_kwargs: Any) -> _QualifiedSession:
        session: _QualifiedSession
        session = _BrokenEntry() if not sessions else _QualifiedSession()
        sessions.append(session)
        return session

    app.console_speculative_voice_session_factory = _factory

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)

        console.action_toggle_console_hands_free()
        await pilot.pause()

        assert console._console_hands_free is None
        assert sessions[0].close_reasons == [ControlKind.TEARDOWN]

        console.action_toggle_console_hands_free()
        await pilot.pause()
        assert console._console_hands_free is not None
        assert sessions[1].entered_with == [False]


@pytest.mark.asyncio
async def test_qualified_runtime_failure_closes_session_and_switch(monkeypatch):
    factory_calls: list[dict[str, Any]] = []
    session = _QualifiedSession()
    monkeypatch.setattr(
        hands_free_module,
        "speculative_voice_qualified",
        lambda: True,
    )
    app, host = _ready_host()

    def _factory(**kwargs: Any) -> _QualifiedSession:
        factory_calls.append(kwargs)
        return session

    app.console_speculative_voice_session_factory = _factory

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console.action_toggle_console_hands_free()
        await pilot.pause()

        factory_calls[0]["on_runtime_failure"](RuntimeError("audio pump failed"))
        await pilot.pause()

        assert console._console_hands_free is None
        assert session.close_reasons == [ControlKind.TEARDOWN]
        assert console.query_one("#console-hands-free-switch", Switch).value is False


@pytest.mark.asyncio
async def test_qualified_facade_tolerates_legacy_key_and_send_callbacks(monkeypatch):
    session = _QualifiedSession()
    monkeypatch.setattr(
        hands_free_module,
        "speculative_voice_qualified",
        lambda: True,
    )
    app, host = _ready_host()
    app.console_speculative_voice_session_factory = lambda **_kwargs: session

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console.action_toggle_console_hands_free()
        await pilot.pause()

        console._console_hands_free.controller.on_composer_key()
        console._hands_free._console_hands_free_force_immediate_send()

        assert console._console_hands_free is not None
        assert session.close_reasons == []


@pytest.mark.asyncio
async def test_stale_async_entry_failure_cannot_close_replacement(monkeypatch):
    release_failure = asyncio.Event()
    sessions: list[_QualifiedSession] = []
    monkeypatch.setattr(
        hands_free_module,
        "speculative_voice_qualified",
        lambda: True,
    )
    app, host = _ready_host()

    class _DelayedBrokenEntry(_QualifiedSession):
        async def enter(self, *, capture_live: bool) -> None:
            self.entered_with.append(capture_live)
            await release_failure.wait()
            raise RuntimeError("late startup failure")

    def _factory(**_kwargs: Any) -> _QualifiedSession:
        session: _QualifiedSession
        session = _DelayedBrokenEntry() if not sessions else _QualifiedSession()
        sessions.append(session)
        return session

    app.console_speculative_voice_session_factory = _factory

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console.action_toggle_console_hands_free()
        await pilot.pause()

        console.action_exit_console_hands_free()
        console.action_toggle_console_hands_free()
        await pilot.pause()
        replacement = console._console_hands_free
        release_failure.set()
        await pilot.pause()

        assert console._console_hands_free is replacement
        assert sessions[0].close_reasons == [ControlKind.ESCAPE]
        assert sessions[1].close_reasons == []


@pytest.mark.asyncio
async def test_confirmed_navigation_fences_qualified_view_before_unmount(monkeypatch):
    session = _QualifiedSession()
    monkeypatch.setattr(
        hands_free_module,
        "speculative_voice_qualified",
        lambda: True,
    )
    app, host = _ready_host()
    app.console_speculative_voice_session_factory = lambda **_kwargs: session

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console.action_toggle_console_hands_free()
        await pilot.pause()

        assert await console.confirm_navigation() is True

        assert session.close_reasons == [ControlKind.NAVIGATION]
        assert console._console_hands_free is None
