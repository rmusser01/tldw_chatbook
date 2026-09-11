"""Provider readiness through the mounted Hands-free control, without audio."""

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual.widgets import Switch

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from Tests.UI.test_console_speculative_voice_wiring import _QualifiedSession
from tldw_chatbook.UI.Console_Modules import hands_free as hands_free_module
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


SECRET = "secret-token https://private.invalid/provider voice_provider_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result", ["unavailable", "missing", "unknown", "timeout", "ready"]
)
async def test_visible_control_validates_before_factory_and_preserves_draft(
    monkeypatch, result
):
    monkeypatch.setattr(hands_free_module, "speculative_voice_qualified", lambda: True)
    app, host = _ready_host()
    calls = []
    engine = _QualifiedSession()
    app.console_speculative_voice_session_factory = lambda **kw: (
        calls.append(kw) or engine
    )
    notices = []
    app.notify = lambda message, **kw: notices.append(message)
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        composer = console.query_one(ConsoleComposerBar)
        probe_calls = []

        async def resolve(selection):
            probe_calls.append(selection)
            if result == "unknown":
                raise RuntimeError(SECRET)
            if result == "timeout":
                await asyncio.Future()
            return SimpleNamespace(ready=result == "ready", visible_copy=SECRET)

        monkeypatch.setattr(controller.provider_gateway, "resolve_for_send", resolve)
        controller.PROVIDER_VALIDATION_TIMEOUT_SECONDS = 0.01
        if result == "missing":
            # Simulate ownership disappearing at the real readiness read. A
            # global navigation to None also rebuilds/clears the composer and
            # tests ordinary session switching, not refusal's draft behavior.
            validate = controller.validate_speculative_voice_entry

            async def missing_owner():
                owner = controller.store.active_session_id
                controller.store.active_session_id = None
                try:
                    return await validate()
                finally:
                    controller.store.active_session_id = owner

            monkeypatch.setattr(
                controller, "validate_speculative_voice_entry", missing_owner
            )
        composer.load_draft("keep this draft exactly")
        notices.clear()
        await pilot.click("#console-hands-free-switch")
        await pilot.pause()
        async with asyncio.timeout(2):
            while console._hands_free._qualified_voice_startup_generation is not None:
                await pilot.pause()
        switch = console.query_one("#console-hands-free-switch", Switch)
        assert composer.draft_text() == "keep this draft exactly"
        assert len(probe_calls) == (0 if result == "missing" else 1)
        if result == "ready":
            assert switch.value and len(calls) == 1
            assert engine.entered_with == [False]
            assert isinstance(
                console._console_hands_free,
                hands_free_module.ConsoleSpeculativeHandsFreeSession,
            )
            assert notices == []
        else:
            assert calls == []
            assert engine.entered_with == []
            assert not switch.value
            assert console._console_hands_free is None
            assert len(notices) == 1
            assert SECRET not in notices[0]
            assert (
                "provider" in notices[0].lower() or "conversation" in notices[0].lower()
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "off",
        "teardown",
        "session",
        "session_aba",
        "provider",
        "model",
        "settings_aba",
        "config",
        "config_aba",
        "replacement",
        "legacy_starting",
        "legacy_recording",
    ],
)
@pytest.mark.parametrize("fails", [False, True])
async def test_pending_readiness_cannot_enter_or_notify_after_owner_changes(
    monkeypatch, change, fails
):
    monkeypatch.setattr(hands_free_module, "speculative_voice_qualified", lambda: True)
    app, host = _ready_host()
    calls = []
    engine = _QualifiedSession()
    app.console_speculative_voice_session_factory = lambda **kw: (
        calls.append(kw) or engine
    )
    notices = []
    app.notify = lambda message, **kw: notices.append(message)
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        store = controller.store

        async def resolve(selection):
            if entered.is_set():
                return SimpleNamespace(ready=True)
            entered.set()
            await release.wait()
            finished.set()
            if fails:
                raise RuntimeError(SECRET)
            return SimpleNamespace(ready=True)

        monkeypatch.setattr(controller.provider_gateway, "resolve_for_send", resolve)
        try:
            await pilot.click("#console-hands-free-switch")
            await asyncio.wait_for(entered.wait(), 0.5)
            if change in {"off", "replacement"}:
                await pilot.click("#console-hands-free-switch")
                if change == "replacement":
                    await pilot.click("#console-hands-free-switch")
                    await pilot.pause()
                    assert engine.entered_with == [False]
            elif change == "teardown":
                console._hands_free.teardown()
            elif change.startswith("legacy_"):
                console._console_dictation_state = change.removeprefix("legacy_")
            elif change.startswith("session"):
                old_id = store.active_session_id
                store.create_session()
                if change == "session_aba":
                    store._activate_session(old_id)
            elif change.startswith("config"):
                from tldw_chatbook import config

                original = config.get_cli_setting(
                    "dictation", "response_eagerness_ms", 700
                )
                assert config.save_setting_to_cli_config(
                    "dictation", "response_eagerness_ms", 800
                )
                if change == "config_aba":
                    assert config.save_setting_to_cli_config(
                        "dictation", "response_eagerness_ms", original
                    )
            else:
                session_id = store.active_session_id
                original = store.session_settings(session_id)
                changed = replace(
                    original,
                    **(
                        {"provider": "deepseek"}
                        if change == "provider"
                        else {"model": "changed-model"}
                    ),
                )
                store.replace_session_settings(session_id, changed)
                if change == "settings_aba":
                    store.replace_session_settings(session_id, original)
            notices.clear()
            release.set()
            await asyncio.wait_for(finished.wait(), 0.5)
            await pilot.pause()
            assert notices == []
            assert len(calls) == (1 if change == "replacement" else 0)
            if change.startswith("legacy_"):
                assert console._console_dictation_state == change.removeprefix(
                    "legacy_"
                )
            assert console.query_one("#console-hands-free-switch", Switch).value is (
                change == "replacement"
            )
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("changes_during_load", [False, True])
async def test_default_factory_and_voice_worker_wait_for_current_readiness(
    monkeypatch, changes_during_load
):
    monkeypatch.setattr(hands_free_module, "speculative_voice_qualified", lambda: True)
    app, host = _ready_host()
    monkeypatch.delattr(app, "console_speculative_voice_session_factory", raising=False)
    loaded, release = threading.Event(), threading.Event()
    starts = []

    def load():
        loaded.set()
        release.wait(2)
        return lambda **kw: starts.append("factory") or _QualifiedSession()

    monkeypatch.setattr(
        hands_free_module, "_load_default_speculative_voice_factory", load
    )
    try:
        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            controller = console._ensure_console_chat_controller()

            async def resolve(selection):
                return SimpleNamespace(ready=changes_during_load)

            monkeypatch.setattr(
                controller.provider_gateway, "resolve_for_send", resolve
            )
            monkeypatch.setattr(
                type(console._console_runtime()),
                "voice_worker",
                property(lambda self: starts.append("worker")),
            )
            await pilot.click("#console-hands-free-switch")
            if changes_during_load:
                assert await asyncio.to_thread(loaded.wait, 1)
                session_id = controller.store.active_session_id
                settings = controller.store.session_settings(session_id)
                controller.store.replace_session_settings(
                    session_id, replace(settings, model="changed")
                )
                release.set()
            async with asyncio.timeout(2):
                while (
                    console._hands_free._qualified_voice_startup_generation is not None
                ):
                    await pilot.pause()
            assert starts == []
            assert loaded.is_set() is changes_during_load
            assert not console.query_one("#console-hands-free-switch", Switch).value
    finally:
        release.set()


@pytest.mark.asyncio
async def test_exception_in_final_readiness_check_closes_unentered_engine(monkeypatch):
    monkeypatch.setattr(hands_free_module, "speculative_voice_qualified", lambda: True)
    app, host = _ready_host()
    engine, notices = _QualifiedSession(), []
    app.notify = lambda message, **kw: notices.append(message)
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()

        async def resolve(selection):
            return SimpleNamespace(ready=True)

        def fail_selection(session_id):
            raise RuntimeError(SECRET)

        def factory(**kwargs):
            monkeypatch.setattr(
                controller, "_provider_selection_for_session", fail_selection
            )
            return engine

        monkeypatch.setattr(controller.provider_gateway, "resolve_for_send", resolve)
        app.console_speculative_voice_session_factory = factory
        await pilot.click("#console-hands-free-switch")
        await pilot.pause()
        assert not console.query_one("#console-hands-free-switch", Switch).value
        assert engine.entered_with == []
        assert len(engine.close_reasons) == 1
        assert len(notices) == 1 and SECRET not in notices[0]
